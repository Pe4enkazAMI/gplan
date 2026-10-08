"""Goal-conditioned autoregressive action planner (GPlaner), its sampler, and checkpoint I/O."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


class GPlaner(nn.Module):
    """Predicts a diagonal Gaussian over the next world-model action.

    The network is conditioned on the start state, the goal state and the whole
    fixed-length action buffer of size ``horizon`` (= n_steps * action_dim, laid
    out step-major so ``buffer.view(B, n_steps, action_dim)`` is the plan the world
    model consumes). Slots of the buffer that have not been filled yet are zero.
    It outputs the mean and variance of the ``action_dim`` values of the next
    unfilled step, so the plan is sampled in ``n_steps`` autoregressive steps
    rather than ``horizon`` scalar ones. ``action_dim=1`` recovers the scalar case.

    The std is parametrized through a tanh squash of the head output,

        log_std = log_std_min + (log_std_max - log_std_min) * (tanh(x) + 1) / 2,

    so it is bounded in ``[exp(log_std_min), exp(log_std_max)]`` and can neither
    collapse to zero nor blow up (as in SAC-style policies).
    """

    def __init__(self, state_dim=2, horizon=8, hidden_size=64, n_layers=2,
                 log_std_min=-5.0, log_std_max=2.0, action_dim=1) -> None:
        super().__init__()
        assert horizon % action_dim == 0, f"horizon {horizon} must be a multiple of action_dim {action_dim}"
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_steps = horizon // action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        width = 3 * hidden_size
        self.in_proj = nn.Sequential(nn.Linear(2 * state_dim + horizon, width), nn.GELU())
        self.backbone = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.out_proj = nn.Linear(width, 2 * action_dim)
        # log-partition function log Z(start, goal); the 64 outputs are summed into one scalar
        self.Z = nn.Sequential(nn.Linear(2 * state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, 64))

    def squash_log_std(self, x):
        """Map an unbounded head output to log_std in [log_std_min, log_std_max] via tanh."""
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(x) + 1.0)

    def log_Z(self, start_state, goal_state):
        """Log-partition function of the GFlowNet, conditioned on (start, goal). Shape (B,)."""
        return self.Z(torch.cat([start_state, goal_state], dim=-1)).sum(-1)

    def forward(self, start_state, goal_state, actions):
        """
        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) action buffer, zeros in unfilled slots.

        Returns:
            mean, var: diagonal Gaussian parameters of the next step, each (B, action_dim).
                       var = exp(2 * log_std) with log_std tanh-bounded, so it is
                       always strictly positive and finite.
        """
        h_in = self.in_proj(torch.cat([start_state, goal_state, actions], dim=-1))
        h = h_in
        for layer in self.backbone:
            h = F.gelu(layer(h)) + h_in  # every layer is skip-connected to the input projection
        mean, raw_std = self.out_proj(h).chunk(2, -1)
        return mean, torch.exp(2.0 * self.squash_log_std(raw_std))


class Sampler:
    """Autoregressively fills a fixed-length action buffer using a ``GPlaner``.

    The buffer has shape (B, horizon) = (B, n_steps * action_dim). At step ``t``
    the model sees the buffer with steps ``0..t-1`` (slots ``0..t*action_dim-1``)
    filled and everything after zeroed, predicts a diagonal Gaussian over the
    ``action_dim`` values of step ``t``, and the sample is written to those slots.

    ``reference_var`` is the variance s^2 of the N(0, s^2 I) reference measure
    (the prior over action buffers in the target distribution).
    """

    def __init__(self, model: GPlaner, reference_var=1.0) -> None:
        self.model = model
        self.reference_var = reference_var

    @property
    def n_steps(self):
        return self.model.n_steps

    @property
    def action_dim(self):
        return self.model.action_dim

    def _slots(self, step):
        """Slice of the buffer holding step ``step``."""
        return slice(step * self.action_dim, (step + 1) * self.action_dim)

    def action_dist(self, start_state, goal_state, actions, step) -> Independent:
        """Diagonal Gaussian over the action at step ``step``, given the steps before it.

        Slots at index ``>= step * action_dim`` are zeroed before being fed to the
        model, so it never sees the action it is asked to predict or anything after.
        The returned distribution has event shape (action_dim,): ``sample()`` is
        (B, action_dim) and ``log_prob(x)`` is (B,).
        """
        visible = torch.arange(actions.shape[1], device=actions.device) < step * self.action_dim
        mean, var = self.model(start_state, goal_state, actions * visible.to(actions.dtype))
        return Independent(Normal(mean, var.sqrt()), 1)

    @torch.no_grad()
    def rollout(self, start_state, goal_state, actions=None, n_given=0, mode="sample"):
        """Fill steps ``n_given..n_steps-1`` of the action buffer.

        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) buffer whose first ``n_given`` steps are
                         already filled. Defaults to an all-zero buffer.
            n_given:     number of leading steps to keep as they are.
            mode:        "sample" draws from the policy; "mean" writes the
                         predicted mean at every step (deterministic plan).

        Returns:
            (B, horizon) buffer with every slot filled. The input is not modified.
        """
        assert mode in ("sample", "mean"), mode
        if actions is None:
            actions = start_state.new_zeros(start_state.shape[0], self.model.horizon)
        actions = actions.clone()
        for t in range(n_given, self.n_steps):
            dist = self.action_dist(start_state, goal_state, actions, step=t)
            actions[:, self._slots(t)] = dist.sample() if mode == "sample" else dist.mean
        return actions

    def log_prob(self, start_state, goal_state, actions, n_given=0):
        """Log-probability of each step of ``actions`` under the current model.

        Uses teacher forcing: step ``t`` is scored against the Gaussian the model
        predicts from the true steps ``0..t-1`` (later steps masked), exactly as
        the buffer looked when step ``t`` was sampled in ``rollout``. Works for any
        buffer, not only the model's own samples. Differentiable w.r.t. the model.

        Args:
            actions: (B, horizon) fully filled buffer.
            n_given: number of leading steps that were given rather than sampled; not scored.

        Returns:
            (B, n_steps - n_given) per-step log-probabilities (each summed over the
            step's ``action_dim`` components). Sum over the last dim for the whole plan.
        """
        log_probs = []
        for t in range(n_given, self.n_steps):
            dist = self.action_dist(start_state, goal_state, actions, step=t)
            log_probs.append(dist.log_prob(actions[:, self._slots(t)]))
        return torch.stack(log_probs, dim=-1)

    def reference_log_prob(self, actions):
        """log N(actions; 0, s^2 I) summed over the buffer. actions: (B, horizon) -> (B,)."""
        sigma = torch.tensor(self.reference_var, device=actions.device).sqrt()
        return Normal(torch.zeros_like(actions), sigma).log_prob(actions).sum(dim=-1)


# ----------------------------------------------------------------------------- checkpoints

def save_planner(path, model, model_kwargs, train_args=None):
    """Self-describing checkpoint: the state_dict plus everything needed to rebuild the model."""
    torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "train_args": train_args}, path)


def planner_kwargs_from_state_dict(sd):
    """Recover `GPlaner(...)` constructor kwargs from a bare state_dict.
    Bounds of the log-std squash are not in the state_dict and are assumed to be the defaults."""
    width, in_dim = sd["in_proj.0.weight"].shape            # (3 * hidden, 2 * state_dim + horizon)
    state_dim = sd["Z.0.weight"].shape[1] // 2               # Z: Linear(2 * state_dim, state_dim)
    action_dim = sd["out_proj.weight"].shape[0] // 2         # (2 * action_dim, 3 * hidden)
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("backbone.")})
    return dict(state_dim=state_dim, horizon=in_dim - 2 * state_dim, hidden_size=width // 3,
                n_layers=n_layers, action_dim=action_dim)


def load_planner(path, device="cpu"):
    """Build a frozen `GPlaner` from a checkpoint (self-describing or bare state_dict).

    Returns (model, train_args or None).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        kwargs, sd, train_args = ckpt["model_kwargs"], ckpt["state_dict"], ckpt.get("train_args")
    else:
        kwargs, sd, train_args = planner_kwargs_from_state_dict(ckpt), ckpt, None
    model = GPlaner(**kwargs).to(device)
    model.load_state_dict(sd)
    model.eval().requires_grad_(False)
    print(f"Loaded planner {path}: GPlaner({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    return model, train_args
