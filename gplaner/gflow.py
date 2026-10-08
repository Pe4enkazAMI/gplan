"""Goal-conditioned autoregressive action planner and its rollout sampler."""

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
import math
import torch.nn.functional as F

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

        # self.start_proj = nn.Linear(state_dim, hidden_size)
        # self.goal_proj = nn.Linear(state_dim, hidden_size)
        # self.action_proj = nn.Sequential(nn.Linear(horizon, hidden_size), nn.ReLU())

        self.in_proj = nn.Sequential(nn.Linear(2 * state_dim + horizon, 3 * hidden_size), nn.GELU())

        self.backbone = nn.ModuleList([nn.Linear(3 * hidden_size, 3 * hidden_size) for _ in range(n_layers)])

        self.out_proj = nn.Linear(3 * hidden_size, 2 * action_dim)
        
        # head_in = 3 * hidden_size if self.mixer == "concat" else hidden_size
        # self.mean_head = nn.Linear(head_in, action_dim)
        # self.log_std_head = nn.Linear(head_in, action_dim)  # pre-tanh log-std
        self.Z = nn.Sequential(*[nn.Linear(2 * state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, 64)]) # log-partition function log Z(start, goal)

    def squash_log_std(self, x):
        """Map an unbounded head output to log_std in [log_std_min, log_std_max] via tanh."""
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(x) + 1.0)

    def log_Z(self, start_state, goal_state):
        """Log-partition function of the GFlowNet, conditioned on (start, goal). Shape (B,)."""
        z_stacked = torch.cat([start_state, goal_state], dim=-1)
        return self.Z(z_stacked).sum(-1)

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
        z_stacked = torch.cat([start_state, goal_state, actions], dim=-1)
        h_stacked = self.in_proj(z_stacked)
        hnew = h_stacked
        for layer in self.backbone:
            hnew = F.gelu(layer(hnew)) + h_stacked

        out = self.out_proj(hnew)
        mean, raw_std = out.chunk(2, -1)
        log_std = self.squash_log_std(raw_std)
        return mean, torch.exp(2.0 * log_std)


class Sampler:
    """Autoregressively fills a fixed-length action buffer using a ``GPlaner``.

    The buffer has shape (B, horizon) = (B, n_steps * action_dim). At step ``t``
    the model sees the buffer with steps ``0..t-1`` (slots ``0..t*action_dim-1``)
    filled and everything after zeroed, predicts a diagonal Gaussian over the
    ``action_dim`` values of step ``t``, and the sample is written to those slots.
    """

    def __init__(self, model: GPlaner, reference_var=1) -> None:
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
        masked_actions = actions * visible.to(actions.dtype)
        mean, var = self.model(start_state, goal_state, masked_actions)
        return Independent(Normal(mean, var.sqrt()), 1)  # std already bounded away from 0 by the tanh parametrization

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
        """Log-probability of each sampled step under the current model.

        Uses teacher forcing: step ``t`` is scored against the Gaussian the
        model predicts from the true steps ``0..t-1`` (later steps masked),
        exactly as the buffer looked when step ``t`` was sampled in ``rollout``.
        The result is differentiable w.r.t. the model parameters.

        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) fully filled buffer, e.g. from ``rollout``.
            n_given:     number of leading steps that were given rather than
                         sampled; these are not scored.

        Returns:
            (B, n_steps - n_given) per-step log-probabilities (each already summed
            over the step's ``action_dim`` components). Sum over the last dimension
            to get the log-probability of the whole plan.
        """
        log_probs = []
        for t in range(n_given, self.n_steps):
            dist = self.action_dist(start_state, goal_state, actions, step=t)
            log_probs.append(dist.log_prob(actions[:, self._slots(t)]))
        return torch.stack(log_probs, dim=-1)

    def reference_log_prob(self, actions):
        """
        Compute log-probability of `actions` under N(0, sigma^2 I) reference measure.

        Args:
            actions: (B, N) tensor of action samples (buffer of shape (batch, horizon))

        Returns:
            (B,) tensor: log-probability of the whole trajectory under the reference.
        """
        sigma = torch.tensor(self.reference_var, device=actions.device).sqrt()
        ref_dist = Normal(
            loc=torch.zeros_like(actions), 
            scale=sigma
        )
        # log_prob returns (B, N); sum over the last dimension
        return ref_dist.log_prob(actions).sum(dim=-1)
 
