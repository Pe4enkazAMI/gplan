"""Goal-conditioned autoregressive action planner and its rollout sampler."""

import torch
import torch.nn as nn
from torch.distributions import Normal


class GPlaner(nn.Module):
    """Predicts a Gaussian over the next scalar action.

    The network is conditioned on the start state, the goal state and the whole
    fixed-length action buffer of size ``horizon``. Slots of the buffer that
    have not been filled yet are zero. It outputs the mean and variance of the
    action for the next unfilled slot.
    """

    def __init__(self, state_dim=2, horizon=8, hidden_size=64, n_layers=2) -> None:
        super().__init__()
        self.horizon = horizon

        self.start_proj = nn.Linear(state_dim, hidden_size)
        self.goal_proj = nn.Linear(state_dim, hidden_size)
        self.action_proj = nn.Sequential(nn.Linear(horizon, hidden_size), nn.ReLU())

        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_size, 1)
        self.var_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())
        self.Z = nn.Linear(state_dim, 1)  # log-partition function log Z(start, goal)

    def log_Z(self, start_state, goal_state):
        """Log-partition function of the GFlowNet, conditioned on (start, goal). Shape (B,)."""
        return self.Z(start_state + goal_state).squeeze(-1)

    def forward(self, start_state, goal_state, actions):
        """
        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) action buffer, zeros in unfilled slots.

        Returns:
            mean, var: predicted Gaussian parameters, each of shape (B,).
        """
        h = (
            self.action_proj(actions)
            + self.start_proj(start_state)
            + self.goal_proj(goal_state)
        )
        h = self.backbone(h)
        mean = self.mean_head(h).squeeze(-1)
        var = self.var_head(h).squeeze(-1)
        return mean, var


class Sampler:
    """Autoregressively fills a fixed-length action buffer using a ``GPlaner``.

    The buffer has shape (B, horizon). At step ``t`` the model sees the buffer
    with slots ``0..t-1`` filled and slots ``t..horizon-1`` zeroed, predicts a
    Gaussian over action ``t``, and the sampled value is written to slot ``t``.
    """

    def __init__(self, model: GPlaner, min_var: float = 1e-6) -> None:
        self.model = model
        self.min_var = min_var  # keeps the std strictly positive

    def action_dist(self, start_state, goal_state, actions, step) -> Normal:
        """Gaussian over the action at slot ``step``, given the slots before it.

        Slots at index ``>= step`` are zeroed before being fed to the model, so
        the model never sees the action it is asked to predict or anything after.
        """
        visible = torch.arange(actions.shape[1], device=actions.device) < step
        masked_actions = actions * visible.to(actions.dtype)
        mean, var = self.model(start_state, goal_state, masked_actions)
        return Normal(mean, (var + self.min_var).sqrt())

    @torch.no_grad()
    def rollout(self, start_state, goal_state, actions=None, n_given=0):
        """Fill slots ``n_given..horizon-1`` of the action buffer by sampling.

        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) buffer whose first ``n_given`` slots are
                         already filled. Defaults to an all-zero buffer.
            n_given:     number of leading slots to keep as they are.

        Returns:
            (B, horizon) buffer with every slot filled. The input is not modified.
        """
        if actions is None:
            actions = start_state.new_zeros(start_state.shape[0], self.model.horizon)
        actions = actions.clone()

        for t in range(n_given, self.model.horizon):
            dist = self.action_dist(start_state, goal_state, actions, step=t)
            actions[:, t] = dist.sample()
        return actions

    def log_prob(self, start_state, goal_state, actions, n_given=0):
        """Log-probability of each sampled slot under the current model.

        Uses teacher forcing: slot ``t`` is scored against the Gaussian the
        model predicts from the true slots ``0..t-1`` (later slots masked),
        exactly as the buffer looked when slot ``t`` was sampled in ``rollout``.
        The result is differentiable w.r.t. the model parameters.

        Args:
            start_state: (B, state_dim)
            goal_state:  (B, state_dim)
            actions:     (B, horizon) fully filled buffer, e.g. from ``rollout``.
            n_given:     number of leading slots that were given rather than
                         sampled; these are not scored.

        Returns:
            (B, horizon - n_given) per-slot log-probabilities. Sum over the last
            dimension to get the log-probability of the whole trajectory.
        """
        log_probs = []
        for t in range(n_given, self.model.horizon):
            dist = self.action_dist(start_state, goal_state, actions, step=t)
            log_probs.append(dist.log_prob(actions[:, t]))
        return torch.stack(log_probs, dim=-1)
