"""Goal-conditioned autoregressive action planner and its rollout sampler."""

import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import torch.nn.functional as F

class GPlaner(nn.Module):
    """Predicts a Gaussian over the next scalar action.

    The network is conditioned on the start state, the goal state and the whole
    fixed-length action buffer of size ``horizon``. Slots of the buffer that
    have not been filled yet are zero. It outputs the mean and variance of the
    action for the next unfilled slot.

    The std is parametrized through a tanh squash of the head output,

        log_std = log_std_min + (log_std_max - log_std_min) * (tanh(x) + 1) / 2,

    so it is bounded in ``[exp(log_std_min), exp(log_std_max)]`` and can neither
    collapse to zero nor blow up (as in SAC-style policies).
    """

    def __init__(self, state_dim=2, horizon=8, hidden_size=64, n_layers=2, mixer="concat",
                 log_std_min=-5.0, log_std_max=2.0) -> None:
        super().__init__()
        self.horizon = horizon
        self.mixer = mixer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.start_proj = nn.Linear(state_dim, hidden_size)
        self.goal_proj = nn.Linear(state_dim, hidden_size)
        self.action_proj = nn.Sequential(nn.Linear(horizon, hidden_size), nn.ReLU())

        if self.mixer == "concat":
            self.backbone = nn.ModuleList([nn.Linear(3 * hidden_size, 3 * hidden_size)] * n_layers)
        else: 
            self.backbone = nn.ModuleList([nn.Linear(hidden_size, hidden_size) * n_layers])
        
        head_in = 3 * hidden_size if self.mixer == "concat" else hidden_size
        self.mean_head = nn.Linear(head_in, 1)
        self.log_std_head = nn.Linear(head_in, 1)  # pre-tanh log-std
        self.Z = nn.Linear(state_dim, 1)  # log-partition function log Z(start, goal)

    def squash_log_std(self, x):
        """Map an unbounded head output to log_std in [log_std_min, log_std_max] via tanh."""
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(x) + 1.0)

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
                       var = exp(2 * log_std) with log_std tanh-bounded, so it is
                       always strictly positive and finite.
        """
        if self.mixer == "concat":
            z_act = self.action_proj(actions)
            z_goal = self.goal_proj(goal_state)
            z_start = self.start_proj(start_state)
            h = torch.cat([z_act, z_goal, z_start], dim=-1)
        else:
            h = (
                self.action_proj(actions)
                + self.start_proj(start_state)
                + self.goal_proj(goal_state)
            )
        hnew = h
        for layer in self.backbone:
            hnew = F.relu(layer(hnew)) + h
        mean = self.mean_head(hnew).squeeze(-1)
        log_std = self.squash_log_std(self.log_std_head(hnew).squeeze(-1))
        return mean, torch.exp(2.0 * log_std)


class Sampler:
    """Autoregressively fills a fixed-length action buffer using a ``GPlaner``.

    The buffer has shape (B, horizon). At step ``t`` the model sees the buffer
    with slots ``0..t-1`` filled and slots ``t..horizon-1`` zeroed, predicts a
    Gaussian over action ``t``, and the sampled value is written to slot ``t``.
    """

    def __init__(self, model: GPlaner, reference_var=1) -> None:
        self.model = model
        self.reference_var = reference_var

    def action_dist(self, start_state, goal_state, actions, step) -> Normal:
        """Gaussian over the action at slot ``step``, given the slots before it.

        Slots at index ``>= step`` are zeroed before being fed to the model, so
        the model never sees the action it is asked to predict or anything after.
        """
        visible = torch.arange(actions.shape[1], device=actions.device) < step
        masked_actions = actions * visible.to(actions.dtype)
        mean, var = self.model(start_state, goal_state, masked_actions)
        return Normal(mean, var.sqrt())  # std already bounded away from 0 by the tanh parametrization

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
 
