"""GPlaner in Equinox: the goal-conditioned autoregressive Gaussian policy over action plans.

Same architecture and parameter layout as `gplan.policy.GPlaner` (PyTorch), so weights convert in
both directions (`gplan_jax.convert`). Everything here works on ONE plan; batch with `jax.vmap`.

A plan is a flat buffer of `horizon = n_steps * action_dim` values, step-major. Step t is predicted
from (z_start, z_goal, buffer with steps >= t zeroed).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp


class GPlaner(eqx.Module):
    in_proj: eqx.nn.Linear
    backbone: list[eqx.nn.Linear]
    out_proj: eqx.nn.Linear
    Z: list[eqx.nn.Linear]  # log Z(start, goal): Linear -> ReLU -> Linear, 64 outputs summed
    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    def __init__(self, state_dim=2, horizon=8, hidden_size=64, n_layers=2, log_std_min=-5.0, log_std_max=2.0,
                 action_dim=1, *, key):
        assert horizon % action_dim == 0, f"horizon {horizon} must be a multiple of action_dim {action_dim}"
        width = 3 * hidden_size
        keys = jax.random.split(key, n_layers + 4)
        self.in_proj = eqx.nn.Linear(2 * state_dim + horizon, width, key=keys[0])
        self.backbone = [eqx.nn.Linear(width, width, key=k) for k in keys[1:n_layers + 1]]
        self.out_proj = eqx.nn.Linear(width, 2 * action_dim, key=keys[n_layers + 1])
        self.Z = [eqx.nn.Linear(2 * state_dim, state_dim, key=keys[n_layers + 2]),
                  eqx.nn.Linear(state_dim, 64, key=keys[n_layers + 3])]
        self.horizon = horizon
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    @property
    def n_steps(self):
        return self.horizon // self.action_dim

    def log_Z(self, z_start, z_goal):
        """Log-partition function log Z(start, goal), a scalar."""
        h = jax.nn.relu(self.Z[0](jnp.concatenate([z_start, z_goal])))
        return jnp.sum(self.Z[1](h))

    def __call__(self, z_start, z_goal, actions):
        """Mean and variance (each (action_dim,)) of the next step, given the masked buffer (horizon,)."""
        h_in = jax.nn.gelu(self.in_proj(jnp.concatenate([z_start, z_goal, actions])), approximate=False)
        h = h_in
        for layer in self.backbone:
            h = jax.nn.gelu(layer(h), approximate=False) + h_in  # every layer is skip-connected to the input projection
        mean, raw_std = jnp.split(self.out_proj(h), 2)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(raw_std) + 1.0)
        return mean, jnp.exp(2.0 * log_std)


def normal_log_prob(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - 0.5 * math.log(2 * math.pi)


def step_dist(model, z_start, z_goal, actions, step):
    """Mean and std of step `step`; buffer slots from that step on are zeroed before the forward pass."""
    visible = jnp.arange(model.horizon) < step * model.action_dim
    mean, var = model(z_start, z_goal, jnp.where(visible, actions, 0.0))
    return mean, jnp.sqrt(var)


def rollout(model, key, z_start, z_goal, mode="sample"):
    """Sample one plan (horizon,) step by step. mode="mean" writes the predicted mean instead."""
    A = model.action_dim
    actions = jnp.zeros(model.horizon)
    for t, k in enumerate(jax.random.split(key, model.n_steps)):
        mean, std = step_dist(model, z_start, z_goal, actions, t)
        a = mean if mode == "mean" else mean + std * jax.random.normal(k, mean.shape)
        actions = actions.at[t * A:(t + 1) * A].set(a)
    return actions


def step_log_probs(model, z_start, z_goal, actions):
    """Teacher-forced log P_F of every step of a given plan (any plan, not only the model's samples).

    All n_steps masked buffers go through the model in one vmapped call.
    Returns (log_probs (n_steps,), means (n_steps, A), stds (n_steps, A)).
    """
    n, A = model.n_steps, model.action_dim
    visible = jnp.arange(model.horizon)[None, :] < (jnp.arange(n) * A)[:, None]  # (n_steps, horizon)
    means, variances = jax.vmap(lambda buf: model(z_start, z_goal, buf))(jnp.where(visible, actions, 0.0))
    stds = jnp.sqrt(variances)
    log_probs = normal_log_prob(actions.reshape(n, A), means, stds).sum(-1)
    return log_probs, means, stds


def reference_log_prob(actions, reference_var):
    """log N(actions; 0, s^2 I) summed over the buffer."""
    return normal_log_prob(actions, 0.0, jnp.sqrt(reference_var)).sum(-1)
