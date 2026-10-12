"""GPlaner in Equinox: the goal-conditioned autoregressive policy over action plans.

Same architecture and parameter layout as `gplan.torch.policy.GPlaner` (PyTorch), so weights convert in
both directions (`gplan.jax.convert`). Everything here works on ONE plan; batch with `jax.vmap`.

A plan is a flat buffer of `horizon = n_steps * action_dim` values, step-major. Step t is predicted
from (z_start, z_goal, buffer with steps >= t zeroed) as a mixture of `n_components` diagonal
Gaussians over the step's `action_dim` values (n_components = 1: a single diagonal Gaussian).
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from gplan.jax.distributions import DiagGaussianMixture


class GPlaner(eqx.Module):
    in_proj: eqx.nn.Linear
    backbone: list[eqx.nn.Linear]
    out_proj: eqx.nn.Linear
    Z: list[eqx.nn.Linear]  # log Z(start, goal): Linear -> ReLU -> Linear, 64 outputs summed
    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    n_components: int = eqx.field(static=True)

    def __init__(self, state_dim=2, horizon=8, hidden_size=64, n_layers=2, log_std_min=-5.0, log_std_max=2.0,
                 action_dim=1, n_components=1, *, key):
        assert horizon % action_dim == 0, f"horizon {horizon} must be a multiple of action_dim {action_dim}"
        width = 3 * hidden_size
        keys = jax.random.split(key, n_layers + 4)
        self.in_proj = eqx.nn.Linear(2 * state_dim + horizon, width, key=keys[0])
        self.backbone = [eqx.nn.Linear(width, width, key=k) for k in keys[1:n_layers + 1]]
        self.out_proj = eqx.nn.Linear(width, head_size(action_dim, n_components), key=keys[n_layers + 1])
        self.Z = [eqx.nn.Linear(2 * state_dim, state_dim, key=keys[n_layers + 2]),
                  eqx.nn.Linear(state_dim, 64, key=keys[n_layers + 3])]
        self.horizon = horizon
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.n_components = n_components

    @property
    def n_steps(self):
        return self.horizon // self.action_dim

    def log_Z(self, z_start, z_goal):
        """Log-partition function log Z(start, goal), a scalar."""
        h = jax.nn.relu(self.Z[0](jnp.concatenate([z_start, z_goal])))
        return jnp.sum(self.Z[1](h))

    def __call__(self, z_start, z_goal, actions) -> DiagGaussianMixture:
        """Distribution of the next step (a mixture of K = n_components diagonal Gaussians over
        action_dim values), given the masked buffer (horizon,). With K = 1 the log weight is 0."""
        h_in = jax.nn.gelu(self.in_proj(jnp.concatenate([z_start, z_goal, actions])), approximate=False)
        h = h_in
        for layer in self.backbone:
            h = jax.nn.gelu(layer(h), approximate=False) + h_in  # every layer is skip-connected to the input projection
        out = self.out_proj(h)
        K, A = self.n_components, self.action_dim
        means, raw_std = out[:K * A].reshape(K, A), out[K * A:2 * K * A].reshape(K, A)
        logits = out[2 * K * A:] if K > 1 else jnp.zeros(1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(raw_std) + 1.0)
        return DiagGaussianMixture.from_logits(logits, means, jnp.exp(log_std))


def head_size(action_dim, n_components):
    """Outputs of `out_proj`: means and raw stds of every component, plus K mixture logits if K > 1
    (so n_components = 1 keeps the plain Gaussian layout of older checkpoints)."""
    return 2 * action_dim if n_components == 1 else n_components * (2 * action_dim + 1)


def step_dist(model, z_start, z_goal, actions, step) -> DiagGaussianMixture:
    """Distribution of step `step`; buffer slots from that step on are zeroed first."""
    visible = jnp.arange(model.horizon) < step * model.action_dim
    return model(z_start, z_goal, jnp.where(visible, actions, 0.0))


def rollout(model, key, z_start, z_goal, mode="sample"):
    """Sample one plan (horizon,) step by step.

    mode="sample": draw every step from the policy.
    mode="mean":   write the mean of the most likely component at every step (deterministic plan;
                   the mixture's own mean could fall between modes).
    """
    A = model.action_dim
    actions = jnp.zeros(model.horizon)
    for t, k in enumerate(jax.random.split(key, model.n_steps)):
        dist = step_dist(model, z_start, z_goal, actions, t)
        actions = actions.at[t * A:(t + 1) * A].set(dist.mode() if mode == "mean" else dist.sample(k))
    return actions


def step_log_probs(model, z_start, z_goal, actions):
    """Teacher-forced log P_F of every step of a given plan (any plan, not only the model's samples).

    All n_steps masked buffers go through the model in one vmapped call.
    Returns (log_probs (n_steps,), the step distributions as one DiagGaussianMixture with a
    leading n_steps axis).
    """
    n, A = model.n_steps, model.action_dim
    visible = jnp.arange(model.horizon)[None, :] < (jnp.arange(n) * A)[:, None]  # (n_steps, horizon)
    dists = jax.vmap(lambda buf: model(z_start, z_goal, buf))(jnp.where(visible, actions, 0.0))
    return dists.log_prob(actions.reshape(n, A)), dists
