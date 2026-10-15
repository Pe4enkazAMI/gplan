"""Baselines trained with the same GPlaner, data and optimizer as the GFlowNet, but other objectives.

    bc   behavior cloning: maximize log P_F of the dataset's expert plan between start and goal
         (needs action labels, which the GFlowNet does not use)
    map  amortized MAP planning: the policy's deterministic plan A(c) (most likely component at every
         step) minimizes beta * J(A) + ||A||^2 / (2 s^2), the -log of the GFlowNet target, by
         backpropagating through LeWM. One plan per condition, no distribution: evaluate it with
         `scripts/evaluate.py --gflow-select mean`.

Both are trained by `train_baseline`, which mirrors `gplan.jax.train.train` (same optimizer and
logging). Both objectives are deterministic given the batches, so no random key is involved.
"""

import time
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import wandb

from gplan.jax.policy import rollout, step_log_probs
from gplan.jax.train import make_optimizer, norm_stats


def bc_loss(model, z_start, z_goal, plans):
    """Negative log-likelihood of expert plans. plans: (B, n_steps, A) -> (loss, stats)."""
    flat = plans.reshape(plans.shape[0], -1)
    log_pf_steps, dists = jax.vmap(partial(step_log_probs, model))(z_start, z_goal, flat)
    log_pf = log_pf_steps.sum(-1)
    stats = {"bc/log_pf": log_pf.mean(), "bc/log_pf_per_step": log_pf_steps.mean(),
             "policy/std": dists.stds.mean(), "policy/weight_max": dists.weights.max(-1).mean()}
    return -log_pf.mean(), stats


def map_loss(model, z_start, z_goal, cost_model, beta, reference_var, action_dim):
    """beta * J + ||A||^2 / (2 s^2) of the policy's deterministic plans, differentiated through the
    autoregressive rollout and LeWM. -> (loss, stats)."""
    plans = jax.vmap(partial(rollout, model, jax.random.key(0), mode="mean"))(z_start, z_goal)
    cost = cost_model.cost(z_start, z_goal, plans.reshape(plans.shape[0], -1, action_dim))
    prior = jnp.sum(plans ** 2, axis=-1) / (2 * reference_var)
    stats = {"cost/mean": cost.mean(), "cost/min": cost.min(), "cost/beta_cost": (beta * cost).mean(),
             "map/prior": prior.mean(), "actions/abs_max": jnp.abs(plans).max()}
    return jnp.mean(beta * cost + prior), stats


def train_baseline(model, loss_fn, batches, lr=1e-3, grad_clip=0.0, log_every=10):
    """One optimizer step per batch of `loss_fn(model, *batch) -> (loss, stats)`.

    For bc, batches are (z_start, z_goal, expert plans); for map, (z_start, z_goal) with the cost model
    and beta bound into loss_fn (`partial(map_loss, cost_model=..., beta=..., ...)`). The log Z head
    is not used by either loss and stays at its initial weights. Returns (model, losses).
    """
    optimizer = make_optimizer(model, lr, grad_clip=grad_clip)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step_fn(model, opt_state, *batch):
        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, *batch)
        stats.update(norm_stats(model, grads, grad_clip))
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss, stats

    losses, t_prev = [], time.perf_counter()
    for step, batch in enumerate(batches):
        model, opt_state, loss, stats = step_fn(model, opt_state, *batch)
        stats = {k: float(v) for k, v in jax.device_get(stats).items()}
        losses.append(float(loss))
        t_now = time.perf_counter()
        stats.update({"loss/total": losses[-1], "optim/lr": lr, "time/step_s": t_now - t_prev})
        t_prev = t_now
        if wandb.run is not None:
            wandb.log(stats, step=step)
        if step % log_every == 0:
            print(f"step {step:5d}  loss {losses[-1]:.4f}  "
                  f"grad_norm {stats['grad_norm/global_raw']:.3e} -> {stats['grad_norm/global']:.3e}")
    return model, losses

