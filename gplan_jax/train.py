"""Optimizer, jitted train step and training loop."""

import time
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb

from gplan_jax.losses import tb_loss, vargrad_loss
from gplan_jax.policy import rollout


@dataclass(frozen=True)
class TrainConfig:
    """Static settings of the train step (changing one recompiles it)."""
    loss_type: str = "tb"        # "tb" (learned log Z) or "vargrad" (in-batch log Z, needs n_samples >= 2)
    n_samples: int = 1           # K plans per (start, goal); the effective batch is B * K
    action_dim: int = 1          # world-model action dim per plan step (frameskip * env action dim)
    reference_var: float = 1.0   # s^2 of the N(0, s^2 I) reference measure
    grad_clip: float = 0.0       # max global grad norm; <= 0 disables clipping


def linear_beta_schedule(beta_end, beta_start=0.0, warmup_steps=0):
    """`step -> beta`, linear from `beta_start` to `beta_end` over `warmup_steps`, then constant."""
    if warmup_steps <= 0:
        return lambda step: beta_end
    return lambda step: beta_start + (beta_end - beta_start) * min(1.0, step / warmup_steps)


def make_optimizer(model, lr, lr_z=None, grad_clip=0.0):
    """Global-norm clipping, then Adam on the policy and Adam at `lr_z` on the log Z head (`model.Z`)."""
    params = eqx.filter(model, eqx.is_array)
    labels = jax.tree.map(lambda _: "policy", params)
    labels = eqx.tree_at(lambda m: m.Z, labels, replace=jax.tree.map(lambda _: "z", params.Z))
    clip = optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity()
    # labels is itself a GPlaner (callable), so hand it to optax through a function
    adam = optax.multi_transform({"policy": optax.adam(lr), "z": optax.adam(lr if lr_z is None else lr_z)},
                                 lambda _: labels)
    return optax.chain(clip, adam)


def norm_stats(model, grads, grad_clip):
    """Per-submodule and global L2 norms of gradients (before clipping) and parameters."""
    stats = {}
    for name in ("in_proj", "backbone", "out_proj", "Z"):
        stats[f"grad_norm/{name}"] = optax.tree.norm(eqx.filter(getattr(grads, name), eqx.is_array))
        stats[f"param_norm/{name}"] = optax.tree.norm(eqx.filter(getattr(model, name), eqx.is_array))
    raw = optax.tree.norm(eqx.filter(grads, eqx.is_array))
    stats["grad_norm/global_raw"] = raw
    stats["param_norm/global"] = optax.tree.norm(eqx.filter(model, eqx.is_array))
    if grad_clip > 0:
        scale = jnp.minimum(1.0, grad_clip / (raw + 1e-12))
        stats["grad_norm/clipped"] = (raw > grad_clip).astype(jnp.float32)
        stats["grad_norm/clip_scale"] = scale
        stats["grad_norm/global"] = raw * scale  # the norm the optimizer actually applies
    else:
        stats["grad_norm/global"] = raw
    return stats


def make_train_step(optimizer, cfg: TrainConfig):
    """Build the jitted step: sample K plans per condition, score them, take one optimizer step.

    `cost_model` is anything with `.cost(z_start, z_goal, plans) -> (N,)`, e.g. `LeWMPredictor`.
    """
    loss_fn = tb_loss if cfg.loss_type == "tb" else partial(vargrad_loss, n_samples=cfg.n_samples)

    @eqx.filter_jit
    def train_step(model, opt_state, cost_model, key, z_start, z_goal, beta):
        z_start = jnp.repeat(z_start, cfg.n_samples, axis=0)  # rows i*K .. i*K+K-1 share condition i
        z_goal = jnp.repeat(z_goal, cfg.n_samples, axis=0)
        n = z_start.shape[0]

        # on-policy plans and their cost; neither is differentiated (only `model` in loss_fn is)
        actions = jax.vmap(partial(rollout, model))(jax.random.split(key, n), z_start, z_goal)
        cost = cost_model.cost(z_start, z_goal, actions.reshape(n, -1, cfg.action_dim))

        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, z_start, z_goal, actions, cost, beta, cfg.reference_var)
        stats.update(norm_stats(model, grads, cfg.grad_clip))
        stats.update({"actions/mean": actions.mean(), "actions/std": jnp.std(actions, ddof=1),
                      "actions/abs_max": jnp.abs(actions).max()})

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, stats

    return train_step


def train(model, cost_model, batches, beta, cfg: TrainConfig, lr=1e-3, lr_z=None, seed=0, log_every=10):
    """One optimizer step per (z_start, z_goal) batch. Returns (trained model, list of losses).

    `beta` is a float or a callable `step -> beta`. Every step is logged to wandb when a run is
    active; a console summary is printed every `log_every` steps. Step 0 includes compilation time.
    """
    beta_fn = beta if callable(beta) else (lambda step: beta)
    optimizer = make_optimizer(model, lr, lr_z, cfg.grad_clip)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(optimizer, cfg)
    key = jax.random.key(seed)

    losses = []
    t_prev = time.perf_counter()
    for step, (z_start, z_goal) in enumerate(batches):
        key, step_key = jax.random.split(key)
        beta_t = beta_fn(step)
        model, opt_state, loss, stats = train_step(model, opt_state, cost_model, step_key, z_start, z_goal,
                                                   jnp.float32(beta_t))
        stats = {k: float(v) for k, v in jax.device_get(stats).items()}  # one host transfer per step
        losses.append(float(loss))

        t_now = time.perf_counter()
        stats.update({"loss/total": losses[-1], "tb/beta": beta_t, "optim/lr": lr,
                      "optim/lr_z": lr if lr_z is None else lr_z, "time/step_s": t_now - t_prev,
                      "time/samples_per_s": len(z_start) * cfg.n_samples / (t_now - t_prev)})
        t_prev = t_now
        if wandb.run is not None:
            wandb.log(stats, step=step)
        if step % log_every == 0:
            print(f"step {step:5d}  loss {losses[-1]:.4f}  residual_std {stats['tb/residual_std']:.3f}  "
                  f"beta {beta_t:.3f}  cost {stats['cost/mean']:.3f}  log_z {stats['tb/log_z']:+.3f}  "
                  f"beta_cost {stats['cost/beta_cost']:.3f} "
                  f"grad_norm {stats['grad_norm/global_raw']:.3e} -> {stats['grad_norm/global']:.3e}")
    return model, losses
