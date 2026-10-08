"""Optimization loop for the GPlaner sampler, plus training diagnostics."""

import time
from functools import partial

import torch
import wandb

from gplan.lewm import encode, lewm_cost
from gplan.losses import tb_loss, vargrad_loss


def linear_beta_schedule(beta_end, beta_start=0.0, warmup_steps=0):
    """Return `step -> beta`, linear from `beta_start` to `beta_end` over `warmup_steps`, then constant.

    Annealing beta up from ~0 starts training on the well-posed "match the prior"
    problem and gradually sharpens the target exp(-beta J) N(A), so log Z and the
    policy track a slowly moving target instead of chasing a far-away one from scratch.
    """
    if warmup_steps <= 0:
        return lambda step: beta_end
    return lambda step: beta_start + (beta_end - beta_start) * min(1.0, step / warmup_steps)


@torch.no_grad()
def policy_stats(sampler, z_start, z_goal, actions):
    """Mean / std of the Gaussians the policy predicted along the sampled buffer (teacher forced)."""
    means, stds = [], []
    for t in range(sampler.n_steps):
        dist = sampler.action_dist(z_start, z_goal, actions, step=t)
        means.append(dist.mean)      # (B, action_dim)
        stds.append(dist.stddev)
    means, stds = torch.stack(means, 1), torch.stack(stds, 1)  # (B, n_steps, action_dim)
    return {
        "policy/mean_abs": means.abs().mean().item(),
        "policy/std": stds.mean().item(),
        "policy/std_min": stds.min().item(),
        "policy/std_max": stds.max().item(),
        "policy/entropy": torch.distributions.Normal(means, stds).entropy().flatten(1).sum(-1).mean().item(),
    }


@torch.no_grad()
def grad_and_param_stats(model):
    """Global and per-submodule L2 norms of the gradients and the parameters.

    Must be called after `backward()` and before `zero_grad()`.
    """
    stats = {}
    grad_sq, param_sq = 0.0, 0.0
    for name, child in model.named_children():
        g = [p.grad.flatten() for p in child.parameters() if p.grad is not None]
        w = [p.detach().flatten() for p in child.parameters()]
        if not w:
            continue
        gn = torch.cat(g).norm().item() if g else 0.0
        pn = torch.cat(w).norm().item()
        stats[f"grad_norm/{name}"] = gn
        stats[f"param_norm/{name}"] = pn
        grad_sq += gn ** 2
        param_sq += pn ** 2
    stats["grad_norm/global"] = grad_sq ** 0.5
    stats["param_norm/global"] = param_sq ** 0.5
    return stats


def make_optimizer(model, lr, lr_z=None):
    """Adam with two groups: the policy, and the log Z head (`model.Z`) at its own learning rate.

    Returns (optimizer, all params in group order).
    """
    z_params = list(model.Z.parameters())
    z_ids = {id(p) for p in z_params}
    policy_params = [p for p in model.parameters() if id(p) not in z_ids]
    opt = torch.optim.Adam([
        {"params": policy_params, "lr": lr},
        {"params": z_params, "lr": lr if lr_z is None else lr_z},
    ])
    return opt, policy_params + z_params


def train(sampler, wm, batches, beta, action_dim, lr=1e-3, device="cpu", log_every=10, grad_clip=None, lr_z=None,
          loss_type="tb", n_samples=1, cost_fn=None):
    """Run one optimization step per (start_pixels, goal_pixels) batch.

    Args:
        beta:      cost temperature, either a float or a callable `step -> float`
                   (see `linear_beta_schedule`) for annealing.
        action_dim: world-model action dim per step (frameskip * env action dim).
        grad_clip: max global L2 norm of the gradient; `None` or <= 0 disables clipping.
        lr_z:      learning rate for the log Z head (`model.Z`). GFlowNets are much
                   more stable when log Z can move faster than the policy, since it
                   alone must absorb the constant offset of the TB residual.
                   Defaults to `lr`.
        loss_type: "tb" (learned log Z) or "vargrad" (in-batch log Z, needs n_samples >= 2).
        n_samples: K plans sampled per (start, goal) condition; the effective batch is B*K.
        cost_fn:   `(z_start, z_goal, plans) -> (N,)`; defaults to the LeWM cost of `wm`.

    Every step is logged to wandb when a run is active (`wandb.init` was called);
    otherwise only the console summary every `log_every` steps is printed.
    Returns the list of per-step losses.
    """
    assert loss_type in ("tb", "vargrad"), loss_type
    cost_fn = cost_fn or partial(lewm_cost, wm)
    opt, params = make_optimizer(sampler.model, lr, lr_z)
    beta_fn = beta if callable(beta) else (lambda step: beta)
    losses = []
    t_prev = time.perf_counter()
    for step, (start_pixels, goal_pixels) in enumerate(batches):
        # K plans per condition: rows i*K .. i*K+K-1 all share condition i
        z_start = encode(wm, start_pixels.to(device)).repeat_interleave(n_samples, 0)
        z_goal = encode(wm, goal_pixels.to(device)).repeat_interleave(n_samples, 0)

        beta_t = beta_fn(step)
        actions = sampler.rollout(z_start, z_goal)
        if loss_type == "vargrad":
            loss, stats = vargrad_loss(sampler, cost_fn, z_start, z_goal, actions, beta_t, action_dim, n_samples,
                                       return_stats=True)
        else:
            loss, stats = tb_loss(sampler, cost_fn, z_start, z_goal, actions, beta_t, action_dim, return_stats=True)
        stats.update({
            "tb/beta": beta_t,
            "actions/mean": actions.mean().item(),
            "actions/std": actions.std().item(),
            "actions/abs_max": actions.abs().max().item(),
        })

        opt.zero_grad()
        loss.backward()
        stats.update(grad_and_param_stats(sampler.model))  # raw (pre-clip) norms
        stats["grad_norm/global_raw"] = stats.pop("grad_norm/global")
        if grad_clip is not None and grad_clip > 0:
            pre = torch.nn.utils.clip_grad_norm_(params, grad_clip).item()
            stats["grad_norm/clipped"] = float(pre > grad_clip)         # fraction of clipped steps when averaged
            stats["grad_norm/clip_scale"] = min(1.0, grad_clip / (pre + 1e-12))  # 1 = untouched
        # norm of the gradient the optimizer actually applies (== raw when not clipped)
        stats["grad_norm/global"] = torch.cat([p.grad.flatten() for p in params if p.grad is not None]).norm().item()
        opt.step()

        losses.append(loss.item())
        t_now = time.perf_counter()
        stats.update({
            "loss/total": losses[-1],
            "optim/lr": opt.param_groups[0]["lr"],
            "optim/lr_z": opt.param_groups[1]["lr"],
            "time/step_s": t_now - t_prev,
            "time/samples_per_s": actions.shape[0] / (t_now - t_prev),
        })
        t_prev = t_now
        if wandb.run is not None:
            stats.update(policy_stats(sampler, z_start, z_goal, actions))
            wandb.log(stats, step=step)
        if step % log_every == 0:
            print(f"step {step:5d}  loss {losses[-1]:.4f}  residual_std {stats['tb/residual_std']:.3f}  "
                  f"beta {beta_t:.3f}  cost {stats['cost/mean']:.3f}  log_z {stats['tb/log_z']:+.3f}  "
                  f"beta_cost {stats['cost/beta_cost']:.3f} "
                  f"grad_norm {stats['grad_norm/global_raw']:.3e} -> {stats['grad_norm/global']:.3e}")
    return losses
