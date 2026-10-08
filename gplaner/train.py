"""Train the GPlaner sampler as a conditional GFlowNet over action plans.

Target:  P*(A | c)  ∝  exp(-beta * J(A, c)) * N(A; 0, s^2 I)

where c = (z_start, z_goal) are embeddings from a frozen LeWorldModel, J is the
LeWM planning cost (squared distance between the embedding the world model
predicts after executing A and the goal embedding; O(100s) on the raw 192-dim
latent, so beta is expected to be small) and N(.; 0, s^2 I) is a Gaussian prior
on action buffers (the reference measure).

Two losses, both built on the same per-plan quantity

    xi(A, c) = sum_t log P_F(a_t | a_<t, c) + beta * J(A, c) - log N(A)       (= -log Z(c) at optimum)

  tb:       ( log Z(c) + xi )^2                 with log Z(c) a learned head
  vargrad:  Var_k[ xi(A_k, c) ]  over K plans   sampled for the same c; log Z(c)
            is the in-batch mean -xi, so no partition function has to be learned.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the compression filter used by the LeWM .h5 files
import numpy as np
import torch
import wandb
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                    # gflow
sys.path.insert(0, str(ROOT.parent / "le-wm"))   # jepa / module, needed to unpickle the LeWM ckpt

from gflow import GPlaner, Sampler  # noqa: E402

STABLEWM_HOME = Path(os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def img_transform(img_size=224):
    """uint8 HWC frame -> normalized float CHW tensor, as LeWM was trained."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET),
        transforms.Resize(size=img_size),
    ])


def valid_start_rows(ep_idx, step_idx, goal_offset):
    """Dataset rows with at least `goal_offset` steps left in their episode (so row + goal_offset is the goal)."""
    ep_len = np.zeros(ep_idx.max() + 1, dtype=np.int64)
    np.maximum.at(ep_len, ep_idx, step_idx + 1)
    return np.nonzero(step_idx <= ep_len[ep_idx] - goal_offset - 1)[0]


def make_batches(n_batches, batch_size, img_size=224, dataset="tworoom", goal_offset=25, seed=0):
    """Yield (start_pixels, goal_pixels) batches of shape (B, 3, img_size, img_size) from a LeWM .h5 dataset.

    The start is a random dataset frame and the goal is the frame `goal_offset`
    steps later in the same episode, matching the evaluation protocol.
    """
    h5 = h5py.File(STABLEWM_HOME / f"{dataset}.h5", "r")
    ep_col = "episode_idx" if "episode_idx" in h5 else "ep_idx"
    valid = valid_start_rows(h5[ep_col][:], h5["step_idx"][:], goal_offset)
    transform = img_transform(img_size)
    rng = np.random.default_rng(seed)

    for _ in range(n_batches):
        rows = np.sort(rng.choice(valid, size=batch_size, replace=False))  # h5 needs sorted indices
        start = torch.stack([transform(f) for f in h5["pixels"][rows]])
        goal = torch.stack([transform(f) for f in h5["pixels"][rows + goal_offset]])
        yield start, goal


def load_lewm(ckpt_path, device="cpu"):
    """Load a LeWM `*_object.ckpt` (a pickled `jepa.JEPA`) as a frozen eval model."""
    wm = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.eval().requires_grad_(False)
    return wm


@torch.no_grad()
def encode(wm, pixels):
    """Frozen LeWM embedding of a batch of frames. pixels: (B, C, H, W) -> (B, D)."""
    return wm.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]


@torch.no_grad()
def lewm_cost(wm, z_start, z_goal, actions, history_size=3):
    """J(A, c): LeWM planning cost of an action plan.

    Rolls the frozen predictor forward from z_start through `actions` and returns
    the squared distance of the final predicted embedding to z_goal. This is
    `JEPA.rollout` + `JEPA.criterion` with a single start frame, run directly
    on embeddings.

    Args:
        z_start, z_goal: (B, D)
        actions:         (B, T, A) actions in the world model's action space.
    Returns:
        (B,) cost per plan.
    """
    emb = z_start.unsqueeze(1)  # (B, 1, D)
    for t in range(actions.shape[1]):
        act_emb = wm.action_encoder(actions[:, : t + 1])
        pred = wm.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return (emb[:, -1] - z_goal).pow(2).sum(-1)


def plan_terms(sampler, wm, z_start, z_goal, actions, action_dim):
    """Everything the losses need, per plan. All tensors have leading dim N = actions.shape[0].

    Returns a dict with
        log_pf:       sum_t log P_F(a_t | a_<t, c)                (N,)   differentiable
        log_pf_steps: the per-step terms                          (N, n_steps)
        log_z:        log Z(c) from the model's Z head            (N,)   differentiable
        cost:         J(A, c), the LeWM planning cost             (N,)   no gradient
        log_ref:      log N(A; 0, s^2 I), prior on action buffers (N,)   no gradient
    """
    n = actions.shape[0]
    log_pf_steps = sampler.log_prob(z_start, z_goal, actions)
    return {
        "log_pf": log_pf_steps.sum(-1),
        "log_pf_steps": log_pf_steps,
        "log_z": sampler.model.log_Z(z_start, z_goal),
        "cost": lewm_cost(wm, z_start, z_goal, actions.view(n, -1, action_dim)),
        "log_ref": sampler.reference_log_prob(actions),
    }


def xi(terms, beta):
    """xi = log P_F + beta J - log N(A).  At the optimum xi == -log Z(c) for every plan A."""
    return terms["log_pf"] + beta * terms["cost"] - terms["log_ref"]


def tb_loss(sampler, wm, z_start, z_goal, actions, beta, action_dim, return_stats=False):
    """Trajectory Balance loss  ( log Z(c) + xi(A, c) )^2  with a learned log Z head.

    Args:
        z_start, z_goal: (N, D) conditions, one per plan.
        actions:         (N, T * action_dim) buffers from `sampler.rollout`.
        beta:            cost temperature.
        return_stats:    also return a dict of detached scalar diagnostics.
    """
    terms = plan_terms(sampler, wm, z_start, z_goal, actions, action_dim)
    residual = terms["log_z"] + xi(terms, beta)
    loss = residual.pow(2).mean()
    if not return_stats:
        return loss
    return loss, {**plan_stats(terms, residual, beta), "tb/log_z": terms["log_z"].mean().item()}


def vargrad_loss(sampler, wm, z_start, z_goal, actions, beta, action_dim, n_samples, return_stats=False):
    """VarGrad loss  Var_k[ xi(A_k, c) ]  over K plans sampled for the same condition c.

    Since xi(A, c) = -log Z(c) for all A at the optimum, its variance over plans of
    one condition is zero there, and the per-condition mean -xi is an in-batch
    estimate of log Z(c). No partition function has to be learned, and the mean
    acts as a per-condition baseline for the policy gradient. The model's Z head
    is still fitted (by regression to that estimate) for diagnostics; it gets no
    gradient from the VarGrad term and the policy gets none from the regression.

    Args:
        z_start, z_goal: (B*K, D) conditions, each repeated K times consecutively
                         (`repeat_interleave(K, 0)`), i.e. plan i belongs to condition i // K.
        actions:         (B*K, T * action_dim) buffers, one per row of z_start.
        n_samples:       K >= 2.
    """
    assert n_samples >= 2, "VarGrad needs at least 2 plans per condition"
    terms = plan_terms(sampler, wm, z_start, z_goal, actions, action_dim)
    xi_bk = xi(terms, beta).view(-1, n_samples)                    # (B, K)
    loss = xi_bk.var(dim=1).mean()

    log_z_hat = -xi_bk.mean(dim=1).detach()                        # (B,)  in-batch log Z(c)
    log_z_head = terms["log_z"].view(-1, n_samples)[:, 0]          # (B,)  one per condition
    z_fit_loss = (log_z_head - log_z_hat).pow(2).mean()
    if not return_stats:
        return loss + z_fit_loss
    residual = xi_bk - xi_bk.mean(dim=1, keepdim=True)             # TB residual with log Z = log_z_hat
    stats = plan_stats(terms, residual.flatten(), beta)
    stats.update({
        "loss/vargrad": loss.item(),
        "loss/z_fit": z_fit_loss.item(),
        "tb/log_z": log_z_head.mean().item(),
        "tb/log_z_hat": log_z_hat.mean().item(),
        "tb/log_z_hat_std": log_z_hat.std().item(),                # how much log Z varies across conditions
    })
    return loss + z_fit_loss, stats


@torch.no_grad()
def plan_stats(terms, residual, beta):
    """Detached scalar diagnostics shared by both losses."""
    cost = terms["cost"]
    stats = {
        "tb/residual": residual.mean(),          # signed; should hover around 0
        "tb/residual_abs": residual.abs().mean(),
        "tb/residual_std": residual.std(),
        "tb/log_pf": terms["log_pf"].mean(),
        "tb/log_pf_std": terms["log_pf"].std(),
        "tb/log_pf_per_step": terms["log_pf_steps"].mean(),
        "tb/log_ref": terms["log_ref"].mean(),
        "tb/log_ref_std": terms["log_ref"].std(),
        "tb/neg_log_reward": (beta * cost - terms["log_ref"]).mean(),  # -log R, what log Z + log P_F must match
        "cost/mean": cost.mean(),
        "cost/std": cost.std(),
        "cost/min": cost.min(),
        "cost/max": cost.max(),
        "cost/beta_cost": (beta * cost).mean(),
        "cost/beta_cost_std": (beta * cost).std(),  # spread of -log R across the batch; what the policy must explain
    }
    return {k: v.item() for k, v in stats.items()}


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


def linear_beta_schedule(beta_end, beta_start=0.0, warmup_steps=0):
    """Return `step -> beta`, linear from `beta_start` to `beta_end` over `warmup_steps`, then constant.

    Annealing beta up from ~0 starts training on the well-posed "match the prior"
    problem and gradually sharpens the target exp(-beta J) N(A), so log Z and the
    policy track a slowly moving target instead of chasing a far-away one from scratch.
    """
    if warmup_steps <= 0:
        return lambda step: beta_end
    return lambda step: beta_start + (beta_end - beta_start) * min(1.0, step / warmup_steps)


def train(sampler, wm, batches, beta, action_dim, lr=1e-3, device="cpu", log_every=10, grad_clip=None, lr_z=None,
          loss_type="tb", n_samples=1):
    """Run one optimization step per (start_pixels, goal_pixels) batch.

    Args:
        beta:      cost temperature, either a float or a callable `step -> float`
                   (see `linear_beta_schedule`) for annealing.
        grad_clip: max global L2 norm of the gradient; `None` or <= 0 disables clipping.
        lr_z:      learning rate for the log Z head (`model.Z`). GFlowNets are much
                   more stable when log Z can move faster than the policy, since it
                   alone must absorb the constant offset of the TB residual.
                   Defaults to `lr`.
        loss_type: "tb" (learned log Z) or "vargrad" (in-batch log Z, needs n_samples >= 2).
        n_samples: K plans sampled per (start, goal) condition; the effective batch is B*K.

    Every step is logged to wandb when a run is active (`wandb.init` was called);
    otherwise only the console summary every `log_every` steps is printed.
    """
    assert loss_type in ("tb", "vargrad"), loss_type
    z_params = list(sampler.model.Z.parameters())
    z_ids = {id(p) for p in z_params}
    policy_params = [p for p in sampler.model.parameters() if id(p) not in z_ids]
    opt = torch.optim.Adam([
        {"params": policy_params, "lr": lr},
        {"params": z_params, "lr": lr if lr_z is None else lr_z},
    ])
    params = policy_params + z_params
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
            loss, stats = vargrad_loss(sampler, wm, z_start, z_goal, actions, beta_t, action_dim, n_samples,
                                       return_stats=True)
        else:
            loss, stats = tb_loss(sampler, wm, z_start, z_goal, actions, beta_t, action_dim, return_stats=True)
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
                  f"beta {beta_t:.3f}  cost {stats['cost/mean']:.3f}  log_z {stats['tb/log_z']:+.3f}  beta_cost {stats['cost/beta_cost']:.3f} "
                  f"grad_norm {stats['grad_norm/global_raw']:.3e} -> {stats['grad_norm/global']:.3e}")
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help=".h5 name under $STABLEWM_HOME")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
    p.add_argument("--n-steps", type=int, default=5, help="planning horizon T")
    p.add_argument("--action-dim", type=int, default=10, help="frameskip * env action dim")
    p.add_argument("--loss", default="tb", choices=["vargrad", "tb"])
    p.add_argument("--n-samples", type=int, default=8, help="plans sampled per (start, goal); vargrad needs >= 2")
    p.add_argument("--beta", type=float, default=1.0, help="final cost temperature on the raw LeWM cost")
    p.add_argument("--beta-start", type=float, default=0.0, help="initial beta when annealing")
    p.add_argument("--beta-warmup", type=int, default=0,
                   help="steps to anneal beta linearly from --beta-start to --beta (0 = constant beta)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-z", type=float, default=3e-3, help="learning rate of the log Z head (usually 10-100x --lr)")
    p.add_argument("--grad-clip", type=float, default=10.0, help="max global grad L2 norm; <= 0 disables")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=1000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=ROOT / "gplaner.pt", help="where to save the trained GPlaner state_dict")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None, help="run name (defaults to wandb's random name)")
    p.add_argument("--ref-var", type=float, default=1.0, help="variance of reference gauss")
    args = p.parse_args()
    print("Device:", args.device)
    torch.manual_seed(args.seed)
    wm = load_lewm(args.ckpt, args.device)
    embed_dim = wm.predictor.pos_embedding.shape[-1]
    model_kwargs = dict(state_dim=embed_dim, horizon=args.n_steps * args.action_dim, action_dim=args.action_dim)
    model = GPlaner(**model_kwargs).to(args.device)
    sampler = Sampler(model, reference_var=args.ref_var)

    config = {**vars(args), "embed_dim": embed_dim, "n_params": sum(p.numel() for p in model.parameters())}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, mode=args.wandb_mode,
               job_type="train")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("cost/mean", summary="min")
    batches = make_batches(args.n_batches, args.batch_size, args.img_size, args.dataset, args.goal_offset, args.seed)
    # --beta and --beta-start are given per embedding dimension; J sums over all embed_dim dims
    beta = linear_beta_schedule(args.beta / embed_dim, args.beta_start / embed_dim, args.beta_warmup)
    losses = train(sampler, wm, batches, beta, args.action_dim, args.lr, args.device,
                   grad_clip=args.grad_clip, lr_z=args.lr_z, loss_type=args.loss, n_samples=args.n_samples)
    # self-describing checkpoint so eval.py can rebuild the exact architecture (eval also accepts a bare state_dict)
    torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "train_args": config}, args.out)
    wandb.summary["final_loss"] = losses[-1]
    wandb.save(str(args.out), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
