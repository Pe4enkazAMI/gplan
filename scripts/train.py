"""Train the GPlaner sampler (JAX) as a conditional GFlowNet over LeWM action plans.

Needs precomputed embeddings (scripts/encode_dataset.py). Writes outputs/<name>.{eqx,json,pt};
the .pt is what scripts/evaluate.py loads.

    python scripts/train.py --beta 120 --wandb-name tb-beta120 --out outputs/tb-beta120
    python scripts/train.py --loss vargrad --n-samples 8

    # EXP k-warmup: k = 25 for the first 10k steps, then the largest k grows to 50 over 10k steps;
    # half of the pairs always keep k = 25
    python scripts/train.py --goal-offset 25 --goal-offset-max 50 \
        --goal-offset-warmup-start 10000 --goal-offset-warmup 10000

See gplan_jax/losses.py for the target distribution and the two losses.
"""

import argparse
from pathlib import Path

import jax
import wandb

from gplan.data import STABLEWM_HOME
from gplan_jax import checkpoint
from gplan_jax.convert import load_lewm_predictor
from gplan_jax.data import LatentPairs, goal_offset_schedule
from gplan_jax.policy import GPlaner
from gplan_jax.train import TrainConfig, linear_beta_schedule, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help="dataset name under $STABLEWM_HOME")
    p.add_argument("--latents", type=Path, default=None,
                   help="prefix of the precomputed embeddings (default: $STABLEWM_HOME/<dataset>_latents)")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
    p.add_argument("--goal-offset-max", type=int, default=None,
                   help="EXP k-warmup: largest goal offset k; pairs then use k in [--goal-offset, this] "
                        "(default: --goal-offset, i.e. a fixed k)")
    p.add_argument("--goal-offset-warmup-start", type=int, default=0,
                   help="EXP k-warmup: step at which the k range starts to widen")
    p.add_argument("--goal-offset-warmup", type=int, default=0,
                   help="EXP k-warmup: steps over which the largest k grows linearly from --goal-offset to "
                        "--goal-offset-max (0 = the full range from --goal-offset-warmup-start on)")
    p.add_argument("--goal-offset-fixed-frac", type=float, default=0.5,
                   help="EXP k-warmup: share of pairs that keep k = --goal-offset")
    p.add_argument("--n-steps", type=int, default=5, help="planning horizon T")
    p.add_argument("--action-dim", type=int, default=10, help="frameskip * env action dim")
    p.add_argument("--loss", default="tb", choices=["vargrad", "tb"])
    p.add_argument("--n-samples", type=int, default=8, help="plans sampled per (start, goal); vargrad needs >= 2")
    p.add_argument("--beta", type=float, default=1.0,
                   help="final cost temperature, per embedding dim (the raw cost is scaled by beta / embed_dim)")
    p.add_argument("--beta-start", type=float, default=0.0, help="initial beta when annealing, same units as --beta")
    p.add_argument("--beta-warmup", type=int, default=0,
                   help="steps to anneal beta linearly from --beta-start to --beta (0 = constant beta)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-z", type=float, default=3e-3, help="learning rate of the log Z head (usually 10-100x --lr)")
    p.add_argument("--grad-clip", type=float, default=10.0, help="max global grad L2 norm; <= 0 disables")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-var", type=float, default=1.0, help="variance s^2 of the N(0, s^2 I) reference measure")
    p.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "default"],
                   help="float32 matmul precision on GPU; 'default' allows TF32 (faster, less exact than torch)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("outputs/gplaner"), help="checkpoint prefix")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None, help="run name (defaults to wandb's random name)")
    args = p.parse_args()
    if args.goal_offset_max is None:
        args.goal_offset_max = args.goal_offset
    assert args.goal_offset_max >= args.goal_offset, "--goal-offset-max must be >= --goal-offset"
    assert 0.0 <= args.goal_offset_fixed_frac <= 1.0, "--goal-offset-fixed-frac must be in [0, 1]"
    return args


def main():
    args = parse_args()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print("JAX devices:", jax.devices())

    cost_model = load_lewm_predictor(args.ckpt)  # torch is used only here, on CPU
    pairs = LatentPairs(args.latents or STABLEWM_HOME / f"{args.dataset}_latents", args.goal_offset,
                        expected_ckpt=args.ckpt)
    d = pairs.embed_dim
    model_kwargs = dict(state_dim=d, horizon=args.n_steps * args.action_dim, action_dim=args.action_dim)
    model = GPlaner(**model_kwargs, key=jax.random.key(args.seed))
    cfg = TrainConfig(loss_type=args.loss, n_samples=args.n_samples, action_dim=args.action_dim,
                      reference_var=args.ref_var, grad_clip=args.grad_clip)

    n_params = sum(x.size for x in jax.tree.leaves(model))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"embed_dim": d, "n_params": n_params, "backend": "jax"})
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, mode=args.wandb_mode,
               job_type="train")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("cost/mean", summary="min")

    beta = linear_beta_schedule(args.beta / d, args.beta_start / d, args.beta_warmup)
    k_max = goal_offset_schedule(args.goal_offset, args.goal_offset_max,
                                 args.goal_offset_warmup_start, args.goal_offset_warmup)
    batches = pairs.batches(args.n_batches, args.batch_size, args.seed,
                            max_offset=k_max, fixed_frac=args.goal_offset_fixed_frac)
    model, losses = train(model, cost_model, batches, beta, cfg, lr=args.lr, lr_z=args.lr_z, seed=args.seed,
                          log_every=args.log_every, log_fn=lambda step: {"data/goal_offset_max": k_max(step)})

    checkpoint.save(args.out, model, model_kwargs, train_args=config)
    print(f"saved {args.out}.eqx / .json / .pt")
    wandb.summary["final_loss"] = losses[-1]
    wandb.save(f"{args.out}.pt", policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
