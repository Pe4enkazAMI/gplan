"""Train the GPlaner sampler (JAX) as a conditional GFlowNet over LeWM action plans.

Needs precomputed embeddings (scripts/encode_dataset.py). Writes outputs/<name>.{eqx,json,pt};
the .pt is what scripts/evaluate.py loads.

    python scripts/train.py --beta 120 --wandb-name tb-beta120 --out outputs/tb-beta120
    python scripts/train.py --loss vargrad --n-samples 8
    python scripts/train.py --n-components 4             # EXP: mixture of 4 Gaussians per plan step
    python scripts/train.py --ls-steps 10                 # EXP: + 10 MALA steps per plan as off-policy data

See gplan/jax/losses.py for the target distribution and the two losses.
"""

import argparse
import os
from pathlib import Path

import jax
import wandb

from gplan.data import STABLEWM_HOME
from gplan.jax import checkpoint
from gplan.jax.convert import load_lewm_predictor
from gplan.jax.latents import LatentPairs
from gplan.jax.policy import GPlaner
from gplan.jax.train import linear_beta_schedule, train, TrainConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help="dataset name under $STABLEWM_HOME")
    p.add_argument("--latents", type=Path, default=None,
                   help="prefix of the precomputed embeddings (default: $STABLEWM_HOME/<dataset>_latents)")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
    p.add_argument("--n-steps", type=int, default=5, help="planning horizon T")
    p.add_argument("--action-dim", type=int, default=10, help="frameskip * env action dim")
    p.add_argument("--n-components", type=int, default=1,
                   help="EXP mixture policy: Gaussians per plan step (1 = a single diagonal Gaussian)")
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
    p.add_argument("--ls-steps", type=int, default=0,
                   help="EXP MALA local search: MALA steps from each on-policy plan; the results are added to "
                        "the batch as off-policy plans (0 = off)")
    p.add_argument("--ls-every", type=int, default=1, help="EXP MALA: run local search every this many steps")
    p.add_argument("--ls-start", type=int, default=0,
                   help="EXP MALA: first step with local search (e.g. after --beta-warmup)")
    p.add_argument("--ls-step-size", type=float, default=1e-3, help="EXP MALA: initial step size eta")
    p.add_argument("--ls-target-accept", type=float, default=0.574,
                   help="EXP MALA: eta is adapted toward this acceptance rate (0 = keep eta fixed)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-var", type=float, default=1.0, help="variance s^2 of the N(0, s^2 I) reference measure")
    p.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "default"],
                   help="float32 matmul precision on GPU; 'default' allows TF32 (faster, less exact than torch)")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic XLA GPU kernels (--xla_gpu_deterministic_ops): bit-identical reruns, slower")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("outputs/gplaner"), help="checkpoint prefix")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None, help="run name (defaults to wandb's random name)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.deterministic:  # XLA reads its flags when the backend starts, i.e. at the first jax.devices() below
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print("JAX devices:", jax.devices())

    cost_model = load_lewm_predictor(args.ckpt)  # torch is used only here, on CPU
    pairs = LatentPairs(args.latents or STABLEWM_HOME / f"{args.dataset}_latents", args.goal_offset,
                        expected_ckpt=args.ckpt)
    d = pairs.embed_dim
    model_kwargs = dict(state_dim=d, horizon=args.n_steps * args.action_dim, action_dim=args.action_dim,
                        n_components=args.n_components)
    model = GPlaner(**model_kwargs, key=jax.random.key(args.seed))
    cfg = TrainConfig(loss_type=args.loss, n_samples=args.n_samples, action_dim=args.action_dim,
                      reference_var=args.ref_var, grad_clip=args.grad_clip, ls_steps=args.ls_steps,
                      ls_every=args.ls_every, ls_start=args.ls_start, ls_step_size=args.ls_step_size,
                      ls_target_accept=args.ls_target_accept)

    n_params = sum(x.size for x in jax.tree.leaves(model))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"embed_dim": d, "n_params": n_params, "backend": "jax"})
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, mode=args.wandb_mode,
               job_type="train")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("cost/mean", summary="min")

    beta = linear_beta_schedule(args.beta / d, args.beta_start / d, args.beta_warmup)
    model, losses = train(model, cost_model, pairs.batches(args.n_batches, args.batch_size, args.seed), beta, cfg,
                          lr=args.lr, lr_z=args.lr_z, seed=args.seed, log_every=args.log_every)

    checkpoint.save(args.out, model, model_kwargs, train_args=config)
    print(f"saved {args.out}.eqx / .json / .pt")
    wandb.summary["final_loss"] = losses[-1]
    wandb.save(f"{args.out}.pt", policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
