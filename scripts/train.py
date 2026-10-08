"""Train the GPlaner sampler as a conditional GFlowNet over LeWM action plans.

    python scripts/train.py --beta 120 --wandb-name tb-beta120
    python scripts/train.py --loss vargrad --n-samples 8

See gplan/losses.py for the target distribution and the two losses.
"""

import argparse
from pathlib import Path

import torch
import wandb

from gplan.data import make_batches
from gplan.lewm import STABLEWM_HOME, embed_dim, load_lewm
from gplan.policy import GPlaner, Sampler, save_planner
from gplan.trainer import linear_beta_schedule, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help=".h5 name under $STABLEWM_HOME")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
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
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/gplaner.pt"), help="where to save the trained planner")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None, help="run name (defaults to wandb's random name)")
    p.add_argument("--ref-var", type=float, default=1.0, help="variance s^2 of the N(0, s^2 I) reference measure")
    return p.parse_args()


def main():
    args = parse_args()
    print("Device:", args.device)
    torch.manual_seed(args.seed)
    wm = load_lewm(args.ckpt, args.device)
    d = embed_dim(wm)
    model_kwargs = dict(state_dim=d, horizon=args.n_steps * args.action_dim, action_dim=args.action_dim)
    model = GPlaner(**model_kwargs).to(args.device)
    sampler = Sampler(model, reference_var=args.ref_var)

    config = {**vars(args), "embed_dim": d, "n_params": sum(p.numel() for p in model.parameters())}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, mode=args.wandb_mode,
               job_type="train")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("cost/mean", summary="min")

    batches = make_batches(args.n_batches, args.batch_size, args.img_size, args.dataset, args.goal_offset, args.seed)
    beta = linear_beta_schedule(args.beta / d, args.beta_start / d, args.beta_warmup)
    losses = train(sampler, wm, batches, beta, args.action_dim, args.lr, args.device,
                   grad_clip=args.grad_clip, lr_z=args.lr_z, loss_type=args.loss, n_samples=args.n_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_planner(args.out, model, model_kwargs, train_args=config)  # eval rebuilds the exact architecture from it
    wandb.summary["final_loss"] = losses[-1]
    wandb.save(str(args.out), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
