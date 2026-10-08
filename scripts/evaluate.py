"""Evaluate a trained GPlaner (GFlowNet) against CEM as MPC planners on TwoRoom or PushT.

    # head-to-head, one wandb run with a comparison table
    python scripts/evaluate.py --policy gflow cem --planner outputs/gplaner.pt
    python scripts/evaluate.py --task pusht --policy gflow cem --planner outputs/pusht.pt

    # just the GFlowNet, no cost re-ranking (1 sample per replanning step)
    python scripts/evaluate.py --policy gflow --planner outputs/gplaner.pt --gflow-select sample

    python scripts/evaluate.py --policy cem        # LeWM paper planner
    python scripts/evaluate.py --policy random     # floor

Per policy we log success_rate, wall-clock time, the LeWM cost of the plans that
were actually executed (recomputed identically for every solver) and the number
of plans the world model had to score per replanning step, i.e. the planner's
compute (CEM = num_samples * cem_steps, GFlow best-of-N = num_samples, GFlow
sample/mean = 0). Protocol details: gplan/evaluation.py.

--task picks the env, dataset, LeWM checkpoint and reset keys (gplan.evaluation.TASKS);
--env / --dataset / --wm override single fields.
"""

import argparse
from pathlib import Path

import stable_worldmodel as swm
import torch
import wandb

from gplan.data import STABLEWM_HOME
from gplan.evaluation import TASKS, build_policy, evaluate, fit_normalizers, sample_eval_starts
from gplan.policy import load_planner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", choices=["gflow", "cem", "random"], default=["gflow", "cem"],
                   help="policies to evaluate on the same episodes")
    p.add_argument("--planner", default=None, help="[gflow] scripts/train.py checkpoint (.pt)")
    p.add_argument("--gflow-select", default="min-cost", choices=["min-cost", "sample", "mean"],
                   help="[gflow] execute the cheapest of --num-samples plans, a single sample, or the mean plan")
    p.add_argument("--task", default="tworoom", choices=sorted(TASKS), help="evaluation task preset")
    p.add_argument("--wm", default=None, help="LeWM ckpt name relative to $STABLEWM_HOME (default: from --task)")
    p.add_argument("--dataset", default=None, help="dataset name (default: from --task)")
    p.add_argument("--env", default=None, help="gymnasium env id (default: from --task)")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5, help="frameskip")
    p.add_argument("--num-samples", type=int, default=None,
                   help="plans sampled per replanning step (default: 64 for gflow, 300 for cem as in the paper)")
    p.add_argument("--cem-steps", type=int, default=30, help="[cem] optimization iterations")
    p.add_argument("--cem-topk", type=int, default=30, help="[cem] elites kept per iteration")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--video-dir", default=None, help="save rollout videos under <dir>/<policy>/ (off by default)")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()
    task = TASKS[args.task]
    args.wm = args.wm or task.wm
    args.dataset = args.dataset or task.dataset
    args.env = args.env or task.env
    return args, task


def check_planner_matches(planner, train_args, args, env_action_dim):
    """Warn about train/eval protocol mismatches and assert the plan layout fits the env."""
    if train_args is not None:
        wandb.config.update({"planner_train_args": train_args})
        for k, eval_v in (("goal_offset", args.goal_offset), ("n_steps", args.horizon), ("dataset", args.dataset)):
            if k in train_args and train_args[k] != eval_v:
                print(f"WARNING: planner was trained with {k}={train_args[k]} but eval uses {eval_v}")
        eval_ckpt = (STABLEWM_HOME / f"{args.wm}_object.ckpt").resolve()
        if "ckpt" in train_args and Path(train_args["ckpt"]).resolve() != eval_ckpt:
            print(f"WARNING: planner was trained against {train_args['ckpt']} but eval scores plans with {eval_ckpt}")
    action_dim = env_action_dim * args.action_block
    assert planner.horizon == args.horizon * action_dim, (
        f"planner horizon {planner.horizon} != --horizon {args.horizon} x action dim {action_dim}")


def print_comparison(rows, successes, policies, num_eval):
    """Console + wandb comparison table, and per-episode agreement when exactly two policies ran."""
    cols = ["policy", "success_rate", "plan_cost", "plans_per_replan", "solve_time_s", "eval_time_s"]
    table = wandb.Table(columns=cols)
    print(f"\n{'policy':8s} {'success%':>9s} {'plan_cost':>10s} {'plans/replan':>13s} {'solve_s':>8s} {'total_s':>8s}")
    for name, m in rows:
        vals = [m.get(c, float("nan")) for c in cols[1:]]
        table.add_data(name, *vals)
        print(f"{name:8s} {vals[0]:9.1f} {vals[1]:10.2f} {vals[2]:13d} {vals[3]:8.1f} {vals[4]:8.1f}")
    if len(successes) == 2:
        a, b = policies
        both = (successes[a] & successes[b]).sum()
        only_a = (successes[a] & ~successes[b]).sum()
        only_b = (~successes[a] & successes[b]).sum()
        print(f"\nepisodes solved by both: {both}, only {a}: {only_a}, only {b}: {only_b}, "
              f"neither: {num_eval - both - only_a - only_b}")
        wandb.log({"both_solved": int(both), f"only_{a}": int(only_a), f"only_{b}": int(only_b)})
    wandb.log({"comparison": table})


def main():
    args, task = parse_args()
    torch.manual_seed(args.seed)
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode,
               job_type="eval")

    # -- data: start/goal pairs (shared by every policy) and normalization stats
    dataset = swm.data.HDF5Dataset(args.dataset, keys_to_cache=list(task.cache_keys))
    process = fit_normalizers(dataset, task.cache_keys)
    episodes, start_steps = sample_eval_starts(dataset, args.num_eval, args.goal_offset, args.seed)

    # -- shared frozen world model and (optionally) the trained planner
    wm = planner = None
    if any(name != "random" for name in args.policy):
        wm = swm.policy.AutoCostModel(args.wm).to(args.device)
        wm.eval().requires_grad_(False)
    if "gflow" in args.policy:
        planner, train_args = load_planner(args.planner, args.device)
        check_planner_matches(planner, train_args, args, dataset.get_col_data("action").shape[1])

    # -- evaluate every policy on the same episodes
    rows, successes = [], {}
    for name in args.policy:
        print(f"\n===== {name} =====")
        policy, solver, plans_per_replan = build_policy(name, args, wm, process, planner)
        res = evaluate(name, policy, args, task, dataset, episodes, start_steps)
        successes[name] = res.pop("episode_successes")
        metrics = {**res, "plans_per_replan": plans_per_replan, **(solver.summary() if solver else {})}
        rows.append((name, metrics))
        wandb.log({f"{name}/{k}": v for k, v in metrics.items()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    print_comparison(rows, successes, args.policy, args.num_eval)
    wandb.finish()


if __name__ == "__main__":
    main()
