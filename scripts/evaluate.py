"""Evaluate a trained GPlaner (GFlowNet) against CEM as MPC planners on TwoRoom or PushT.

    # head-to-head, one wandb run with a comparison table (CEM always uses its paper settings
    # unless --cem-* says otherwise)
    python scripts/evaluate.py --policy gflow cem --planner outputs/gplaner.pt
    python scripts/evaluate.py --task pusht --policy gflow cem --planner outputs/pusht.pt

    # just the GFlowNet, no cost re-ranking (1 sample per replanning step)
    python scripts/evaluate.py --policy gflow --planner outputs/gplaner.pt --gflow-select sample

    # test-time refinement diagnostic: best-of-64 vs the same samples refined through LeWM, plus CEM
    python scripts/evaluate.py --task pusht --policy gflow gflow-refine cem --planner outputs/pusht.pt

    python scripts/evaluate.py --policy cem        # LeWM paper planner
    python scripts/evaluate.py --policy random     # floor

Per policy we log success_rate, wall-clock time, the LeWM cost of the plans that
were actually executed (recomputed identically for every solver; overall, first replan and
per replan), the largest |action| of the executed plans (z-scored, so in dataset stds) and
the planner's compute per replanning step as world-model passes:
    fwd/replan  forward rollouts: CEM = cem_samples * cem_steps, GFlow best-of-N = gflow_samples,
                GFlow sample/mean = 0, GFlow refine = gflow_samples * (refine_steps + 1)
    bwd/replan  backward passes, ~2x a forward each: GFlow refine = gflow_samples * refine_steps
Protocol details: gplan/torch/evaluation.py.

Each policy has its own settings (--gflow-*, --refine-*, --cem-*), so CEM runs with the
same configuration whatever GFlow checkpoint or flags are evaluated next to it.

--task picks the env, dataset, LeWM checkpoint and reset keys (gplan.torch.evaluation.TASKS);
--env / --dataset / --wm override single fields.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import stable_worldmodel as swm
import torch
import wandb

from gplan.data import STABLEWM_HOME
from gplan.torch.evaluation import (
    build_policy,
    eval_episode_pool,
    evaluate,
    fit_normalizers,
    sample_eval_starts,
    TASKS,
    trained_on,
    use_deterministic_algorithms,
)
from gplan.torch.lewm import embed_dim
from gplan.torch.policy import load_planner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", action="extend",
                   choices=["gflow", "gflow-refine", "cem", "prior", "prior-refine", "random"],
                   help="policies to evaluate on the same episodes, in this order (default: gflow cem); "
                        "repeating the flag adds to the list")
    p.add_argument("--planner", default=None, help="[gflow] scripts/train.py checkpoint (.pt)")
    p.add_argument("--gflow-select", default="min-cost", choices=["min-cost", "sample", "mean"],
                   help="[gflow] execute the cheapest of --gflow-samples plans, a single sample, or the mean plan")
    p.add_argument("--prior-samples", type=int, default=64,
                   help="[prior, prior-refine] plans drawn from N(0, --prior-var I) per replanning step")
    p.add_argument("--prior-var", type=float, default=1.0, help="[prior, prior-refine] variance of the prior plans")
    p.add_argument("--refine-steps", type=int, default=20, help="[*-refine] Adam steps on the actions")
    p.add_argument("--refine-lr", type=float, default=0.05, help="[*-refine] Adam step size (z-scored units)")
    p.add_argument("--refine-objective", default="target", choices=["target", "cost"],
                   help="[*-refine] minimize beta*J - log N(A) or J alone")
    p.add_argument("--refine-beta", type=float, default=None,
                   help="[*-refine] per-dim beta of the target objective (default: the planner's training beta)")
    p.add_argument("--task", default="tworoom", choices=sorted(TASKS), help="evaluation task preset")
    p.add_argument("--wm", default=None, help="LeWM ckpt name relative to $STABLEWM_HOME (default: from --task)")
    p.add_argument("--dataset", default=None, help="dataset name (default: from --task)")
    p.add_argument("--env", default=None, help="gymnasium env id (default: from --task)")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="draw episodes only from the held-out side of this episode split (train.py --holdout-frac); "
                        "0 = all episodes, the LeWM protocol")
    p.add_argument("--split-seed", type=int, default=0, help="seed of the episode split (train.py --split-seed)")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5, help="frameskip")
    p.add_argument("--gflow-samples", type=int, default=64,
                   help="[gflow, gflow-refine] plans sampled from the GPlaner per replanning step")
    p.add_argument("--cem-samples", type=int, default=300, help="[cem] samples per iteration (LeWM paper: 300)")
    p.add_argument("--cem-steps", type=int, default=30, help="[cem] optimization iterations")
    p.add_argument("--cem-topk", type=int, default=30, help="[cem] elites kept per iteration")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic cuDNN / cuBLAS kernels: bit-identical GPU reruns (slower)")
    p.add_argument("--video-dir", default=None, help="save rollout videos under <dir>/<policy>/ (off by default)")
    p.add_argument("--results-json", type=Path, default=None,
                   help="also write every policy's metrics and per-episode successes to this .json")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()
    args.policy = list(dict.fromkeys(args.policy or ["gflow", "cem"]))  # keep the order, drop repeats
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


def check_no_leakage(dataset, episodes, train_args, args):
    """Evaluating on episodes the planner trained on is an error with a held-out split and a warning
    under the LeWM protocol (--holdout-frac 0), where it is unavoidable for a planner trained on all."""
    if (train_args or {}).get("dataset", args.dataset) != args.dataset:
        return  # different datasets: episode ids do not correspond (check_planner_matches warned)
    seen = trained_on(dataset, episodes, train_args)
    if not seen:
        return
    msg = (f"{len(set(seen))} of the {len(set(episodes))} evaluation episodes were in the planner's training data "
           f"(planner: holdout_frac={(train_args or {}).get('holdout_frac', 0.0)}, "
           f"split_seed={(train_args or {}).get('split_seed', 0)}; eval: --holdout-frac {args.holdout_frac} "
           f"--split-seed {args.split_seed})")
    if args.holdout_frac > 0:
        raise SystemExit(f"ERROR: {msg}. Evaluate with the planner's split, or train with this one.")
    print(f"WARNING: {msg}")


def write_results(path, args, rows, successes, episodes, start_steps, planner_train_args):
    """Metrics and per-episode successes of every policy, the evaluated (episode, start step) pairs and
    the planner's training config: everything scripts/aggregate_results.py needs."""
    out = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "episodes": [[int(e), int(s)] for e, s in zip(episodes, start_steps)],
           "planner_train_args": planner_train_args,
           "policies": {name: {**m, "episode_successes": successes[name].tolist()} for name, m in rows}}
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


def print_comparison(rows, successes, policies, num_eval):
    """Console + wandb comparison table, and per-episode agreement for every pair of policies."""
    cols = ["policy", "success_rate", "plan_cost", "action_absmax", "plans_per_replan", "backward_per_replan",
            "solve_time_s", "eval_time_s"]
    table = wandb.Table(columns=cols)
    print(f"\n{'policy':12s} {'success%':>9s} {'plan_cost':>10s} {'max|a|':>7s} {'fwd/replan':>11s} "
          f"{'bwd/replan':>11s} {'solve_s':>8s} {'total_s':>8s}")
    for name, m in rows:
        vals = [m.get(c, float("nan")) for c in cols[1:]]
        table.add_data(name, *vals)
        print(f"{name:12s} {vals[0]:9.1f} {vals[1]:10.2f} {vals[2]:7.2f} {vals[3]:11d} {vals[4]:11d} "
              f"{vals[5]:8.1f} {vals[6]:8.1f}")
    for a, b in combinations(policies, 2):
        both = (successes[a] & successes[b]).sum()
        only_a = (successes[a] & ~successes[b]).sum()
        only_b = (~successes[a] & successes[b]).sum()
        print(f"{a} vs {b}: solved by both {both}, only {a} {only_a}, only {b} {only_b}, "
              f"neither {num_eval - both - only_a - only_b}")
        wandb.log({f"{a}_vs_{b}/both": int(both), f"{a}_vs_{b}/only_{a}": int(only_a),
                   f"{a}_vs_{b}/only_{b}": int(only_b)})
    wandb.log({"comparison": table})


def refine_settings(args, train_args, d):
    """Settings shared by gflow-refine and prior-refine; beta and s^2 default to the planner's training
    values (s^2 = --prior-var without a planner). d: LeWM embedding dim (beta is per dim)."""
    train_args = train_args or {}
    beta = args.refine_beta if args.refine_beta is not None else train_args.get("beta")
    if args.refine_objective == "target" and beta is None:
        raise ValueError("--refine-beta is needed: no planner, or its checkpoint does not record its training beta")
    return dict(steps=args.refine_steps, lr=args.refine_lr, objective=args.refine_objective,
                beta=(beta or 0.0) / d, reference_var=train_args.get("ref_var", args.prior_var))


def main():
    args, task = parse_args()
    if args.deterministic:
        use_deterministic_algorithms()
    torch.manual_seed(args.seed)
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode,
               job_type="eval")

    # -- data: start/goal pairs (shared by every policy) and normalization stats
    dataset = swm.data.HDF5Dataset(args.dataset, keys_to_cache=list(task.cache_keys))
    process = fit_normalizers(dataset, task.cache_keys)
    pool = eval_episode_pool(dataset, args.holdout_frac, args.split_seed)
    episodes, start_steps = sample_eval_starts(dataset, args.num_eval, args.goal_offset, args.seed, pool)

    # -- shared frozen world model and (optionally) the trained planner
    wm = planner = refine = train_args = None
    if any(name != "random" for name in args.policy):
        wm = swm.policy.AutoCostModel(args.wm).to(args.device)
        wm.eval().requires_grad_(False)
    if any(name.startswith("gflow") for name in args.policy):
        planner, train_args = load_planner(args.planner, args.device)
        check_planner_matches(planner, train_args, args, dataset.get_col_data("action").shape[1])
        check_no_leakage(dataset, episodes, train_args, args)
    if any(name.endswith("-refine") for name in args.policy):  # one setting for gflow- and prior-refine
        refine = refine_settings(args, train_args, embed_dim(wm))
        print(f"refine: {refine}")

    # -- evaluate every policy on the same episodes
    rows, successes = [], {}
    for name in args.policy:
        print(f"\n===== {name} =====")
        policy, solver, compute = build_policy(name, args, wm, process, planner, refine)
        res = evaluate(name, policy, args, task, dataset, episodes, start_steps)
        successes[name] = res.pop("episode_successes")
        metrics = {**res, **compute, **(solver.summary() if solver else {})}
        rows.append((name, metrics))
        wandb.log({f"{name}/{k}": v for k, v in metrics.items()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    print_comparison(rows, successes, args.policy, args.num_eval)
    if args.results_json:
        write_results(args.results_json, args, rows, successes, episodes, start_steps, train_args)
    wandb.finish()


if __name__ == "__main__":
    main()
