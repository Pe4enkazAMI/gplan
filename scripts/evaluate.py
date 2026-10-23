"""Evaluate a trained GPlanner (GFlowNet) against CEM as MPC planners on TwoRoom, PushT, Reacher or Cube.
Settings: gplan/config.py EvalArgs (configs/evaluate.yaml), as Hydra overrides.

    # head-to-head, one wandb run with a comparison table (CEM always uses its paper settings
    # unless cem_* says otherwise)
    python scripts/evaluate.py "policy=[gflow,cem]" planner=outputs/gplaner.pt
    python scripts/evaluate.py task=pusht "policy=[gflow,cem]" planner=outputs/pusht.pt

    # just the GFlowNet, no cost re-ranking (1 sample per replanning step)
    python scripts/evaluate.py "policy=[gflow]" planner=outputs/gplaner.pt gflow_select=sample

    # test-time refinement diagnostic: best-of-64 vs the same samples refined through LeWM, plus CEM
    python scripts/evaluate.py task=pusht "policy=[gflow,gflow-refine,cem]" planner=outputs/pusht.pt

    python scripts/evaluate.py "policy=[cem]"      # LeWM paper planner
    python scripts/evaluate.py "policy=[random]"   # floor

Per policy we log success_rate, wall-clock time, the LeWM cost of the plans that
were actually executed (recomputed identically for every solver; overall, first replan and
per replan), the largest |action| of the executed plans (z-scored, so in dataset stds) and
the planner's compute per replanning step as world-model passes:
    fwd/replan  forward rollouts: CEM = cem_samples * cem_steps, GFlow best-of-N = gflow_samples,
                GFlow sample/mean = 0, GFlow refine = gflow_samples * (refine_steps + 1)
    bwd/replan  backward passes, ~2x a forward each: GFlow refine = gflow_samples * refine_steps
Protocol details: gplan/torch/evaluation.py.

Each policy has its own settings (gflow_*, refine_*, cem_*), so CEM runs with the
same configuration whatever GFlow checkpoint or settings are evaluated next to it.

task picks the env (and its settings), dataset, LeWM checkpoint and reset calls
(gplan.torch.evaluation.TASKS); env / dataset / wm override single fields.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import stable_worldmodel as swm
import torch
import wandb

from gplan.config import compose_config, EvalArgs, script_overrides, to_args
from gplan.torch.evaluation import (
    build_policy,
    check_no_leakage,
    check_planner_matches,
    eval_episode_pool,
    evaluate,
    fit_normalizers,
    refine_settings,
    sample_eval_starts,
    TASKS,
    use_deterministic_algorithms,
)
from gplan.torch.lewm import check_history, embed_dim
from gplan.torch.policy import load_planner


def parse_args(argv=None):
    """(settings namespace, task preset) from the command line (Hydra overrides). Unset policy, wm,
    dataset, env and device take their defaults here: [gflow, cem], the task preset's, cuda if available."""
    overrides = script_overrides(sys.argv[1:] if argv is None else argv, __doc__, "evaluate")
    args = to_args(compose_config("evaluate", overrides), EvalArgs)
    if args.task not in TASKS:
        raise SystemExit(f"task={args.task!r} is not one of {sorted(TASKS)}")
    args.policy = list(dict.fromkeys(args.policy or ["gflow", "cem"]))  # keep the order, drop repeats
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    task = TASKS[args.task]
    args.wm = args.wm or task.wm
    args.dataset = args.dataset or task.dataset
    args.env = args.env or task.env
    return args, task


def write_results(path, args, rows, successes, episodes, start_steps, planner_train_args):
    """Metrics and per-episode successes of every policy, the evaluated (episode, start step) pairs and
    the planner's training config: everything scripts/aggregate_results.py needs."""
    out = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "episodes": [[int(e), int(s)] for e, s in zip(episodes, start_steps)],
           "planner_train_args": planner_train_args,
           "policies": {name: {**m, "episode_successes": successes[name].tolist()} for name, m in rows}}
    tmp = Path(f"{path}.tmp")  # complete or absent: scripts/run.py skips existing results
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
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
        wm = check_history(swm.policy.AutoCostModel(args.wm).to(args.device))
        wm.eval().requires_grad_(False)
    if any(name.startswith("gflow") for name in args.policy):
        planner, train_args = load_planner(args.planner, args.device)
        if train_args is not None:
            wandb.config.update({"planner_train_args": train_args})
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
