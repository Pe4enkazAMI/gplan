"""Evaluate a trained GPlaner (GFlowNet) against CEM as MPC planners on TwoRoom.

Follows the LeWorldModel evaluation protocol (le-wm/config/eval/tworoom.yaml):
start states and goals are taken from the expert dataset (goal = state
`goal_offset` steps ahead), the agent has `eval_budget` env steps to reach it,
and an episode is a success if the env terminates (agent within 16 px of the
target). Every policy is run on the *same* (episode, start step) pairs and env
seeds, so success rates are directly comparable.

    # head-to-head, one wandb run with a comparison table
    python gplaner/eval.py --policy gflow cem --planner gplaner/gplaner.pt

    # just the GFlowNet, no cost re-ranking (1 sample per replanning, no LeWM cost evaluated at all)
    python gplaner/eval.py --policy gflow --planner gplaner/gplaner.pt --gflow-select sample

    python gplaner/eval.py --policy cem        # LeWM paper planner
    python gplaner/eval.py --policy random     # floor

Per policy we log success_rate, wall-clock time, the LeWM cost of the plans that
were actually executed (recomputed identically for every solver) and the number
of plans the world model had to score per replanning step, i.e. the planner's
compute (CEM = num_samples * cem_steps, GFlow best-of-N = num_samples, GFlow
sample/mean = 0).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import encode, img_transform, lewm_cost, valid_start_rows  # noqa: E402  (puts gflow / le-wm on sys.path)
from gflow import GPlaner, Sampler  # noqa: E402

import stable_worldmodel as swm  # noqa: E402


# ----------------------------------------------------------------------------- planner checkpoint

def planner_kwargs_from_state_dict(sd):
    """Recover `GPlaner(...)` constructor kwargs from a bare state_dict (checkpoints saved before
    train.py stored `model_kwargs`). Bounds of the log-std squash are not in the state_dict and
    are assumed to be the defaults."""
    hidden, state_dim = sd["start_proj.weight"].shape
    horizon = sd["action_proj.0.weight"].shape[1]
    action_dim, head_in = sd["mean_head.weight"].shape
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("backbone.")})
    return dict(state_dim=state_dim, horizon=horizon, hidden_size=hidden, n_layers=n_layers, action_dim=action_dim)


def load_planner(path, device):
    """Build a `GPlaner` from a train.py checkpoint. Returns (model, train_args or None)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:               # train.py >= self-describing format
        kwargs, sd, train_args = ckpt["model_kwargs"], ckpt["state_dict"], ckpt.get("train_args")
    else:                                                             # bare state_dict
        kwargs, sd, train_args = planner_kwargs_from_state_dict(ckpt), ckpt, None
    model = GPlaner(**kwargs).to(device)
    model.load_state_dict(sd)
    model.eval().requires_grad_(False)
    print(f"Loaded planner {path}: GPlaner({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    return model, train_args


# ----------------------------------------------------------------------------- solvers

class GFlowSolver:
    """`stable_worldmodel` Solver that plans with the GFlowNet sampler.

    For every env: encode start/goal with the frozen LeWM, sample `num_samples`
    action plans from the GPlaner and pick one according to `select`:
        "min-cost": score all plans with the LeWM cost and execute the cheapest (best-of-N);
        "sample":   execute the first sample as is (num_samples=1 -> zero cost evaluations);
        "mean":     execute the policy's mean plan (deterministic).
    """

    def __init__(self, wm, sampler, num_samples=64, select="min-cost", device="cpu"):
        assert select in ("min-cost", "sample", "mean"), select
        self.wm = wm
        self.sampler = sampler
        self.num_samples = num_samples if select == "min-cost" else 1
        self.select = select
        self.device = device

    @property
    def plans_per_replan(self):
        """Plans the world model scores per replanning step (0 if the plan is executed unscored)."""
        return self.num_samples if self.select == "min-cost" else 0

    def configure(self, *, action_space, n_envs, config):
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._n_envs = n_envs
        self._config = config
        assert self.sampler.model.horizon == self.horizon * self.action_dim, (
            f"GPlaner horizon {self.sampler.model.horizon} != "
            f"plan horizon {self.horizon} x action dim {self.action_dim}"
        )
        assert self.sampler.model.action_dim in (1, self.action_dim), (
            f"GPlaner action_dim {self.sampler.model.action_dim} != env action dim x action_block {self.action_dim}"
        )

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        z_start = encode(self.wm, info_dict["pixels"][:, -1].to(self.device))  # (E, D)
        z_goal = encode(self.wm, info_dict["goal"][:, -1].to(self.device))
        E, N = self.n_envs, self.num_samples

        z_start = z_start.repeat_interleave(N, dim=0)  # (E*N, D)
        z_goal = z_goal.repeat_interleave(N, dim=0)
        mode = "mean" if self.select == "mean" else "sample"
        plans = self.sampler.rollout(z_start, z_goal, mode=mode).view(E * N, self.horizon, self.action_dim)
        costs = lewm_cost(self.wm, z_start, z_goal, plans).view(E, N)  # also used for logging when N == 1

        best = costs.argmin(dim=1)
        actions = plans.view(E, N, self.horizon, self.action_dim)[torch.arange(E, device=best.device), best]
        return {"actions": actions.cpu(), "costs": costs.min(dim=1).values.cpu().tolist()}


class CostLoggingSolver:
    """Wraps any swm Solver and records the LeWM cost of the plan it returns on every replanning step.

    The cost is recomputed here with `lewm_cost` on the executed plan, so it means the same
    thing for every solver (CEM's own `outputs["costs"]` is the mean over its elites, not the
    cost of the mean plan it executes). Also accumulates solver wall-clock time.
    """

    def __init__(self, solver, wm, device="cpu"):
        self.solver = solver
        self.wm = wm
        self.device = device
        self.costs = []      # one (n_envs,) array per replanning step
        self.solve_time = 0.0

    def configure(self, **kwargs):
        self.solver.configure(**kwargs)

    @property
    def n_envs(self):
        return self.solver.n_envs

    @property
    def action_dim(self):
        return self.solver.action_dim

    @property
    def horizon(self):
        return self.solver.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        t0 = time.time()
        out = self.solver.solve(info_dict, init_action=init_action)
        self.solve_time += time.time() - t0
        z_start = encode(self.wm, info_dict["pixels"][:, -1].to(self.device))
        z_goal = encode(self.wm, info_dict["goal"][:, -1].to(self.device))
        plans = out["actions"].to(self.device).view(self.n_envs, self.horizon, self.action_dim)
        self.costs.append(lewm_cost(self.wm, z_start, z_goal, plans).cpu().numpy())
        return out

    def summary(self):
        if not self.costs:
            return {"plan_cost": float("nan"), "replans": 0, "solve_time_s": self.solve_time}
        costs = np.stack(self.costs)  # (replans, n_envs)
        return {"plan_cost": float(costs.mean()), "plan_cost_first": float(costs[0].mean()),
                "replans": int(costs.shape[0]), "solve_time_s": self.solve_time}


# ----------------------------------------------------------------------------- data helpers

class Standardizer:
    """Z-score normalizer with the sklearn StandardScaler interface used by swm policies."""

    def __init__(self, data):
        data = data[~np.isnan(data).any(axis=1)]
        self.mean, self.std = data.mean(0), data.std(0)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def fit_normalizers(dataset, cols):
    """Normalizers for dataset columns; the goal copy of a column shares its normalizer."""
    process = {}
    for col in cols:
        process[col] = Standardizer(dataset.get_col_data(col))
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_starts(dataset, num_eval, goal_offset, seed):
    """Pick `num_eval` dataset rows that have at least `goal_offset` steps left in their episode."""
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    valid = valid_start_rows(dataset.get_col_data(ep_col), dataset.get_col_data("step_idx"), goal_offset)
    rows = np.sort(np.random.default_rng(seed).choice(valid, size=num_eval, replace=False))
    rows = dataset.get_row_data(rows)
    return rows[ep_col].tolist(), rows["step_idx"].tolist()


# ----------------------------------------------------------------------------- evaluation

def build_policy(name, args, wm, dataset, process, planner):
    """(policy, cost-logging solver or None, plans scored by the WM per replanning step)."""
    if name == "random":
        return swm.policy.RandomPolicy(seed=args.seed), None, 0

    if name == "gflow":
        assert planner is not None, "--planner is required for the gflow policy"
        solver = GFlowSolver(wm, Sampler(planner), args.num_samples or 64, args.gflow_select, args.device)
        plans_per_replan = solver.plans_per_replan
    else:  # LeWM paper planner (le-wm/config/eval/solver/cem.yaml); batch_size=1 is required by JEPA.criterion
        solver = swm.solver.CEMSolver(
            model=wm, batch_size=1, num_samples=args.num_samples or 300, var_scale=1.0,
            n_steps=args.cem_steps, topk=args.cem_topk, device=args.device, seed=args.seed,
        )
        plans_per_replan = solver.num_samples * solver.n_steps

    solver = CostLoggingSolver(solver, wm, args.device)
    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.horizon, action_block=args.action_block)
    transform = img_transform(args.img_size)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process,
        transform={"pixels": transform, "goal": transform},
    )
    return policy, solver, plans_per_replan


def evaluate(name, policy, args, dataset, episodes, start_steps):
    """Run the LeWM protocol for one policy on the given episodes. Fresh World -> identical env seeds."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    world = swm.World(
        env_name=args.env, num_envs=args.num_eval, image_shape=(224, 224),
        max_episode_steps=max(2 * args.eval_budget, args.goal_offset),  # swm requires >= goal_offset
    )
    world.set_policy(policy)
    t0 = time.time()
    results = world.evaluate_from_dataset(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=start_steps,
        goal_offset_steps=args.goal_offset,
        eval_budget=args.eval_budget,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
        save_video=args.video_dir is not None,
        video_path=str(Path(args.video_dir) / name) if args.video_dir else "./",
    )
    eval_time = time.time() - t0
    world.close()
    return {"success_rate": float(results["success_rate"]), "eval_time_s": eval_time,
            "episode_successes": np.asarray(results["episode_successes"], dtype=bool)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", nargs="+", choices=["gflow", "cem", "random"], default=["gflow", "cem"],
                   help="policies to evaluate on the same episodes")
    p.add_argument("--planner", default=None, help="[gflow] train.py checkpoint (.pt)")
    p.add_argument("--gflow-select", default="min-cost", choices=["min-cost", "sample", "mean"],
                   help="[gflow] execute the cheapest of --num-samples plans, a single sample, or the mean plan")
    p.add_argument("--wm", default="tworoom/lewm", help="LeWM ckpt name relative to $STABLEWM_HOME")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--env", default="swm/TwoRoom-v1")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=3)
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

    torch.manual_seed(args.seed)
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode,
               job_type="eval")

    # -- data: start/goal pairs (shared by every policy) and normalization stats
    dataset = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action", "proprio"])
    process = fit_normalizers(dataset, ["action", "proprio"])
    episodes, start_steps = sample_eval_starts(dataset, args.num_eval, args.goal_offset, args.seed)

    # -- shared frozen world model and (optionally) the trained planner
    wm = planner = None
    if any(name != "random" for name in args.policy):
        wm = swm.policy.AutoCostModel(args.wm).to(args.device)
        wm.eval().requires_grad_(False)
    if "gflow" in args.policy:
        planner, train_args = load_planner(args.planner, args.device)
        if train_args is not None:
            wandb.config.update({"planner_train_args": train_args})
            for k, eval_v in (("goal_offset", args.goal_offset), ("n_steps", args.horizon)):
                if k in train_args and train_args[k] != eval_v:
                    print(f"WARNING: planner was trained with {k}={train_args[k]} but eval uses {eval_v}")
        action_dim = dataset.get_col_data("action").shape[1] * args.action_block
        assert planner.horizon == args.horizon * action_dim, (
            f"planner horizon {planner.horizon} != --horizon {args.horizon} x action dim {action_dim}")

    # -- evaluate every policy on the same episodes
    rows, successes = [], {}
    for name in args.policy:
        print(f"\n===== {name} =====")
        policy, solver, plans_per_replan = build_policy(name, args, wm, dataset, process, planner)
        res = evaluate(name, policy, args, dataset, episodes, start_steps)
        successes[name] = res.pop("episode_successes")
        metrics = {**res, "plans_per_replan": plans_per_replan, **(solver.summary() if solver else {})}
        rows.append((name, metrics))
        wandb.log({f"{name}/{k}": v for k, v in metrics.items()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    # -- comparison
    cols = ["policy", "success_rate", "plan_cost", "plans_per_replan", "solve_time_s", "eval_time_s"]
    table = wandb.Table(columns=cols)
    print(f"\n{'policy':8s} {'success%':>9s} {'plan_cost':>10s} {'plans/replan':>13s} {'solve_s':>8s} {'total_s':>8s}")
    for name, m in rows:
        vals = [m.get(c, float("nan")) for c in cols[1:]]
        table.add_data(name, *vals)
        print(f"{name:8s} {vals[0]:9.1f} {vals[1]:10.2f} {vals[2]:13d} {vals[3]:8.1f} {vals[4]:8.1f}")
    if len(successes) == 2:  # per-episode agreement between the two planners
        a, b = args.policy
        both, only_a, only_b = (successes[a] & successes[b]).sum(), (successes[a] & ~successes[b]).sum(), \
            (~successes[a] & successes[b]).sum()
        print(f"\nepisodes solved by both: {both}, only {a}: {only_a}, only {b}: {only_b}, neither: "
              f"{args.num_eval - both - only_a - only_b}")
        wandb.log({"both_solved": int(both), f"only_{a}": int(only_a), f"only_{b}": int(only_b)})
    wandb.log({"comparison": table})
    wandb.finish()


if __name__ == "__main__":
    main()
