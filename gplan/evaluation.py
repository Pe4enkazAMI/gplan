"""LeWorldModel evaluation protocol on TwoRoom (le-wm/config/eval/tworoom.yaml).

Start states and goals are taken from the expert dataset (goal = state
`goal_offset` steps ahead), the agent has `eval_budget` env steps to reach it,
and an episode is a success if the env terminates (agent within 16 px of the
target). Every policy is run on the *same* (episode, start step) pairs and env
seeds, so success rates are directly comparable.
"""

import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from gplan.data import episode_column, valid_start_rows
from gplan.lewm import img_transform
from gplan.policy import Sampler
from gplan.solvers import CostLoggingSolver, GFlowSolver


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
    ep_col = episode_column(dataset.column_names)
    valid = valid_start_rows(dataset.get_col_data(ep_col), dataset.get_col_data("step_idx"), goal_offset)
    rows = np.sort(np.random.default_rng(seed).choice(valid, size=num_eval, replace=False))
    rows = dataset.get_row_data(rows)
    return rows[ep_col].tolist(), rows["step_idx"].tolist()


def build_policy(name, args, wm, process, planner):
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
