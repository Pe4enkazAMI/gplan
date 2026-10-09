"""LeWorldModel evaluation protocol (le-wm/eval.py with le-wm/config/eval/{tworoom,pusht}.yaml).

Start states and goals are taken from the expert dataset (goal = state
`goal_offset` steps ahead), the agent has `eval_budget` env steps to reach it,
and an episode is a success if the env terminates (TwoRoom: agent within 16 px
of the target; PushT: block and agent within 20 px and block angle within 20 deg).
Every policy is run on the *same* (episode, start step) pairs and env seeds, so
success rates are directly comparable, and the pairs are drawn exactly as
le-wm/eval.py draws them, so they match LeWM's reported CEM numbers.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from gplan.data import episode_column, valid_start_rows
from gplan.lewm import img_transform
from gplan.policy import Sampler
from gplan.solvers import CostLoggingSolver, GFlowSolver


@dataclass(frozen=True)
class Task:
    """Everything that differs between evaluation tasks (values from le-wm/config/eval/<task>.yaml)."""
    env: str                  # gymnasium id
    dataset: str              # .h5 name under $STABLEWM_HOME
    wm: str                   # LeWM checkpoint name under $STABLEWM_HOME
    state_key: str            # dataset column the env is reset to (env._set_state)
    goal_state_key: str       # column of the goal row passed to env._set_goal_state
    cache_keys: tuple         # dataset columns kept in memory; each also gets a normalizer


TASKS = {
    # TwoRoom: proprio = agent position, which is the whole state
    "tworoom": Task(env="swm/TwoRoom-v1", dataset="tworoom", wm="tworoom/lewm",
                    state_key="proprio", goal_state_key="goal_proprio", cache_keys=("action", "proprio")),
    # PushT: proprio is only the agent (position, velocity); the reset needs the full state
    # (agent position, block position and angle, velocity) and success is judged on it
    "pusht": Task(env="swm/PushT-v1", dataset="pusht_expert_train", wm="pusht/lewm",
                  state_key="state", goal_state_key="goal_state", cache_keys=("action", "proprio", "state")),
}


class Standardizer:
    """Z-score normalizer with the sklearn StandardScaler interface used by swm policies.

    Same statistics as the StandardScaler in le-wm/eval.py: rows with NaN dropped, ddof=0,
    and a std of 0 replaced by 1."""

    def __init__(self, data):
        data = data[~np.isnan(data).any(axis=1)]
        self.mean, self.std = data.mean(0), data.std(0)
        self.std = np.where(self.std == 0, 1.0, self.std)

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
    """Pick `num_eval` dataset rows that have at least `goal_offset` steps left in their episode.

    Draws exactly like le-wm/eval.py (including its `len(valid) - 1`), so the same seed gives
    the same (episode, start step) pairs LeWM was evaluated on."""
    ep_col = episode_column(dataset.column_names)
    valid = valid_start_rows(dataset.get_col_data(ep_col), dataset.get_col_data("step_idx"), goal_offset)
    picked = np.random.default_rng(seed).choice(len(valid) - 1, size=num_eval, replace=False)
    rows = dataset.get_row_data(np.sort(valid[picked]))
    return rows[ep_col].tolist(), rows["step_idx"].tolist()


def build_policy(name, args, wm, process, planner, refine=None):
    """(policy, cost-logging solver or None, plans scored by the WM per replanning step).

    "gflow" uses --gflow-select; "gflow-refine" is best-of-N after test-time refinement with the
    `refine` settings (see gplan.solvers.refine_plans)."""
    if name == "random":
        return swm.policy.RandomPolicy(seed=args.seed), None, 0

    if name in ("gflow", "gflow-refine"):
        assert planner is not None, f"--planner is required for the {name} policy"
        select = "refine" if name == "gflow-refine" else args.gflow_select
        solver = GFlowSolver(wm, Sampler(planner), args.num_samples or 64, select, args.device, refine=refine)
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


def evaluate(name, policy, args, task, dataset, episodes, start_steps):
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
            {"method": "_set_state", "args": {"state": {"value": task.state_key}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": task.goal_state_key}}},
        ],
        save_video=args.video_dir is not None,
        video_path=str(Path(args.video_dir) / name) if args.video_dir else "./",
    )
    eval_time = time.time() - t0
    world.close()
    return {"success_rate": float(results["success_rate"]), "eval_time_s": eval_time,
            "episode_successes": np.asarray(results["episode_successes"], dtype=bool)}
