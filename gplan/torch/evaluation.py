"""LeWorldModel evaluation protocol (le-wm/eval.py with le-wm/config/eval/{tworoom,pusht}.yaml).

Start states and goals are taken from the expert dataset (goal = state
`goal_offset` steps ahead), the agent has `eval_budget` env steps to reach it,
and an episode is a success if the env terminates (TwoRoom: agent within 16 px
of the target; PushT: block and agent within 20 px and block angle within 20 deg).
Every policy is run on the *same* (episode, start step) pairs and env seeds, so
success rates are directly comparable, and the pairs are drawn exactly as
le-wm/eval.py draws them, so they match LeWM's reported CEM numbers.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from gplan.data import episode_column, split_episodes, valid_start_rows
from gplan.torch.lewm import img_transform
from gplan.torch.policy import PriorSampler, Sampler
from gplan.torch.solvers import CostLoggingSolver, GFlowSolver


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
    and a std of 0 replaced by 1. (LeWM's training normalizer uses ddof=1; see
    `gplan.jax.probe.lewm_action_stats` for why the difference does not matter.)"""

    def __init__(self, data):
        data = data[~np.isnan(data).any(axis=1)]
        self.mean, self.std = data.mean(0), data.std(0)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def use_deterministic_algorithms():
    """Bit-reproducible PyTorch on GPU (deterministic cuDNN / cuBLAS kernels, slower). Call it before the
    first CUDA operation. Ops without a deterministic kernel warn instead of failing (warn_only)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_normalizers(dataset, cols):
    """Normalizers for dataset columns; the goal copy of a column shares its normalizer."""
    process = {}
    for col in cols:
        process[col] = Standardizer(dataset.get_col_data(col))
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_starts(dataset, num_eval, goal_offset, seed, episodes=None):
    """Pick `num_eval` dataset rows that have at least `goal_offset` steps left in their episode.

    Draws exactly like le-wm/eval.py (including its `len(valid) - 1`), so with `episodes=None` the
    same seed gives the same (episode, start step) pairs LeWM was evaluated on. `episodes` restricts
    the draw to those episode ids (e.g. the held-out side of `gplan.data.split_episodes`)."""
    ep_col = episode_column(dataset.column_names)
    ep_idx = dataset.get_col_data(ep_col)
    valid = valid_start_rows(ep_idx, dataset.get_col_data("step_idx"), goal_offset)
    if episodes is not None:
        valid = valid[np.isin(ep_idx[valid], episodes)]
    if len(valid) - 1 < num_eval:
        raise ValueError(f"only {len(valid)} valid start rows for {num_eval} evaluation episodes")
    picked = np.random.default_rng(seed).choice(len(valid) - 1, size=num_eval, replace=False)
    rows = dataset.get_row_data(np.sort(valid[picked]))
    return rows[ep_col].tolist(), rows["step_idx"].tolist()


def eval_episode_pool(dataset, holdout_frac, split_seed):
    """Episodes evaluation may draw from: all of them (holdout_frac = 0, the LeWM protocol) or the
    held-out side of the split training used."""
    if holdout_frac <= 0:
        return None
    return split_episodes(dataset.get_col_data(episode_column(dataset.column_names)), holdout_frac, split_seed)[1]


def trained_on(dataset, episodes, train_args):
    """Which of `episodes` were in a planner's training data, from the split recorded in its
    train_args (no split recorded = it trained on every episode)."""
    train_args = train_args or {}
    ep_idx = dataset.get_col_data(episode_column(dataset.column_names))
    train_eps, _ = split_episodes(ep_idx, train_args.get("holdout_frac", 0.0), train_args.get("split_seed", 0))
    return sorted(set(episodes) & set(train_eps.tolist()))


def build_policy(name, args, wm, process, planner, refine=None):
    """(policy, cost-logging solver or None, compute per replanning step).

    compute = {"plans_per_replan": forward rollouts of the world model,
               "backward_per_replan": backward passes through it (refine policies only)}.
    "gflow" uses --gflow-select; "gflow-refine" is best-of-N after test-time refinement with the
    `refine` settings (see gplan.torch.solvers.refine_plans). "prior" / "prior-refine" do the same with
    --prior-samples plans from N(0, --prior-var I) instead of the GPlaner: random shooting, and
    gradient planning from random starts (the control for gflow-refine)."""
    if name == "random":
        return swm.policy.RandomPolicy(seed=args.seed), None, {"plans_per_replan": 0, "backward_per_replan": 0}

    if name in ("gflow", "gflow-refine", "prior", "prior-refine"):
        if name.startswith("gflow"):
            assert planner is not None, f"--planner is required for the {name} policy"
            sampler, n_samples, select = Sampler(planner), args.gflow_samples, args.gflow_select
        else:
            action_dim = process["action"].mean.shape[0] * args.action_block
            sampler = PriorSampler(args.horizon * action_dim, action_dim, args.prior_var)
            sampler, n_samples, select = sampler, args.prior_samples, "min-cost"
        if name.endswith("-refine"):
            select = "refine"
        solver = GFlowSolver(wm, sampler, n_samples, select, args.device, refine=refine, seed=args.seed)
        compute = {"plans_per_replan": solver.plans_per_replan, "backward_per_replan": solver.backward_per_replan}
    else:  # LeWM paper planner (le-wm/config/eval/solver/cem.yaml); batch_size=1 is required by JEPA.criterion
        solver = swm.solver.CEMSolver(
            model=wm, batch_size=1, num_samples=args.cem_samples, var_scale=1.0,
            n_steps=args.cem_steps, topk=args.cem_topk, device=args.device, seed=args.seed,
        )
        compute = {"plans_per_replan": solver.num_samples * solver.n_steps, "backward_per_replan": 0}

    solver = CostLoggingSolver(solver, wm, args.device)
    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.horizon, action_block=args.action_block)
    transform = img_transform(args.img_size)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process,
        transform={"pixels": transform, "goal": transform},
    )
    return policy, solver, compute


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
