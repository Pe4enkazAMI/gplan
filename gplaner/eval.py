"""Evaluate the GPlaner sampler as an MPC planner on TwoRoom, logging success rate to wandb.

Follows the LeWorldModel evaluation protocol (le-wm/config/eval/tworoom.yaml):
start states and goals are taken from the expert dataset (goal = state 25 steps
ahead), the agent has 50 env steps to reach it, and an episode is a success if
the env terminates (agent within 16 px of the target).

    python gplaner/eval.py --policy gflow --planner gplaner/gplaner.pt   # GFlowNet sampler + LeWM cost
    python gplaner/eval.py --policy cem                                   # LeWM paper planner (CEM)
    python gplaner/eval.py --policy random
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


class GFlowSolver:
    """`stable_worldmodel` Solver that plans by sampling from the GFlowNet.

    For every env: encode start/goal with the frozen LeWM, sample `num_samples`
    action plans from the GPlaner sampler, score them with the LeWM cost and
    return the cheapest one.
    """

    def __init__(self, wm, sampler, num_samples=64, device="cpu"):
        self.wm = wm
        self.sampler = sampler
        self.num_samples = num_samples
        self.device = device

    def configure(self, *, action_space, n_envs, config):
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._n_envs = n_envs
        self._config = config
        assert self.sampler.model.horizon == self.horizon * self.action_dim, (
            f"GPlaner horizon {self.sampler.model.horizon} != "
            f"plan horizon {self.horizon} x action dim {self.action_dim}"
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
        plans = self.sampler.rollout(z_start, z_goal).view(E * N, self.horizon, self.action_dim)
        costs = lewm_cost(self.wm, z_start, z_goal, plans).view(E, N)

        best = costs.argmin(dim=1)
        actions = plans.view(E, N, self.horizon, self.action_dim)[torch.arange(E), best]
        return {"actions": actions.cpu(), "costs": costs.min(dim=1).values.cpu()}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["gflow", "cem", "random"], default="gflow")
    p.add_argument("--planner", default=None, help="[gflow] GPlaner state_dict (.pt); random init if omitted")
    p.add_argument("--wm", default="tworoom/lewm", help="LeWM ckpt name relative to $STABLEWM_HOME")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--env", default="swm/TwoRoom-v1")
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
    p.add_argument("--video-dir", default=None, help="save rollout videos here (off by default)")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)

    # -- data: start/goal pairs and normalization stats
    dataset = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action", "proprio"])
    process = fit_normalizers(dataset, ["action", "proprio"])
    episodes, start_steps = sample_eval_starts(dataset, args.num_eval, args.goal_offset, args.seed)

    # -- policy
    if args.policy == "random":
        policy = swm.policy.RandomPolicy(seed=args.seed)
    else:
        wm = swm.policy.AutoCostModel(args.wm).to(args.device)
        wm.eval().requires_grad_(False)

        if args.policy == "gflow":
            embed_dim = wm.predictor.pos_embedding.shape[-1]
            action_dim = dataset.get_col_data("action").shape[1] * args.action_block
            planner = GPlaner(state_dim=embed_dim, horizon=args.horizon * action_dim).to(args.device)
            if args.planner:
                planner.load_state_dict(torch.load(args.planner, map_location=args.device))
            planner.eval()
            solver = GFlowSolver(wm, Sampler(planner), args.num_samples or 64, args.device)
        else:  # LeWM paper planner (le-wm/config/eval/solver/cem.yaml); batch_size=1 is required by JEPA.criterion
            solver = swm.solver.CEMSolver(
                model=wm, batch_size=1, num_samples=args.num_samples or 300, var_scale=1.0,
                n_steps=args.cem_steps, topk=args.cem_topk, device=args.device, seed=args.seed,
            )

        config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.horizon, action_block=args.action_block)
        transform = img_transform(args.img_size)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process,
            transform={"pixels": transform, "goal": transform},
        )

    # -- environment (one env per evaluated episode, as in the LeWM protocol)
    world = swm.World(
        env_name=args.env, num_envs=args.num_eval, image_shape=(224, 224),
        max_episode_steps=2 * args.eval_budget,
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
        video_path=args.video_dir or "./",
    )
    eval_time = time.time() - t0

    metrics = {"success_rate": results["success_rate"], "num_eval": args.num_eval, "eval_time_s": eval_time}
    print(metrics)
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
