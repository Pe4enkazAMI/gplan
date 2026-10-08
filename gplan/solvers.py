"""`stable_worldmodel`-compatible solvers: the GFlowNet planner and a cost-logging wrapper."""

import time

import numpy as np
import torch

from gplan.lewm import encode, lewm_cost


class GFlowSolver:
    """`stable_worldmodel` Solver that plans with the GFlowNet sampler.

    For every env: encode start/goal with the frozen LeWM, sample `num_samples`
    action plans from the GPlaner and pick one according to `select`:
        "min-cost": score all plans with the LeWM cost and execute the cheapest (best-of-N);
        "sample":   execute a single sample as is;
        "mean":     execute the policy's mean plan (deterministic).
    With "sample"/"mean" the cost of the single plan is still computed, for logging only.
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
        costs = lewm_cost(self.wm, z_start, z_goal, plans).view(E, N)

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
