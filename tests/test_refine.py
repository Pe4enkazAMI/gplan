"""Test-time refinement in the PyTorch solver: same procedure as the JAX probe, and it lowers the cost."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import A, B, D, IMG, T
from test_jax_lewm import make_wm

from gplan.lewm import lewm_cost
from gplan.policy import GPlaner, Sampler
from gplan.solvers import GFlowSolver, refine_plans


def lewm_with_action_effect():
    """Randomized weights: a fresh LeWM has zero AdaLN gates, so its cost ignores the actions."""
    return make_wm(embed=D, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A)


@pytest.mark.parametrize("objective", ["target", "cost"])
def test_torch_refinement_matches_jax_probe(objective):
    jax_probe = pytest.importorskip("gplan_jax.probe")
    from gplan_jax.convert import lewm_from_torch

    wm = lewm_with_action_effect()
    g = torch.Generator().manual_seed(0)
    zs, zg, plans = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g), torch.randn(B, T, A, generator=g)
    settings = dict(steps=10, lr=0.05, beta=0.5, reference_var=1.0, objective=objective)

    ours = refine_plans(wm, zs, zg, plans, **settings).numpy()
    ref = np.asarray(jax_probe.refine_plans(lewm_from_torch(wm), zs.numpy(), zg.numpy(), plans.numpy(), **settings))
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-5)


def test_refine_solver_lowers_the_cost_of_the_executed_plan():
    wm = lewm_with_action_effect()
    torch.manual_seed(0)
    planner = GPlaner(state_dim=D, horizon=T * A, action_dim=A)
    config = SimpleNamespace(horizon=T, action_block=1)
    info = {"pixels": torch.randn(B, 1, 3, IMG, IMG), "goal": torch.randn(B, 1, 3, IMG, IMG)}
    refine = dict(steps=30, lr=0.05, beta=0.5, reference_var=1.0, objective="cost")

    costs = {}
    for select in ("min-cost", "refine"):
        solver = GFlowSolver(wm, Sampler(planner), num_samples=8, select=select, refine=refine)
        solver.configure(action_space=SimpleNamespace(shape=(B, A)), n_envs=B, config=config)
        torch.manual_seed(1)  # the same 8 samples per env for both solvers
        out = solver.solve(info)
        assert out["actions"].shape == (B, T, A)
        costs[select] = np.asarray(out["costs"])
    assert costs["refine"].mean() < costs["min-cost"].mean()
    assert GFlowSolver(wm, None, 8, "refine", refine=refine).plans_per_replan == 8 * 31


def test_refine_does_not_touch_world_model_weights():
    wm = lewm_with_action_effect()
    before = {k: v.clone() for k, v in wm.state_dict().items()}
    zs, zg, plans = torch.randn(B, D), torch.randn(B, D), torch.randn(B, T, A)
    refined = refine_plans(wm, zs, zg, plans, steps=5, lr=0.05, beta=0.5, reference_var=1.0, objective="cost")
    assert not refined.requires_grad
    assert lewm_cost(wm, zs, zg, refined).shape == (B,)
    for k, v in wm.state_dict().items():
        assert torch.equal(v, before[k]), k
