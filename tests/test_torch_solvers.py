"""Solvers: GFlow best-of-N selection, test-time refinement (same procedure as the JAX probe), cost logging."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import A, B, D, IMG, make_wm, T

from gplan.torch.lewm import encode, lewm_cost
from gplan.torch.policy import GPlanner, Sampler
from gplan.torch.solvers import CostLoggingSolver, GFlowSolver, refine_plans


def lewm_with_action_effect():
    """Randomized weights: a fresh LeWM has zero AdaLN gates, so its cost ignores the actions."""
    return make_wm(embed=D, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A)


@pytest.mark.parametrize("objective", ["target", "cost"])
def test_torch_refinement_matches_jax_probe(objective):
    jax_probe = pytest.importorskip("gplan.jax.probe")
    from gplan.jax.convert import lewm_from_torch

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
    planner = GPlanner(state_dim=D, horizon=T * A, action_dim=A)
    config = SimpleNamespace(horizon=T, action_block=1)
    info = {"pixels": torch.randn(B, 1, 3, IMG, IMG), "goal": torch.randn(B, 1, 3, IMG, IMG)}
    refine = dict(steps=30, lr=0.05, beta=0.5, reference_var=1.0, objective="cost")

    costs = {}
    for select in ("min-cost", "refine"):
        solver = GFlowSolver(wm, Sampler(planner), num_samples=8, select=select, refine=refine)
        solver.configure(action_space=SimpleNamespace(shape=(B, A)), n_envs=B, config=config)
        # both solvers draw the same 8 samples per env: same default seed
        out = solver.solve(info)
        assert out["actions"].shape == (B, T, A)
        costs[select] = np.asarray(out["costs"])
    assert costs["refine"].mean() < costs["min-cost"].mean()
    assert GFlowSolver(wm, None, 8, "refine", refine=refine).plans_per_replan == 8 * 31
    assert GFlowSolver(wm, None, 8, "refine", refine=refine).backward_per_replan == 8 * 30


def test_refine_does_not_touch_world_model_weights():
    wm = lewm_with_action_effect()
    before = {k: v.clone() for k, v in wm.state_dict().items()}
    zs, zg, plans = torch.randn(B, D), torch.randn(B, D), torch.randn(B, T, A)
    refined = refine_plans(wm, zs, zg, plans, steps=5, lr=0.05, beta=0.5, reference_var=1.0, objective="cost")
    assert not refined.requires_grad
    assert lewm_cost(wm, zs, zg, refined).shape == (B,)
    for k, v in wm.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_gflow_solver_owns_its_random_stream():
    """Same seed -> same plans, whatever happened to torch's global random state in between."""
    wm = lewm_with_action_effect()
    torch.manual_seed(0)
    planner = GPlanner(state_dim=D, horizon=T * A, action_dim=A, n_components=2)
    config = SimpleNamespace(horizon=T, action_block=1)
    info = {"pixels": torch.randn(B, 1, 3, IMG, IMG), "goal": torch.randn(B, 1, 3, IMG, IMG)}

    def plan(seed, disturb):
        solver = GFlowSolver(wm, Sampler(planner), num_samples=8, select="min-cost", seed=seed)
        solver.configure(action_space=SimpleNamespace(shape=(B, A)), n_envs=B, config=config)
        torch.manual_seed(disturb)
        torch.rand(disturb + 1)  # consume the global stream differently each time
        return [solver.solve(info)["actions"] for _ in range(2)]  # two replans

    first, second = plan(seed=3, disturb=0), plan(seed=3, disturb=5)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    assert not torch.equal(first[0], first[1])  # the stream advances between replans
    assert not torch.equal(first[0], plan(seed=4, disturb=0)[0])


def configured(solver, n_envs=B):
    solver.configure(action_space=SimpleNamespace(shape=(n_envs, A)), n_envs=n_envs,
                     config=SimpleNamespace(horizon=T, action_block=1))
    return solver


def frames(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"pixels": torch.randn(B, 1, 3, IMG, IMG, generator=g), "goal": torch.randn(B, 1, 3, IMG, IMG, generator=g)}


def test_min_cost_executes_the_cheapest_of_its_samples():
    """Redo the solver's draw by hand (same seed, same generator) and score every sample."""
    wm, info, N = lewm_with_action_effect(), frames(), 6
    torch.manual_seed(0)
    sampler = Sampler(GPlanner(state_dim=D, horizon=T * A, action_dim=A, n_components=2))
    out = configured(GFlowSolver(wm, sampler, num_samples=N, select="min-cost", seed=11)).solve(info)

    zs = encode(wm, info["pixels"][:, -1]).repeat_interleave(N, 0)
    zg = encode(wm, info["goal"][:, -1]).repeat_interleave(N, 0)
    plans = sampler.rollout(zs, zg, generator=torch.Generator().manual_seed(11)).view(B, N, T, A)
    costs = lewm_cost(wm, zs, zg, plans.view(B * N, T, A)).view(B, N)
    torch.testing.assert_close(out["actions"], plans[torch.arange(B), costs.argmin(1)])
    np.testing.assert_allclose(out["costs"], costs.min(1).values.numpy(), rtol=1e-6)


def test_sample_and_mean_execute_a_single_unscored_plan():
    wm, info = lewm_with_action_effect(), frames()
    torch.manual_seed(0)
    sampler = Sampler(GPlanner(state_dim=D, horizon=T * A, action_dim=A, n_components=2))
    mean = configured(GFlowSolver(wm, sampler, num_samples=64, select="mean"))
    assert mean.num_samples == 1 and mean.plans_per_replan == 0 == mean.backward_per_replan
    zs, zg = encode(wm, info["pixels"][:, -1]), encode(wm, info["goal"][:, -1])
    torch.testing.assert_close(mean.solve(info)["actions"], sampler.rollout(zs, zg, mode="mean").view(B, T, A))
    torch.testing.assert_close(mean.solve(info)["actions"], mean.solve(info)["actions"])  # deterministic
    single = configured(GFlowSolver(wm, sampler, num_samples=64, select="sample"))
    assert single.num_samples == 1 and single.plans_per_replan == 0


def test_cost_logging_records_what_was_executed():
    wm = lewm_with_action_effect()
    torch.manual_seed(0)
    inner = GFlowSolver(wm, Sampler(GPlanner(state_dim=D, horizon=T * A, action_dim=A)), num_samples=4)
    logged = configured(CostLoggingSolver(inner, wm))
    outs = [logged.solve(frames(seed)) for seed in (0, 1)]  # two replanning steps
    for out, cost, absmax, seed in zip(outs, logged.costs, logged.action_absmax, (0, 1)):
        info = frames(seed)
        ref = lewm_cost(wm, encode(wm, info["pixels"][:, -1]), encode(wm, info["goal"][:, -1]), out["actions"])
        np.testing.assert_allclose(cost, ref.numpy(), rtol=1e-6)
        np.testing.assert_allclose(absmax, out["actions"].abs().amax(dim=(1, 2)).numpy())
    s = logged.summary()
    assert s["replans"] == 2 and s["plan_cost_first"] == s["plan_cost_replan0"]
    assert np.isclose(s["plan_cost"], np.mean([s["plan_cost_replan0"], s["plan_cost_replan1"]]))
    assert s["action_absmax_max"] == max(a.max() for a in logged.action_absmax)
