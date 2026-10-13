"""Headroom probe: refinement lowers the objective, expert plans have the right layout, the probe runs."""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import A, D, make_wm, T
from toy_costs import QuadraticCost

from gplan.jax import policy as jp
from gplan.jax.convert import lewm_from_torch
from gplan.jax.probe import make_probe, refine_plans


def test_refine_reaches_the_minimum_of_a_quadratic():
    """Both objectives have their minimum at A = 0; Adam with a constant step ends within ~lr of it."""
    plans = jax.random.normal(jax.random.key(0), (8, T, A)) * 3
    z = jnp.zeros((8, D))
    for objective in ("target", "cost"):
        refined = refine_plans(QuadraticCost(), z, z, plans, beta=2.0, reference_var=1.0, steps=300, lr=0.05,
                               objective=objective)
        assert float(jnp.abs(refined).max()) < 0.2, objective
        assert float(QuadraticCost().cost(z, z, refined).mean()) < 1e-2 * float(QuadraticCost().cost(z, z, plans).mean())


def lewm_with_action_effect():
    """A small LeWM with randomized weights: a freshly built one starts with zero AdaLN gates, which
    makes the predicted embedding (and so the cost) independent of the actions."""
    return lewm_from_torch(make_wm(embed=D, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A))


def test_refine_lowers_the_lewm_cost():
    lewm = lewm_with_action_effect()
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    zs, zg = jax.random.normal(k1, (16, D)), jax.random.normal(k2, (16, D))
    plans = jax.random.normal(k3, (16, T, A))
    refined = refine_plans(lewm, zs, zg, plans, beta=1.0, reference_var=1.0, steps=30, lr=0.05, objective="cost")
    assert float(lewm.cost(zs, zg, refined).mean()) < float(lewm.cost(zs, zg, plans).mean())


def test_probe_runs_and_refinement_helps_on_average():
    lewm = lewm_with_action_effect()
    planner = jp.GPlaner(state_dim=D, horizon=T * A, action_dim=A, key=jax.random.key(0))
    # objective="cost": with "target" the prior term may rightly trade a little J for smaller actions
    probe = make_probe(planner, lewm, (T, A), n_samples=8, beta=0.5, reference_var=1.0, steps=30, lr=0.05,
                       objective="cost")
    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    out = probe(k3, jax.random.normal(k1, (4, D)), jax.random.normal(k2, (4, D)), jnp.zeros((4, T, A)))
    for k in ("sampled_cost", "refined_cost", "prior_cost", "expert_cost", "sampled_mean_cost"):
        assert out[k].shape == (4,), k
    assert np.all(out["sampled_cost"] <= out["sampled_mean_cost"])  # best of N <= mean of N
    assert float(out["refined_cost"].mean()) < float(out["sampled_cost"].mean())
