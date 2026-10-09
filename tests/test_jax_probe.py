"""Headroom probe: refinement lowers the objective, expert plans have the right layout, the probe runs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from conftest import A, D, T
from test_jax_lewm import make_wm

from gplan_jax import policy as jp
from gplan_jax.convert import lewm_from_torch
from gplan_jax.probe import expert_plans, lewm_action_stats, make_probe, refine_plans


class QuadraticCost(eqx.Module):
    def cost(self, z_start, z_goal, plans):
        return jnp.sum(plans ** 2, axis=(1, 2))


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


def test_expert_plans_layout():
    """Step k of the plan holds env actions r + k*frameskip .. r + (k+1)*frameskip - 1, flattened in order."""
    actions = np.stack([np.arange(40.0), -np.arange(40.0)], axis=1)  # action[i] = (i, -i)
    plans = expert_plans(actions, np.array([3, 20]), goal_offset=10, n_steps=2, mean=0.0, std=1.0)
    assert plans.shape == (2, 2, 10)
    np.testing.assert_array_equal(plans[0, 0], [3, -3, 4, -4, 5, -5, 6, -6, 7, -7])
    np.testing.assert_array_equal(plans[0, 1], [8, -8, 9, -9, 10, -10, 11, -11, 12, -12])
    np.testing.assert_array_equal(plans[1, 1, :2], [25, -25])


def test_lewm_action_stats_drop_nan_rows_and_use_ddof_1():
    actions = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 0.0], [5.0, 9.0]])
    mean, std = lewm_action_stats(actions)
    np.testing.assert_allclose(mean, [3.0, 5.0])
    np.testing.assert_allclose(std, np.std([[1, 2], [3, 4], [5, 9]], axis=0, ddof=1))


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
