"""The losses are exactly zero at the closed-form optimum, for any plan (deterministic, no training).

With J(A) = ||A||^2 and the reference N(0, s^2 I), the target P*(A) ∝ exp(-beta J) N(A; 0, s^2 I) is
N(0, sigma^2 I) with 1 / sigma^2 = 2 beta + 1 / s^2, and log Z = -(H / 2) log(1 + 2 beta s^2). A GPlaner
whose every step is that Gaussian and whose log Z head outputs that constant makes the TB residual
    log Z + log P_F(A) + beta J(A) - log N(A)
vanish for EVERY plan A, on-policy or not: the quadratic terms and the constants cancel. So TB is
exactly 0 there and its gradient vanishes; moving the policy away makes it positive.
This pins down that the code's objective has its minimum at the analytic answer, independently of
whether an optimizer gets there (tests/test_jax_train.py checks that part).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import A, D, T
from toy_costs import QuadraticCost

from gplan.jax import policy as jp
from gplan.jax.losses import tb_loss
from gplan.jax.mala import mala

BETA, S2, H = 2.0, 1.5, T * A
SIGMA2 = 1 / (2 * BETA + 1 / S2)
LOG_Z = -H / 2 * math.log(1 + 2 * BETA * S2)


def optimal_planer(n_components, std=math.sqrt(SIGMA2), log_z=LOG_Z):
    """A GPlaner that ignores its inputs: every step is N(0, std^2) (all mixture components equal,
    arbitrary weights) and log Z(c) = log_z."""
    model = jp.GPlaner(state_dim=D, horizon=H, action_dim=A, n_components=n_components, key=jax.random.key(0))
    lo, hi = model.log_std_min, model.log_std_max
    raw_std = math.atanh(2 * (math.log(std) - lo) / (hi - lo) - 1)  # inverts the tanh squash
    K = n_components
    bias = [jnp.zeros(K * A), jnp.full(K * A, raw_std)] + ([jnp.array([0.3, -1.0, 2.0][:K])] if K > 1 else [])
    out = model.out_proj
    model = eqx.tree_at(lambda m: (m.out_proj.weight, m.out_proj.bias), model,
                        (jnp.zeros_like(out.weight), jnp.concatenate(bias)))
    z_out = model.Z[1]
    return eqx.tree_at(lambda m: (m.Z[1].weight, m.Z[1].bias), model,
                       (jnp.zeros_like(z_out.weight), jnp.full(z_out.bias.shape, log_z / z_out.bias.shape[0])))


def batch(n=32, scale=1.0, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    zs, zg = jax.random.normal(k1, (n, D)), jax.random.normal(k2, (n, D))
    return zs, zg, scale * jax.random.normal(k3, (n, H))


def losses(model, zs, zg, plans):
    """(TB loss, TB stats)."""
    cost = QuadraticCost().cost(zs, zg, plans.reshape(len(plans), -1, A))
    return tb_loss(model, zs, zg, plans, cost, BETA, S2)


def grad_norm(model, zs, zg, plans):
    return float(optax.tree.norm(eqx.filter_grad(lambda m: losses(m, zs, zg, plans)[0])(model)))


@pytest.mark.parametrize("n_components", [1, 3])
@pytest.mark.parametrize("plans_from", ["policy", "far off-policy", "mala"])
def test_losses_vanish_at_the_optimum_for_any_plan(n_components, plans_from):
    model = optimal_planer(n_components)
    zs, zg, noise = batch(scale=3.0 if plans_from == "far off-policy" else 1.0)
    if plans_from == "policy":
        plans = jax.vmap(lambda k, s, g: jp.rollout(model, k, s, g))(jax.random.split(jax.random.key(1), 32), zs, zg)
    elif plans_from == "mala":
        plans, _, _ = mala(QuadraticCost(), jax.random.key(2), zs, zg, noise, BETA, S2, A, steps=5, step_size=0.05)
    else:
        plans = noise
    tb, stats = losses(model, zs, zg, plans)
    assert float(stats["tb/residual_abs"]) < 1e-4  # every plan's residual, not just the mean
    assert float(tb) < 1e-8
    np.testing.assert_allclose(float(stats["tb/log_z"]), LOG_Z, rtol=1e-6)

    # float32 leaves residuals of ~1e-6, so the gradient is not exactly 0; it is ~3e-5 of the gradient
    # 1% away from the optimum (measured: 0.0025 vs 86 for these plans, 0.41 vs 12468 for 3x wider ones)
    near = optimal_planer(n_components, std=1.01 * math.sqrt(SIGMA2))
    assert grad_norm(model, zs, zg, plans) < 1e-3 * grad_norm(near, zs, zg, plans)


def test_moving_away_from_the_optimum_costs_loss():
    zs, zg, plans = batch(scale=1.5)
    opt_tb, _ = losses(optimal_planer(1), zs, zg, plans)
    # a wrong log Z shifts every residual by the same amount: TB = shift^2
    tb, _ = losses(optimal_planer(1, log_z=LOG_Z + 0.1), zs, zg, plans)
    np.testing.assert_allclose(float(tb), 0.01, rtol=1e-3)
    # a wrong std (here: the prior's, i.e. ignoring the cost) is penalized
    for std in (math.sqrt(S2), 0.8 * math.sqrt(SIGMA2), 1.25 * math.sqrt(SIGMA2)):
        tb, _ = losses(optimal_planer(1, std=std), zs, zg, plans)
        assert float(tb) > 100 * float(opt_tb) + 1e-3, std
