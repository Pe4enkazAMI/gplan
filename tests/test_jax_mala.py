"""MALA local search samples the GFlowNet target, and off-policy training on its plans works."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import D
from toy_costs import DoubleWell, QuadraticCost

from gplan.jax import policy as jp
from gplan.jax.mala import mala
from gplan.jax.train import adapt_step_size, init_key, train, TrainConfig


def test_mala_samples_the_gaussian_target():
    """J = ||A||^2 with N(0, s^2 I): the target is N(0, 1 / (2 beta + 1 / s^2)) per action."""
    beta, s2, n, H, A = 2.0, 1.0, 4000, 4, 2
    z = jnp.zeros((n, D))
    start = jnp.full((n, H), 3.0)  # far from the target's bulk
    plans, cost, accept = mala(QuadraticCost(), jax.random.key(0), z, z, start, beta, s2, A, steps=400,
                               step_size=0.05)
    target_var = 1 / (2 * beta + 1 / s2)
    assert abs(float(plans.mean())) < 0.02
    assert abs(float(plans.var()) / target_var - 1) < 0.05
    np.testing.assert_allclose(np.asarray(cost), np.asarray(jnp.sum(plans ** 2, -1)), rtol=1e-5)
    assert 0.5 < float(accept) < 1.0


def test_mala_with_tiny_steps_accepts_almost_everything():
    z = jnp.zeros((256, D))
    start = jax.random.normal(jax.random.key(1), (256, 4))
    _, _, accept = mala(QuadraticCost(), jax.random.key(0), z, z, start, 2.0, 1.0, 2, steps=20, step_size=1e-6)
    assert float(accept) > 0.99


def test_step_size_adaptation():
    assert adapt_step_size(0.1, 0.9, 0.574) > 0.1 > adapt_step_size(0.1, 0.2, 0.574)
    assert adapt_step_size(0.1, 0.9, 0.0) == 0.1  # target 0 keeps the step size fixed


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mixture_with_local_search_gives_both_modes_their_mass(seed):
    """Double well, 2-component mixture, TB with MALA plans: about half of the samples on each side.

    Without MALA, on-policy TB keeps a single well in about half of the seeds (it never samples the
    other one, and its loss stays low). With MALA every seed we tried gets the mass right; what
    can still go wrong is the shape: on some seeds both components sit at 0 with a wide std,
    covering both wells instead of fitting each (a symmetric stationary point), so only the mass
    is asserted per seed."""
    model = jp.GPlaner(state_dim=D, horizon=1, hidden_size=32, n_components=2, key=init_key(seed))
    batch = (np.zeros((64, D), np.float32), np.zeros((64, D), np.float32))
    cfg = TrainConfig(loss_type="tb", n_samples=1, action_dim=1, reference_var=4.0, grad_clip=10.0,
                      ls_steps=10, ls_step_size=0.01)
    model, _ = train(model, DoubleWell(), [batch] * 1500, 1.0, cfg, lr=1e-3, lr_z=1e-2, seed=seed, log_every=10**9)
    z = jnp.zeros((4000, D))
    a = jax.vmap(lambda k, s: jp.rollout(model, k, s, s))(jax.random.split(jax.random.key(99), 4000), z)[:, 0]
    assert abs(float((a > 0).mean()) - 0.5) < 0.1
