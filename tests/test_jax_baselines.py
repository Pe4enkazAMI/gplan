"""Baselines reach their closed-form optima: behavior cloning recovers the expert plan distribution,
amortized MAP planning recovers the minimizer of beta * J + ||A||^2 / (2 s^2)."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import A, D, T
from toy_costs import ShiftedQuadratic

from gplan.jax import policy as jp
from gplan.jax.baselines import bc_loss, map_loss, train_baseline
from gplan.jax.train import init_key, lr_schedule

H = T * A
MEAN = np.linspace(-1.0, 1.5, H).astype(np.float32)   # per action dimension of the plan
STD = np.linspace(0.3, 0.8, H).astype(np.float32)


def test_bc_loss_is_the_negative_log_likelihood():
    model = jp.GPlaner(state_dim=D, horizon=H, action_dim=A, n_components=2, key=init_key(0))
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    zs, zg, plans = jax.random.normal(k1, (5, D)), jax.random.normal(k2, (5, D)), jax.random.normal(k3, (5, T, A))
    loss, _ = bc_loss(model, zs, zg, plans)
    ref = jax.vmap(lambda s, g, p: jp.step_log_probs(model, s, g, p)[0].sum())(zs, zg, plans.reshape(5, H))
    np.testing.assert_allclose(float(loss), -float(ref.mean()), rtol=1e-6)


@pytest.mark.slow
def test_bc_recovers_the_expert_distribution():
    """Expert plans ~ N(MEAN, diag STD^2), independent of the condition: the max-likelihood policy has
    exactly those means and stds at every step."""
    rng = np.random.default_rng(0)

    def batches(n):
        for _ in range(n):
            plans = (MEAN + STD * rng.normal(size=(256, H))).astype(np.float32).reshape(256, T, A)
            yield rng.normal(size=(256, D)).astype(np.float32), rng.normal(size=(256, D)).astype(np.float32), plans

    steps = 2000
    model = jp.GPlaner(state_dim=D, horizon=H, action_dim=A, key=init_key(0))
    model, losses = train_baseline(model, bc_loss, batches(steps), lr=lr_schedule(3e-3, "cosine", steps),
                                   grad_clip=10.0, log_every=10**9)
    zs, zg, plans = next(batches(1))
    _, dists = jax.vmap(lambda s, g, p: jp.step_log_probs(model, s, g, p))(zs, zg, plans.reshape(256, H))
    # Measured on 3 seeds: averaged over conditions, means within 0.017 and stds within 0.010 of the
    # expert's; the NLL within 0.05 nats of its minimum, the expert's entropy. Single conditions vary
    # more (the network also sees the noisy previous actions), so they are not asserted.
    np.testing.assert_allclose(np.asarray(dists.means).reshape(256, H).mean(0), MEAN, atol=0.05)
    np.testing.assert_allclose(np.asarray(dists.stds).reshape(256, H).mean(0), STD, atol=0.05)
    entropy = float(np.sum(0.5 * np.log(2 * np.pi * np.e * STD ** 2)))
    assert abs(np.mean(losses[-100:]) - entropy) < 0.1


@pytest.mark.slow
def test_map_recovers_the_minimizer_of_the_target():
    beta, s2, steps = 2.0, 1.5, 3000
    target = jnp.asarray(MEAN.reshape(T, A))
    minimizer = 2 * beta * MEAN / (2 * beta + 1 / s2)
    rng = np.random.default_rng(0)
    batches = ((rng.normal(size=(64, D)).astype(np.float32), rng.normal(size=(64, D)).astype(np.float32))
               for _ in range(steps))
    loss_fn = partial(map_loss, cost_model=ShiftedQuadratic(target), beta=beta, reference_var=s2, action_dim=A)
    model = jp.GPlaner(state_dim=D, horizon=H, action_dim=A, key=init_key(0))
    model, _ = train_baseline(model, loss_fn, batches, lr=lr_schedule(1e-2, "cosine", steps), grad_clip=10.0,
                              log_every=10**9)
    zs, zg = (jnp.asarray(rng.normal(size=(256, D)).astype(np.float32)) for _ in range(2))  # unseen conditions
    plans = np.asarray(jax.vmap(lambda s, g: jp.rollout(model, jax.random.key(0), s, g, mode="mean"))(zs, zg))
    # measured on two seeds: mean over conditions within 2e-4 of the minimizer, every condition within 0.016
    np.testing.assert_allclose(plans.mean(0), minimizer, atol=2e-3)
    np.testing.assert_allclose(plans, np.broadcast_to(minimizer, plans.shape), atol=0.08)
