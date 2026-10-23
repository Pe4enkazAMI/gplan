"""PPO baseline: GAE, the clipped losses, rewards, the critic, and convergence to closed-form optima."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import A, D, T
from toy_costs import QuadraticCost, ShiftedQuadratic

from gplan.jax import policy as jp
from gplan.jax.distributions import normal_log_prob
from gplan.jax.ppo import (
    clipped_surrogate,
    Critic,
    critic_key,
    gae,
    PPOConfig,
    step_rewards,
    train_ppo,
)
from gplan.jax.train import init_key


def test_gae_matches_hand_computation():
    r = jnp.array([[1.0, 0.0, 2.0]])
    v = jnp.array([[0.5, 1.0, 1.5]])
    # lambda = 1, gamma = 1: Monte-Carlo returns (3, 2, 2); advantages = returns - V
    adv, ret = gae(r, v, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret, [[3.0, 2.0, 2.0]])
    np.testing.assert_allclose(adv, [[2.5, 1.0, 0.5]])
    # lambda = 0: one-step TD errors r_t + V(s_t+1) - V(s_t), with V after the last step = 0
    adv, ret = gae(r, v, gamma=1.0, lam=0.0)
    np.testing.assert_allclose(adv, [[1.0 + 1.0 - 0.5, 0.0 + 1.5 - 1.0, 2.0 + 0.0 - 1.5]])
    np.testing.assert_allclose(ret, adv + v)
    # general case against the definition A_t = sum_k (gamma lambda)^k delta_{t+k}
    g, lam = 0.9, 0.7
    delta = r + g * jnp.array([[1.0, 1.5, 0.0]]) - v
    expected = [sum((g * lam) ** k * delta[0, t + k] for k in range(3 - t)) for t in range(3)]
    np.testing.assert_allclose(gae(r, v, g, lam)[0][0], expected, rtol=1e-6)


def test_clipped_surrogate_value_and_gradient():
    eps = 0.2
    # positive advantage: the objective grows with the ratio up to 1 + eps, then is flat (zero gradient)
    for ratio, value, grad in [(1.1, -1.1, -1.0), (1.5, -1.2, 0.0), (0.5, -0.5, -1.0)]:
        loss, g = jax.value_and_grad(lambda r: clipped_surrogate(r, jnp.array([1.0]), eps))(jnp.array([ratio]))
        assert np.isclose(loss, value) and np.isclose(g[0], grad), (ratio, loss, g)
    # negative advantage: flat below 1 - eps, linear above (no clipping of a harmful increase)
    for ratio, value, grad in [(0.5, 0.8, 0.0), (1.5, 1.5, 1.0), (0.9, 0.9, 1.0)]:
        loss, g = jax.value_and_grad(lambda r: clipped_surrogate(r, jnp.array([-1.0]), eps))(jnp.array([ratio]))
        assert np.isclose(loss, value) and np.isclose(g[0], grad), (ratio, loss, g)


def test_rewards_sum_to_the_log_target_minus_the_entropy_term():
    rng = np.random.default_rng(0)
    steps = jnp.asarray(rng.normal(size=(4, 3, 2)), dtype=jnp.float32)
    cost, log_pi = jnp.asarray(rng.random(4), dtype=jnp.float32), jnp.asarray(rng.normal(size=(4, 3)), dtype=jnp.float32)
    log_prior = normal_log_prob(steps, 0.0, jnp.sqrt(2.0)).sum((1, 2))
    target = step_rewards(steps, cost, log_pi, 1.5, PPOConfig(action_dim=2, reference_var=2.0, ent_coef=0.5))
    np.testing.assert_allclose(target.sum(1), log_prior - 1.5 * cost - 0.5 * log_pi.sum(1), rtol=1e-5)
    np.testing.assert_allclose(target[:, :-1], normal_log_prob(steps[:, :-1], 0.0, jnp.sqrt(2.0)).sum(-1)
                               - 0.5 * log_pi[:, :-1], rtol=1e-5)  # the cost is paid at the last step only
    cost_only = step_rewards(steps, cost, log_pi, 1.5, PPOConfig(action_dim=2, reward="cost"))
    np.testing.assert_allclose(cost_only, jnp.zeros((4, 3)).at[:, -1].set(-1.5 * cost))


def test_critic_sees_only_the_steps_taken_so_far():
    H, n_steps = T * A, T
    critic = Critic(D, H, n_steps, key=critic_key(0))
    zs, zg = jnp.ones(D), -jnp.ones(D)
    a, b = jnp.arange(H, dtype=jnp.float32), jnp.arange(H, dtype=jnp.float32).at[A:].set(99.0)  # differ from step 1 on
    assert critic.all_steps(zs, zg, a).shape == (n_steps,)
    assert critic(zs, zg, a, 1) == critic(zs, zg, b, 1)  # V(s_1) depends on step 0 only
    assert critic(zs, zg, a, 2) != critic(zs, zg, b, 2)
    assert critic(zs, zg, a, 0) != critic(zs, zg, a, 1)  # the step index is an input


def run_ppo(cost, cfg, iters, seed=0, n_cond=64):
    """PPO on a fixed batch of `n_cond` random conditions (blocked layout: T steps of A values);
    returns (policy, critic, per-iteration plan returns, conditions repeated n_samples times)."""
    H = T * A
    policy = jp.GPlaner(state_dim=D, horizon=H, action_dim=A, key=init_key(seed))
    critic = Critic(D, H, T, key=critic_key(seed))
    rng = np.random.default_rng(seed)
    batch = (rng.normal(size=(n_cond, D)).astype(np.float32), rng.normal(size=(n_cond, D)).astype(np.float32))
    policy, critic, returns = train_ppo(policy, critic, cost, [batch] * iters, 2.0, cfg, lr=3e-4, seed=seed,
                                        log_every=10**9)
    zs, zg = (jnp.repeat(jnp.asarray(x), cfg.n_samples, axis=0) for x in batch)
    return policy, critic, returns, zs, zg


def sampled_step_dists(policy, zs, zg):
    """Step distributions along plans sampled from the policy: means and variances (N, T * A)."""
    plans = jax.vmap(lambda k, s, g: jp.rollout(policy, k, s, g))(jax.random.split(jax.random.key(1), len(zs)), zs, zg)
    _, d = jax.vmap(lambda s, g, p: jp.step_log_probs(policy, s, g, p))(zs, zg, plans)
    return np.asarray(d.means)[:, :, 0].reshape(len(zs), -1), (np.asarray(d.stds)[:, :, 0] ** 2).reshape(len(zs), -1)


@pytest.mark.slow
def test_maxent_ppo_converges_to_the_gflownet_target():
    """ent_coef = 1 and the target reward: the optimum is P* itself. With J = ||A||^2, beta 2 and N(0, I)
    every action is N(0, 0.2) and the mean plan return is log Z = -(H / 2) log 5 (as in the TB test).
    Calibrated over 4 seeds (400 iterations): every |mean| <= 0.04, |mean var - 0.2| <= 0.027, every
    |var - 0.2| <= 0.066, last-20 return within 0.018 of log Z. Asserted at ~2.5x."""
    cfg = PPOConfig(n_samples=8, action_dim=A, ent_coef=1.0)
    policy, _, returns, zs, zg = run_ppo(QuadraticCost(), cfg, iters=400)
    means, variances = sampled_step_dists(policy, zs, zg)
    log_z = -(T * A / 2) * np.log(1 + 2 * 2.0)
    assert np.abs(means).max() < 0.1
    assert abs(variances.mean() - 0.2) < 0.07 and np.abs(variances - 0.2).max() < 0.15
    assert abs(np.mean(returns[-20:]) - log_z) < 0.05


@pytest.mark.slow
def test_classic_ppo_reaches_the_minimizer():
    """ent_coef = 0 (classic PPO), target reward, J = ||A - mu||^2: the policy concentrates on the minimizer
    of beta J + ||A||^2 / 2, A* = 2 beta mu / (2 beta + 1) (the point map's test checks). PPO's plans are
    noisier per condition than TB's, so the check is on the typical condition: over 4 seeds the mean over
    conditions was within 0.088 of A* and the median entry within 0.051."""
    mu = np.linspace(-1.0, 1.5, T * A).astype(np.float32)
    cfg = PPOConfig(n_samples=8, action_dim=A)
    policy, _, returns, zs, zg = run_ppo(ShiftedQuadratic(jnp.asarray(mu.reshape(T, A))), cfg, iters=400)
    means, variances = sampled_step_dists(policy, zs, zg)
    a_star = 2 * 2.0 * mu / (2 * 2.0 + 1)
    assert np.abs(means.mean(0) - a_star).max() < 0.2 and np.median(np.abs(means - a_star)) < 0.12
    assert np.sqrt(variances).mean() < 0.1  # concentrated (it started at ~0.3)
    best = -2.0 * np.sum((a_star - mu) ** 2) + float(normal_log_prob(jnp.asarray(a_star), 0.0, 1.0).sum())
    assert best - 1.0 < np.mean(returns[-20:]) <= best + 1e-3  # observed: 0.12-0.39 below the best return


def test_ppo_is_reproducible_and_seeds_matter():
    cfg = PPOConfig(n_samples=2, action_dim=A, epochs=2, minibatches=2, target_kl=0.01)
    runs = [run_ppo(QuadraticCost(), cfg, iters=3, seed=s, n_cond=8) for s in (0, 0, 1)]
    (p1, c1, r1, *_), (p2, c2, r2, *_), (_, _, r3, *_) = runs
    assert r1 == r2 and r1 != r3
    for a, b in zip(jax.tree.leaves(eqx.filter((p1, c1), eqx.is_array)), jax.tree.leaves(eqx.filter((p2, c2), eqx.is_array))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_ppo_minibatches_must_split_the_plans_evenly():
    """3 conditions x 2 plans = 6 plans cannot be cut into 4 equal minibatches: an error, not a
    silently dropped remainder."""
    cfg = PPOConfig(n_samples=2, action_dim=A, minibatches=4)
    with pytest.raises(ValueError, match="6 plans per iteration .* do not split into 4"):
        run_ppo(QuadraticCost(), cfg, iters=1, n_cond=3)
