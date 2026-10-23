"""Training: the jitted loop reaches the closed-form TB optimum (with and without MALA plans),
and a run trained on the tiny LeWM exports to the PyTorch checkpoint evaluation loads."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from conftest import A, B, D, T
from toy_costs import QuadraticCost

from gplan.jax import checkpoint
from gplan.jax import policy as jp
from gplan.jax import train as train_module
from gplan.jax.convert import gplaner_to_torch, lewm_from_torch
from gplan.jax.train import (
    init_key,
    linear_beta_schedule,
    make_optimizer,
    MALA_STREAM,
    norm_stats,
    train,
    TRAIN_STREAM,
    TrainConfig,
)
from gplan.torch.policy import load_planner, Sampler


@pytest.mark.slow
@pytest.mark.parametrize("ls_steps", [0, 5], ids=["tb", "tb-mala"])
def test_train_converges_to_analytic_solution(ls_steps):
    """Training from a random init reaches the closed-form optimum. With J = ||A||^2 and N(0, s^2 I):
    each action ~ N(0, 1 / (2 beta + 1 / s^2)) = N(0, 0.2), log Z = -(H / 2) log(1 + 2 beta s^2), for
    every one of the 64 conditions, with and without MALA plans in the batch (off-policy).

    Constant learning rates (as in scripts/train.py): the last iterate keeps jittering around the
    optimum, by about the learning rate, so the rates are low (3e-4 / 3e-3) and the run long (6000
    steps). Tolerances are 2-6x the worst error over 8 seeds x 2 variants (calibration below); the
    analytic numbers are exact, see tests/test_jax_optimum.py for the objective itself."""
    beta, s2, H, steps = 2.0, 1.0, T * A, 6000
    model = jp.GPlaner(state_dim=D, horizon=H, hidden_size=64, key=init_key(0))  # scalar layout
    rng = np.random.default_rng(0)
    batch = (rng.normal(size=(64, D)).astype(np.float32), rng.normal(size=(64, D)).astype(np.float32))
    cfg = TrainConfig(n_samples=1, action_dim=1, reference_var=s2, grad_clip=10.0, ls_steps=ls_steps, ls_step_size=0.05)
    model, losses = train(model, QuadraticCost(), [batch] * steps, beta, cfg, lr=3e-4, lr_z=3e-3, log_every=10**9)

    zs, zg = (jnp.asarray(x) for x in batch)
    plans = jax.vmap(lambda k, s, g: jp.rollout(model, k, s, g))(jax.random.split(jax.random.key(1), len(zs)), zs, zg)
    _, dists = jax.vmap(lambda s, g, p: jp.step_log_probs(model, s, g, p))(zs, zg, plans)
    variances, log_z = np.asarray(dists.stds) ** 2, np.asarray(jax.vmap(model.log_Z)(zs, zg))
    target_var, target_log_z = 1 / (2 * beta + 1 / s2), -H / 2 * math.log(1 + 2 * beta * s2)  # 0.2, -4.83

    # worst over 8 seeds x {without, with MALA} (calibration): last-100 loss 0.0064, every |mean| 0.016,
    # every |var - 0.2| 0.022, |mean var - 0.2| 0.0017, every |log Z - target| 0.028. (Dropping the
    # prior term would give var 0.25 and log Z +1.35.)
    assert np.mean(losses[-100:]) < 0.04 < losses[0]
    assert np.abs(np.asarray(dists.means)).max() < 0.05
    assert np.abs(variances - target_var).max() < 0.1
    assert abs(variances.mean() - target_var) < 0.01
    assert np.abs(log_z - target_log_z).max() < 0.06

@pytest.mark.parametrize("n_components", [1, 3])
def test_train_with_lewm_then_export_to_torch(wm, tmp_path, n_components):
    """End to end on the tiny LeWM: train a few steps, save, reload in JAX, and load the .pt with the
    torch eval loader; all three hold the same weights and the torch policy scores plans the same way."""
    model_kwargs = dict(state_dim=D, horizon=T * A, action_dim=A, n_components=n_components)
    model = jp.GPlaner(**model_kwargs, key=init_key(0))
    rng = np.random.default_rng(0)
    batches = [(rng.normal(size=(B, D)).astype(np.float32), rng.normal(size=(B, D)).astype(np.float32))] * 3
    cfg = TrainConfig(n_samples=2, action_dim=A, grad_clip=10.0)
    model, losses = train(model, lewm_from_torch(wm), batches, 0.5, cfg, log_every=100)
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)

    checkpoint.save(tmp_path / "run", model, model_kwargs, train_args={"n_steps": T})
    reloaded, kwargs, train_args = checkpoint.load(tmp_path / "run")
    assert kwargs == model_kwargs and train_args == {"n_steps": T}
    for a, b in zip(jax.tree.leaves(eqx.filter(model, eqx.is_array)), jax.tree.leaves(eqx.filter(reloaded, eqx.is_array))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    torch_model, train_args = load_planner(tmp_path / "run.pt")
    assert train_args == {"n_steps": T} and torch_model.n_components == n_components
    for name, p in torch_model.state_dict().items():
        np.testing.assert_array_equal(p.numpy(), gplaner_to_torch(reloaded)[name].numpy(), err_msg=name)
    zs, zg, plans = (rng.normal(size=(B, n)).astype(np.float32) for n in (D, D, T * A))
    theirs = Sampler(torch_model).log_prob(*(torch.from_numpy(x) for x in (zs, zg, plans))).numpy()
    ours = jax.vmap(lambda s, g, p: jp.step_log_probs(reloaded, s, g, p)[0])(zs, zg, plans)
    np.testing.assert_allclose(np.asarray(ours), theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_training_is_reproducible_and_seeds_matter():
    """Same seed -> bit-identical weights and losses (CPU); another seed -> a different run.
    The init and training streams of one seed are different keys."""
    def run(seed):
        model = jp.GPlaner(state_dim=D, horizon=T * A, action_dim=A, n_components=2, key=init_key(seed))
        rng = np.random.default_rng(0)
        batches = [(rng.normal(size=(B, D)).astype(np.float32), rng.normal(size=(B, D)).astype(np.float32))] * 5
        cfg = TrainConfig(n_samples=2, action_dim=A, grad_clip=10.0, ls_steps=2, ls_every=2)
        return train(model, QuadraticCost(), batches, 0.5, cfg, seed=seed, log_every=10**9)

    (m1, l1), (m2, l2), (_, l3) = run(0), run(0), run(1)
    assert l1 == l2 and l1 != l3
    for a, b in zip(jax.tree.leaves(eqx.filter(m1, eqx.is_array)), jax.tree.leaves(eqx.filter(m2, eqx.is_array))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(jax.random.key_data(init_key(0)),
                              jax.random.key_data(jax.random.fold_in(jax.random.key(0), TRAIN_STREAM)))


def test_mala_draws_from_its_own_random_stream(monkeypatch):
    """MALA's key at each training step comes from MALA_STREAM, not from the key the on-policy plans are
    drawn with (the plans use split(step_key, n): MALA must not reuse any of those keys)."""
    seen = []

    def spy(cost_model, key, *args, **kwargs):
        jax.debug.callback(lambda k: seen.append(np.asarray(k)), jax.random.key_data(key))
        return mala(cost_model, key, *args, **kwargs)

    mala = train_module.mala
    monkeypatch.setattr(train_module, "mala", spy)
    model = jp.GPlaner(state_dim=D, horizon=T * A, action_dim=A, key=init_key(0))
    batch = (np.zeros((B, D), np.float32), np.zeros((B, D), np.float32))
    cfg = TrainConfig(n_samples=2, action_dim=A, ls_steps=1)
    train(model, QuadraticCost(), [batch] * 2, 0.5, cfg, seed=0, log_every=10**9)

    key = jax.random.fold_in(jax.random.key(0), TRAIN_STREAM)
    ls_key = jax.random.fold_in(jax.random.key(0), MALA_STREAM)
    assert len(seen) == 2
    for step in range(2):
        key, step_key = jax.random.split(key)
        ls_key, step_ls_key = jax.random.split(ls_key)
        np.testing.assert_array_equal(seen[step], jax.random.key_data(step_ls_key))
        plan_keys = jax.random.key_data(jax.random.split(step_key, B * 2))
        assert not any(np.array_equal(seen[step], k) for k in plan_keys)


def test_beta_schedule():
    sched = linear_beta_schedule(beta_end=2.0, beta_start=0.5, warmup_steps=10)
    assert [sched(s) for s in (0, 5, 10, 100)] == [0.5, 1.25, 2.0, 2.0]
    assert linear_beta_schedule(2.0)(0) == 2.0  # no warmup: constant


def test_log_z_head_gets_its_own_learning_rate():
    """Adam's first step moves every parameter by its learning rate (gradient of ones, no clipping)."""
    model = jp.GPlaner(state_dim=D, horizon=T * A, action_dim=A, key=jax.random.key(0))
    params = eqx.filter(model, eqx.is_array)
    optimizer = make_optimizer(model, lr=1e-3, lr_z=1e-1, grad_clip=0.0)
    updates, _ = optimizer.update(jax.tree.map(jnp.ones_like, params), optimizer.init(params), params)
    for leaf in jax.tree.leaves(updates.Z):
        np.testing.assert_allclose(np.asarray(leaf), -1e-1, rtol=1e-4)
    for name in ("in_proj", "backbone", "out_proj"):
        for leaf in jax.tree.leaves(getattr(updates, name)):
            np.testing.assert_allclose(np.asarray(leaf), -1e-3, rtol=1e-4)


def test_gradient_clipping_and_its_stats():
    model = jp.GPlaner(state_dim=D, horizon=T * A, action_dim=A, key=jax.random.key(0))
    params = eqx.filter(model, eqx.is_array)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 3.0), params)
    raw = float(optax.tree.norm(grads))
    stats = norm_stats(model, grads, grad_clip=raw / 4)
    assert np.isclose(float(stats["grad_norm/global_raw"]), raw)
    assert float(stats["grad_norm/clipped"]) == 1.0 and np.isclose(float(stats["grad_norm/clip_scale"]), 0.25)
    assert np.isclose(float(stats["grad_norm/global"]), raw / 4)
    assert float(norm_stats(model, grads, grad_clip=0.0)["grad_norm/global"]) == float(stats["grad_norm/global_raw"])

