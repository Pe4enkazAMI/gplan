"""Training: the jitted loop reaches the closed-form TB / VarGrad optimum (with and without MALA plans),
and a run trained on the tiny LeWM exports to the PyTorch checkpoint evaluation loads."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import A, B, D, T
from toy_costs import QuadraticCost

from gplan.jax import checkpoint
from gplan.jax import policy as jp
from gplan.jax.convert import gplaner_to_torch, lewm_from_torch
from gplan.jax.train import train, TrainConfig
from gplan.torch.policy import load_planner


@pytest.mark.parametrize("loss_type, ls_steps", [("tb", 0), ("vargrad", 0), ("tb", 5)],
                         ids=["tb", "vargrad", "tb-mala"])
def test_train_converges_to_analytic_solution(loss_type, ls_steps):
    """With J = ||A||^2 and N(0, s^2 I): each action ~ N(0, 1 / (2 beta + 1 / s^2)),
    log Z = -(H / 2) * log(1 + 2 beta s^2). With MALA plans in the batch (off-policy) the
    optimum must be the same."""
    beta, s2, H = 2.0, 1.0, T * A
    K = 1 if loss_type == "tb" else 8
    n_cond = 64 // K
    model = jp.GPlaner(state_dim=D, horizon=H, hidden_size=64, key=jax.random.key(0))  # scalar layout
    rng = np.random.default_rng(0)
    batch = (rng.normal(size=(n_cond, D)).astype(np.float32), rng.normal(size=(n_cond, D)).astype(np.float32))
    cfg = TrainConfig(loss_type=loss_type, n_samples=K, action_dim=1, reference_var=s2, grad_clip=10.0,
                      ls_steps=ls_steps, ls_step_size=0.05)

    # the loss is noisy step to step (on-policy samples), so judge it over the last 100 steps
    model, losses = train(model, QuadraticCost(), [batch] * 2000, beta, cfg, lr=1e-3, lr_z=1e-2, log_every=10**9)
    assert np.mean(losses[-100:]) < 0.05 < losses[0]

    zs, zg = (jnp.repeat(jnp.asarray(x), K, axis=0) for x in batch)
    keys = jax.random.split(jax.random.key(1), len(zs))
    plans = jax.vmap(lambda k, s, g: jp.rollout(model, k, s, g))(keys, zs, zg)
    _, dists = jax.vmap(lambda s, g, p: jp.step_log_probs(model, s, g, p))(zs, zg, plans)
    means, stds = dists.means, dists.stds
    assert float(jnp.abs(means).max()) < 0.1
    # The optimum is a statement about the distribution, so check the variance averaged over all
    # conditions tightly (dropping the prior would give 0.25, i.e. 0.05 off) and each condition loosely:
    # after a finite stochastic run one of a few conditions can still lag (seen: 0.23 on one of 8), and
    # float round-off differences between CPUs change which one.
    target_var = 1 / (2 * beta + 1 / s2)
    variances = np.asarray(stds) ** 2
    assert abs(variances.mean() - target_var) < 0.01
    np.testing.assert_allclose(variances, target_var, atol=0.05)
    log_z = jax.vmap(model.log_Z)(zs, zg)
    target = -H / 2 * math.log(1 + 2 * beta * s2)
    assert abs(float(log_z.mean()) - target) < 0.1  # target -4.83; a wrong prior term would give +1.35


def test_train_with_lewm_then_export_to_torch(wm, tmp_path):
    """End to end on the tiny LeWM: train a few steps, save, and load the .pt with the torch eval loader."""
    model_kwargs = dict(state_dim=D, horizon=T * A, action_dim=A)
    model = jp.GPlaner(**model_kwargs, key=jax.random.key(0))
    rng = np.random.default_rng(0)
    batches = [(rng.normal(size=(B, D)).astype(np.float32), rng.normal(size=(B, D)).astype(np.float32))] * 3
    cfg = TrainConfig(loss_type="tb", n_samples=2, action_dim=A, grad_clip=10.0)
    model, losses = train(model, lewm_from_torch(wm), batches, 0.5, cfg, log_every=100)
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)

    checkpoint.save(tmp_path / "run", model, model_kwargs, train_args={"n_steps": T})
    reloaded, kwargs, _ = checkpoint.load(tmp_path / "run")
    assert kwargs == model_kwargs
    torch_model, train_args = load_planner(tmp_path / "run.pt")
    assert train_args == {"n_steps": T}
    for name, p in torch_model.state_dict().items():
        np.testing.assert_array_equal(p.numpy(), gplaner_to_torch(reloaded)[name].numpy(), err_msg=name)
