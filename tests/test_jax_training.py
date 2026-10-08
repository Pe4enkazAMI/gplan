"""JAX losses equal the PyTorch formulas (values and gradients); training reaches the closed-form optimum."""

import json
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import A, B, D, T

from gplan.policy import GPlaner as TorchGPlaner
from gplan.policy import Sampler, load_planner
from gplan_jax import checkpoint
from gplan_jax import policy as jp
from gplan_jax.convert import gplaner_from_torch, gplaner_to_torch, lewm_from_torch
from gplan_jax.data import LatentPairs
from gplan_jax.losses import tb_loss, vargrad_loss
from gplan_jax.train import TrainConfig, train

BETA, REF_VAR = 0.7, 1.5


class QuadraticCost(eqx.Module):
    """J(A) = ||A||^2: the target becomes a product of Gaussians with a closed-form optimum."""

    def cost(self, z_start, z_goal, plans):
        return jnp.sum(plans ** 2, axis=(1, 2))


def torch_xi(sampler, z_start, z_goal, plans, cost):
    return sampler.log_prob(z_start, z_goal, plans).sum(-1) + BETA * cost - sampler.reference_log_prob(plans)


@pytest.fixture
def setup():
    torch.manual_seed(0)
    torch_model = TorchGPlaner(state_dim=D, horizon=T * A, hidden_size=8, action_dim=A)
    g = torch.Generator().manual_seed(1)
    z_start, z_goal = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)
    plans, cost = torch.randn(B, T * A, generator=g), 10 * torch.rand(B, generator=g)
    jax_args = tuple(jnp.asarray(x.numpy()) for x in (z_start, z_goal, plans, cost))
    return torch_model, gplaner_from_torch(torch_model), (z_start, z_goal, plans, cost), jax_args


def assert_grads_match(jax_grads, torch_model):
    """Equal up to float32 round-off, measured against each tensor's largest entry (the
    gradients span ~4 orders of magnitude, so small entries carry cancellation error)."""
    ours = gplaner_to_torch(jax_grads)
    for name, p in torch_model.named_parameters():
        ref = p.grad.numpy()
        np.testing.assert_allclose(ours[name].numpy(), ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max(), err_msg=name)


def test_tb_loss_and_grads_match_torch(setup):
    torch_model, jax_model, (zs, zg, plans, cost), jargs = setup
    sampler = Sampler(torch_model, reference_var=REF_VAR)
    ref = (torch_model.log_Z(zs, zg) + torch_xi(sampler, zs, zg, plans, cost)).pow(2).mean()
    ref.backward()

    (loss, _), grads = eqx.filter_value_and_grad(tb_loss, has_aux=True)(jax_model, *jargs, BETA, REF_VAR)
    np.testing.assert_allclose(float(loss), ref.item(), rtol=1e-5)
    assert_grads_match(grads, torch_model)


def test_vargrad_loss_and_grads_match_torch(setup):
    torch_model, jax_model, (zs, zg, plans, cost), jargs = setup
    K = 2  # B = 4 rows = 2 conditions x 2 plans; repeat conditions so the grouping is real
    zs, zg = zs[::K].repeat_interleave(K, 0), zg[::K].repeat_interleave(K, 0)
    jargs = (jnp.asarray(zs.numpy()), jnp.asarray(zg.numpy())) + jargs[2:]

    sampler = Sampler(torch_model, reference_var=REF_VAR)
    xi_bk = torch_xi(sampler, zs, zg, plans, cost).view(-1, K)
    log_z_hat = -xi_bk.mean(1).detach()
    ref = xi_bk.var(1).mean() + (torch_model.log_Z(zs, zg).view(-1, K)[:, 0] - log_z_hat).pow(2).mean()
    ref.backward()

    (loss, _), grads = eqx.filter_value_and_grad(vargrad_loss, has_aux=True)(jax_model, *jargs, BETA, REF_VAR, K)
    np.testing.assert_allclose(float(loss), ref.item(), rtol=1e-5)
    assert_grads_match(grads, torch_model)


@pytest.mark.parametrize("loss_type", ["tb", "vargrad"])
def test_train_converges_to_analytic_solution(loss_type):
    """With J = ||A||^2 and N(0, s^2 I): each action ~ N(0, 1 / (2 beta + 1 / s^2)),
    log Z = -(H / 2) * log(1 + 2 beta s^2)."""
    beta, s2, H = 2.0, 1.0, T * A
    K = 1 if loss_type == "tb" else 8
    n_cond = 64 // K
    model = jp.GPlaner(state_dim=D, horizon=H, hidden_size=64, key=jax.random.key(0))  # scalar layout
    rng = np.random.default_rng(0)
    batch = (rng.normal(size=(n_cond, D)).astype(np.float32), rng.normal(size=(n_cond, D)).astype(np.float32))
    cfg = TrainConfig(loss_type=loss_type, n_samples=K, action_dim=1, reference_var=s2, grad_clip=10.0)

    # the loss is noisy step to step (on-policy samples), so judge it over the last 100 steps
    model, losses = train(model, QuadraticCost(), [batch] * 2000, beta, cfg, lr=1e-3, lr_z=1e-2, log_every=10**9)
    assert np.mean(losses[-100:]) < 0.05 < losses[0]

    zs, zg = (jnp.repeat(jnp.asarray(x), K, axis=0) for x in batch)
    keys = jax.random.split(jax.random.key(1), len(zs))
    plans = jax.vmap(lambda k, s, g: jp.rollout(model, k, s, g))(keys, zs, zg)
    _, means, stds = jax.vmap(lambda s, g, p: jp.step_log_probs(model, s, g, p))(zs, zg, plans)
    assert float(jnp.abs(means).max()) < 0.1
    np.testing.assert_allclose(np.asarray(stds) ** 2, 1 / (2 * beta + 1 / s2), atol=0.03)
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


def test_latent_pairs_stay_within_episode(tmp_path):
    n, offset = 12, 3
    latents = np.arange(n, dtype=np.float32)[:, None].repeat(4, axis=1)  # row i has value i
    np.save(tmp_path / "toy_latents.npy", latents)
    np.savez(tmp_path / "toy_latents_meta.npz", ep_idx=np.repeat([0, 1], 6), step_idx=np.tile(np.arange(6), 2),
             action=np.zeros((n, 2)))
    (tmp_path / "toy_latents.json").write_text(json.dumps({"ckpt": str(tmp_path / "x.ckpt")}))

    pairs = LatentPairs(tmp_path / "toy_latents", offset)
    for z_start, z_goal in pairs.batches(5, 4):
        start_rows = z_start[:, 0].astype(int)
        np.testing.assert_array_equal(z_goal[:, 0], start_rows + offset)
        assert np.all(start_rows % 6 <= 6 - offset - 1)  # goal is inside the same episode
    with pytest.raises(ValueError):
        LatentPairs(tmp_path / "toy_latents", offset, expected_ckpt=tmp_path / "other.ckpt")
