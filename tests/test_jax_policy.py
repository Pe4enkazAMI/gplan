"""The JAX GPlaner must be the same function as the PyTorch one, and weights must round-trip."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import A, B, D, T

from gplan.jax import policy as jp
from gplan.jax.convert import gplaner_from_torch, gplaner_to_torch
from gplan.torch.policy import GPlaner as TorchGPlaner
from gplan.torch.policy import Sampler


@pytest.fixture(params=[(1, 1), (A, 1), (A, 3)], ids=["scalar", "blocked", "blocked-mixture"])
def models(request):
    action_dim, n_components = request.param
    torch.manual_seed(0)
    torch_model = TorchGPlaner(state_dim=D, horizon=T * A, hidden_size=8, n_layers=2, action_dim=action_dim,
                               n_components=n_components)
    return torch_model, gplaner_from_torch(torch_model)


def test_step_log_probs_match_torch(models):
    torch_model, jax_model = models
    g = torch.Generator().manual_seed(1)
    z_start, z_goal, plans = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g), torch.randn(B, T * A, generator=g)

    sampler = Sampler(torch_model)
    with torch.no_grad():
        ref = sampler.log_prob(z_start, z_goal, plans).numpy()
        dists = [sampler.action_dist(z_start, z_goal, plans, t) for t in range(sampler.n_steps)]
        ref_log_w = torch.stack([d.mixture_distribution.logits for d in dists], 1)          # (B, n, K)
        ref_means = torch.stack([d.component_distribution.base_dist.loc for d in dists], 1)  # (B, n, K, A)
        ref_stds = torch.stack([d.component_distribution.base_dist.scale for d in dists], 1)

    log_probs, log_w, means, stds = jax.vmap(lambda s, g_, p: jp.step_log_probs(jax_model, s, g_, p))(
        z_start.numpy(), z_goal.numpy(), plans.numpy())
    np.testing.assert_allclose(np.asarray(log_probs), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(log_w), ref_log_w.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(means), ref_means.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stds), ref_stds.numpy(), rtol=1e-5, atol=1e-6)

    log_z = jax.vmap(jax_model.log_Z)(z_start.numpy(), z_goal.numpy())
    np.testing.assert_allclose(np.asarray(log_z), torch_model.log_Z(z_start, z_goal).detach().numpy(), rtol=1e-5, atol=1e-5)


def test_mean_rollout_matches_torch(models):
    torch_model, jax_model = models
    g = torch.Generator().manual_seed(2)
    z_start, z_goal = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)
    ref = Sampler(torch_model).rollout(z_start, z_goal, mode="mean").numpy()
    keys = jax.random.split(jax.random.key(0), B)
    ours = jax.vmap(lambda k, s, g_: jp.rollout(jax_model, k, s, g_, mode="mean"))(keys, z_start.numpy(), z_goal.numpy())
    np.testing.assert_allclose(np.asarray(ours), ref, rtol=1e-5, atol=1e-6)


def test_weights_round_trip(models):
    torch_model, jax_model = models
    sd = gplaner_to_torch(jax_model)
    assert sd.keys() == torch_model.state_dict().keys()
    for k, v in torch_model.state_dict().items():
        assert torch.equal(sd[k], v), k


def test_reference_log_prob_matches_torch():
    plans = torch.randn(B, T * A)
    ref = Sampler(TorchGPlaner(state_dim=D, horizon=T * A), reference_var=2.0).reference_log_prob(plans).numpy()
    ours = jp.reference_log_prob(jnp.asarray(plans.numpy()), 2.0)
    np.testing.assert_allclose(np.asarray(ours), ref, rtol=1e-6)


def test_single_component_is_the_plain_gaussian():
    """n_components=1 scores a step exactly as a diagonal Gaussian (the pre-mixture policy)."""
    model = jp.GPlaner(state_dim=D, horizon=T * A, hidden_size=8, action_dim=A, key=jax.random.key(0))
    zs, zg, plan = (jax.random.normal(k, s) for k, s in zip(jax.random.split(jax.random.key(1), 3), [(D,), (D,), (T * A,)]))
    log_probs, log_w, means, stds = jp.step_log_probs(model, zs, zg, plan)
    gauss = jp.normal_log_prob(plan.reshape(T, A), means[:, 0], stds[:, 0]).sum(-1)
    np.testing.assert_allclose(np.asarray(log_probs), np.asarray(gauss), rtol=1e-6)
    assert np.all(np.asarray(log_w) == 0.0)


def test_mixture_samples_follow_the_weights():
    """Hand-set mixture head: two well-separated components with weights 0.8 / 0.2 and no input dependence."""
    import equinox as eqx
    model = jp.GPlaner(state_dim=D, horizon=A, hidden_size=8, action_dim=A, n_components=2, key=jax.random.key(0))
    bias = jnp.concatenate([jnp.full(A, -5.0), jnp.full(A, 5.0),       # component means -5 and +5
                            jnp.full(2 * A, -3.0),                        # small stds
                            jnp.log(jnp.array([0.8, 0.2]))])              # logits
    model = eqx.tree_at(lambda m: (m.out_proj.weight, m.out_proj.bias), model,
                        (jnp.zeros_like(model.out_proj.weight), bias))
    zs = jnp.zeros((4000, D))
    plans = jax.vmap(lambda k, s: jp.rollout(model, k, s, s))(jax.random.split(jax.random.key(2), 4000), zs)
    assert abs(float((plans[:, 0] < 0).mean()) - 0.8) < 0.03
    assert np.all(np.abs(np.abs(np.asarray(plans)) - 5.0) < 0.5)  # every sample sits on one of the modes
    mode = jp.rollout(model, jax.random.key(3), zs[0], zs[0], mode="mean")
    np.testing.assert_allclose(np.asarray(mode), -5.0)  # the most likely component's mean
