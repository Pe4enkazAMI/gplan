"""The JAX GPlaner must be the same function as the PyTorch one, and weights must round-trip."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import A, B, D, T

from gplan.policy import GPlaner as TorchGPlaner
from gplan.policy import Sampler
from gplan_jax import policy as jp
from gplan_jax.convert import gplaner_from_torch, gplaner_to_torch


@pytest.fixture(params=[1, A], ids=["scalar", "blocked"])
def models(request):
    torch.manual_seed(0)
    torch_model = TorchGPlaner(state_dim=D, horizon=T * A, hidden_size=8, n_layers=2, action_dim=request.param)
    return torch_model, gplaner_from_torch(torch_model)


def test_step_log_probs_match_torch(models):
    torch_model, jax_model = models
    g = torch.Generator().manual_seed(1)
    z_start, z_goal, plans = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g), torch.randn(B, T * A, generator=g)

    sampler = Sampler(torch_model)
    with torch.no_grad():
        ref = sampler.log_prob(z_start, z_goal, plans).numpy()
        ref_mean = torch.stack([sampler.action_dist(z_start, z_goal, plans, t).mean for t in range(sampler.n_steps)], 1)

    log_probs, means, _ = jax.vmap(lambda s, g_, p: jp.step_log_probs(jax_model, s, g_, p))(
        z_start.numpy(), z_goal.numpy(), plans.numpy())
    np.testing.assert_allclose(np.asarray(log_probs), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(means), ref_mean.numpy(), rtol=1e-5, atol=1e-6)

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
