"""The JAX TB loss equals its formula computed in PyTorch (the PyTorch GPlanner with the same weights,
torch.distributions densities): values and gradients, for a Gaussian and a mixture policy."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from conftest import A, B, D, T, torch_log_z, torch_step_log_probs
from torch.distributions import Normal

from gplan.jax.convert import gplanner_from_torch, gplanner_to_torch
from gplan.jax.losses import tb_loss
from gplan.torch.policy import GPlanner as TorchGPlanner
from gplan.torch.policy import Sampler

BETA, REF_VAR = 0.7, 1.5


def torch_xi(model, z_start, z_goal, plans, cost):
    """log P_F(A) + beta J(A) - log N(A; 0, s^2 I) per plan."""
    log_reference = Normal(0.0, REF_VAR ** 0.5).log_prob(plans).sum(-1)
    return torch_step_log_probs(model, z_start, z_goal, plans).sum(-1) + BETA * cost - log_reference


@pytest.fixture(params=[1, 3], ids=["gaussian", "mixture"])
def setup(request):
    torch.manual_seed(0)
    torch_model = TorchGPlanner(state_dim=D, horizon=T * A, hidden_size=8, action_dim=A, n_components=request.param)
    g = torch.Generator().manual_seed(1)
    z_start, z_goal = torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)
    plans, cost = torch.randn(B, T * A, generator=g), 10 * torch.rand(B, generator=g)
    jax_args = tuple(jnp.asarray(x.numpy()) for x in (z_start, z_goal, plans, cost))
    return torch_model, gplanner_from_torch(torch_model), (z_start, z_goal, plans, cost), jax_args


def assert_grads_match(jax_grads, torch_model):
    """Equal up to float32 round-off, measured against each tensor's largest entry (the
    gradients span ~4 orders of magnitude, so small entries carry cancellation error)."""
    ours = gplanner_to_torch(jax_grads)
    for name, p in torch_model.named_parameters():
        ref = p.grad.numpy()
        np.testing.assert_allclose(ours[name].numpy(), ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max(), err_msg=name)


def test_tb_loss_and_grads_match_torch(setup):
    torch_model, jax_model, (zs, zg, plans, cost), jargs = setup
    ref = (torch_log_z(torch_model, zs, zg) + torch_xi(torch_model, zs, zg, plans, cost)).pow(2).mean()
    ref.backward()

    (loss, _), grads = eqx.filter_value_and_grad(tb_loss, has_aux=True)(jax_model, *jargs, BETA, REF_VAR)
    np.testing.assert_allclose(float(loss), ref.item(), rtol=1e-5)
    assert_grads_match(grads, torch_model)


def test_policy_stats_for_gaussian_and_mixture(setup):
    """With one component the entropy statistic is the Gaussian entropy and the weight is 1;
    a mixture reports weights below 1 and a lower-bound entropy, sum_k w_k H(component k)."""
    torch_model, jax_model, _, jargs = setup
    _, stats = tb_loss(jax_model, *jargs, BETA, REF_VAR)
    sampler = Sampler(torch_model)
    zs, zg, plans = (torch.from_numpy(np.asarray(x)) for x in jargs[:3])
    with torch.no_grad():
        dists = [sampler.action_dist(zs, zg, plans, t) for t in range(sampler.n_steps)]
        entropy = sum((d.weights * Normal(d.means, d.stds).entropy().sum(-1)).sum(-1) for d in dists).mean().item()
    np.testing.assert_allclose(float(stats["policy/entropy"]), entropy, rtol=1e-5)
    if torch_model.n_components == 1:
        assert float(stats["policy/weight_max"]) == 1.0 and float(stats["policy/weight_entropy"]) == 0.0
        gauss = sum(0.5 * torch.log(2 * torch.pi * torch.e * d.stds[:, 0] ** 2).sum(-1) for d in dists).mean()
        np.testing.assert_allclose(float(stats["policy/entropy"]), gauss.item(), rtol=1e-5)
    else:
        assert 1 / torch_model.n_components <= float(stats["policy/weight_max"]) < 1.0
