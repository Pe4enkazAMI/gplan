import math
from functools import partial

import torch
from conftest import A, B, D, IMG, T

from gplan.lewm import encode, lewm_cost
from gplan.losses import tb_loss, vargrad_loss
from gplan.trainer import train


def quadratic_cost(z_start, z_goal, plans):
    """J(A) = ||A||^2: makes the target a product of Gaussians with a closed-form optimum."""
    return plans.flatten(1).pow(2).sum(-1)


def test_tb_loss_value_and_grads(wm, sampler, batch):
    z_start, z_goal = encode(wm, batch[0]), encode(wm, batch[1])
    actions = sampler.rollout(z_start, z_goal)
    assert actions.shape == (B, T * A)

    cost_fn = partial(lewm_cost, wm)
    loss = tb_loss(sampler, cost_fn, z_start, z_goal, actions, beta=0.5, action_dim=A)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # matches ( log Z + sum_t log P_F + beta * J - log N(A) )^2 averaged over the batch
    log_pf = sampler.log_prob(z_start, z_goal, actions).sum(-1)
    log_z = sampler.model.log_Z(z_start, z_goal)
    cost = lewm_cost(wm, z_start, z_goal, actions.view(B, T, A))
    log_ref = sampler.reference_log_prob(actions)
    assert torch.allclose(loss, (log_z + log_pf + 0.5 * cost - log_ref).pow(2).mean())

    loss.backward()
    for name, p in sampler.model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(p.grad is None for p in wm.parameters())  # world model stays frozen


def test_vargrad_loss_is_zero_when_xi_is_constant(sampler):
    """If every plan of a condition has the same xi, the VarGrad term vanishes."""
    K = 4
    z = torch.randn(2, D).repeat_interleave(K, 0)
    actions = sampler.rollout(z, z)
    log_pf = sampler.log_prob(z, z, actions).sum(-1).detach()
    log_ref = sampler.reference_log_prob(actions)
    cost_fn = lambda zs, zg, plans: (log_ref - log_pf + 3.0) / 0.5  # makes xi == 3 for every plan
    loss, stats = vargrad_loss(sampler, cost_fn, z, z, actions, beta=0.5, action_dim=A, n_samples=K,
                               return_stats=True)
    assert stats["loss/vargrad"] < 1e-6
    assert abs(stats["tb/log_z_hat"] + 3.0) < 1e-4


def test_train_runs_end_to_end(wm, sampler, batch):
    """Smoke test of the full loop (encode -> rollout -> TB loss -> step) on the dummy WM."""
    losses = train(sampler, wm, [batch] * 3, beta=0.5, action_dim=A, log_every=100)
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)


def test_train_converges_to_analytic_solution(wm, sampler):
    """With J(A) = ||A||^2 and the N(0, s^2 I) reference, the target is a product of Gaussians:
    each action ~ N(0, 1 / (2 beta + 1 / s^2)) and log Z = -(H / 2) * log(1 + 2 beta s^2)."""
    beta, H, n = 2.0, T * A, 64
    s2 = sampler.reference_var

    torch.manual_seed(0)
    batch = (torch.randn(n, 3, IMG, IMG), torch.randn(n, 3, IMG, IMG))
    losses = train(sampler, wm, [batch] * 600, beta=beta, action_dim=1, lr=1e-3, lr_z=1e-2,
                   grad_clip=10.0, log_every=10**9, cost_fn=quadratic_cost)
    assert sum(losses[-20:]) / 20 < 0.05 < losses[0]

    z_start, z_goal = encode(wm, batch[0]), encode(wm, batch[1])
    with torch.no_grad():
        actions = sampler.rollout(z_start, z_goal)
        for t in range(H):
            dist = sampler.action_dist(z_start, z_goal, actions, t)
            assert dist.mean.abs().max() < 0.1
            assert torch.allclose(dist.variance, torch.full_like(dist.variance, 1 / (2 * beta + 1 / s2)), atol=0.03)
        log_z = sampler.model.log_Z(z_start, z_goal)
        target_log_z = -H / 2 * math.log(1 + 2 * beta * s2)
        assert abs(log_z.mean().item() - target_log_z) < 0.05
        assert (log_z - target_log_z).abs().max() < 0.3
