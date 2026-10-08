"""Unit tests for train.py on a dummy batch with a tiny LeWorldModel.

Run with:  conda run -n DLA python -m pytest gplaner/test_train.py -q
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import train
from gflow import GPlaner, Sampler
from jepa import JEPA
from module import MLP, ARPredictor, Embedder

B, T, A, D, HS, IMG = 4, 3, 2, 16, 3, 8  # batch, horizon, action dim, embed dim, history, image size
N_ANALYTIC_STEPS = 600


class DummyEncoder(nn.Module):
    """Mimics the HF ViT interface used by JEPA.encode: returns .last_hidden_state (B, N, D)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3 * IMG * IMG, D)

    def forward(self, pixels, interpolate_pos_encoding=False):
        return SimpleNamespace(last_hidden_state=self.proj(pixels.flatten(1)).unsqueeze(1))


@pytest.fixture
def wm():
    torch.manual_seed(0)
    model = JEPA(
        encoder=DummyEncoder(),
        predictor=ARPredictor(num_frames=HS, depth=1, heads=2, mlp_dim=32,
                              input_dim=D, hidden_dim=D, dim_head=8),
        action_encoder=Embedder(input_dim=A, smoothed_dim=A, emb_dim=D),
        projector=MLP(D, 16, D, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(D, 16, D, norm_fn=nn.BatchNorm1d),
    )
    return model.eval().requires_grad_(False)


@pytest.fixture
def sampler():
    torch.manual_seed(0)
    return Sampler(GPlaner(state_dim=D, horizon=T * A))


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(B, 3, IMG, IMG), torch.randn(B, 3, IMG, IMG)


def test_encode_shape(wm, batch):
    z = train.encode(wm, batch[0])
    assert z.shape == (B, D)
    assert torch.isfinite(z).all()


def test_lewm_cost_matches_jepa_rollout(wm, batch):
    """Our embedding-space cost must equal JEPA.rollout + JEPA.criterion from pixels."""
    start_pixels, goal_pixels = batch
    z_start, z_goal = train.encode(wm, start_pixels), train.encode(wm, goal_pixels)
    actions = torch.randn(B, T, A)

    ours = train.lewm_cost(wm, z_start, z_goal, actions, history_size=HS)

    info = {"pixels": start_pixels[:, None, None]}          # (B, S=1, H=1, C, h, w)
    info = wm.rollout(info, actions[:, None], history_size=HS)  # actions: (B, S=1, T, A)
    info["goal_emb"] = z_goal[:, None, None]                 # (B, S=1, 1, D)
    ref = wm.criterion(info)[:, 0]                           # (B,)

    assert ours.shape == (B,)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_tb_loss_value_and_grads(wm, sampler, batch):
    z_start, z_goal = train.encode(wm, batch[0]), train.encode(wm, batch[1])
    actions = sampler.rollout(z_start, z_goal)
    assert actions.shape == (B, T * A)

    loss = train.tb_loss(sampler, wm, z_start, z_goal, actions, beta=0.5, action_dim=A)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # matches the formula ( log Z + sum_t log P_F + beta * J - log N(A) )^2 averaged over the batch
    log_pf = sampler.log_prob(z_start, z_goal, actions).sum(-1)
    log_z = sampler.model.log_Z(z_start, z_goal)
    cost = train.lewm_cost(wm, z_start, z_goal, actions.view(B, T, A))
    log_ref = sampler.reference_log_prob(actions)
    assert torch.allclose(loss, (log_z + log_pf + 0.5 * cost - log_ref).pow(2).mean())

    loss.backward()
    for name, p in sampler.model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(p.grad is None for p in wm.parameters())  # world model stays frozen


def test_cost_is_constant_target(wm, sampler, batch):
    """The cost must not carry gradients: it is the (log) reward, not a learnable quantity."""
    z_start, z_goal = train.encode(wm, batch[0]), train.encode(wm, batch[1])
    actions = sampler.rollout(z_start, z_goal)
    cost = train.lewm_cost(wm, z_start, z_goal, actions.view(B, T, A))
    assert not cost.requires_grad


def test_train_runs_end_to_end(wm, sampler, batch):
    """Smoke test of the full loop (encode -> rollout -> TB loss -> step) on the dummy WM."""
    losses = train.train(sampler, wm, [batch] * 3, beta=0.5, action_dim=A, log_every=100)
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)


def test_train_converges_to_analytic_solution(wm, sampler, monkeypatch):
    """With J(A) = ||A||^2 and the N(0, s^2 I) reference, the target is a product of Gaussians:
    each action ~ N(0, 1 / (2 beta + 1 / s^2)) and log Z = -(H / 2) * log(1 + 2 beta s^2).
    (The random dummy WM cannot be used here: its cost is ~constant in the actions.)"""
    beta, H, n = 2.0, T * A, 64
    s2 = sampler.reference_var
    monkeypatch.setattr(train, "lewm_cost", lambda wm, zs, zg, a: a.flatten(1).pow(2).sum(-1))

    torch.manual_seed(0)
    batch = (torch.randn(n, 3, IMG, IMG), torch.randn(n, 3, IMG, IMG))
    losses = train.train(sampler, wm, [batch] * N_ANALYTIC_STEPS, beta=beta, action_dim=1, lr=1e-3, lr_z=1e-2,
                         grad_clip=10.0, log_every=10**9)
    assert sum(losses[-20:]) / 20 < 0.05 < losses[0]

    z_start, z_goal = train.encode(wm, batch[0]), train.encode(wm, batch[1])
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


@pytest.mark.parametrize("action_dim", [1, A])
def test_rollout_and_log_prob_shapes(action_dim):
    """Both the scalar (action_dim=1) and the blocked layout produce a full buffer and per-step log-probs."""
    torch.manual_seed(0)
    s = Sampler(GPlaner(state_dim=D, horizon=T * A, action_dim=action_dim))
    z_start, z_goal = torch.randn(B, D), torch.randn(B, D)
    actions = s.rollout(z_start, z_goal)
    assert actions.shape == (B, T * A)
    assert s.log_prob(z_start, z_goal, actions).shape == (B, T * A // action_dim)


def test_log_z_keeps_batch_dim_for_single_condition():
    model = GPlaner(state_dim=D, horizon=T * A, action_dim=A)
    assert model.log_Z(torch.randn(1, D), torch.randn(1, D)).shape == (1,)
