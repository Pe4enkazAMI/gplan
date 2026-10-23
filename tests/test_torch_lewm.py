import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import A, ARPredictor, B, D, DummyEncoder, Embedder, HS, JEPA, make_wm, MLP, randomize_, T

from gplan.torch.lewm import check_history, encode, history_length, lewm_cost


def test_encode_shape(wm, batch):
    z = encode(wm, batch[0])
    assert z.shape == (B, D)
    assert torch.isfinite(z).all()


def test_lewm_cost_matches_jepa_rollout(wm, batch):
    """Our embedding-space cost must equal JEPA.rollout + JEPA.criterion from pixels."""
    start_pixels, goal_pixels = batch
    z_start, z_goal = encode(wm, start_pixels), encode(wm, goal_pixels)
    actions = torch.randn(B, T, A)

    ours = lewm_cost(wm, z_start, z_goal, actions, history_size=HS)

    info = {"pixels": start_pixels[:, None, None]}             # (B, S=1, H=1, C, h, w)
    info = wm.rollout(info, actions[:, None], history_size=HS)  # actions: (B, S=1, T, A)
    info["goal_emb"] = z_goal[:, None, None]                    # (B, S=1, 1, D)
    ref = wm.criterion(info)[:, 0]                              # (B,)

    assert ours.shape == (B,)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_history_size_comes_from_the_checkpoint(batch):
    """A predictor with 2 position embeddings rolls out with a context of 2, here and in JAX."""
    torch.manual_seed(0)
    wm = randomize_(JEPA(
        encoder=DummyEncoder(),
        predictor=ARPredictor(num_frames=2, depth=1, heads=2, mlp_dim=32, input_dim=D, hidden_dim=D, dim_head=8),
        action_encoder=Embedder(input_dim=A, smoothed_dim=A, emb_dim=D),
        projector=MLP(D, 16, D, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(D, 16, D, norm_fn=nn.BatchNorm1d),
    ))
    assert history_length(wm) == 2
    with pytest.raises(ValueError, match="context is 2 frames"):  # CEM would plan with 3: a different cost
        check_history(wm)
    z_start, z_goal = encode(wm, batch[0]), encode(wm, batch[1])
    actions = torch.randn(B, T, A)
    ours = lewm_cost(wm, z_start, z_goal, actions)
    torch.testing.assert_close(ours, lewm_cost(wm, z_start, z_goal, actions, history_size=2))
    info = wm.rollout({"pixels": batch[0][:, None, None]}, actions[:, None], history_size=2)
    info["goal_emb"] = z_goal[:, None, None]
    torch.testing.assert_close(ours, wm.criterion(info)[:, 0], atol=1e-5, rtol=1e-5)
    jax_lewm = pytest.importorskip("gplan.jax.convert").lewm_from_torch(wm)
    np.testing.assert_allclose(np.asarray(jax_lewm.cost(z_start.numpy(), z_goal.numpy(), actions.numpy())),
                               ours.numpy(), rtol=1e-5)


def test_cost_is_differentiable_in_the_actions_and_constant_under_no_grad(batch):
    """Refinement differentiates J w.r.t. the actions; the solvers score plans under no_grad (same value)."""
    wm = make_wm(embed=D, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A)
    z_start, z_goal = torch.randn(B, D), torch.randn(B, D)
    actions = torch.randn(B, T, A, requires_grad=True)
    value = lewm_cost(wm, z_start, z_goal, actions)
    with torch.no_grad():
        constant = lewm_cost(wm, z_start, z_goal, actions)
    assert not constant.requires_grad
    torch.testing.assert_close(value.detach(), constant, rtol=0, atol=0)
    value.sum().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0


def test_check_history_passes_the_lewm_context(wm):
    assert history_length(wm) == 3 and check_history(wm) is wm
