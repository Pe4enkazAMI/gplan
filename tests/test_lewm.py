import torch
from conftest import B, D, HS, T, A

from gplan.lewm import encode, lewm_cost


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


def test_cost_is_constant_target(wm, sampler, batch):
    """The cost must not carry gradients: it is the (log) reward, not a learnable quantity."""
    z_start, z_goal = encode(wm, batch[0]), encode(wm, batch[1])
    actions = sampler.rollout(z_start, z_goal)
    assert not lewm_cost(wm, z_start, z_goal, actions.view(B, T, A)).requires_grad
