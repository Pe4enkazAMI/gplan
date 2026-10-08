"""The Equinox LeWM predictor must reproduce the PyTorch planning cost on identical weights."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import A, B, D, HS, T, DummyEncoder

from gplan.lewm import lewm_cost
from gplan_jax.convert import lewm_from_torch
from jepa import JEPA
from module import MLP, ARPredictor, Embedder


def randomize_(wm):
    """Give every parameter and BatchNorm statistic a non-trivial value.

    LeWM zero-initializes the AdaLN layers and BatchNorm starts at mean 0 / var 1, so an
    untrained model would hide conversion mistakes in exactly those places."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in wm.parameters():
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
        for m in wm.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.copy_(0.5 * torch.randn(m.running_mean.shape, generator=g))
                m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g))
    return wm.eval().requires_grad_(False)


def make_wm(embed, depth, heads, dim_head, mlp_dim, proj_hidden, action_dim):
    torch.manual_seed(0)
    return randomize_(JEPA(
        encoder=DummyEncoder(),
        predictor=ARPredictor(num_frames=HS, depth=depth, heads=heads, mlp_dim=mlp_dim,
                              input_dim=embed, hidden_dim=embed, dim_head=dim_head),
        action_encoder=Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=embed),
        projector=MLP(embed, proj_hidden, embed, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(embed, proj_hidden, embed, norm_fn=nn.BatchNorm1d),
    ))


@pytest.mark.parametrize("config", [
    dict(embed=D, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A, n_steps=T),
    # the real LeWM predictor: ViT-tiny width, 6 blocks, 16 heads of 64, frameskip 5 x 2-d actions, 5 steps
    dict(embed=192, depth=6, heads=16, dim_head=64, mlp_dim=2048, proj_hidden=2048, action_dim=10, n_steps=5),
])
def test_jax_cost_matches_torch(config):
    n_steps = config.pop("n_steps")
    wm = make_wm(**config)
    lewm = lewm_from_torch(wm, history_size=HS)

    g = torch.Generator().manual_seed(1)
    z_start, z_goal = torch.randn(B, config["embed"], generator=g), torch.randn(B, config["embed"], generator=g)
    plans = torch.randn(B, n_steps, config["action_dim"], generator=g)

    ours = np.asarray(lewm.cost(z_start.numpy(), z_goal.numpy(), plans.numpy()))
    ref = lewm_cost(wm, z_start, z_goal, plans, history_size=HS).numpy()
    np.testing.assert_allclose(ours, ref, rtol=1e-5)
