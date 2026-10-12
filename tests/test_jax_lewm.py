"""The Equinox LeWM predictor must reproduce the PyTorch planning cost on identical weights."""

import numpy as np
import pytest
import torch
from conftest import A, B, D, HS, make_wm, T

from gplan.jax.convert import lewm_from_torch
from gplan.torch.lewm import lewm_cost


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
