"""Shared fixtures: a tiny random LeWorldModel and a GPlaner, small enough to run on CPU in seconds."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import gplan.lewm  # noqa: F401  puts the le-wm submodule on sys.path
from gplan.policy import GPlaner, Sampler
from jepa import JEPA
from module import MLP, ARPredictor, Embedder

B, T, A, D, HS, IMG = 4, 3, 2, 16, 3, 8  # batch, horizon, action dim, embed dim, history, image size

try:
    import jax
except ImportError:  # torch-only test runs
    pass
else:
    # The parity tests compare JAX against float32 PyTorch on CPU. On an NVIDIA GPU, JAX's default
    # float32 matmul runs in TF32 (~3 significant digits), which alone produces 1e-4..1e-2 relative
    # differences. The tests check the math, so they run at full float32 precision, like scripts/train.py.
    jax.config.update("jax_default_matmul_precision", "highest")


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
    """Scalar layout (action_dim=1): one autoregressive step per action component."""
    torch.manual_seed(0)
    return Sampler(GPlaner(state_dim=D, horizon=T * A))


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(B, 3, IMG, IMG), torch.randn(B, 3, IMG, IMG)
