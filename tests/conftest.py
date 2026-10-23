"""Shared fixtures: a tiny random LeWorldModel and a GPlanner, small enough to run on CPU in seconds, and
PyTorch references for the JAX policy."""

import os

# The tests run on CPU, and so does every subprocess they start (scripts/*.py inherit this
# environment). On a GPU, JAX pre-allocates 75% of the GPU's memory per process; the script and
# end-to-end tests start several JAX processes while this one is alive, and on a GPU that shares
# memory with the CPU (DGX Spark / GB10) that exhausts the machine's RAM and freezes it.
# Set GPLAN_TESTS_ON_GPU=1 to run on the GPU anyway (memory is then allocated on demand).
if os.environ.get("GPLAN_TESTS_ON_GPU") != "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # PyTorch: no CUDA context either
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

import gplan.torch.lewm  # noqa: F401  puts the le-wm submodule on sys.path
from gplan.torch.policy import Sampler

# isort: split
from jepa import JEPA  # le-wm modules, importable once gplan.torch.lewm is imported
from module import ARPredictor, Embedder, MLP

B, T, A, D, HS, IMG = 4, 3, 2, 16, 3, 8  # batch, horizon, action dim, embed dim, history, image size
REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    """scripts/<name>.py as a module, without running its main()."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_env(home=None):
    """Environment of a scripts/*.py subprocess: this process's (CPU only, see above), the repository and
    tests/ on PYTHONPATH (the tiny LeWM checkpoints pickle conftest's classes), and STABLEWM_HOME=home."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}"}
    if home is not None:
        env["STABLEWM_HOME"] = str(home)
    return env

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


def torch_mixture(dist):
    """A step distribution of the PyTorch GPlanner as a torch.distributions object (an independent
    implementation of the mixture density)."""
    return MixtureSameFamily(Categorical(logits=dist.log_weights), Independent(Normal(dist.means, dist.stds), 1))


def torch_step_log_probs(model, z_start, z_goal, plans):
    """Reference for `gplan.jax.policy.step_log_probs`: teacher-forced log P_F of every step of `plans`
    (B, horizon) under the PyTorch GPlanner `model`, scored with torch.distributions. -> (B, n_steps)"""
    sampler, A = Sampler(model), model.action_dim
    return torch.stack([torch_mixture(sampler.action_dist(z_start, z_goal, plans, t)).log_prob(plans[:, t * A:(t + 1) * A])
                        for t in range(model.n_steps)], dim=-1)


def torch_log_z(model, z_start, z_goal):
    """Reference for `GPlanner.log_Z` (JAX): the PyTorch GPlanner's log Z head, summed. -> (B,)"""
    return model.Z(torch.cat([z_start, z_goal], dim=-1)).sum(-1)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(B, 3, IMG, IMG), torch.randn(B, 3, IMG, IMG)


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
