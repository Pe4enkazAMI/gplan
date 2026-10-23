"""Frozen LeWorldModel: loading, encoding frames, and the planning cost J(A, c).

LeWM checkpoints (`*_object.ckpt`) are pickled `jepa.JEPA` objects, so unpickling
needs the le-wm source importable under its own top-level module names (`jepa`,
`module`). Importing this module puts the `le-wm` git submodule on `sys.path`.
"""

import sys
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms

from gplan.data import STABLEWM_HOME  # noqa: F401  (re-exported for scripts)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_DIR = REPO_ROOT / "le-wm"
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if not (LEWM_DIR / "jepa.py").exists():
    raise ImportError(f"le-wm sources not found in {LEWM_DIR}; run `git submodule update --init le-wm`")
if str(LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(LEWM_DIR))


def img_transform(img_size=224, batch=False):
    """uint8 frames -> normalized float tensors, as LeWM was trained: one HWC frame (as swm hands it to the
    policy) -> CHW, or with batch=True a (B, C, H, W) uint8 tensor -> (B, C, H, W) (dataset encoding)."""
    steps = [transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(**IMAGENET),
             transforms.Resize(size=img_size)]
    return transforms.Compose(steps if batch else [transforms.ToImage(), *steps])


def load_lewm(ckpt_path, device="cpu"):
    """Load a LeWM `*_object.ckpt` (a pickled `jepa.JEPA`) as a frozen eval model."""
    wm = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.eval().requires_grad_(False)
    return wm


def embed_dim(wm):
    """Dimension D of the LeWM embedding."""
    return wm.predictor.pos_embedding.shape[-1]


def history_length(wm):
    """Context length of the predictor (its number of position embeddings; 3 for every LeWM config)."""
    return wm.predictor.pos_embedding.shape[1]


CEM_HISTORY = 3  # the context swm's CEMSolver plans with: JEPA.get_cost -> JEPA.rollout(history_size=3)


def check_history(wm):
    """Our planning cost uses the predictor's own context length (`history_length`), CEM the fixed 3 of
    JEPA.rollout: stop if they differ, so every planner is scored by the same J. Returns `wm`."""
    if history_length(wm) != CEM_HISTORY:
        raise ValueError(f"LeWM predictor context is {history_length(wm)} frames, but swm's CEM plans with "
                         f"{CEM_HISTORY} (JEPA.rollout): the planners would be scored by different costs")
    return wm


@torch.no_grad()
def encode(wm, pixels):
    """Frozen LeWM embedding of a batch of frames. pixels: (B, C, H, W) -> (B, D)."""
    return wm.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]


def lewm_cost(wm, z_start, z_goal, actions, history_size=None):
    """J(A, c): LeWM planning cost of an action plan.

    Rolls the frozen predictor forward from z_start through `actions` and returns
    the squared distance of the final predicted embedding to z_goal. This is
    `JEPA.rollout` + `JEPA.criterion` with a single start frame, run directly
    on embeddings. Differentiable w.r.t. the actions (test-time refinement); the
    solvers call it under torch.no_grad.

    Args:
        z_start, z_goal: (B, D)
        actions:         (B, T, A) actions in the world model's (z-scored) action space.
        history_size:    predictor context (default: the checkpoint's, `history_length(wm)`).
    Returns:
        (B,) cost per plan.
    """
    history_size = history_size or history_length(wm)
    emb = z_start.unsqueeze(1)  # (B, 1, D)
    for t in range(actions.shape[1]):
        act_emb = wm.action_encoder(actions[:, : t + 1])
        pred = wm.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return (emb[:, -1] - z_goal).pow(2).sum(-1)
