"""Frozen world models of the LeWorldModel architecture: loading, encoding frames, the planning cost J(A, c).

Two world models have this architecture: LeWM itself and PLDM as stable-worldmodel implements it (the same
encoder, predictor and planning cost, trained with a VICReg-style loss instead of SIGReg). Their
checkpoints (`*_object.ckpt`) are pickled modules: LeWM's are `jepa.JEPA` objects of the le-wm sources,
stable-worldmodel's are its own copies of those classes (`stable_worldmodel.wm.lewm`, `.pldm`). `load_wm`
loads either into the le-wm classes, so unpickling needs the le-wm source importable under its own
top-level module names (`jepa`, `module`): importing this module puts the `le-wm` git submodule on
`sys.path`.
"""

import pickle
import sys
import types
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

import module as lewm_modules  # noqa: E402  (le-wm)
from jepa import JEPA  # noqa: E402  (le-wm)

# stable-worldmodel's classes of this architecture compute exactly what le-wm's do (they only store their
# constructor arguments as attributes as well); its `Predictor` is le-wm's `ARPredictor` and its LeWM / PLDM
# are `jepa.JEPA`. Unpickling them as the le-wm classes lets every checkpoint load whatever version of
# stable-worldmodel is installed, and plan with the same `JEPA.rollout` / `JEPA.get_cost` as LeWM.
SWM_MODULES = ("stable_worldmodel.wm.lewm", "stable_worldmodel.wm.pldm")
SWM_NAMES = {"LeWM": ("jepa", "JEPA"), "PLDM": ("jepa", "JEPA"), "Predictor": ("module", "ARPredictor")}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module == m or module.startswith(f"{m}.") for m in SWM_MODULES):
            target_module, target = SWM_NAMES.get(name, ("module", name))
            if target_module == "module" and not hasattr(lewm_modules, target):
                raise ValueError(f"{module}.{name} has no counterpart in le-wm/module.py: not a checkpoint of "
                                 f"the LeWM architecture")
            module, name = target_module, target
        return super().find_class(module, name)


_PICKLE = types.ModuleType("gplan_lewm_pickle")  # what torch.load needs of a pickle module
_PICKLE.Unpickler = _Unpickler
_PICKLE.load = lambda f, **kwargs: _Unpickler(f, **kwargs).load()


def load_wm(ckpt_path, device="cpu"):
    """Load a world-model `*_object.ckpt` of the LeWM architecture (LeWM, PLDM) as a frozen eval `jepa.JEPA`.

    The checkpoint may hold the model inside a larger module; its first `jepa.JEPA` is used, as
    `swm.policy.AutoCostModel` takes the first module with a planning cost. A checkpoint without one is an
    error: other architectures (e.g. DINO-WM) need their own encoder, cost and JAX port."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False, pickle_module=_PICKLE)
    wm = next((m for m in obj.modules() if isinstance(m, JEPA)), None) if isinstance(obj, torch.nn.Module) else None
    if wm is None:
        raise ValueError(f"{ckpt_path} holds a {type(obj).__name__}, not a world model of the LeWM architecture "
                         f"(LeWM, PLDM): other world models are not supported")
    wm.eval().requires_grad_(False)
    return wm


def img_transform(img_size=224, batch=False):
    """uint8 frames -> normalized float tensors, as LeWM was trained: one HWC frame (as swm hands it to the
    policy) -> CHW, or with batch=True a (B, C, H, W) uint8 tensor -> (B, C, H, W) (dataset encoding)."""
    steps = [transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(**IMAGENET),
             transforms.Resize(size=img_size)]
    return transforms.Compose(steps if batch else [transforms.ToImage(), *steps])


def embed_dim(wm):
    """Dimension D of the world model's embedding."""
    return wm.predictor.pos_embedding.shape[-1]


def history_length(wm):
    """Context length of the predictor (its number of position embeddings; 3 for every LeWM and PLDM config)."""
    return wm.predictor.pos_embedding.shape[1]


CEM_HISTORY = 3  # the context swm's CEMSolver plans with: JEPA.get_cost -> JEPA.rollout(history_size=3)


def check_history(wm):
    """Our planning cost uses the predictor's own context length (`history_length`), CEM the fixed 3 of
    JEPA.rollout: stop if they differ, so every planner is scored by the same J. Returns `wm`."""
    if history_length(wm) != CEM_HISTORY:
        raise ValueError(f"world model predictor context is {history_length(wm)} frames, but swm's CEM plans "
                         f"with {CEM_HISTORY} (JEPA.rollout): the planners would be scored by different costs")
    return wm


@torch.no_grad()
def encode(wm, pixels):
    """Frozen embedding of a batch of frames. pixels: (B, C, H, W) -> (B, D)."""
    return wm.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]


def lewm_cost(wm, z_start, z_goal, actions, history_size=None):
    """J(A, c): planning cost of an action plan under a world model of the LeWM architecture.

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
