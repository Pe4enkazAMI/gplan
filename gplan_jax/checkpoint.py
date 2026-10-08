"""Save / load JAX planner checkpoints, and export them to the PyTorch format evaluation loads.

A checkpoint with prefix `outputs/run` is three files:
    outputs/run.eqx    Equinox weights (eqx.tree_serialise_leaves)
    outputs/run.json   {"model_kwargs": ..., "train_args": ...}
    outputs/run.pt     the same weights as a `gplan.policy` checkpoint, for scripts/evaluate.py
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import torch

from gplan_jax.convert import gplaner_to_torch
from gplan_jax.policy import GPlaner


def export_torch(path, model, model_kwargs, train_args=None):
    """Write `model` as a `gplan.policy.save_planner`-style .pt file."""
    torch.save({"state_dict": gplaner_to_torch(model), "model_kwargs": model_kwargs, "train_args": train_args}, path)


def save(prefix, model, model_kwargs, train_args=None):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(f"{prefix}.eqx", model)
    Path(f"{prefix}.json").write_text(json.dumps({"model_kwargs": model_kwargs, "train_args": train_args}, indent=2))
    export_torch(f"{prefix}.pt", model, model_kwargs, train_args)


def load(prefix):
    """Returns (model, model_kwargs, train_args)."""
    meta = json.loads(Path(f"{prefix}.json").read_text())
    skeleton = GPlaner(**meta["model_kwargs"], key=jax.random.key(0))
    model = eqx.tree_deserialise_leaves(f"{prefix}.eqx", skeleton)
    return model, meta["model_kwargs"], meta["train_args"]
