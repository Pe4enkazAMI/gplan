"""Save / load JAX planner checkpoints, and export them to the PyTorch format evaluation loads.

A checkpoint with prefix `outputs/run` is three files:
    outputs/run.eqx    Equinox weights (eqx.tree_serialise_leaves)
    outputs/run.json   {"model_kwargs": ..., "train_args": ...}
    outputs/run.pt     the same weights as a `gplan.torch.policy` checkpoint, for scripts/evaluate.py
Each file is complete or absent, and the .pt is written last (scripts/run.py skips runs whose .pt exists).
"""

import json
import os
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import torch

from gplan.jax.convert import gplanner_to_torch
from gplan.jax.policy import GPlanner


def export_torch(path, model, model_kwargs, train_args=None):
    """Write `model` as a `gplan.torch.policy.save_planner`-style .pt file."""
    torch.save({"state_dict": gplanner_to_torch(model), "model_kwargs": model_kwargs, "train_args": train_args}, path)


def write_atomically(path, write):
    """`write(p)` into a temporary folder next to `path`, under the same file name (torch.save records it
    in the archive, so the bytes are those of a direct write), then move the file into place."""
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as tmp:
        write(Path(tmp) / path.name)
        os.replace(Path(tmp) / path.name, path)


def save(prefix, model, model_kwargs, train_args=None):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"model_kwargs": model_kwargs, "train_args": train_args}, indent=2)
    write_atomically(f"{prefix}.eqx", lambda p: eqx.tree_serialise_leaves(p, model))
    write_atomically(f"{prefix}.json", lambda p: p.write_text(meta))
    write_atomically(f"{prefix}.pt", lambda p: export_torch(p, model, model_kwargs, train_args))


def load(prefix):
    """Returns (model, model_kwargs, train_args)."""
    meta = json.loads(Path(f"{prefix}.json").read_text())
    skeleton = GPlanner(**meta["model_kwargs"], key=jax.random.key(0))
    model = eqx.tree_deserialise_leaves(f"{prefix}.eqx", skeleton)
    return model, meta["model_kwargs"], meta["train_args"]
