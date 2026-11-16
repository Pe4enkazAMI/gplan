"""Loading world models of the LeWM architecture (gplan.torch.lewm.load_wm): LeWM checkpoints as before, and
stable-worldmodel's LeWM / PLDM checkpoints (its own copies of the classes) as the same le-wm model."""

import sys
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import A, B, D, make_wm, save_as_swm, T

from gplan.torch.lewm import encode, JEPA, lewm_cost, load_wm


def toy_wm():
    return make_wm(embed=D, depth=2, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=A)


def same_model(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def pickled_names(path):
    """The module paths a torch checkpoint's pickle refers to (its data.pkl, as text)."""
    with zipfile.ZipFile(path) as z:
        return z.read(next(n for n in z.namelist() if n.endswith("data.pkl"))).decode("latin-1")


def test_a_lewm_checkpoint_loads_as_before(tmp_path):
    wm = toy_wm()
    torch.save(wm, tmp_path / "lewm_object.ckpt")
    loaded = load_wm(tmp_path / "lewm_object.ckpt")
    plain = torch.load(tmp_path / "lewm_object.ckpt", weights_only=False)
    assert type(loaded) is type(plain) is JEPA and same_model(loaded, plain)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


@pytest.mark.parametrize("family", ["pldm", "lewm"])
def test_a_stable_worldmodel_checkpoint_is_the_same_model(tmp_path, batch, family):
    """swm's PLDM / LeWM classes are le-wm's: a model saved under their names loads as the le-wm JEPA with the
    same weights, embeddings and planning cost, whether or not stable-worldmodel is installed."""
    wm = toy_wm()
    path = save_as_swm(wm, tmp_path / f"{family}_object.ckpt", family)
    assert f"stable_worldmodel.wm.{family}" in pickled_names(path) and "\njepa\n" not in pickled_names(path)
    assert f"stable_worldmodel.wm.{family}" not in sys.modules  # the stand-ins are gone
    loaded = load_wm(path)
    assert type(loaded) is JEPA and same_model(loaded, wm) and not loaded.training
    z_start, z_goal = encode(wm, batch[0]), encode(wm, batch[1])
    assert torch.equal(encode(loaded, batch[0]), z_start)
    actions = torch.randn(B, T, A)
    assert torch.equal(lewm_cost(loaded, z_start, z_goal, actions), lewm_cost(wm, z_start, z_goal, actions))
    # the JAX training cost (gplan.jax.convert.load_lewm_predictor): the same as from the le-wm checkpoint
    load_predictor = pytest.importorskip("gplan.jax.convert").load_lewm_predictor
    torch.save(wm, tmp_path / "lewm_object.ckpt")
    costs = [np.asarray(load_predictor(p).cost(z_start.numpy(), z_goal.numpy(), actions.numpy()))
             for p in (path, tmp_path / "lewm_object.ckpt")]
    np.testing.assert_array_equal(*costs)


def test_the_world_model_inside_a_larger_module_is_used(tmp_path):
    """A training checkpoint may hold the model next to other modules: its first JEPA is the world model."""
    wm = toy_wm()
    holder = nn.Module()
    holder.head, holder.model = nn.Linear(2, 2), wm
    torch.save(holder, tmp_path / "wrapped_object.ckpt")
    assert same_model(load_wm(tmp_path / "wrapped_object.ckpt"), wm)


def test_other_checkpoints_are_refused(tmp_path):
    torch.save(nn.Linear(2, 2), tmp_path / "other_object.ckpt")
    with pytest.raises(ValueError, match="not a world model of the LeWM architecture"):
        load_wm(tmp_path / "other_object.ckpt")
    # a class in swm's PLDM package that le-wm does not have: a different architecture
    import module as lewm_modules
    wm = toy_wm()
    lewm_modules.Unknown = type("Unknown", (type(wm.projector),), {"__module__": "module", "__qualname__": "Unknown"})
    wm.projector.__class__ = lewm_modules.Unknown
    try:
        path = save_as_swm(wm, tmp_path / "pldm_object.ckpt")
    finally:
        del lewm_modules.Unknown
    with pytest.raises(ValueError, match="no counterpart in le-wm/module.py"):
        load_wm(path)
