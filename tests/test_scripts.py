"""The command-line scripts, run as on the cluster: each in a fresh Python process, chained like the
real workflow on a toy STABLEWM_HOME (an 8x8-pixel dataset and the tiny LeWM from conftest)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from conftest import IMG, make_wm

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow
EMBED, ENV_ACTION_DIM, FRAMESKIP = 16, 2, 5


def run(args, home=None, timeout=600):
    """Run a script (or `-c` code) in a fresh interpreter; returns stdout."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}"}
    if home is not None:
        env["STABLEWM_HOME"] = str(home)
    proc = subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def script(name):
    return str(REPO / "scripts" / name)


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """$STABLEWM_HOME with a raw toy dataset (3 episodes of 40 frames) and a tiny LeWM checkpoint."""
    home = tmp_path_factory.mktemp("stablewm_home")
    rng = np.random.default_rng(0)
    n_eps, ep_len = 3, 40
    with h5py.File(home / "toy.h5", "w") as h5:
        h5["pixels"] = rng.integers(0, 256, size=(n_eps * ep_len, IMG, IMG, 3), dtype=np.uint8)
        h5["ep_idx"] = np.repeat(np.arange(n_eps), ep_len)
        h5["step_idx"] = np.tile(np.arange(ep_len), n_eps)
        h5["action"] = rng.normal(size=(n_eps * ep_len, ENV_ACTION_DIM)).astype(np.float32)
    (home / "toy").mkdir()
    wm = make_wm(embed=EMBED, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16,
                 action_dim=ENV_ACTION_DIM * FRAMESKIP)
    torch.save(wm, home / "toy" / "lewm_object.ckpt")
    return home


@pytest.fixture(scope="session")
def latents(home):
    """scripts/encode_dataset.py on the toy dataset."""
    run([script("encode_dataset.py"), "--dataset", "toy", "--ckpt", str(home / "toy" / "lewm_object.ckpt"),
         "--img-size", str(IMG), "--batch-size", "32", "--device", "cpu"], home)
    return home / "toy_latents"


@pytest.fixture(scope="session")
def trained(home, latents, tmp_path_factory):
    """scripts/train.py for a few steps: mixture policy, MALA, deterministic mode, 1 of 3 episodes held out."""
    out = tmp_path_factory.mktemp("outputs") / "run"
    run([script("train.py"), "--dataset", "toy", "--ckpt", str(home / "toy" / "lewm_object.ckpt"),
         "--n-components", "2", "--ls-steps", "2", "--ls-every", "2", "--n-batches", "4", "--batch-size", "4",
         "--n-samples", "2", "--beta", "1", "--deterministic", "--holdout-frac", "0.34", "--lr-schedule", "cosine",
         "--wandb-mode", "disabled",
         "--out", str(out)], home)
    return out


@pytest.mark.parametrize("method", ["bc", "map"])
def test_train_baselines(home, latents, tmp_path, method):
    out = tmp_path / method
    run([script("train.py"), "--dataset", "toy", "--ckpt", str(home / "toy" / "lewm_object.ckpt"), "--method", method,
         "--n-batches", "3", "--batch-size", "4", "--beta", "1", "--holdout-frac", "0.34", "--wandb-mode", "disabled",
         "--out", str(out)], home)
    meta = json.loads(Path(f"{out}.json").read_text())
    assert meta["train_args"]["method"] == method and Path(f"{out}.pt").exists()


def test_encode_dataset(home, latents):
    z = np.load(f"{latents}.npy")
    meta = np.load(f"{latents}_meta.npz")
    info = json.loads(Path(f"{latents}.json").read_text())
    assert z.shape == (120, EMBED) and np.all(np.isfinite(z))
    assert meta["action"].shape == (120, ENV_ACTION_DIM) and info["n_frames"] == 120
    assert Path(info["ckpt"]) == (home / "toy" / "lewm_object.ckpt").resolve()


def test_train_writes_all_three_checkpoint_files(trained):
    for suffix in (".eqx", ".json", ".pt"):
        assert Path(f"{trained}{suffix}").exists(), suffix
    meta = json.loads(Path(f"{trained}.json").read_text())
    assert meta["model_kwargs"] == {"state_dim": EMBED, "horizon": 5 * FRAMESKIP * ENV_ACTION_DIM,
                                    "action_dim": FRAMESKIP * ENV_ACTION_DIM, "n_components": 2}
    assert meta["train_args"]["ls_steps"] == 2 and meta["train_args"]["deterministic"] is True
    assert meta["train_args"]["holdout_frac"] == 0.34 and meta["train_args"]["split_seed"] == 0
    assert meta["train_args"]["lr_schedule"] == "cosine"


def test_export_planner_reproduces_the_training_export(trained, tmp_path):
    run([script("export_planner.py"), str(trained), "--out", str(tmp_path / "copy.pt")])
    a = torch.load(f"{trained}.pt", weights_only=False)
    b = torch.load(tmp_path / "copy.pt", weights_only=False)
    assert a["model_kwargs"] == b["model_kwargs"] and a["train_args"] == b["train_args"]
    for k, v in a["state_dict"].items():
        assert torch.equal(v, b["state_dict"][k]), k


def test_probe_headroom(home, trained):
    out = run([script("probe_headroom.py"), str(trained), "--n-pairs", "8", "--chunk", "4", "--n-samples", "4",
               "--steps", "2"], home)
    assert "relative drop" in out and "expert (dataset actions)" in out
    assert "pairs from holdout episodes" in out  # the planner was trained with a held-out split


def test_importing_the_training_script_does_not_start_jax():
    """XLA reads XLA_FLAGS when its backend starts, so `train.py --deterministic` only works if no import
    (gplan.jax.* included) creates a JAX array before main() sets the flag."""
    out = run(["-c", "import runpy; from jax._src import xla_bridge as xb; "
                     f"runpy.run_path({script('train.py')!r}, run_name='not_main'); "
                     "print('started' if xb.backends_are_initialized() else 'idle')"])
    assert out.strip().endswith("idle")


def test_evaluate_policy_flag_keeps_order_and_drops_repeats():
    pytest.importorskip("stable_worldmodel")
    out = run(["-c", "import runpy, sys; sys.argv = ['evaluate.py', '--policy', 'gflow', 'cem', '--policy', "
                     "'gflow', 'random']; "
                     f"args, _ = runpy.run_path({script('evaluate.py')!r}, run_name='not_main')['parse_args'](); "
                     "print(args.policy)"])
    assert out.strip().endswith("['gflow', 'cem', 'random']")


def test_convert_lewm_builds_a_jepa_from_an_hf_config(tmp_path):
    """build_lewm on a small config shaped like the HF config.json (no download): the model loads its
    own state_dict strictly, pickles, and encodes frames."""
    pytest.importorskip("transformers")
    config = {
        "encoder": {"_target_": "vit", "size": "tiny", "image_size": 8, "patch_size": 4},
        "predictor": {"_target_": "p", "num_frames": 3, "depth": 1, "heads": 2, "mlp_dim": 32, "input_dim": 192,
                      "hidden_dim": 192, "dim_head": 8},
        "action_encoder": {"_target_": "a", "input_dim": 10, "smoothed_dim": 10, "emb_dim": 192},
        "projector": {"_target_": "m", "input_dim": 192, "hidden_dim": 16, "output_dim": 192},
        "pred_proj": {"_target_": "m", "input_dim": 192, "hidden_dim": 16, "output_dim": 192},
    }
    out = run(["-c", "import json, runpy, torch; "
                     f"m = runpy.run_path({script('convert_lewm.py')!r}, run_name='not_main'); "
                     f"wm = m['build_lewm'](json.loads({json.dumps(json.dumps(config))})); "
                     "wm.load_state_dict(wm.state_dict(), strict=True); "
                     f"torch.save(wm.eval(), {str(tmp_path / 'x_object.ckpt')!r}); "
                     "from gplan.torch.lewm import encode, load_lewm; "
                     f"wm = load_lewm({str(tmp_path / 'x_object.ckpt')!r}); "
                     "print(tuple(encode(wm, torch.randn(2, 3, 8, 8)).shape))"])
    assert out.strip().endswith("(2, 192)")
