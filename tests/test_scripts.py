"""The command-line scripts, run as on the cluster: each in a fresh Python process, chained like the
real workflow on a toy STABLEWM_HOME (an 8x8-pixel dataset and the tiny LeWM from conftest)."""

import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from conftest import IMG, load_script, make_wm, REPO, save_as_swm, script_env

from gplan.torch.lewm import load_wm
from gplan.torch.policy import load_planner

pytestmark = pytest.mark.slow
EMBED, ENV_ACTION_DIM, FRAMESKIP = 16, 2, 5


def run(args, home=None, timeout=600):
    """Run a script (or `-c` code) in a fresh interpreter; returns stdout."""
    proc = subprocess.run([sys.executable, *args], env=script_env(home), capture_output=True, text=True,
                          timeout=timeout)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def script(name):
    return str(REPO / "scripts" / name)


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """$STABLEWM_HOME with a raw toy dataset (3 episodes of 40 frames) and a tiny LeWM checkpoint, also where
    scripts/run.py task=tworoom looks for it (tworoom/lewm_object.ckpt, a link)."""
    home = tmp_path_factory.mktemp("stablewm_home")
    write_toy_dataset(home / "toy.h5", ENV_ACTION_DIM)
    (home / "toy").mkdir()
    wm = make_wm(embed=EMBED, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16,
                 action_dim=ENV_ACTION_DIM * FRAMESKIP)
    torch.save(wm, home / "toy" / "lewm_object.ckpt")
    (home / "tworoom").mkdir()
    (home / "tworoom" / "lewm_object.ckpt").symlink_to(home / "toy" / "lewm_object.ckpt")
    return home


def write_toy_dataset(path, env_action_dim, n_eps=3, ep_len=40, seed=0):
    """A raw LeWM-format .h5: random 8x8 frames and actions, `n_eps` episodes of `ep_len` frames."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5["pixels"] = rng.integers(0, 256, size=(n_eps * ep_len, IMG, IMG, 3), dtype=np.uint8)
        h5["ep_idx"] = np.repeat(np.arange(n_eps), ep_len)
        h5["step_idx"] = np.tile(np.arange(ep_len), n_eps)
        h5["action"] = rng.normal(size=(n_eps * ep_len, env_action_dim)).astype(np.float32)


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
    run([script("train.py"), "dataset=toy", f"ckpt={home / 'toy' / 'lewm_object.ckpt'}",
         "n_components=2", "ls_steps=2", "ls_every=2", "n_batches=4", "batch_size=4",
         "n_samples=2", "beta=1", "deterministic=true", "holdout_frac=0.34",
         "wandb_mode=disabled",
         f"out={out}"], home)
    return out


@pytest.mark.parametrize("method, extra", [("bc", []), ("map", []), ("ppo", ["ent_coef=1", "target_kl=0.02"])])
def test_train_baselines(home, latents, tmp_path, method, extra):
    """Each baseline trains a few steps and exports a planner evaluate.py loads (a GPlanner of the same shape)."""
    out = tmp_path / method
    run([script("train.py"), "dataset=toy", f"ckpt={home / 'toy' / 'lewm_object.ckpt'}", f"method={method}",
         "n_batches=3", "batch_size=4", "beta=1", "holdout_frac=0.34", "wandb_mode=disabled",
         f"out={out}", *extra], home)
    meta = json.loads(Path(f"{out}.json").read_text())
    assert meta["train_args"]["method"] == method and meta["model_kwargs"]["horizon"] == 5 * FRAMESKIP * ENV_ACTION_DIM
    planner, train_args = load_planner(f"{out}.pt")
    assert train_args["method"] == method and planner.horizon == 5 * FRAMESKIP * ENV_ACTION_DIM


@pytest.fixture(scope="session")
def cube_like(home):
    """A Cube-like dataset (5-d actions, in a subfolder like ogbench/...), a tiny LeWM whose action
    encoder takes 5 x 5 = 25 values, and its latents. Returns the checkpoint path."""
    write_toy_dataset(home / "ogbench" / "toy5.h5", 5)
    (home / "toy25").mkdir()
    ckpt = home / "toy25" / "lewm_object.ckpt"
    torch.save(make_wm(embed=EMBED, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=25), ckpt)
    run([script("encode_dataset.py"), "--dataset", "ogbench/toy5", "--ckpt", str(ckpt), "--img-size", str(IMG),
         "--batch-size", "32", "--device", "cpu"], home)
    return ckpt


@pytest.mark.parametrize("method", ["gfn", "bc"])
def test_train_infers_a_25_value_plan_step_from_5d_actions(home, cube_like, tmp_path, method):
    """A plan step is frameskip 5 x 5 = 25 values, read from the data and matching the LeWM action
    encoder; gfn (mixture + MALA) and bc train on it."""
    out = tmp_path / method
    extra = ["n_components=2", "ls_steps=2", "ls_every=2"] if method == "gfn" else []
    run([script("train.py"), "dataset=ogbench/toy5", f"ckpt={cube_like}", f"method={method}",
         "n_batches=3", "batch_size=4", "n_samples=2", "beta=1", "wandb_mode=disabled",
         f"out={out}", *extra], home)
    meta = json.loads(Path(f"{out}.json").read_text())
    assert meta["train_args"]["action_dim"] == 25 and meta["model_kwargs"]["horizon"] == 5 * 25


def test_train_refuses_a_plan_step_the_lewm_cannot_read(home, cube_like, tmp_path):
    """frameskip 2 x 5-d actions = 10 values, but the LeWM action encoder takes 25: a clear error."""
    proc = subprocess.run([sys.executable, script("train.py"), "dataset=ogbench/toy5", f"ckpt={cube_like}",
                           "frameskip=2", "n_batches=1", "wandb_mode=disabled",
                           f"out={tmp_path / 'x'}"], env=script_env(home), capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode != 0 and "action encoder takes 25" in proc.stdout + proc.stderr, proc.stderr[-2000:]


def test_run_trains_each_model_once_and_saves_its_config(home, latents, tmp_path):
    """scripts/run.py stage=train on the toy data: one tiny gfn run with its composed config next to the
    checkpoint; a second call skips it."""
    overrides = ["stage=train", "task.dataset=toy", "task.gfn.n_batches=2", "models=[gfn]",
                 "seeds=[0]", "ablations=false", "holdout_frac=0.34", "wandb_mode=disabled", f"out={tmp_path}",
                 "train_overrides={batch_size:4,n_samples:2}"]
    first = run([script("run.py"), *overrides], home)
    assert f"=== train {tmp_path}/gfn-s0" in first and (tmp_path / "gfn-s0.pt").exists()
    config = (tmp_path / "gfn-s0.yaml").read_text()
    assert config.startswith("# overrides: task.dataset=toy task.gfn.n_batches=2 wm=lewm model=gfn seed=0")
    assert "n_batches: 2\n" in config and "batch_size: 4\n" in config and "dataset: toy\n" in config
    meta = json.loads((tmp_path / "gfn-s0.json").read_text())["train_args"]
    assert (meta["method"], meta["n_batches"], meta["beta"], meta["wandb_name"]) == ("gfn", 2, 90.0, "tworoom-gfn-s0")
    assert f"skip {tmp_path}/gfn-s0 (exists)" in run([script("run.py"), *overrides], home)


@pytest.fixture(scope="session")
def pldm(home, latents):
    """The toy LeWM saved as stable-worldmodel saves a PLDM (toy/pldm_object.ckpt: the same weights), and its
    latents, encoded under the default PLDM name toy_pldm_latents. Returns the checkpoint path."""
    wm = torch.load(home / "toy" / "lewm_object.ckpt", weights_only=False)
    ckpt = save_as_swm(wm, home / "toy" / "pldm_object.ckpt")
    run([script("encode_dataset.py"), "--dataset", "toy", "--ckpt", str(ckpt), "--img-size", str(IMG),
         "--batch-size", "32", "--device", "cpu"], home)
    return ckpt


def test_a_pldm_checkpoint_trains_like_lewm_with_the_same_weights(home, latents, pldm, tmp_path):
    """Latents and training through a PLDM checkpoint: its own files, and the same numbers as LeWM gives when
    the weights are the same (deterministic training, step logs and weights compared exactly)."""
    np.testing.assert_array_equal(np.load(home / "toy_pldm_latents.npy"), np.load(f"{latents}.npy"))
    assert Path(json.loads((home / "toy_pldm_latents.json").read_text())["ckpt"]) == pldm.resolve()
    logs = {}
    for name in ("lewm", "pldm"):
        out = run([script("train.py"), "dataset=toy", f"ckpt={home / 'toy' / f'{name}_object.ckpt'}", "n_components=2",
                   "ls_steps=2", "ls_every=2", "n_batches=4", "batch_size=4", "n_samples=2", "beta=1", "log_every=1",
                   "wandb_mode=disabled", f"out={tmp_path / name}"], home)
        logs[name] = [line for line in out.splitlines() if line.startswith("step")]
    assert len(logs["pldm"]) == 4 and logs["pldm"] == logs["lewm"]
    assert (tmp_path / "pldm.eqx").read_bytes() == (tmp_path / "lewm.eqx").read_bytes()
    # PLDM with the LeWM latents: refused
    proc = subprocess.run([sys.executable, script("train.py"), "dataset=toy", f"ckpt={pldm}", f"latents={latents}",
                           "n_batches=1", "wandb_mode=disabled", f"out={tmp_path / 'x'}"], env=script_env(home),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0 and "re-run scripts/encode_dataset.py" in proc.stdout + proc.stderr


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
    """XLA reads XLA_FLAGS when its backend starts, so `train.py deterministic=true` only works if no import
    (gplan.jax.* included) creates a JAX array before main() sets the flag."""
    out = run(["-c", "import runpy; from jax._src import xla_bridge as xb; "
                     f"runpy.run_path({script('train.py')!r}, run_name='not_main'); "
                     "print('started' if xb.backends_are_initialized() else 'idle')"])
    assert out.strip().endswith("idle")


def test_evaluate_policy_list_keeps_order_and_drops_repeats():
    pytest.importorskip("stable_worldmodel")
    out = run(["-c", "import runpy, sys; sys.argv = ['evaluate.py', 'policy=[gflow,cem,gflow,random]']; "
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
                     "from gplan.torch.lewm import encode, load_wm; "
                     f"wm = load_wm({str(tmp_path / 'x_object.ckpt')!r}); "
                     "print(tuple(encode(wm, torch.randn(2, 3, 8, 8)).shape))"])
    assert out.strip().endswith("(2, 192)")


def test_convert_lewm_reads_a_stable_worldmodel_training_run(tmp_path):
    """--run-dir: the config.json and last weights_epoch_<n>.pt that stable-worldmodel's training scripts write
    (here a PLDM's, its ViT under transformers-5 weight names) become <name>_object.ckpt with the same weights."""
    pytest.importorskip("transformers")
    config = {
        "_target_": "stable_worldmodel.wm.pldm.pldm.PLDM",
        "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf", "size": "tiny", "patch_size": 4,
                    "image_size": 8, "pretrained": False, "use_mask_token": False},
        "predictor": {"_target_": "stable_worldmodel.wm.pldm.module.Predictor", "num_frames": 3, "input_dim": 192,
                      "hidden_dim": 192, "output_dim": 192, "depth": 1, "heads": 2, "mlp_dim": 32, "dim_head": 8,
                      "dropout": 0.1, "emb_dropout": 0.0},
        "action_encoder": {"_target_": "stable_worldmodel.wm.pldm.module.Embedder", "input_dim": 10, "emb_dim": 192},
        "projector": {"_target_": "stable_worldmodel.wm.pldm.module.MLP", "input_dim": 192, "output_dim": 192,
                      "hidden_dim": 16, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
        "pred_proj": {"_target_": "stable_worldmodel.wm.pldm.module.MLP", "input_dim": 192, "output_dim": 192,
                      "hidden_dim": 16, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
    }
    v5_names = [  # transformers-4 ViT weight names -> transformers 5 (as stable-worldmodel's environment saves them)
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
        (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
        (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
        (r"^encoder\.encoder\.layer\.(\d+)\.(layernorm_before|layernorm_after)\.", r"encoder.layers.\1.\2."),
    ]
    wm = load_script("convert_lewm").build_lewm(config)
    for p in wm.parameters():
        torch.nn.init.normal_(p)
    run_dir = tmp_path / "checkpoints" / "toy" / "pldm"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    torch.save({k: torch.zeros_like(v) for k, v in wm.state_dict().items()}, run_dir / "weights_epoch_95.pt")
    renamed = {}
    for key, value in wm.state_dict().items():
        for pattern, repl in v5_names:
            key = re.sub(pattern, repl, key)
        renamed[key] = value
    assert any(".q_proj." in k for k in renamed)
    torch.save(renamed, run_dir / "weights_epoch_100.pt")

    run([script("convert_lewm.py"), "--run-dir", str(run_dir), "--name", "toy/pldm"], tmp_path)
    loaded, ref = load_wm(tmp_path / "toy" / "pldm_object.ckpt").state_dict(), wm.state_dict()
    assert list(loaded) == list(ref) and all(torch.equal(loaded[k], ref[k]) for k in ref)  # epoch 100, names mapped
