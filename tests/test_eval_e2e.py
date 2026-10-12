"""End to end: scripts/evaluate.py on a tiny real stable_worldmodel setup (CPU, about a minute).

A short TwoRoom dataset is recorded with a random policy, LeWM is the tiny random model from
conftest (8x8 frames, so --img-size 8), and the GFlow checkpoints are untrained GPlaners. The
numbers mean nothing; what is checked is that the evaluation is reproducible and that CEM does
not depend on anything GFlow-related.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_wm

swm = pytest.importorskip("stable_worldmodel")
from gplan.torch.policy import GPlaner, save_planner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EMBED, ACTION_DIM = 16, 10  # 10 = frameskip 5 x 2-d env actions


def record(env_name, dataset, home, episodes=6, steps=60):
    world = swm.World(env_name=env_name, num_envs=2, image_shape=(224, 224), max_episode_steps=steps, verbose=0)
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    world.record_dataset(dataset, episodes=episodes, seed=0, cache_dir=home)
    world.close()


@pytest.fixture(scope="session")
def home(tmp_path_factory):
    """A STABLEWM_HOME with toy datasets, a tiny LeWM checkpoint and two GFlow checkpoints."""
    home = tmp_path_factory.mktemp("stablewm_home")
    record("swm/TwoRoom-v1", "toy_tworoom", home)
    record("swm/PushT-v1", "toy_pusht", home, episodes=4)
    (home / "toy").mkdir()
    torch.save(make_wm(embed=EMBED, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=ACTION_DIM),
               home / "toy" / "lewm_object.ckpt")
    for name, seed, n_components in (("gauss", 1, 1), ("mixture", 2, 3)):
        torch.manual_seed(seed)
        kwargs = dict(state_dim=EMBED, horizon=5 * ACTION_DIM, action_dim=ACTION_DIM, n_components=n_components)
        save_planner(home / f"{name}.pt", GPlaner(**kwargs), kwargs, train_args={"beta": 90.0})
    return home


def evaluate(home, tmp_path, *args, task="tworoom", dataset="toy_tworoom"):
    """Run scripts/evaluate.py in a fresh process (as on the cluster); returns the --results-json content."""
    out = tmp_path / f"results_{len(list(tmp_path.glob('results_*')))}.json"
    env = {**os.environ, "STABLEWM_HOME": str(home), "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}"}
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--task", task, "--wm", "toy/lewm",
           "--dataset", dataset, "--num-eval", "4", "--img-size", "8", "--device", "cpu",
           "--wandb-mode", "disabled", "--results-json", str(out), *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    # swm only warns when a reset column is missing and then keeps the env's own random start
    assert "skipping callable" not in proc.stdout + proc.stderr
    return json.loads(out.read_text())["policies"]


def outcome(metrics):
    """Everything that must be identical between two runs of the same policy (timings excluded;
    the random policy has no planner, so no plan costs)."""
    return {k: metrics.get(k) for k in ("success_rate", "episode_successes", "plan_cost", "plan_cost_first", "replans",
                                    "plans_per_replan")}


def test_cem_does_not_depend_on_gflow(home, tmp_path):
    """Same CEM result with either checkpoint, in either order, alone, and whatever --gflow-samples is."""
    a = evaluate(home, tmp_path, "--policy", "gflow", "cem", "--planner", str(home / "gauss.pt"))
    b = evaluate(home, tmp_path, "--policy", "cem", "gflow", "--planner", str(home / "mixture.pt"),
                 "--gflow-samples", "128")
    c = evaluate(home, tmp_path, "--policy", "cem")
    assert outcome(a["cem"]) == outcome(b["cem"]) == outcome(c["cem"])
    assert a["cem"]["plans_per_replan"] == 300 * 30  # the LeWM paper's CEM, untouched by GFlow flags
    assert b["gflow"]["plans_per_replan"] == 128


def test_every_policy_is_reproducible(home, tmp_path):
    args = ("--policy", "gflow", "gflow-refine", "cem", "random", "--planner", str(home / "mixture.pt"),
            "--refine-steps", "3")
    # the second run also checks that --deterministic changes nothing on CPU
    first, second = evaluate(home, tmp_path, *args), evaluate(home, tmp_path, *args, "--deterministic")
    assert list(first) == ["gflow", "gflow-refine", "cem", "random"]
    for name in first:
        assert outcome(first[name]) == outcome(second[name]), name
    for name in ("gflow", "gflow-refine", "cem"):
        assert np.isfinite(first[name]["plan_cost"]) and first[name]["replans"] == 2, name  # budget 50 = 2 x 25 steps
    assert first["gflow-refine"]["plans_per_replan"] == 64 * (3 + 1)


def test_pusht_resets_from_the_full_state(home, tmp_path):
    """PushT starts and goals come from the `state` / `goal_state` columns (agent, block pose)."""
    res = evaluate(home, tmp_path, "--policy", "cem", "random", "--cem-steps", "2", task="pusht", dataset="toy_pusht")
    assert set(res) == {"cem", "random"}
    assert len(res["cem"]["episode_successes"]) == 4
