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
from gplan.data import episode_column, split_episodes  # noqa: E402
from gplan.results import comparison_table, group_runs, load_runs  # noqa: E402
from gplan.torch.policy import GPlaner, save_planner  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow
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
    # trained (nominally) with half of toy_tworoom's episodes held out
    split_args = {"beta": 90.0, "dataset": "toy_tworoom", "holdout_frac": 0.5, "split_seed": 0}
    save_planner(home / "gauss_split.pt", GPlaner(**kwargs), kwargs, train_args=split_args)
    return home


def evaluate(home, tmp_path, *args, task="tworoom", dataset="toy_tworoom", full=False, expect_failure=False):
    """Run scripts/evaluate.py in a fresh process (as on the cluster); returns the policies of the
    --results-json (the whole file with full=True, the console output with expect_failure=True)."""
    out = tmp_path / f"results_{len(list(tmp_path.glob('results_*')))}.json"
    env = {**os.environ, "STABLEWM_HOME": str(home), "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}"}
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--task", task, "--wm", "toy/lewm",
           "--dataset", dataset, "--num-eval", "4", "--img-size", "8", "--device", "cpu",
           "--wandb-mode", "disabled", "--results-json", str(out), *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if expect_failure:
        assert proc.returncode != 0
        return proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    # swm only warns when a reset column is missing and then keeps the env's own random start
    assert "skipping callable" not in proc.stdout + proc.stderr
    results = json.loads(out.read_text())
    return results if full else results["policies"]


def outcome(metrics):
    """Everything that must be identical between two runs of the same policy (timings excluded;
    the random policy has no planner, so no plan costs)."""
    return {k: metrics.get(k) for k in ("success_rate", "episode_successes", "plan_cost", "plan_cost_first",
                                        "plan_cost_replan0", "plan_cost_replan1", "action_absmax", "replans",
                                        "plans_per_replan", "backward_per_replan")}


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
    args = ("--policy", "gflow", "gflow-refine", "cem", "prior", "prior-refine", "random",
            "--planner", str(home / "mixture.pt"), "--refine-steps", "3", "--prior-samples", "32")
    # the second run also checks that --deterministic changes nothing on CPU
    first, second = evaluate(home, tmp_path, *args), evaluate(home, tmp_path, *args, "--deterministic")
    assert list(first) == ["gflow", "gflow-refine", "cem", "prior", "prior-refine", "random"]
    for name in first:
        assert outcome(first[name]) == outcome(second[name]), name
    for name in ("gflow", "gflow-refine", "cem", "prior", "prior-refine"):
        assert np.isfinite(first[name]["plan_cost"]) and first[name]["replans"] == 2, name  # budget 50 = 2 x 25 steps
    assert first["gflow-refine"]["plans_per_replan"] == 64 * (3 + 1)
    assert first["gflow-refine"]["backward_per_replan"] == 64 * 3

    # the aggregation reads these files: a repeated run is one result per policy, paired with CEM
    rows = {r["policy"]: r for r in comparison_table(group_runs(load_runs(sorted(tmp_path.glob("results_*.json")))))}
    assert set(rows) == set(first) and all(r["episodes"] == 4 for r in rows.values())
    assert rows["gflow"]["label"] == "mixture" and "only_reference" in rows["gflow"]
    assert first["cem"]["backward_per_replan"] == first["gflow"]["backward_per_replan"] == 0
    assert first["prior"]["plans_per_replan"] == 32 and first["prior"]["backward_per_replan"] == 0
    assert (first["prior-refine"]["plans_per_replan"], first["prior-refine"]["backward_per_replan"]) == (32 * 4, 32 * 3)
    for name in ("gflow", "cem"):  # per-replan costs average to the overall one (equal env counts)
        m = first[name]
        assert np.isclose((m["plan_cost_replan0"] + m["plan_cost_replan1"]) / 2, m["plan_cost"])
        assert m["plan_cost_first"] == m["plan_cost_replan0"]
        assert 0 < m["action_absmax"] <= m["action_absmax_max"]


def test_pusht_resets_from_the_full_state(home, tmp_path):
    """PushT starts and goals come from the `state` / `goal_state` columns (agent, block pose)."""
    res = evaluate(home, tmp_path, "--policy", "cem", "random", "--cem-steps", "2", task="pusht", dataset="toy_pusht")
    assert set(res) == {"cem", "random"}
    assert len(res["cem"]["episode_successes"]) == 4


def test_held_out_evaluation_uses_only_unseen_episodes(home, tmp_path):
    """With --holdout-frac, episodes come from the held-out side of the split; a planner whose
    recorded split does not hold them out is refused instead of being evaluated on its training data."""
    res = evaluate(home, tmp_path, "--policy", "gflow", "cem", "--planner", str(home / "gauss_split.pt"),
                   "--holdout-frac", "0.5", "--split-seed", "0", full=True)
    dataset = swm.data.HDF5Dataset("toy_tworoom", cache_dir=home)
    _, held_out = split_episodes(dataset.get_col_data(episode_column(dataset.column_names)), 0.5, 0)
    assert {ep for ep, _ in res["episodes"]} <= set(held_out.tolist())
    assert res["planner_train_args"]["holdout_frac"] == 0.5 and set(res["policies"]) == {"gflow", "cem"}

    out = evaluate(home, tmp_path, "--policy", "gflow", "--planner", str(home / "gauss.pt"),
                   "--holdout-frac", "0.5", expect_failure=True)
    assert "were in the planner's training data" in out
    out = evaluate(home, tmp_path, "--policy", "gflow", "--planner", str(home / "gauss_split.pt"),
                   "--holdout-frac", "0.5", "--split-seed", "1", expect_failure=True)  # a different split
    assert "were in the planner's training data" in out
