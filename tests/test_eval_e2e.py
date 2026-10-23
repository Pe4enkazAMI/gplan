"""End to end: scripts/evaluate.py on a tiny real stable_worldmodel setup (CPU, about a minute).

A short TwoRoom dataset is recorded with a random policy, LeWM is the tiny random model from
conftest (8x8 frames, so --img-size 8), and the GFlow checkpoints are untrained GPlanners. The
numbers mean nothing; what is checked is that the evaluation is reproducible and that CEM does
not depend on anything GFlow-related.
"""

import json
import shutil
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest
import torch
from conftest import make_wm, REPO, script_env

swm = pytest.importorskip("stable_worldmodel")
from gplan.data import episode_column, split_episodes  # noqa: E402
from gplan.results import comparison_table, group_runs, load_runs  # noqa: E402
from gplan.torch.evaluation import evaluate as run_protocol  # noqa: E402
from gplan.torch.evaluation import sample_eval_starts, TASKS  # noqa: E402
from gplan.torch.policy import GPlanner, save_planner  # noqa: E402

pytestmark = pytest.mark.slow
EMBED, ACTION_DIM = 16, 10  # 10 = frameskip 5 x 2-d env actions


def record(env_name, dataset, home, episodes=6, steps=60, **world_kwargs):
    (home / dataset).parent.mkdir(parents=True, exist_ok=True)
    world = swm.World(env_name=env_name, num_envs=2, image_shape=(224, 224), max_episode_steps=steps, verbose=0,
                      **world_kwargs)
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
        save_planner(home / f"{name}.pt", GPlanner(**kwargs), kwargs, train_args={"beta": 90.0})
    # trained (nominally) with half of toy_tworoom's episodes held out
    split_args = {"beta": 90.0, "dataset": "toy_tworoom", "holdout_frac": 0.5, "split_seed": 0}
    save_planner(home / "gauss_split.pt", GPlanner(**kwargs), kwargs, train_args=split_args)
    return home


def evaluate(home, tmp_path, *args, task="tworoom", dataset="toy_tworoom", wm="toy/lewm", full=False,
             expect_failure=False):
    """Run scripts/evaluate.py in a fresh process (as on the cluster); returns the policies of the
    --results-json (the whole file with full=True, the console output with expect_failure=True)."""
    out = tmp_path / f"results_{len(list(tmp_path.glob('results_*')))}.json"
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"), f"task={task}", f"wm={wm}",
           f"dataset={dataset}", "num_eval=4", "img_size=8", "device=cpu",
           "wandb_mode=disabled", f"results_json={out}", *args]
    proc = subprocess.run(cmd, env=script_env(home), capture_output=True, text=True, timeout=600)
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
    a = evaluate(home, tmp_path, "policy=[gflow,cem]", f"planner={home / 'gauss.pt'}")
    b = evaluate(home, tmp_path, "policy=[cem,gflow]", f"planner={home / 'mixture.pt'}", "gflow_samples=128")
    c = evaluate(home, tmp_path, "policy=[cem]")
    assert outcome(a["cem"]) == outcome(b["cem"]) == outcome(c["cem"])
    assert a["cem"]["plans_per_replan"] == 300 * 30  # the LeWM paper's CEM, untouched by GFlow flags
    assert b["gflow"]["plans_per_replan"] == 128


def test_every_policy_is_reproducible(home, tmp_path):
    args = ("policy=[gflow,gflow-refine,cem,prior,prior-refine,random]", f"planner={home / 'mixture.pt'}",
            "refine_steps=3", "prior_samples=32")
    # the second run also checks that --deterministic changes nothing on CPU
    first, second = evaluate(home, tmp_path, *args), evaluate(home, tmp_path, *args, "deterministic=true")
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
    res = evaluate(home, tmp_path, "policy=[cem,random]", "cem_steps=2", task="pusht", dataset="toy_pusht")
    assert set(res) == {"cem", "random"}
    assert len(res["cem"]["episode_successes"]) == 4


def test_held_out_evaluation_uses_only_unseen_episodes(home, tmp_path):
    """With --holdout-frac, episodes come from the held-out side of the split; a planner whose
    recorded split does not hold them out is refused instead of being evaluated on its training data."""
    res = evaluate(home, tmp_path, "policy=[gflow,cem]", f"planner={home / 'gauss_split.pt'}",
                   "holdout_frac=0.5", "split_seed=0", full=True)
    dataset = swm.data.HDF5Dataset("toy_tworoom", cache_dir=home)
    _, held_out = split_episodes(dataset.get_col_data(episode_column(dataset.column_names)), 0.5, 0)
    assert {ep for ep, _ in res["episodes"]} <= set(held_out.tolist())
    assert res["planner_train_args"]["holdout_frac"] == 0.5 and set(res["policies"]) == {"gflow", "cem"}

    out = evaluate(home, tmp_path, "policy=[gflow]", f"planner={home / 'gauss.pt'}",
                   "holdout_frac=0.5", expect_failure=True)
    assert "were in the planner's training data" in out
    out = evaluate(home, tmp_path, "policy=[gflow]", f"planner={home / 'gauss_split.pt'}",
                   "holdout_frac=0.5", "split_seed=1", expect_failure=True)  # a different split
    assert "were in the planner's training data" in out


# ---------------------------------------------------------------------------- MuJoCo tasks (Reacher, Cube)

MUJOCO = {  # task: (toy dataset, env action dim)
    "reacher": ("dmc/toy_reacher", 2),
    "cube": ("ogbench/toy_cube", 5),
}


@pytest.fixture(scope="session")
def mujoco_home(tmp_path_factory):
    """Toy Reacher and Cube datasets recorded with the real envs (random actions, so the cube rarely
    moves), tiny LeWMs whose action encoders take 5 x 2 and 5 x 5 values, and untrained planners."""
    for module in ("mujoco", "dm_control", "ogbench"):
        pytest.importorskip(module)
    home = tmp_path_factory.mktemp("stablewm_home_mujoco")
    for name, (dataset, env_dim) in MUJOCO.items():
        task = TASKS[name]
        record(task.env, dataset, home, episodes=4, steps=40, **task.world_kwargs)
        (home / f"toy_{name}").mkdir()
        torch.save(make_wm(embed=EMBED, depth=1, heads=2, dim_head=8, mlp_dim=32, proj_hidden=16, action_dim=5 * env_dim),
                   home / f"toy_{name}" / "lewm_object.ckpt")
        torch.manual_seed(0)
        kwargs = dict(state_dim=EMBED, horizon=5 * 5 * env_dim, action_dim=5 * env_dim, n_components=2)
        save_planner(home / f"{name}.pt", GPlanner(**kwargs), kwargs, train_args={"beta": 90.0})
    return home


class Replay(swm.policy.BasePolicy):
    """Executes given env actions, one row per env and step (then zeros)."""

    def __init__(self, actions):
        super().__init__()
        self.actions, self.t = actions, 0  # (n_envs, steps, env action dim)

    def get_action(self, obs, **kwargs):
        a = self.actions[:, self.t] if self.t < self.actions.shape[1] else np.zeros_like(self.actions[:, 0])
        self.t += 1
        return a


def replay_success(home, name, dataset_name, actions_of, n=4):
    """Success of `Replay(actions_of(dataset actions from each start row))` under the task's protocol."""
    task = TASKS[name]
    dataset = swm.data.HDF5Dataset(dataset_name, keys_to_cache=list(task.cache_keys), cache_dir=home)
    episodes, starts = sample_eval_starts(dataset, n, 25, seed=0)
    ep_idx = dataset.get_col_data(episode_column(dataset.column_names))
    step_idx, actions = dataset.get_col_data("step_idx"), dataset.get_col_data("action")
    rows = [int(np.nonzero((ep_idx == e) & (step_idx == s))[0][0]) for e, s in zip(episodes, starts)]
    plans = actions_of(np.stack([actions[r:r + 25] for r in rows]))
    args = types.SimpleNamespace(env=task.env, num_eval=n, eval_budget=50, goal_offset=25, seed=0, video_dir=None)
    return run_protocol("replay", Replay(plans), args, task, dataset, episodes, starts)["episode_successes"]


def test_reacher_resets_to_the_dataset_state_and_goal(mujoco_home):
    """Replaying the recorded actions from the dataset's start state reaches the dataset's goal (the
    joint angles 25 steps later): every episode succeeds. Doing nothing reaches none of them."""
    assert replay_success(mujoco_home, "reacher", "dmc/toy_reacher", lambda a: a).all()
    assert not replay_success(mujoco_home, "reacher", "dmc/toy_reacher", np.zeros_like).any()


def test_cube_resets_to_the_dataset_state_and_goal(mujoco_home, tmp_path):
    """Random actions rarely move the cube, so in the toy data the goal pose is the start pose: doing
    nothing succeeds only if the env is reset to the dataset's cube pose AND given the dataset's goal.
    With the goal pose moved 10 cm in the data, doing nothing fails."""
    assert replay_success(mujoco_home, "cube", "ogbench/toy_cube", np.zeros_like).all()
    shutil.copytree(mujoco_home, tmp_path / "home")
    with h5py.File(tmp_path / "home" / "ogbench" / "toy_cube.h5", "r+") as f:
        f["privileged_block_0_pos"][:, 0] += 0.1
    assert not replay_success(tmp_path / "home", "cube", "ogbench/toy_cube", np.zeros_like).any()


@pytest.mark.parametrize("name", sorted(MUJOCO))
def test_evaluate_runs_on_mujoco_tasks(mujoco_home, tmp_path, name):
    """scripts/evaluate.py task=reacher / cube end to end: GFlow (an untrained planner of the right
    plan size), CEM and random, with every reset call applied (no swm "skipping callable" fallback)."""
    dataset, env_dim = MUJOCO[name]
    res = evaluate(mujoco_home, tmp_path, "policy=[gflow,cem,random]", f"planner={mujoco_home / f'{name}.pt'}",
                   "cem_samples=16", "cem_steps=2", "cem_topk=4", "num_eval=2",  # 2 envs: faster
                   task=name, dataset=dataset, wm=f"toy_{name}/lewm", full=True)
    assert set(res["policies"]) == {"gflow", "cem", "random"} and len(res["episodes"]) == 2
    for policy in ("gflow", "cem"):
        m = res["policies"][policy]
        assert np.isfinite(m["plan_cost"]) and m["replans"] == 2, policy  # budget 50 = 2 x 25 steps
