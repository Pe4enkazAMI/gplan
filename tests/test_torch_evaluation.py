"""Evaluation protocol: task presets and LeWM-identical start sampling (needs the `eval` extra)."""

import types

import numpy as np
import pytest
from conftest import load_script

from gplan.data import split_episodes

pytest.importorskip("stable_worldmodel")
from gplan.torch import evaluation  # noqa: E402
from gplan.torch.evaluation import (  # noqa: E402
    check_planner_matches,
    sample_eval_starts,
    Standardizer,
    TASKS,
    trained_on,
)
from gplan.torch.lewm import LEWM_DIR  # noqa: E402


def evaluate_script():
    """scripts/evaluate.py as a module (its argument parser holds the protocol defaults)."""
    return load_script("evaluate")


class FakeDataset:
    """The two methods of swm.data.HDF5Dataset the sampler uses."""

    def __init__(self, ep_idx, step_idx):
        self.cols = {"ep_idx": np.asarray(ep_idx), "step_idx": np.asarray(step_idx)}
        self.column_names = list(self.cols)

    def get_col_data(self, col):
        return self.cols[col]

    def get_row_data(self, rows):
        return {k: v[rows] for k, v in self.cols.items()}


def lewm_eval_starts(dataset, num_eval, goal_offset, seed):
    """The start sampling of le-wm/eval.py, copied as written there."""
    col_name = "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_idx, step_idx = dataset.get_col_data(col_name), dataset.get_col_data("step_idx")
    episode_len = np.array([np.max(step_idx[episode_idx == e]) + 1 for e in ep_indices])
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_indices = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    rows = dataset.get_row_data(random_episode_indices)
    return rows[col_name].tolist(), rows["step_idx"].tolist()


@pytest.mark.parametrize("seed", [0, 42])
def test_eval_starts_match_lewm(seed):
    lengths = [40, 80, 33, 120, 60]
    ds = FakeDataset(np.repeat(np.arange(len(lengths)), lengths), np.concatenate([np.arange(n) for n in lengths]))
    assert sample_eval_starts(ds, 50, 25, seed) == lewm_eval_starts(ds, 50, 25, seed)


@pytest.mark.parametrize("name", ["tworoom", "pusht", "reacher", "cube"])
def test_task_presets_match_lewm_configs(name):
    """Every preset is what le-wm/config/eval/<task>.yaml says: env and its settings, dataset, cached
    columns, the reset calls, and the protocol defaults of scripts/evaluate.py."""
    yaml = pytest.importorskip("yaml")
    path = LEWM_DIR / "config" / "eval" / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not checked out (git submodule update --init le-wm)")
    cfg = yaml.safe_load(path.read_text())
    task = TASKS[name]
    assert task.env == cfg["world"]["env_name"] and task.dataset == cfg["eval"]["dataset_name"]
    assert task.world_kwargs == {k: v for k, v in cfg["world"].items()
                                 if k not in ("env_name", "num_envs", "max_episode_steps")}
    assert set(task.cache_keys) == set(cfg["dataset"]["keys_to_cache"])
    assert list(task.resets) == cfg["eval"]["callables"]
    assert task.wm == f"{name}/lewm"

    defaults = vars(evaluate_script().parse_args([])[0])  # scripts/evaluate.py with no flags
    assert cfg["plan_config"] == {"horizon": defaults["horizon"], "receding_horizon": defaults["horizon"],
                                  "action_block": defaults["action_block"]}
    assert (cfg["eval"]["num_eval"], cfg["eval"]["goal_offset_steps"], cfg["eval"]["eval_budget"],
            cfg["eval"]["img_size"], cfg["seed"]) == tuple(
        defaults[k] for k in ("num_eval", "goal_offset", "eval_budget", "img_size", "seed"))


def test_standardizer_matches_sklearn_conventions():
    data = np.array([[1.0, 5.0], [3.0, 5.0], [np.nan, 0.0], [2.0, 5.0]])
    s = Standardizer(data)
    np.testing.assert_allclose(s.mean, [2.0, 5.0])
    np.testing.assert_allclose(s.std, [np.std([1.0, 3.0, 2.0]), 1.0])  # ddof=0; constant column -> 1
    x = np.array([[2.5, 7.0]])
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)


def test_eval_starts_can_be_restricted_to_held_out_episodes():
    lengths = [40, 80, 33, 120, 60, 70, 90, 50]
    ds = FakeDataset(np.repeat(np.arange(len(lengths)), lengths), np.concatenate([np.arange(n) for n in lengths]))
    eps, steps = sample_eval_starts(ds, 20, 25, seed=0, episodes=[1, 6])
    assert set(eps) <= {1, 6} and len(eps) == 20
    with pytest.raises(ValueError):
        sample_eval_starts(ds, 200, 25, seed=0, episodes=[2])  # episode 2 has 33 - 25 = 8 valid starts


def test_trained_on_follows_the_planners_recorded_split():
    lengths = [30] * 10
    ds = FakeDataset(np.repeat(np.arange(10), lengths), np.concatenate([np.arange(n) for n in lengths]))
    assert trained_on(ds, [0, 5], None) == [0, 5]  # no split recorded: trained on everything
    train_eps, held = split_episodes(ds.get_col_data("ep_idx"), 0.3, 1)
    assert trained_on(ds, held.tolist(), {"holdout_frac": 0.3, "split_seed": 1}) == []
    assert trained_on(ds, [int(train_eps[0])], {"holdout_frac": 0.3, "split_seed": 1}) == [int(train_eps[0])]


def test_planner_trained_for_another_protocol_is_an_error_unless_allowed(monkeypatch, tmp_path, capsys):
    """Dataset, LeWM checkpoint, goal offset, plan steps or frameskip differing from the evaluation's stop
    it; --allow-mismatch turns the errors into warnings. Keys the planner did not record are not compared."""
    monkeypatch.setattr(evaluation, "STABLEWM_HOME", tmp_path)
    args, _ = evaluate_script().parse_args([])  # TwoRoom: dataset tworoom, wm tworoom/lewm, goal offset 25, 5 steps, frameskip 5
    planner = types.SimpleNamespace(horizon=5 * 5 * 2)  # 5 steps x frameskip 5 x 2-d env actions
    matching = {"dataset": "tworoom", "ckpt": str(tmp_path / "tworoom" / "lewm_object.ckpt"), "goal_offset": 25,
                "n_steps": 5, "frameskip": 5, "beta": 90.0}
    for train_args in (matching, {"beta": 90.0}, None):
        check_planner_matches(planner, train_args, args, env_action_dim=2)
    assert "WARNING" not in capsys.readouterr().out
    for key, value in [("dataset", "pusht_expert_train"), ("ckpt", str(tmp_path / "pusht" / "lewm_object.ckpt")),
                       ("goal_offset", 24), ("n_steps", 4), ("frameskip", 1)]:
        with pytest.raises(SystemExit, match="ERROR: planner was trained .*--allow-mismatch"):
            check_planner_matches(planner, {**matching, key: value}, args, env_action_dim=2)
    allowed, _ = evaluate_script().parse_args(["--allow-mismatch"])
    check_planner_matches(planner, {**matching, "goal_offset": 24, "dataset": "x"}, allowed, env_action_dim=2)
    out = capsys.readouterr().out
    assert "WARNING: planner was trained with goal_offset=24 but eval uses 25" in out
    assert "WARNING: planner was trained with dataset=x but eval uses tworoom" in out


def test_refine_settings_default_to_the_planners_training_values():
    """beta (per embedding dim) and s^2 come from the planner's train_args unless --refine-beta is given;
    without a planner, s^2 is --prior-var and the target objective needs --refine-beta."""
    args, _ = evaluate_script().parse_args(["--prior-var", "2.0"])
    planner_args = {"beta": 96.0, "ref_var": 0.5}
    assert evaluation.refine_settings(args, planner_args, d=192) == dict(
        steps=20, lr=0.05, objective="target", beta=0.5, reference_var=0.5)
    given, _ = evaluate_script().parse_args(["--refine-beta", "48", "--refine-steps", "5"])
    assert evaluation.refine_settings(given, planner_args, d=192)["beta"] == 0.25
    assert evaluation.refine_settings(given, None, d=192)["reference_var"] == 1.0  # --prior-var default
    with pytest.raises(ValueError, match="--refine-beta is needed"):
        evaluation.refine_settings(args, None, d=192)
    cost, _ = evaluate_script().parse_args(["--refine-objective", "cost"])
    assert evaluation.refine_settings(cost, None, d=192)["beta"] == 0.0  # J alone: beta unused
