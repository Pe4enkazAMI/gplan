"""Evaluation protocol: task presets and LeWM-identical start sampling (needs the `eval` extra)."""

import numpy as np
import pytest

pytest.importorskip("stable_worldmodel")
from gplan.evaluation import TASKS, Standardizer, sample_eval_starts  # noqa: E402


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


def test_task_presets_match_lewm_configs():
    """Values from le-wm/config/eval/{tworoom,pusht}.yaml."""
    assert TASKS["tworoom"].env == "swm/TwoRoom-v1"
    assert (TASKS["tworoom"].state_key, TASKS["tworoom"].goal_state_key) == ("proprio", "goal_proprio")
    assert TASKS["pusht"].env == "swm/PushT-v1" and TASKS["pusht"].dataset == "pusht_expert_train"
    assert (TASKS["pusht"].state_key, TASKS["pusht"].goal_state_key) == ("state", "goal_state")
    assert set(TASKS["pusht"].cache_keys) == {"action", "proprio", "state"}


def test_standardizer_matches_sklearn_conventions():
    data = np.array([[1.0, 5.0], [3.0, 5.0], [np.nan, 0.0], [2.0, 5.0]])
    s = Standardizer(data)
    np.testing.assert_allclose(s.mean, [2.0, 5.0])
    np.testing.assert_allclose(s.std, [np.std([1.0, 3.0, 2.0]), 1.0])  # ddof=0; constant column -> 1
    x = np.array([[2.5, 7.0]])
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)
