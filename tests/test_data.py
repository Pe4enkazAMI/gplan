"""Shared dataset helpers: rows left in each episode and valid start rows for a goal offset."""

import numpy as np

from gplan.data import episode_column, split_episodes, steps_left, valid_start_rows

# three episodes of lengths 4, 1 and 3, rows contiguous
EP = np.array([0, 0, 0, 0, 1, 2, 2, 2])
STEP = np.array([0, 1, 2, 3, 0, 0, 1, 2])


def test_steps_left_counts_the_rows_after_each_frame():
    np.testing.assert_array_equal(steps_left(EP, STEP), [3, 2, 1, 0, 0, 2, 1, 0])


def test_valid_start_rows_keep_the_goal_inside_the_episode():
    np.testing.assert_array_equal(valid_start_rows(EP, STEP, 0), np.arange(8))
    np.testing.assert_array_equal(valid_start_rows(EP, STEP, 2), [0, 1, 5])
    np.testing.assert_array_equal(valid_start_rows(EP, STEP, 3), [0])  # offset = episode length - 1
    assert len(valid_start_rows(EP, STEP, 4)) == 0
    for offset in range(5):  # every goal row is in its start row's episode
        rows = valid_start_rows(EP, STEP, offset)
        assert np.all(EP[rows + offset] == EP[rows])


def test_episode_column_accepts_both_spellings():
    assert episode_column(["pixels", "episode_idx", "step_idx"]) == "episode_idx"
    assert episode_column(["pixels", "ep_idx", "step_idx"]) == "ep_idx"


def test_episode_split_is_a_deterministic_partition():
    ep_idx = np.repeat(np.arange(20), 7)
    train, held = split_episodes(ep_idx, 0.25, seed=3)
    assert len(held) == 5 and len(train) == 15
    assert set(train) | set(held) == set(range(20)) and not set(train) & set(held)
    np.testing.assert_array_equal(held, split_episodes(ep_idx[::-1], 0.25, seed=3)[1])  # row order is irrelevant
    assert not np.array_equal(held, split_episodes(ep_idx, 0.25, seed=4)[1])
    assert len(split_episodes(ep_idx, 0.01, seed=0)[1]) == 1  # at least one episode once frac > 0
    train, held = split_episodes(ep_idx, 0.0)
    assert len(held) == 0 and len(train) == 20
