"""Shared dataset helpers: rows left in each episode and valid start rows for a goal offset."""

import numpy as np

from gplan.data import episode_column, steps_left, valid_start_rows

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
