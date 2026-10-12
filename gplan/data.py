"""Dataset helpers shared by precompute, JAX training and evaluation (NumPy only, no torch)."""

import os
from pathlib import Path

import numpy as np

STABLEWM_HOME = Path(os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()


def episode_column(names):
    """Name of the episode-index column; datasets use either spelling."""
    return "episode_idx" if "episode_idx" in names else "ep_idx"


def valid_start_rows(ep_idx, step_idx, goal_offset):
    """Dataset rows with at least `goal_offset` steps left in their episode (so row + goal_offset is the goal).

    Assumes `step_idx` counts 0, 1, 2, ... within each episode and rows of an episode are contiguous.
    """
    ep_len = np.zeros(ep_idx.max() + 1, dtype=np.int64)
    np.maximum.at(ep_len, ep_idx, step_idx + 1)
    return np.nonzero(step_idx <= ep_len[ep_idx] - goal_offset - 1)[0]
