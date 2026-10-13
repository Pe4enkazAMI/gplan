"""Dataset helpers shared by precompute, JAX training and evaluation (NumPy only, no torch)."""

import os
from pathlib import Path

import numpy as np

STABLEWM_HOME = Path(os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()


def episode_column(names):
    """Name of the episode-index column; datasets use either spelling."""
    return "episode_idx" if "episode_idx" in names else "ep_idx"


def steps_left(ep_idx, step_idx):
    """For every dataset row, how many rows of its episode come after it (0 for the last frame).

    Assumes `step_idx` counts 0, 1, 2, ... within each episode and rows of an episode are contiguous.
    """
    ep_len = np.zeros(ep_idx.max() + 1, dtype=np.int64)
    np.maximum.at(ep_len, ep_idx, step_idx + 1)
    return ep_len[ep_idx] - step_idx - 1


def valid_start_rows(ep_idx, step_idx, goal_offset):
    """Dataset rows with at least `goal_offset` steps left in their episode (so row + goal_offset is the goal)."""
    return np.nonzero(steps_left(ep_idx, step_idx) >= goal_offset)[0]


def split_episodes(ep_idx, holdout_frac, seed=0):
    """Episode-level train / held-out split: (train episode ids, held-out episode ids), both sorted.

    round(holdout_frac * n_episodes) episodes (at least one if holdout_frac > 0), chosen by a
    permutation seeded with `seed`, are held out. The split depends only on the set of episode ids,
    so training (latents' metadata) and evaluation (the .h5 dataset) compute the same one.
    """
    episodes = np.unique(ep_idx)
    if holdout_frac <= 0:
        return episodes, episodes[:0]
    n_held = max(1, int(round(holdout_frac * len(episodes))))
    assert n_held < len(episodes), f"holdout_frac {holdout_frac} leaves no training episodes"
    held = np.sort(episodes[np.random.default_rng(seed).permutation(len(episodes))[:n_held]])
    return np.setdiff1d(episodes, held), held
