"""(start, goal) frame pairs from the LeWM .h5 datasets."""

import h5py
import hdf5plugin  # noqa: F401  registers the compression filter used by the LeWM .h5 files
import numpy as np
import torch

from gplan.lewm import STABLEWM_HOME, img_transform


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


def make_batches(n_batches, batch_size, img_size=224, dataset="tworoom", goal_offset=25, seed=0):
    """Yield (start_pixels, goal_pixels) batches of shape (B, 3, img_size, img_size) from a LeWM .h5 dataset.

    The start is a random dataset frame and the goal is the frame `goal_offset`
    steps later in the same episode, matching the evaluation protocol.
    """
    transform = img_transform(img_size)
    rng = np.random.default_rng(seed)
    with h5py.File(STABLEWM_HOME / f"{dataset}.h5", "r") as h5:
        valid = valid_start_rows(h5[episode_column(h5)][:], h5["step_idx"][:], goal_offset)
        for _ in range(n_batches):
            rows = np.sort(rng.choice(valid, size=batch_size, replace=False))  # h5 needs sorted indices
            start = torch.stack([transform(f) for f in h5["pixels"][rows]])
            goal = torch.stack([transform(f) for f in h5["pixels"][rows + goal_offset]])
            yield start, goal
