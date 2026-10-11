"""(start, goal) latent pairs from embeddings precomputed by `scripts/encode_dataset.py`."""

import json
from pathlib import Path

import numpy as np

from gplan.data import steps_left, valid_start_rows


def goal_offset_schedule(goal_offset, max_offset, start=0, warmup=0):
    """k-warmup: the largest goal offset allowed at each training step.

    Stays at `goal_offset` until step `start`, then grows linearly to `max_offset` over `warmup`
    steps (warmup = 0: jumps to `max_offset` at `start`). Returns a function step -> int.
    """
    def k_max(step):
        if warmup <= 0:
            frac = float(step >= start)
        else:
            frac = min(max((step - start) / warmup, 0.0), 1.0)
        return int(round(goal_offset + frac * (max_offset - goal_offset)))
    return k_max


class LatentPairs:
    """Samples (z_start, z_goal = z of the frame k steps later in the same episode).

    By default k = `goal_offset` for every pair. `batches(..., max_offset=...)` makes k variable
    (see `sample_offsets`). Start rows always have at least `goal_offset` steps left, so the
    start distribution does not depend on k.

    Latents are memory-mapped, so a batch costs two small fancy-index reads.
    """

    def __init__(self, prefix, goal_offset, expected_ckpt=None):
        prefix = Path(prefix)
        info = json.loads(Path(f"{prefix}.json").read_text())
        if expected_ckpt is not None and Path(expected_ckpt).resolve() != Path(info["ckpt"]):
            raise ValueError(f"latents in {prefix}.npy were made with {info['ckpt']}, not {expected_ckpt}; "
                             f"re-run scripts/encode_dataset.py")
        self.latents = np.load(f"{prefix}.npy", mmap_mode="r")
        meta = np.load(f"{prefix}_meta.npz")
        self.goal_offset = goal_offset
        self.valid = valid_start_rows(meta["ep_idx"], meta["step_idx"], goal_offset)
        self.steps_left = steps_left(meta["ep_idx"], meta["step_idx"])  # per row
        self.actions = meta["action"]  # (N, env_action_dim), raw dataset actions (not normalized)

    @property
    def embed_dim(self):
        return self.latents.shape[1]

    def sample_rows(self, n, rng):
        """`n` distinct valid start rows, sorted."""
        return np.sort(rng.choice(self.valid, size=n, replace=False))

    def sample_offsets(self, rows, max_offset, fixed_frac, rng):
        """A goal offset k per start row.

        With probability `fixed_frac` k = goal_offset; otherwise k is uniform in
        [goal_offset, min(max_offset, steps left in the episode)], so the goal stays in the episode.
        """
        hi = np.minimum(max_offset, self.steps_left[rows])
        k = rng.integers(self.goal_offset, hi + 1)
        return np.where(rng.random(len(rows)) < fixed_frac, self.goal_offset, k)

    def pairs(self, rows, offsets=None):
        """(z_start, z_goal) for the given start rows, each (len(rows), D) float32.
        `offsets`: per-row goal offsets (default: `goal_offset` for all)."""
        offsets = self.goal_offset if offsets is None else offsets
        return (np.asarray(self.latents[rows], dtype=np.float32),
                np.asarray(self.latents[rows + offsets], dtype=np.float32))

    def batches(self, n_batches, batch_size, seed=0, max_offset=None, fixed_frac=0.5):
        """Yield `n_batches` of (z_start, z_goal), each (batch_size, D) float32.

        max_offset: None for a fixed k = goal_offset, an int, or a function step -> int
        (`goal_offset_schedule`). Until it exceeds goal_offset, the batches are identical to a
        fixed-k run with the same seed. fixed_frac: share of pairs kept at k = goal_offset.
        """
        rng = np.random.default_rng(seed)
        for step in range(n_batches):
            rows = self.sample_rows(batch_size, rng)
            k_max = max_offset(step) if callable(max_offset) else max_offset
            if k_max is None or k_max <= self.goal_offset:
                yield self.pairs(rows)
            else:
                yield self.pairs(rows, self.sample_offsets(rows, k_max, fixed_frac, rng))
