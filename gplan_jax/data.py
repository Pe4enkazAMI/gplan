"""(start, goal) latent pairs from embeddings precomputed by `scripts/encode_dataset.py`."""

import json
from pathlib import Path

import numpy as np

from gplan.data import valid_start_rows


class LatentPairs:
    """Samples (z_start, z_goal = z of the frame `goal_offset` steps later in the same episode).

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
        self.actions = meta["action"]  # (N, env_action_dim), raw dataset actions (not normalized)

    @property
    def embed_dim(self):
        return self.latents.shape[1]

    def sample_rows(self, n, rng):
        """`n` distinct valid start rows, sorted."""
        return np.sort(rng.choice(self.valid, size=n, replace=False))

    def pairs(self, rows):
        """(z_start, z_goal) for the given start rows, each (len(rows), D) float32."""
        return (np.asarray(self.latents[rows], dtype=np.float32),
                np.asarray(self.latents[rows + self.goal_offset], dtype=np.float32))

    def batches(self, n_batches, batch_size, seed=0):
        """Yield `n_batches` of (z_start, z_goal), each (batch_size, D) float32."""
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            yield self.pairs(self.sample_rows(batch_size, rng))
