"""(start, goal) latent pairs from embeddings precomputed by `scripts/encode_dataset.py`, and the
dataset's expert actions between them as plans (for behavior cloning and the headroom probe)."""

import json
from pathlib import Path

import numpy as np

from gplan.data import valid_start_rows


def lewm_action_stats(actions):
    """Per-dimension mean and std of the raw dataset actions, as LeWM's training normalizer computes
    them (rows with NaN dropped, unbiased std; le-wm/utils.py).

    Evaluation normalizes with ddof=0 instead (`gplan.torch.evaluation.Standardizer`, as le-wm/eval.py's
    StandardScaler does). The stds differ by a factor sqrt(N / (N - 1)), i.e. ~1 / (2N) relative:
    negligible for datasets of thousands of frames, so plans mean the same thing on both sides."""
    a = actions[~np.isnan(actions).any(axis=1)]
    return a.mean(0), a.std(0, ddof=1)


def expert_plans(actions, rows, goal_offset, n_steps, mean, std):
    """Dataset actions from each start row to its goal row as plans (B, n_steps, frameskip * env_dim).

    Rows r .. r + goal_offset - 1 hold the actions taken between frame r and frame r + goal_offset;
    consecutive groups of `frameskip` env actions form one plan step (the layout swm's
    WorldModelPolicy unpacks). Assumes action[r] is the action taken at frame r.
    """
    assert goal_offset % n_steps == 0, f"goal_offset {goal_offset} must be a multiple of n_steps {n_steps}"
    seg = actions[rows[:, None] + np.arange(goal_offset)]           # (B, goal_offset, env_dim)
    seg = (seg - mean) / std
    return seg.reshape(len(rows), n_steps, -1).astype(np.float32)


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
        self.ep_idx = meta["ep_idx"]
        self.valid = valid_start_rows(meta["ep_idx"], meta["step_idx"], goal_offset)
        self.actions = meta["action"]  # (N, env_action_dim), raw dataset actions (not normalized)

    def keep_episodes(self, episodes):
        """Draw pairs only from these episodes (e.g. the training side of `gplan.data.split_episodes`).
        Start and goal of a pair are always in the same episode."""
        self.valid = self.valid[np.isin(self.ep_idx[self.valid], episodes)]
        assert len(self.valid) > 0, "no valid start rows left in the kept episodes"

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

    def expert_batches(self, n_batches, batch_size, n_steps, seed=0):
        """Yield `n_batches` of (z_start, z_goal, expert plan): the dataset's actions between start and
        goal, z-scored with LeWM's action statistics, as (batch_size, n_steps, goal_offset / n_steps *
        env action dim) plans. Same row sampling as `batches`."""
        mean, std = lewm_action_stats(self.actions)
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            rows = self.sample_rows(batch_size, rng)
            yield (*self.pairs(rows), expert_plans(self.actions, rows, self.goal_offset, n_steps, mean, std))
