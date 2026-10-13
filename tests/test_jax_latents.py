"""(start, goal) latent pairs: goals stay inside the start's episode, stale latents are rejected."""

import json

import numpy as np
import pytest

from gplan.jax.latents import LatentPairs


def test_latent_pairs_stay_within_episode(tmp_path):
    n, offset = 12, 3
    latents = np.arange(n, dtype=np.float32)[:, None].repeat(4, axis=1)  # row i has value i
    np.save(tmp_path / "toy_latents.npy", latents)
    np.savez(tmp_path / "toy_latents_meta.npz", ep_idx=np.repeat([0, 1], 6), step_idx=np.tile(np.arange(6), 2),
             action=np.zeros((n, 2)))
    (tmp_path / "toy_latents.json").write_text(json.dumps({"ckpt": str(tmp_path / "x.ckpt")}))

    pairs = LatentPairs(tmp_path / "toy_latents", offset)
    for z_start, z_goal in pairs.batches(5, 4):
        start_rows = z_start[:, 0].astype(int)
        np.testing.assert_array_equal(z_goal[:, 0], start_rows + offset)
        assert np.all(start_rows % 6 <= 6 - offset - 1)  # goal is inside the same episode
    with pytest.raises(ValueError):
        LatentPairs(tmp_path / "toy_latents", offset, expected_ckpt=tmp_path / "other.ckpt")


def test_keep_episodes_restricts_starts_and_goals(tmp_path):
    n_eps, ep_len, offset = 6, 10, 3
    n = n_eps * ep_len
    np.save(tmp_path / "toy_latents.npy", np.arange(n, dtype=np.float32)[:, None].repeat(4, axis=1))
    np.savez(tmp_path / "toy_latents_meta.npz", ep_idx=np.repeat(np.arange(n_eps), ep_len),
             step_idx=np.tile(np.arange(ep_len), n_eps), action=np.zeros((n, 2)))
    (tmp_path / "toy_latents.json").write_text(json.dumps({"ckpt": str(tmp_path / "x.ckpt")}))
    pairs = LatentPairs(tmp_path / "toy_latents", offset)
    pairs.keep_episodes([1, 4])
    for z_start, z_goal in pairs.batches(20, 4):
        assert set(z_start[:, 0].astype(int) // ep_len) <= {1, 4}
        assert set(z_goal[:, 0].astype(int) // ep_len) <= {1, 4}
