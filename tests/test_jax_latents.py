"""(start, goal) latent pairs: goals stay inside the start's episode, stale latents are rejected."""

import json

import numpy as np
import pytest

from gplan.jax.latents import expert_plans, LatentPairs, lewm_action_stats


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


def test_expert_plans_layout():
    """Step k of the plan holds env actions r + k*frameskip .. r + (k+1)*frameskip - 1, flattened in order."""
    actions = np.stack([np.arange(40.0), -np.arange(40.0)], axis=1)  # action[i] = (i, -i)
    plans = expert_plans(actions, np.array([3, 20]), goal_offset=10, n_steps=2, mean=0.0, std=1.0)
    assert plans.shape == (2, 2, 10)
    np.testing.assert_array_equal(plans[0, 0], [3, -3, 4, -4, 5, -5, 6, -6, 7, -7])
    np.testing.assert_array_equal(plans[0, 1], [8, -8, 9, -9, 10, -10, 11, -11, 12, -12])
    np.testing.assert_array_equal(plans[1, 1, :2], [25, -25])


def test_lewm_action_stats_drop_nan_rows_and_use_ddof_1():
    actions = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 0.0], [5.0, 9.0]])
    mean, std = lewm_action_stats(actions)
    np.testing.assert_allclose(mean, [3.0, 5.0])
    np.testing.assert_allclose(std, np.std([[1, 2], [3, 4], [5, 9]], axis=0, ddof=1))


def test_lewm_action_stats_reject_a_constant_action_dimension():
    """LeWM z-scores actions with these stds: a constant dimension would divide by zero."""
    actions = np.array([[1.0, 2.0, 0.5], [3.0, 2.0, 0.5], [np.nan, 0.0, 1.0], [5.0, 2.0, 0.5]])  # NaN row dropped
    with pytest.raises(ValueError, match=r"dimension\(s\) \[1, 2\] are constant"):
        lewm_action_stats(actions)


def test_expert_batches_hold_the_actions_between_start_and_goal(tmp_path):
    """Actions encode their row (a[r] = (r, -r)), so every plan can be traced back to its start row."""
    n_eps, ep_len, offset = 3, 30, 10
    n = n_eps * ep_len
    rows = np.arange(n, dtype=np.float32)
    np.save(tmp_path / "toy_latents.npy", rows[:, None].repeat(4, axis=1))
    np.savez(tmp_path / "toy_latents_meta.npz", ep_idx=np.repeat(np.arange(n_eps), ep_len),
             step_idx=np.tile(np.arange(ep_len), n_eps), action=np.stack([rows, -rows], axis=1))
    (tmp_path / "toy_latents.json").write_text(json.dumps({"ckpt": str(tmp_path / "x.ckpt")}))
    pairs = LatentPairs(tmp_path / "toy_latents", offset)
    mean, std = lewm_action_stats(pairs.actions)
    for z_start, z_goal, plans in pairs.expert_batches(5, 4, n_steps=2):
        start = z_start[:, 0].astype(int)
        assert plans.shape == (4, 2, 5 * 2)
        raw = plans.reshape(4, offset, 2) * std + mean  # back to env actions, one row per env step
        np.testing.assert_allclose(raw[:, :, 0], start[:, None] + np.arange(offset), atol=1e-3)
