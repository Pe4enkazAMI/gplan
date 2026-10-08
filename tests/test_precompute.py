import json

import h5py
import numpy as np
import torch
from conftest import IMG

from gplan.lewm import encode, img_transform
from gplan.precompute import encode_dataset


def test_encode_dataset_matches_per_frame_encoding(wm, tmp_path):
    """Batched precompute == encoding each frame with the training/eval transform; metadata is copied."""
    rng = np.random.default_rng(0)
    n = 10
    h5_path = tmp_path / "toy.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["pixels"] = rng.integers(0, 256, size=(n, IMG, IMG, 3), dtype=np.uint8)
        h5["ep_idx"] = np.repeat([0, 1], 5)
        h5["step_idx"] = np.tile(np.arange(5), 2)
        h5["action"] = rng.normal(size=(n, 2)).astype(np.float32)

    out = encode_dataset(wm, h5_path, tmp_path / "toy_latents", {"note": "test"}, batch_size=4, img_size=IMG)

    latents = np.load(f"{out}.npy")
    with h5py.File(h5_path, "r") as h5:
        frames = h5["pixels"][:]
        expected = encode(wm, torch.stack([img_transform(IMG)(f) for f in frames])).numpy()
        meta = np.load(f"{out}_meta.npz")
        assert np.array_equal(meta["ep_idx"], h5["ep_idx"][:])
        assert np.array_equal(meta["action"], h5["action"][:])
    assert latents.shape == expected.shape
    np.testing.assert_allclose(latents, expected, atol=1e-5)
    assert json.loads(open(f"{out}.json").read())["n_frames"] == n
