"""Precompute frozen LeWM embeddings for every frame of a dataset.

The LeWM encoder is frozen, so each frame's embedding is a constant. Encoding the
whole dataset once lets training run on latents only (no images, no ViT).

Output, for `out_prefix = $STABLEWM_HOME/tworoom_latents`:
    tworoom_latents.npy        (N, D) embeddings, row i = dataset row i
    tworoom_latents_meta.npz   ep_idx (N,), step_idx (N,), action (N, A) raw (not normalized)
    tworoom_latents.json       where the latents came from, to detect a stale cache
"""

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the compression filter used by the LeWM .h5 files
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

from gplan.data import episode_column
from gplan.lewm import IMAGENET, encode


def batch_transform(img_size=224):
    """Same preprocessing as `gplan.lewm.img_transform`, applied to a (B, C, H, W) uint8 batch."""
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET),
        transforms.Resize(size=img_size),
    ])


def cache_info(ckpt_path, h5_path, img_size):
    """Everything that determines the latents; stored next to them and checked when loading."""
    ckpt = Path(ckpt_path).stat()
    return {"ckpt": str(Path(ckpt_path).resolve()), "ckpt_bytes": ckpt.st_size, "ckpt_mtime": ckpt.st_mtime,
            "dataset": str(Path(h5_path).resolve()), "img_size": img_size}


@torch.no_grad()
def encode_dataset(wm, h5_path, out_prefix, info, batch_size=256, img_size=224, dtype="float32", device="cpu"):
    """Encode every frame of `h5_path` with the frozen LeWM `wm` and write the three output files."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    transform = batch_transform(img_size)

    with h5py.File(h5_path, "r") as h5:
        pixels = h5["pixels"]  # (N, H, W, C) uint8
        n = pixels.shape[0]
        latents = None
        for start in range(0, n, batch_size):
            chunk = torch.from_numpy(pixels[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
            z = encode(wm, transform(chunk)).float().cpu().numpy()
            if latents is None:  # allocate once the embedding dim is known
                latents = np.lib.format.open_memmap(f"{out_prefix}.npy", mode="w+", dtype=dtype, shape=(n, z.shape[1]))
            latents[start:start + len(z)] = z
            print(f"encoded {min(start + batch_size, n)}/{n} frames", end="\r")
        latents.flush()
        print()

        np.savez(f"{out_prefix}_meta.npz", ep_idx=h5[episode_column(h5)][:], step_idx=h5["step_idx"][:],
                 action=h5["action"][:])

    Path(f"{out_prefix}.json").write_text(json.dumps({**info, "n_frames": n, "embed_dim": latents.shape[1],
                                                      "dtype": dtype}, indent=2))
    return out_prefix
