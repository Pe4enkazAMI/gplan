"""Precompute frozen LeWM embeddings for a dataset (run once per dataset / LeWM checkpoint).

    python scripts/encode_dataset.py --dataset tworoom
    -> $STABLEWM_HOME/tworoom_latents.npy, tworoom_latents_meta.npz, tworoom_latents.json

    python scripts/encode_dataset.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt
"""

import argparse
from pathlib import Path

import torch

from gplan.lewm import STABLEWM_HOME, load_lewm
from gplan.precompute import cache_info, encode_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="tworoom", help=".h5 name under $STABLEWM_HOME")
    p.add_argument("--ckpt", type=Path, default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--out", type=Path, default=None, help="output prefix (default: $STABLEWM_HOME/<dataset>_latents)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    h5_path = STABLEWM_HOME / f"{args.dataset}.h5"
    out = args.out or STABLEWM_HOME / f"{args.dataset}_latents"
    wm = load_lewm(args.ckpt, args.device)
    info = cache_info(args.ckpt, h5_path, args.img_size)
    encode_dataset(wm, h5_path, out, info, args.batch_size, args.img_size, args.dtype, args.device)
    print(f"wrote {out}.npy, {out}_meta.npz, {out}.json")


if __name__ == "__main__":
    main()
