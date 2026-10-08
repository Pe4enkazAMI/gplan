"""Export a JAX planner checkpoint to the PyTorch .pt that scripts/evaluate.py loads.

scripts/train.py already writes <prefix>.pt next to <prefix>.eqx; use this for checkpoints
saved without it or to write the .pt somewhere else.

    python scripts/export_planner.py outputs/tb-beta120 --out outputs/tb-beta120.pt
"""

import argparse
from pathlib import Path

from gplan_jax import checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prefix", type=Path, help="checkpoint prefix (<prefix>.eqx + <prefix>.json)")
    p.add_argument("--out", type=Path, default=None, help="output .pt (default: <prefix>.pt)")
    args = p.parse_args()

    model, model_kwargs, train_args = checkpoint.load(args.prefix)
    out = args.out or Path(f"{args.prefix}.pt")
    checkpoint.export_torch(out, model, model_kwargs, train_args)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
