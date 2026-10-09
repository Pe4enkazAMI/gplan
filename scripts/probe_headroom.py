"""Headroom probe: how much lower can the LeWM cost get near the policy's own samples?

Measures, on (start, goal) latent pairs, the LeWM cost J of best-of-N policy samples before and after
a few gradient steps through the frozen LeWM, next to prior shooting and the dataset's expert actions.
No env is involved; refinement is a measurement, not part of the planner. See gplan_jax/probe.py.

    python scripts/probe_headroom.py outputs/tb-beta90-warm1k-20k
    python scripts/probe_headroom.py outputs/pusht-run --n-pairs 4096 --steps 50 --objective cost

Reading the result:
    big   sampled -> refined drop  : better plans are reachable from the policy's samples;
                                     training-time local search / replay should help
    small sampled -> refined drop  : the policy already sits near the low-cost region
    refined << expert              : refinement may be exploiting LeWM (check max|a| and env success)
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gplan.data import STABLEWM_HOME
from gplan_jax import checkpoint
from gplan_jax.convert import load_lewm_predictor
from gplan_jax.data import LatentPairs
from gplan_jax.probe import expert_plans, lewm_action_stats, make_probe


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("planner", type=Path, help="checkpoint prefix from scripts/train.py (<prefix>.eqx + .json)")
    p.add_argument("--ckpt", type=Path, default=None, help="LeWM *_object.ckpt (default: the one used in training)")
    p.add_argument("--latents", type=Path, default=None, help="latents prefix (default: the training dataset's)")
    p.add_argument("--goal-offset", type=int, default=None, help="default: as in training")
    p.add_argument("--beta", type=float, default=None, help="per-dim beta of the target objective (default: training)")
    p.add_argument("--n-pairs", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=64, help="pairs per jitted call (memory: chunk * n_samples plans)")
    p.add_argument("--n-samples", type=int, default=64, help="N in best-of-N")
    p.add_argument("--steps", type=int, default=20, help="refinement steps (Adam on the actions)")
    p.add_argument("--lr", type=float, default=0.05, help="Adam step size, in z-scored action units")
    p.add_argument("--objective", default="target", choices=["target", "cost"],
                   help="refine beta*J - log N(A) (stays near the prior) or J alone")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "default"])
    p.add_argument("--out", type=Path, default=None, help="save per-pair results to this .npz")
    return p.parse_args()


def summarize(res, n_samples):
    rows = [("expert (dataset actions)", "expert"), (f"prior, best of {n_samples}", "prior"),
            ("policy, mean over samples", "sampled_mean"), (f"policy, best of {n_samples}", "sampled"),
            (f"policy, best of {n_samples} + refine", "refined")]
    print(f"\n{'plans':34s} {'J median':>10s} {'J mean':>10s} {'max|a| median':>14s}")
    for label, key in rows:
        absmax = res.get(f"{key}_absmax")
        a = f"{np.median(absmax):14.2f}" if absmax is not None else f"{'':14s}"
        print(f"{label:34s} {np.median(res[key + '_cost']):10.2f} {np.mean(res[key + '_cost']):10.2f} {a}")

    drop = 1 - res["refined_cost"] / res["sampled_cost"]
    print(f"\nrelative drop best-of-{n_samples} -> refined: median {np.median(drop):.1%}, "
          f"pairs with > 10%: {np.mean(drop > 0.10):.1%}, > 25%: {np.mean(drop > 0.25):.1%}")
    print(f"pairs where best-of-{n_samples} beats the expert plan: {np.mean(res['sampled_cost'] < res['expert_cost']):.1%}"
          f"   (after refinement: {np.mean(res['refined_cost'] < res['expert_cost']):.1%})")


def main():
    args = parse_args()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    planner, _, train_args = checkpoint.load(args.planner)
    train_args = train_args or {}

    ckpt = args.ckpt or Path(train_args["ckpt"])
    latents = args.latents or (Path(train_args["latents"]) if train_args.get("latents")
                               else STABLEWM_HOME / f"{train_args['dataset']}_latents")
    goal_offset = args.goal_offset or train_args.get("goal_offset", 25)
    plan_shape = (train_args.get("n_steps", 5), train_args.get("action_dim", 10))
    reference_var = train_args.get("ref_var", 1.0)

    cost_model = load_lewm_predictor(ckpt)  # torch is used only here, on CPU
    pairs = LatentPairs(latents, goal_offset, expected_ckpt=ckpt)
    beta = (args.beta if args.beta is not None else train_args["beta"]) / pairs.embed_dim

    mean, std = lewm_action_stats(pairs.actions)
    rng = np.random.default_rng(args.seed)
    rows = pairs.sample_rows(args.n_pairs, rng)
    probe = make_probe(planner, cost_model, plan_shape, args.n_samples, beta, reference_var,
                       args.steps, args.lr, args.objective)
    print(f"{args.n_pairs} pairs, best of {args.n_samples}, refine: {args.steps} Adam steps (lr {args.lr}) "
          f"on the {args.objective} objective, beta per dim {beta * pairs.embed_dim:g} (scaled {beta:.4f})")

    key = jax.random.key(args.seed)
    chunks = []
    for i in range(0, len(rows), args.chunk):
        r = rows[i:i + args.chunk]
        z_start, z_goal = pairs.pairs(r)
        expert = expert_plans(pairs.actions, r, goal_offset, plan_shape[0], mean, std)
        key, sub = jax.random.split(key)
        chunks.append(jax.device_get(probe(sub, jnp.asarray(z_start), jnp.asarray(z_goal), jnp.asarray(expert))))
        print(f"probed {min(i + args.chunk, len(rows))}/{len(rows)} pairs", end="\r")
    res = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    summarize(res, args.n_samples)
    if args.out:
        np.savez(args.out, rows=rows, **res)
        print(f"\nsaved per-pair results to {args.out}")


if __name__ == "__main__":
    main()
