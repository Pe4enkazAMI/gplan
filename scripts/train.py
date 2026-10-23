"""Train the GPlanner sampler (JAX) as a conditional GFlowNet over LeWM action plans (--method gfn),
or the same network as a baseline: behavior cloning on the dataset's expert actions (--method bc),
amortized MAP planning through LeWM (--method map; evaluate with --gflow-select mean) or PPO with a
learned critic (--method ppo, gplan/jax/ppo.py).

Needs precomputed embeddings (scripts/encode_dataset.py). Writes outputs/<name>.{eqx,json,pt};
the .pt is what scripts/evaluate.py loads.

    python scripts/train.py --beta 120 --wandb-name tb-beta120 --out outputs/tb-beta120
    python scripts/train.py --n-components 4             # EXP: mixture of 4 Gaussians per plan step
    python scripts/train.py --ls-steps 10                 # EXP: + 10 MALA steps per plan as off-policy data
    python scripts/train.py --method bc                   # baseline: behavior cloning on expert actions
    python scripts/train.py --method map --beta 120       # baseline: one plan per condition, min beta J - log N
    python scripts/train.py --method ppo --beta 120       # baseline: PPO maximizing -beta J + log N (classic)
    python scripts/train.py --method ppo --ent-coef 1     # PPO whose optimum is the GFlowNet target (max-ent)

See gplan/jax/losses.py for the target distribution and the loss.
"""

import argparse
import os
from functools import partial
from pathlib import Path

import jax
import wandb

from gplan.data import split_episodes, STABLEWM_HOME
from gplan.jax import checkpoint
from gplan.jax.baselines import bc_loss, map_loss, train_baseline
from gplan.jax.convert import load_lewm_predictor
from gplan.jax.latents import LatentPairs
from gplan.jax.policy import GPlanner
from gplan.jax.ppo import Critic, critic_key, PPOConfig, train_ppo
from gplan.jax.train import init_key, linear_beta_schedule, train, TrainConfig


def plan_step_dim(frameskip, env_action_dim, cost_model):
    """Values per plan step: frameskip x the env's action dim (TwoRoom, PushT, Reacher: 5 x 2 = 10;
    Cube: 5 x 5 = 25), checked against the input of LeWM's action encoder so a dataset / checkpoint
    mismatch fails here instead of training on a world model that reads the actions differently."""
    action_dim = frameskip * env_action_dim
    lewm_dim = cost_model.action_encoder.patch_embed.in_features
    if action_dim != lewm_dim:
        raise SystemExit(f"a plan step is --frameskip {frameskip} x {env_action_dim}-d dataset actions = {action_dim} "
                         f"values, but the LeWM checkpoint's action encoder takes {lewm_dim}: wrong --dataset / "
                         f"--ckpt pair or --frameskip?")
    return action_dim


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="gfn", choices=["gfn", "bc", "map", "ppo"],
                   help="gfn: the GFlowNet; bc: behavior cloning on expert plans; map: amortized MAP planning "
                        "(gplan/jax/baselines.py); ppo: PPO with a critic (gplan/jax/ppo.py, flags [ppo]). bc / map / "
                        "ppo ignore the GFlowNet-only flags (MALA, beta warmup)")
    p.add_argument("--ckpt", type=Path, default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help="dataset name under $STABLEWM_HOME")
    p.add_argument("--latents", type=Path, default=None,
                   help="prefix of the precomputed embeddings (default: $STABLEWM_HOME/<dataset>_latents)")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="hold this fraction of episodes out of training (gplan.data.split_episodes), so "
                        "evaluate.py --holdout-frac / --split-seed can test on unseen episodes (0 = train on all)")
    p.add_argument("--split-seed", type=int, default=0, help="seed of the episode split")
    p.add_argument("--n-steps", type=int, default=5, help="planning horizon T")
    p.add_argument("--frameskip", type=int, default=5,
                   help="env steps per plan step: a plan step holds frameskip x the dataset's action dim values, "
                        "which must be the input size of LeWM's action encoder")
    p.add_argument("--n-components", type=int, default=1,
                   help="EXP mixture policy: Gaussians per plan step (1 = a single diagonal Gaussian)")
    p.add_argument("--n-samples", type=int, default=8, help="plans sampled per (start, goal)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="final cost temperature, per embedding dim (the raw cost is scaled by beta / embed_dim)")
    p.add_argument("--beta-start", type=float, default=0.0, help="initial beta when annealing, same units as --beta")
    p.add_argument("--beta-warmup", type=int, default=0,
                   help="steps to anneal beta linearly from --beta-start to --beta (0 = constant beta)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-z", type=float, default=3e-3, help="learning rate of the log Z head (usually 10-100x --lr)")
    p.add_argument("--grad-clip", type=float, default=10.0, help="max global grad L2 norm; <= 0 disables")
    p.add_argument("--ls-steps", type=int, default=0,
                   help="EXP MALA local search: MALA steps from each on-policy plan; the results are added to "
                        "the batch as off-policy plans (0 = off)")
    p.add_argument("--ls-every", type=int, default=1, help="EXP MALA: run local search every this many steps")
    p.add_argument("--ls-start", type=int, default=0,
                   help="EXP MALA: first step with local search (e.g. after --beta-warmup)")
    p.add_argument("--ls-step-size", type=float, default=1e-3, help="EXP MALA: initial step size eta")
    p.add_argument("--ls-target-accept", type=float, default=0.574,
                   help="EXP MALA: eta is adapted toward this acceptance rate (0 = keep eta fixed)")
    p.add_argument("--ppo-reward", default="target", choices=["target", "cost"],
                   help="[ppo] plan return: -beta J + log N(A; 0, s^2 I) (the GFlowNet's log target) or -beta J")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="[ppo] c in the per-step reward -c log pi(a_t | s_t): 0 = classic PPO, 1 = max-ent (optimum: "
                        "the GFlowNet target)")
    p.add_argument("--ppo-epochs", type=int, default=4, help="[ppo] passes over each rollout")
    p.add_argument("--ppo-minibatches", type=int, default=4, help="[ppo] minibatches per pass")
    p.add_argument("--clip-eps", type=float, default=0.2, help="[ppo] probability-ratio clipping")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="[ppo] GAE lambda (gamma = 1: 5-step episodes)")
    p.add_argument("--vf-coef", type=float, default=0.5, help="[ppo] weight of the critic's loss")
    p.add_argument("--target-kl", type=float, default=0.0,
                   help="[ppo] skip an iteration's remaining epochs once an epoch's approx KL exceeds this (0 = off)")
    p.add_argument("--ppo-grad-clip", type=float, default=0.5, help="[ppo] max global grad norm (policy + critic)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-var", type=float, default=1.0, help="variance s^2 of the N(0, s^2 I) reference measure")
    p.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "default"],
                   help="float32 matmul precision on GPU; 'default' allows TF32 (faster, less exact than torch)")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic XLA GPU kernels (--xla_gpu_deterministic_ops): bit-identical reruns, slower")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("outputs/gplaner"), help="checkpoint prefix")
    p.add_argument("--wandb-project", default="gplan")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-name", default=None, help="run name (defaults to wandb's random name)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.deterministic:  # XLA reads its flags when the backend starts, i.e. at the first jax.devices() below
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print("JAX devices:", jax.devices())

    cost_model = load_lewm_predictor(args.ckpt)  # torch is used only here, on CPU
    pairs = LatentPairs(args.latents or STABLEWM_HOME / f"{args.dataset}_latents", args.goal_offset,
                        expected_ckpt=args.ckpt)
    if args.holdout_frac > 0:
        train_eps, held_out = split_episodes(pairs.ep_idx, args.holdout_frac, args.split_seed)
        pairs.keep_episodes(train_eps)
        print(f"training on {len(train_eps)} episodes, {len(held_out)} held out for evaluation")
    d = pairs.embed_dim
    args.action_dim = plan_step_dim(args.frameskip, pairs.actions.shape[1], cost_model)  # recorded in train_args
    model_kwargs = dict(state_dim=d, horizon=args.n_steps * args.action_dim, action_dim=args.action_dim,
                        n_components=args.n_components)
    model = GPlanner(**model_kwargs, key=init_key(args.seed))
    cfg = TrainConfig(n_samples=args.n_samples, action_dim=args.action_dim,
                      reference_var=args.ref_var, grad_clip=args.grad_clip, ls_steps=args.ls_steps,
                      ls_every=args.ls_every, ls_start=args.ls_start, ls_step_size=args.ls_step_size,
                      ls_target_accept=args.ls_target_accept)

    n_params = sum(x.size for x in jax.tree.leaves(model))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"embed_dim": d, "n_params": n_params, "backend": "jax"})
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, mode=args.wandb_mode,
               job_type="train")
    wandb.define_metric("loss/total", summary="min")
    wandb.define_metric("cost/mean", summary="min")

    if args.method == "gfn":
        beta = linear_beta_schedule(args.beta / d, args.beta_start / d, args.beta_warmup)
        model, losses = train(model, cost_model, pairs.batches(args.n_batches, args.batch_size, args.seed), beta,
                              cfg, lr=args.lr, lr_z=args.lr_z, seed=args.seed, log_every=args.log_every)
    elif args.method == "bc":
        env_dim = pairs.actions.shape[1]
        assert args.goal_offset == args.n_steps * args.frameskip, (
            f"bc needs --goal-offset = --n-steps x --frameskip (the expert plan is the {args.goal_offset} dataset "
            f"actions between start and goal, {args.frameskip} {env_dim}-d actions per plan step)")
        batches = pairs.expert_batches(args.n_batches, args.batch_size, args.n_steps, args.seed)
        model, losses = train_baseline(model, bc_loss, batches, lr=args.lr, grad_clip=args.grad_clip,
                                       log_every=args.log_every)
    elif args.method == "map":  # the final beta, no warmup
        loss_fn = partial(map_loss, cost_model=cost_model, beta=args.beta / d, reference_var=args.ref_var,
                          action_dim=args.action_dim)
        model, losses = train_baseline(model, loss_fn, pairs.batches(args.n_batches, args.batch_size, args.seed),
                                       lr=args.lr, grad_clip=args.grad_clip, log_every=args.log_every)
    else:  # ppo: --n-batches iterations of --batch-size conditions x --n-samples plans; the final beta
        critic = Critic(d, model.horizon, model.n_steps, key=critic_key(args.seed))
        ppo_cfg = PPOConfig(n_samples=args.n_samples, action_dim=args.action_dim, reference_var=args.ref_var,
                            reward=args.ppo_reward, ent_coef=args.ent_coef, gae_lambda=args.gae_lambda,
                            clip_eps=args.clip_eps, vf_coef=args.vf_coef, epochs=args.ppo_epochs,
                            minibatches=args.ppo_minibatches, target_kl=args.target_kl, grad_clip=args.ppo_grad_clip)
        model, _, returns = train_ppo(model, critic, cost_model,
                                      pairs.batches(args.n_batches, args.batch_size, args.seed), args.beta / d,
                                      ppo_cfg, lr=args.lr, seed=args.seed, log_every=args.log_every)
        losses = [-r for r in returns]  # final_loss below = minus the last iteration's mean plan return

    checkpoint.save(args.out, model, model_kwargs, train_args=config)
    print(f"saved {args.out}.eqx / .json / .pt")
    wandb.summary["final_loss"] = losses[-1]
    wandb.save(f"{args.out}.pt", policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
