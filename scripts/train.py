"""Train the GPlanner sampler (JAX) as a conditional GFlowNet over LeWM action plans (method=gfn),
or the same network as a baseline: behavior cloning on the dataset's expert actions (method=bc),
amortized MAP planning through LeWM (method=map; evaluate with gflow_select=mean) or PPO with a
learned critic (method=ppo, gplan/jax/ppo.py).

Needs precomputed embeddings (scripts/encode_dataset.py). Writes <out>.{eqx,json,pt} and the composed
config <out>.yaml; the .pt is what scripts/evaluate.py loads. Settings: configs/train.yaml (task, model)
and gplan/config.py TrainArgs, as Hydra overrides:

    python scripts/train.py task=pusht model=gfn-mala seed=1 out=outputs/pusht/gfn-mala-s1   # a paper run
    python scripts/train.py beta=120 wandb_name=tb-beta120 out=outputs/tb-beta120
    python scripts/train.py n_components=4               # EXP: mixture of 4 Gaussians per plan step
    python scripts/train.py ls_steps=10                  # EXP: + 10 MALA steps per plan as off-policy data
    python scripts/train.py method=bc                    # baseline: behavior cloning on expert actions
    python scripts/train.py method=map beta=120          # baseline: one plan per condition, min beta J - log N
    python scripts/train.py method=ppo beta=120          # baseline: PPO maximizing -beta J + log N (classic)
    python scripts/train.py method=ppo ent_coef=1        # PPO whose optimum is the GFlowNet target (max-ent)
    python scripts/train.py task=pusht model=gfn --help  # the composed settings

See gplan/jax/losses.py for the target distribution and the loss.
"""

import os
import sys
from functools import partial
from pathlib import Path

import jax
import wandb

from gplan.config import compose_config, script_overrides, to_args, TrainArgs, write_config
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
        raise SystemExit(f"a plan step is frameskip {frameskip} x {env_action_dim}-d dataset actions = {action_dim} "
                         f"values, but the LeWM checkpoint's action encoder takes {lewm_dim}: wrong dataset / "
                         f"ckpt pair or frameskip?")
    return action_dim


def parse_args(argv=None):
    """(settings namespace, composed config, overrides) from the command line (Hydra overrides)."""
    overrides = script_overrides(sys.argv[1:] if argv is None else argv, __doc__, "train")
    cfg = compose_config("train", overrides)
    return to_args(cfg, TrainArgs), cfg, overrides


def main():
    args, cfg, overrides = parse_args()
    write_config(f"{args.out}.yaml", overrides, cfg)
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
            f"bc needs goal_offset = n_steps x frameskip (the expert plan is the {args.goal_offset} dataset "
            f"actions between start and goal, {args.frameskip} {env_dim}-d actions per plan step)")
        batches = pairs.expert_batches(args.n_batches, args.batch_size, args.n_steps, args.seed)
        model, losses = train_baseline(model, bc_loss, batches, lr=args.lr, grad_clip=args.grad_clip,
                                       log_every=args.log_every)
    elif args.method == "map":  # the final beta, no warmup
        loss_fn = partial(map_loss, cost_model=cost_model, beta=args.beta / d, reference_var=args.ref_var,
                          action_dim=args.action_dim)
        model, losses = train_baseline(model, loss_fn, pairs.batches(args.n_batches, args.batch_size, args.seed),
                                       lr=args.lr, grad_clip=args.grad_clip, log_every=args.log_every)
    else:  # ppo: n_batches iterations of batch_size conditions x n_samples plans; the final beta
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
