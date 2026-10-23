"""PPO baseline: the same GPlaner trained as an RL policy (PPO-clip with GAE and a learned critic).

Planning a 5-step plan for one (start, goal) is an episode:
    state   s_t = (z_start, z_goal, plan steps < t)                    t = 0 .. n_steps - 1
    action  a_t = plan step t (action_dim values), drawn from the GPlaner
    reward  r_t = log N(a_t; 0, s^2 I)   (reward="target"; 0 for reward="cost")
                  - c log pi(a_t | s_t)  (entropy, c = ent_coef; see below)
                  - beta J(A)            (at the last step only; J is the LeWM cost of the whole plan)
Transitions are deterministic and the episode ends after n_steps, so the return of a whole plan is
log N(A) - beta J(A) - c log pi(A): with reward="target" that is the log of the GFlowNet's (and
map's) unnormalized target. ent_coef = 0 is classic PPO, a reward maximizer. With ent_coef = 1 the
expected return is -KL(pi || P*) + log Z, so its optimum is the GFlowNet target P* itself: the entropy
enters through the reward (not as a loss bonus) so that later steps' entropy is credited to earlier
choices, which makes this exact, also for mixture policies whose entropy has no closed form.

One iteration (Schulman et al. 2017; the structure of CleanRL's ppo_continuous_action):
    1. rollout   n_samples plans for each of the batch's (start, goal), with the current policy;
                 per step: log pi_old, V_old; per plan: J (one LeWM forward)
    2. GAE       advantages and value targets (gamma, lambda), with V(s_n_steps) = 0
    3. update    `epochs` passes over the plans in `minibatches` shuffled minibatches: clipped surrogate
                 + vf_coef x squared error of the critic (not clipped: with returns of tens of units,
                 PPO2's value clipping keeps the critic behind them), advantages normalized per
                 minibatch, global-norm gradient clipping, one Adam over policy and critic; optionally
                 the remaining epochs are skipped once an epoch's approx KL exceeds target_kl
The critic is a separate MLP; only the policy is exported (scripts/evaluate.py samples it like any
GPlaner).
"""

import time
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb

from gplan.jax.distributions import normal_log_prob
from gplan.jax.policy import rollout, step_log_probs
from gplan.jax.train import CRITIC_STREAM, TRAIN_STREAM


def critic_key(seed):
    return jax.random.fold_in(jax.random.key(seed), CRITIC_STREAM)


@dataclass(frozen=True)
class PPOConfig:
    """Static settings of a PPO iteration (changing one recompiles it)."""
    n_samples: int = 8           # plans per (start, goal) in a rollout
    action_dim: int = 1          # values per plan step
    reference_var: float = 1.0   # s^2 of the N(0, s^2 I) prior
    reward: str = "target"       # "target": -beta J + log N(A; 0, s^2 I); "cost": -beta J
    ent_coef: float = 0.0        # c in the reward term -c log pi(a_t | s_t); 0 = classic PPO
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2        # probability-ratio clipping
    vf_coef: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    target_kl: float = 0.0       # stop an iteration's remaining epochs once an epoch's mean approx KL exceeds
                                 # this (Spinning Up's early stopping); <= 0 disables
    grad_clip: float = 0.5       # max global grad norm; <= 0 disables clipping


class Critic(eqx.Module):
    """V(s_t): an MLP on (z_start, z_goal, plan buffer with steps >= t zeroed, one-hot t)."""
    layers: list[eqx.nn.Linear]
    n_steps: int = eqx.field(static=True)

    def __init__(self, state_dim, horizon, n_steps, hidden_size=64, n_layers=2, *, key):
        width = 3 * hidden_size  # as the GPlaner
        sizes = [2 * state_dim + horizon + n_steps] + [width] * (n_layers + 1) + [1]
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(i, o, key=k) for i, o, k in zip(sizes[:-1], sizes[1:], keys)]
        self.n_steps = n_steps

    def __call__(self, z_start, z_goal, actions, step):
        """V of one state: actions is the full plan buffer (horizon,); slots of steps >= `step` are ignored."""
        action_dim = actions.shape[0] // self.n_steps
        visible = jnp.arange(actions.shape[0]) < step * action_dim
        h = jnp.concatenate([z_start, z_goal, jnp.where(visible, actions, 0.0), jax.nn.one_hot(step, self.n_steps)])
        for layer in self.layers[:-1]:
            h = jax.nn.gelu(layer(h), approximate=False)
        return self.layers[-1](h)[0]

    def all_steps(self, z_start, z_goal, actions):
        """V(s_0) .. V(s_{n_steps - 1}) of one plan, (n_steps,)."""
        return jax.vmap(lambda t: self(z_start, z_goal, actions, t))(jnp.arange(self.n_steps))


def step_rewards(actions, cost, log_pi, beta, cfg: PPOConfig):
    """Per-step rewards (N, n_steps). actions (N, n_steps, action_dim), cost (N,), log_pi (N, n_steps)."""
    rewards = -cfg.ent_coef * log_pi
    if cfg.reward == "target":
        rewards = rewards + normal_log_prob(actions, 0.0, jnp.sqrt(cfg.reference_var)).sum(-1)
    else:
        assert cfg.reward == "cost", cfg.reward
    return rewards.at[:, -1].add(-beta * cost)


def gae(rewards, values, gamma, lam):
    """Generalized advantage estimates and value targets for finite episodes that all end after the
    last step (V after it = 0). rewards, values (N, T) -> (advantages, returns), both (N, T)."""
    T = rewards.shape[1]
    next_values = jnp.concatenate([values[:, 1:], jnp.zeros_like(values[:, :1])], axis=1)
    deltas = rewards + gamma * next_values - values
    advantages, running = [], jnp.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        running = deltas[:, t] + gamma * lam * running
        advantages.append(running)
    advantages = jnp.stack(advantages[::-1], axis=1)
    return advantages, advantages + values


def clipped_surrogate(ratio, advantages, eps):
    """PPO-clip policy loss (to minimize): -mean(min(r A, clip(r, 1 - eps, 1 + eps) A))."""
    return -jnp.mean(jnp.minimum(ratio * advantages, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages))


def ppo_loss(params, batch, cfg: PPOConfig):
    """Loss of one minibatch; batch = (z_start, z_goal, actions (n, horizon), old log pi, old V,
    advantages, returns), the last four (n, n_steps)."""
    policy, critic = params
    z_start, z_goal, actions, old_log_pi, old_values, advantages, returns = batch
    log_pi, dists = jax.vmap(partial(step_log_probs, policy))(z_start, z_goal, actions)
    values = jax.vmap(critic.all_steps)(z_start, z_goal, actions)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = log_pi - old_log_pi
    ratio = jnp.exp(log_ratio)
    pg_loss = clipped_surrogate(ratio, advantages, cfg.clip_eps)
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    stats = {"ppo/pg_loss": pg_loss, "ppo/value_loss": v_loss,
             "ppo/approx_kl": jnp.mean(ratio - 1.0 - log_ratio),  # k3 estimator of KL(old || new)
             "ppo/clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > cfg.clip_eps),
             "policy/entropy_lb": dists.entropy_lower_bound().mean(), "policy/std": dists.stds.mean()}
    return pg_loss + cfg.vf_coef * v_loss, stats


def make_optimizer(lr, grad_clip):
    clip = optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity()
    return optax.chain(clip, optax.adam(lr))


def make_iteration(optimizer, cfg: PPOConfig):
    """Build the jitted PPO iteration: rollout, GAE, then epochs x minibatches updates."""

    @eqx.filter_jit
    def iteration(params, opt_state, cost_model, key, z_start, z_goal, beta):
        policy, critic = params
        z_start = jnp.repeat(z_start, cfg.n_samples, axis=0)
        z_goal = jnp.repeat(z_goal, cfg.n_samples, axis=0)
        n = z_start.shape[0]
        if n % cfg.minibatches:  # shapes are static: this runs once, when the iteration is compiled
            raise ValueError(f"{n} plans per iteration (batch x n_samples) do not split into {cfg.minibatches} "
                             f"equal minibatches")
        k_rollout, k_shuffle = jax.random.split(key)

        # 1. rollout with the current (= old) policy; nothing here is differentiated
        actions = jax.vmap(partial(rollout, policy))(jax.random.split(k_rollout, n), z_start, z_goal)
        steps = actions.reshape(n, -1, cfg.action_dim)
        cost = cost_model.cost(z_start, z_goal, steps)
        old_log_pi, _ = jax.vmap(partial(step_log_probs, policy))(z_start, z_goal, actions)
        old_values = jax.vmap(critic.all_steps)(z_start, z_goal, actions)
        rewards = step_rewards(steps, cost, old_log_pi, beta, cfg)

        # 2. advantages and value targets
        advantages, returns = gae(rewards, old_values, cfg.gamma, cfg.gae_lambda)
        data = (z_start, z_goal, actions, old_log_pi, old_values, advantages, returns)

        # 3. epochs x minibatches of clipped updates over a fresh permutation per epoch; with target_kl > 0,
        # the epochs after one whose mean approx KL exceeds it are skipped
        perms = jax.vmap(lambda k: jax.random.permutation(k, n))(jax.random.split(k_shuffle, cfg.epochs))
        order = perms.reshape(cfg.epochs, cfg.minibatches, n // cfg.minibatches)
        arrays, static = eqx.partition(params, eqx.is_array)  # lax.scan carries arrays only

        def minibatch(carry, idx):
            arrays, opt_state = carry
            params = eqx.combine(arrays, static)
            batch = jax.tree.map(lambda x: x[idx], data)
            (loss, stats), grads = eqx.filter_value_and_grad(ppo_loss, has_aux=True)(params, batch, cfg)
            stats["loss/total"] = loss
            stats["grad_norm/global_raw"] = optax.tree.norm(eqx.filter(grads, eqx.is_array))
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
            params = eqx.apply_updates(params, updates)
            return (eqx.filter(params, eqx.is_array), opt_state), stats

        def epoch(carry, idx):
            arrays, opt_state, stopped = carry
            run = lambda: jax.lax.scan(minibatch, (arrays, opt_state), idx)
            skip = lambda: ((arrays, opt_state), jax.tree.map(jnp.zeros_like, jax.eval_shape(run)[1]))
            (arrays, opt_state), stats = jax.lax.cond(stopped, skip, run)
            ran = jnp.logical_not(stopped)
            stopped = stopped | ((cfg.target_kl > 0) & (stats["ppo/approx_kl"].mean() > cfg.target_kl))
            return (arrays, opt_state, stopped), (jax.tree.map(jnp.mean, stats), ran)

        (arrays, opt_state, _), (stats, ran) = jax.lax.scan(epoch, (arrays, opt_state, jnp.array(False)), order)
        params = eqx.combine(arrays, static)
        stats = {k: jnp.sum(v * ran) / jnp.sum(ran) for k, v in stats.items()}  # mean over the epochs that ran
        stats["ppo/epochs_run"] = jnp.sum(ran)

        stats.update({"cost/mean": cost.mean(), "cost/min": cost.min(),
                      "cost/beta_cost": (beta * cost).mean(), "ppo/return": rewards.sum(axis=1).mean(),
                      "ppo/value_0": old_values[:, 0].mean(),
                      "ppo/explained_var": 1.0 - jnp.var(returns - old_values) / (jnp.var(returns) + 1e-8),
                      "actions/abs_max": jnp.abs(actions).max()})
        return params, opt_state, stats

    return iteration


def train_ppo(policy, critic, cost_model, batches, beta, cfg: PPOConfig, lr=3e-4, seed=0, log_every=10):
    """One PPO iteration per (z_start, z_goal) batch. `beta` is the cost temperature (per embedding dim,
    as in the GFlowNet's training). Returns (policy, critic, mean plan return per iteration).

    `seed` drives rollouts and minibatch shuffling (stream TRAIN_STREAM, as for the GFlowNet);
    initialize the policy with `gplan.jax.train.init_key(seed)` and the critic with `critic_key(seed)`."""
    optimizer = make_optimizer(lr, cfg.grad_clip)
    params = (policy, critic)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    iteration = make_iteration(optimizer, cfg)
    key = jax.random.fold_in(jax.random.key(seed), TRAIN_STREAM)

    plan_returns, t_prev = [], time.perf_counter()
    for step, (z_start, z_goal) in enumerate(batches):
        key, step_key = jax.random.split(key)
        params, opt_state, stats = iteration(params, opt_state, cost_model, step_key, z_start, z_goal,
                                             jnp.float32(beta))
        stats = {k: float(v) for k, v in jax.device_get(stats).items()}
        plan_returns.append(stats["ppo/return"])
        t_now = time.perf_counter()
        stats.update({"optim/lr": lr, "time/step_s": t_now - t_prev})
        t_prev = t_now
        if wandb.run is not None:
            wandb.log(stats, step=step)
        if step % log_every == 0:
            print(f"iter {step:5d}  return {stats['ppo/return']:.3f}  cost {stats['cost/mean']:.3f}  "
                  f"value_loss {stats['ppo/value_loss']:.3f}  kl {stats['ppo/approx_kl']:.4f}  "
                  f"clip_frac {stats['ppo/clip_frac']:.3f}  epochs {stats['ppo/epochs_run']:.0f}  "
                  f"std {stats['policy/std']:.3f}  "
                  f"explained_var {stats['ppo/explained_var']:.3f}")
    return params[0], params[1], plan_returns
