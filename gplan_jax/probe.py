"""Headroom probe: are there clearly better plans close to the ones the policy samples?

For each (start, goal) pair we compare the LeWM cost J of
    sampled      the best of N policy samples (what best-of-N evaluation executes)
    refined      the best of the same N samples after K gradient steps on the objective
    prior        the best of N draws from the N(0, s^2 I) prior (prior shooting, a floor)
    expert       the dataset's own actions between start and goal (a reference point)

Refinement is a measurement only, never part of the planner. A large sampled -> refined gap
means low-cost plans are reachable from the policy's own samples, which training-time local
search / replay could teach the sampler; a small gap means they would not help much.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gplan_jax.policy import rollout


def refine_plans(cost_model, z_start, z_goal, plans, beta, reference_var, steps, lr, objective="target"):
    """K Adam steps on every plan; plans (N, T, A) in the world model's z-scored action space.

    objective="target": minimize beta * J(A) + ||A||^2 / (2 s^2), i.e. -log of the GFlowNet target
                        (up to a constant); the prior term keeps actions near the data range.
    objective="cost":   minimize J(A) alone (no prior; can drift to actions LeWM never saw).
    Returns the refined plans.
    """
    def loss(p):
        j = cost_model.cost(z_start, z_goal, p)
        if objective == "target":
            return jnp.sum(beta * j + jnp.sum(p ** 2, axis=(1, 2)) / (2 * reference_var))
        return jnp.sum(j)  # plans are independent, so the gradient of the sum is per-plan

    opt = optax.adam(lr)

    def step(carry, _):
        p, state = carry
        updates, state = opt.update(jax.grad(loss)(p), state)
        return (optax.apply_updates(p, updates), state), None

    (plans, _), _ = jax.lax.scan(step, (plans, opt.init(plans)), None, length=steps)
    return plans


def lewm_action_stats(actions):
    """Per-dimension mean and std of the raw dataset actions, as LeWM's training normalizer computes
    them (rows with NaN dropped, unbiased std; le-wm/utils.py)."""
    a = actions[~np.isnan(actions).any(axis=1)]
    return a.mean(0), a.std(0, ddof=1)


def expert_plans(actions, rows, goal_offset, n_steps, mean, std):
    """Dataset actions from each start row to its goal row as plans (B, n_steps, frameskip * env_dim).

    Rows r .. r + goal_offset - 1 hold the actions taken between frame r and frame r + goal_offset;
    consecutive groups of `frameskip` env actions form one plan step (the layout swm's
    WorldModelPolicy unpacks). Assumes action[r] is the action taken at frame r.
    """
    assert goal_offset % n_steps == 0, f"goal_offset {goal_offset} must be a multiple of n_steps {n_steps}"
    seg = actions[rows[:, None] + np.arange(goal_offset)]           # (B, goal_offset, env_dim)
    seg = (seg - mean) / std
    return seg.reshape(len(rows), n_steps, -1).astype(np.float32)


def make_probe(planner, cost_model, plan_shape, n_samples, beta, reference_var, steps, lr, objective):
    """Jitted function: one chunk of pairs -> per-pair costs (each (B,)) of every plan source.

    plan_shape = (T, A): world-model plan steps and action dim per step (T * A == planner.horizon).
    """
    assert plan_shape[0] * plan_shape[1] == planner.horizon, (plan_shape, planner.horizon)

    @jax.jit
    def probe(key, z_start, z_goal, expert):
        B = z_start.shape[0]
        zs, zg = jnp.repeat(z_start, n_samples, axis=0), jnp.repeat(z_goal, n_samples, axis=0)
        k_policy, k_prior = jax.random.split(key)
        sampled = jax.vmap(partial(rollout, planner))(jax.random.split(k_policy, B * n_samples), zs, zg)
        sampled = sampled.reshape(B * n_samples, *plan_shape)
        prior = jnp.sqrt(reference_var) * jax.random.normal(k_prior, sampled.shape)
        refined = refine_plans(cost_model, zs, zg, sampled, beta, reference_var, steps, lr, objective)

        def best(plans):  # min over the N plans of each pair, plus the mean |action| of that plan
            j = cost_model.cost(zs, zg, plans).reshape(B, n_samples)
            idx = jnp.argmin(j, axis=1)
            chosen = plans.reshape(B, n_samples, -1)[jnp.arange(B), idx]
            return j.min(axis=1), jnp.abs(chosen).max(axis=1)

        out = {}
        for name, plans in (("sampled", sampled), ("refined", refined), ("prior", prior)):
            out[f"{name}_cost"], out[f"{name}_absmax"] = best(plans)
        out["sampled_mean_cost"] = cost_model.cost(zs, zg, sampled).reshape(B, n_samples).mean(axis=1)
        out["expert_cost"] = cost_model.cost(z_start, z_goal, expert)
        out["expert_absmax"] = jnp.abs(expert).reshape(B, -1).max(axis=1)
        return out

    return probe
