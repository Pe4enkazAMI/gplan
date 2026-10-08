"""GFlowNet losses for the conditional plan sampler (JAX port of the PyTorch losses, same formulas and stats).

Target:  P*(A | c)  ∝  exp(-beta * J(A, c)) * N(A; 0, s^2 I),  c = (z_start, z_goal)

    xi(A, c) = sum_t log P_F(a_t | a_<t, c) + beta * J(A, c) - log N(A)       (= -log Z(c) at optimum)

  tb:       ( log Z(c) + xi )^2                 with log Z(c) a learned head
  vargrad:  Var_k[ xi(A_k, c) ]  over K plans   of the same c; log Z(c) is the in-batch mean -xi

The cost J is passed in as data (computed without gradient), so plans and their costs can come
from anywhere. Standard deviations use ddof=1, like torch, so logged numbers are comparable.
"""

from functools import partial

import jax
import jax.numpy as jnp

from gplan_jax.policy import reference_log_prob, step_log_probs


def plan_terms(model, z_start, z_goal, actions, cost, reference_var):
    """Per-plan quantities; every array has leading dim N = actions.shape[0]."""
    log_pf_steps, means, stds = jax.vmap(partial(step_log_probs, model))(z_start, z_goal, actions)
    return {
        "log_pf": log_pf_steps.sum(-1),                             # (N,)  differentiable
        "log_pf_steps": log_pf_steps,                               # (N, n_steps)
        "log_z": jax.vmap(model.log_Z)(z_start, z_goal),            # (N,)  differentiable
        "cost": cost,                                               # (N,)  constant
        "log_ref": reference_log_prob(actions, reference_var),      # (N,)  constant
        "means": means, "stds": stds,                               # (N, n_steps, A) for diagnostics
    }


def xi(terms, beta):
    return terms["log_pf"] + beta * terms["cost"] - terms["log_ref"]


def tb_loss(model, z_start, z_goal, actions, cost, beta, reference_var):
    """Trajectory Balance with a learned log Z head. Returns (loss, stats)."""
    terms = plan_terms(model, z_start, z_goal, actions, cost, reference_var)
    residual = terms["log_z"] + xi(terms, beta)
    loss = jnp.mean(residual ** 2)
    return loss, {**plan_stats(terms, residual, beta), "tb/log_z": terms["log_z"].mean()}


def vargrad_loss(model, z_start, z_goal, actions, cost, beta, reference_var, n_samples):
    """VarGrad: variance of xi over the K plans of each condition, plus a regression of the
    log Z head onto the in-batch estimate (for diagnostics only; the two terms share no gradient).

    Rows are grouped by condition: plan i belongs to condition i // K (`jnp.repeat(z, K, axis=0)`).
    """
    terms = plan_terms(model, z_start, z_goal, actions, cost, reference_var)
    xi_bk = xi(terms, beta).reshape(-1, n_samples)                 # (B, K)
    var_loss = jnp.mean(jnp.var(xi_bk, axis=1, ddof=1))

    log_z_hat = jax.lax.stop_gradient(-xi_bk.mean(axis=1))         # (B,) in-batch log Z(c)
    log_z_head = terms["log_z"].reshape(-1, n_samples)[:, 0]       # (B,) one per condition
    z_fit_loss = jnp.mean((log_z_head - log_z_hat) ** 2)

    residual = (xi_bk - xi_bk.mean(axis=1, keepdims=True)).reshape(-1)
    stats = {**plan_stats(terms, residual, beta),
             "loss/vargrad": var_loss, "loss/z_fit": z_fit_loss,
             "tb/log_z": log_z_head.mean(), "tb/log_z_hat": log_z_hat.mean(),
             "tb/log_z_hat_std": jnp.std(log_z_hat, ddof=1)}
    return var_loss + z_fit_loss, stats


def plan_stats(terms, residual, beta):
    """Scalar diagnostics shared by both losses (same names as the PyTorch version)."""
    terms = jax.lax.stop_gradient(terms)
    residual = jax.lax.stop_gradient(residual)
    cost, stds = terms["cost"], terms["stds"]
    std = partial(jnp.std, ddof=1)
    return {
        "tb/residual": residual.mean(),
        "tb/residual_abs": jnp.abs(residual).mean(),
        "tb/residual_std": std(residual),
        "tb/log_pf": terms["log_pf"].mean(),
        "tb/log_pf_std": std(terms["log_pf"]),
        "tb/log_pf_per_step": terms["log_pf_steps"].mean(),
        "tb/log_ref": terms["log_ref"].mean(),
        "tb/log_ref_std": std(terms["log_ref"]),
        "tb/neg_log_reward": (beta * cost - terms["log_ref"]).mean(),
        "cost/mean": cost.mean(),
        "cost/std": std(cost),
        "cost/min": cost.min(),
        "cost/max": cost.max(),
        "cost/beta_cost": (beta * cost).mean(),
        "cost/beta_cost_std": std(beta * cost),
        # the Gaussians the policy predicted along the plans (teacher forced)
        "policy/mean_abs": jnp.abs(terms["means"]).mean(),
        "policy/std": stds.mean(),
        "policy/std_min": stds.min(),
        "policy/std_max": stds.max(),
        "policy/entropy": (0.5 * jnp.log(2 * jnp.pi * jnp.e * stds ** 2)).sum(axis=(1, 2)).mean(),
    }
