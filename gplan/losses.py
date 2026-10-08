"""GFlowNet losses for the conditional plan sampler.

Target:  P*(A | c)  ∝  exp(-beta * J(A, c)) * N(A; 0, s^2 I)

where c = (z_start, z_goal) are frozen LeWM embeddings, J is the planning cost
(see `gplan.lewm.lewm_cost`) and N(.; 0, s^2 I) is the reference measure.

Both losses are built on the per-plan quantity

    xi(A, c) = sum_t log P_F(a_t | a_<t, c) + beta * J(A, c) - log N(A)       (= -log Z(c) at optimum)

  tb:       ( log Z(c) + xi )^2                 with log Z(c) a learned head
  vargrad:  Var_k[ xi(A_k, c) ]  over K plans   sampled for the same c; log Z(c)
            is the in-batch mean -xi, so no partition function has to be learned.

The DAG is a tree (each prefix has one parent), so P_B = 1. Both losses are
valid for plans from any distribution with full support, not only P_F's own samples.

`cost_fn(z_start, z_goal, plans) -> (N,)` scores plans of shape (N, T, action_dim);
pass `functools.partial(lewm_cost, wm)` for the LeWM cost.
"""

import torch


def plan_terms(sampler, cost_fn, z_start, z_goal, actions, action_dim):
    """Everything the losses need, per plan. All tensors have leading dim N = actions.shape[0].

    Returns a dict with
        log_pf:       sum_t log P_F(a_t | a_<t, c)                (N,)   differentiable
        log_pf_steps: the per-step terms                          (N, n_steps)
        log_z:        log Z(c) from the model's Z head            (N,)   differentiable
        cost:         J(A, c), the planning cost                  (N,)   no gradient
        log_ref:      log N(A; 0, s^2 I), prior on action buffers (N,)   no gradient
    """
    n = actions.shape[0]
    log_pf_steps = sampler.log_prob(z_start, z_goal, actions)
    return {
        "log_pf": log_pf_steps.sum(-1),
        "log_pf_steps": log_pf_steps,
        "log_z": sampler.model.log_Z(z_start, z_goal),
        "cost": cost_fn(z_start, z_goal, actions.view(n, -1, action_dim)),
        "log_ref": sampler.reference_log_prob(actions),
    }


def xi(terms, beta):
    """xi = log P_F + beta J - log N(A).  At the optimum xi == -log Z(c) for every plan A."""
    return terms["log_pf"] + beta * terms["cost"] - terms["log_ref"]


def tb_loss(sampler, cost_fn, z_start, z_goal, actions, beta, action_dim, return_stats=False):
    """Trajectory Balance loss  ( log Z(c) + xi(A, c) )^2  with a learned log Z head.

    Args:
        z_start, z_goal: (N, D) conditions, one per plan.
        actions:         (N, T * action_dim) buffers.
        beta:            cost temperature.
        return_stats:    also return a dict of detached scalar diagnostics.
    """
    terms = plan_terms(sampler, cost_fn, z_start, z_goal, actions, action_dim)
    residual = terms["log_z"] + xi(terms, beta)
    loss = residual.pow(2).mean()
    if not return_stats:
        return loss
    return loss, {**plan_stats(terms, residual, beta), "tb/log_z": terms["log_z"].mean().item()}


def vargrad_loss(sampler, cost_fn, z_start, z_goal, actions, beta, action_dim, n_samples, return_stats=False):
    """VarGrad loss  Var_k[ xi(A_k, c) ]  over K plans sampled for the same condition c.

    Since xi(A, c) = -log Z(c) for all A at the optimum, its variance over plans of
    one condition is zero there, and the per-condition mean -xi is an in-batch
    estimate of log Z(c). No partition function has to be learned, and the mean
    acts as a per-condition baseline for the policy gradient. The model's Z head
    is still fitted (by regression to that estimate) for diagnostics; it gets no
    gradient from the VarGrad term and the policy gets none from the regression.

    Args:
        z_start, z_goal: (B*K, D) conditions, each repeated K times consecutively
                         (`repeat_interleave(K, 0)`), i.e. plan i belongs to condition i // K.
        actions:         (B*K, T * action_dim) buffers, one per row of z_start.
        n_samples:       K >= 2.
    """
    assert n_samples >= 2, "VarGrad needs at least 2 plans per condition"
    terms = plan_terms(sampler, cost_fn, z_start, z_goal, actions, action_dim)
    xi_bk = xi(terms, beta).view(-1, n_samples)                    # (B, K)
    loss = xi_bk.var(dim=1).mean()

    log_z_hat = -xi_bk.mean(dim=1).detach()                        # (B,)  in-batch log Z(c)
    log_z_head = terms["log_z"].view(-1, n_samples)[:, 0]          # (B,)  one per condition
    z_fit_loss = (log_z_head - log_z_hat).pow(2).mean()
    if not return_stats:
        return loss + z_fit_loss
    residual = xi_bk - xi_bk.mean(dim=1, keepdim=True)             # TB residual with log Z = log_z_hat
    stats = plan_stats(terms, residual.flatten(), beta)
    stats.update({
        "loss/vargrad": loss.item(),
        "loss/z_fit": z_fit_loss.item(),
        "tb/log_z": log_z_head.mean().item(),
        "tb/log_z_hat": log_z_hat.mean().item(),
        "tb/log_z_hat_std": log_z_hat.std().item(),                # how much log Z varies across conditions
    })
    return loss + z_fit_loss, stats


@torch.no_grad()
def plan_stats(terms, residual, beta):
    """Detached scalar diagnostics shared by both losses."""
    cost = terms["cost"]
    stats = {
        "tb/residual": residual.mean(),          # signed; should hover around 0
        "tb/residual_abs": residual.abs().mean(),
        "tb/residual_std": residual.std(),
        "tb/log_pf": terms["log_pf"].mean(),
        "tb/log_pf_std": terms["log_pf"].std(),
        "tb/log_pf_per_step": terms["log_pf_steps"].mean(),
        "tb/log_ref": terms["log_ref"].mean(),
        "tb/log_ref_std": terms["log_ref"].std(),
        "tb/neg_log_reward": (beta * cost - terms["log_ref"]).mean(),  # -log R, what log Z + log P_F must match
        "cost/mean": cost.mean(),
        "cost/std": cost.std(),
        "cost/min": cost.min(),
        "cost/max": cost.max(),
        "cost/beta_cost": (beta * cost).mean(),
        "cost/beta_cost_std": (beta * cost).std(),  # spread of -log R across the batch; what the policy must explain
    }
    return {k: v.item() for k, v in stats.items()}
