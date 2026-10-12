"""MALA local search on action plans: off-policy training samples for the GFlowNet.

Targets the same distribution the sampler is trained to match,

    log P*(A | c) = -beta * J(A, c) + log N(A; 0, s^2 I) + const,

using only the world model's gradient (no expert, no external planner). One step from A:

    A' = A + eta * grad log P*(A) + sqrt(2 eta) * xi,            xi ~ N(0, I)
    accept with prob. min(1, P*(A') q(A | A') / (P*(A) q(A' | A))),  q(y | x) = N(y; x + eta grad log P*(x), 2 eta I)

The Metropolis-Hastings correction keeps the chain on the target (it does not drift into
regions where the world model is wrong but J looks good). TB / VarGrad stay valid on these
plans because both losses only need log P_F of a given plan (teacher forcing).
"""

import jax
import jax.numpy as jnp

from gplan.jax.policy import reference_log_prob


def mala(cost_model, key, z_start, z_goal, actions, beta, reference_var, action_dim, steps, step_size):
    """`steps` MALA steps on every plan, all chains in parallel.

    actions: (N, horizon) starting plans (e.g. on-policy samples); z_start / z_goal: (N, D).
    Returns (plans (N, horizon), their costs J (N,), acceptance rate averaged over steps and chains).
    """
    n = actions.shape[0]

    def summed_log_target(a):
        cost = cost_model.cost(z_start, z_goal, a.reshape(n, -1, action_dim))
        log_target = -beta * cost + reference_log_prob(a, reference_var)  # (N,)
        return log_target.sum(), (log_target, cost)  # chains are independent: grad of the sum is per chain

    log_target_and_grad = jax.value_and_grad(summed_log_target, has_aux=True)

    def log_q(to, frm, grad_frm):  # log q(to | from), up to a constant that cancels
        return -jnp.sum((to - frm - step_size * grad_frm) ** 2, axis=-1) / (4 * step_size)

    def step(state, key):
        a, lt, cost, grad = state
        k_noise, k_accept = jax.random.split(key)
        prop = a + step_size * grad + jnp.sqrt(2 * step_size) * jax.random.normal(k_noise, a.shape)
        (_, (lt_p, cost_p)), grad_p = log_target_and_grad(prop)
        log_alpha = lt_p - lt + log_q(a, prop, grad_p) - log_q(prop, a, grad)
        accept = jnp.log(jax.random.uniform(k_accept, (n,))) < log_alpha  # NaN -> rejected
        keep = lambda new, old: jnp.where(accept.reshape(-1, *[1] * (new.ndim - 1)), new, old)
        return (keep(prop, a), keep(lt_p, lt), keep(cost_p, cost), keep(grad_p, grad)), accept.mean()

    (_, (lt, cost)), grad = log_target_and_grad(actions)
    (actions, _, cost, _), accepted = jax.lax.scan(step, (actions, lt, cost, grad), jax.random.split(key, steps))
    return actions, cost, accepted.mean()
