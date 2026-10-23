"""The probability distributions the planner uses, in JAX (`gplan.torch.distributions` has the sampling
part in PyTorch, for evaluation).

    normal_log_prob(x, mean, std)       elementwise log N(x; mean, std^2)
    reference_log_prob(actions, var)    log N(A; 0, s^2 I) of a whole plan buffer
    gaussian_entropy(std)               elementwise entropy of N(., std^2)
    DiagGaussianMixture                 sum_k w_k N(mean_k, diag std_k^2): one plan step of the policy

Arrays may carry leading batch dimensions everywhere except `DiagGaussianMixture.sample`, which
draws one vector (batch it with `jax.vmap`).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

LOG_2PI = math.log(2 * math.pi)


def normal_log_prob(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - 0.5 * LOG_2PI


def reference_log_prob(actions, reference_var):
    """log N(actions; 0, s^2 I) summed over the last axis (the plan buffer)."""
    return normal_log_prob(actions, 0.0, jnp.sqrt(reference_var)).sum(-1)


def gaussian_entropy(std):
    return 0.5 * (LOG_2PI + 1.0) + jnp.log(std)


class DiagGaussianMixture(NamedTuple):
    """Mixture of K diagonal Gaussians over vectors of size A; K = 1 is a plain diagonal Gaussian.

    log_weights (..., K) are normalized log mixture weights; means, stds (..., K, A).
    A NamedTuple, so it is a pytree: it can be returned from jitted / vmapped functions.
    """
    log_weights: jax.Array
    means: jax.Array
    stds: jax.Array

    @classmethod
    def from_logits(cls, logits, means, stds):
        return cls(jax.nn.log_softmax(logits, axis=-1), means, stds)

    @property
    def weights(self):
        return jnp.exp(self.log_weights)

    def log_prob(self, x):
        """log sum_k w_k N(x; mean_k, diag std_k^2), computed with logsumexp. x (..., A) -> (...)."""
        per_component = normal_log_prob(x[..., None, :], self.means, self.stds).sum(-1)  # (..., K)
        return jax.nn.logsumexp(self.log_weights + per_component, axis=-1)

    def sample(self, key):
        """One draw (A,): pick a component by its weight, then sample that Gaussian."""
        k_component, k_noise = jax.random.split(key)
        c = jax.random.categorical(k_component, self.log_weights)
        return self.means[c] + self.stds[c] * jax.random.normal(k_noise, self.means.shape[-1:])

    def mode(self):
        """Mean of the most likely component (..., A). The mixture's own mean can fall between
        its modes; with one component the two coincide."""
        best = jnp.argmax(self.log_weights, axis=-1)
        return jnp.take_along_axis(self.means, best[..., None, None], axis=-2)[..., 0, :]

    def entropy_lower_bound(self):
        """sum_k w_k H(component k) (...): a lower bound on the mixture entropy, exact for K = 1."""
        return jnp.sum(self.weights * gaussian_entropy(self.stds).sum(-1), axis=-1)
