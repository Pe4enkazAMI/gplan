"""The probability distributions the planner uses, in PyTorch (twin of `gplan.jax.distributions`).

    normal_log_prob(x, mean, std)       elementwise log N(x; mean, std^2)
    reference_log_prob(actions, var)    log N(A; 0, s^2 I) of a whole plan buffer
    gaussian_entropy(std)               elementwise entropy of N(., std^2)
    DiagGaussianMixture                 sum_k w_k N(mean_k, diag std_k^2): one plan step of the policy

Sampling takes an optional `torch.Generator`, so a solver can own its random stream.
"""

import math

import torch

LOG_2PI = math.log(2 * math.pi)


def normal_log_prob(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - torch.log(std) - 0.5 * LOG_2PI


def reference_log_prob(actions, reference_var):
    """log N(actions; 0, s^2 I) summed over the last axis (the plan buffer)."""
    std = torch.as_tensor(reference_var, dtype=actions.dtype, device=actions.device).sqrt()
    return normal_log_prob(actions, 0.0, std).sum(-1)


def gaussian_entropy(std):
    return 0.5 * (LOG_2PI + 1.0) + torch.log(std)


class DiagGaussianMixture:
    """Mixture of K diagonal Gaussians over vectors of size A; K = 1 is a plain diagonal Gaussian.

    log_weights (B, K) are normalized log mixture weights; means, stds (B, K, A).
    """

    def __init__(self, log_weights, means, stds):
        self.log_weights, self.means, self.stds = log_weights, means, stds

    @classmethod
    def from_logits(cls, logits, means, stds):
        return cls(logits.log_softmax(-1), means, stds)

    @property
    def weights(self):
        return self.log_weights.exp()

    def log_prob(self, x):
        """log sum_k w_k N(x; mean_k, diag std_k^2), computed with logsumexp. x (B, A) -> (B,)."""
        per_component = normal_log_prob(x[:, None, :], self.means, self.stds).sum(-1)  # (B, K)
        return torch.logsumexp(self.log_weights + per_component, dim=-1)

    def sample(self, generator=None):
        """One draw per row (B, A): pick a component by its weight, then sample that Gaussian."""
        B, K, A = self.means.shape
        rows = torch.arange(B, device=self.means.device)
        c = torch.multinomial(self.weights, 1, generator=generator)[:, 0] if K > 1 else torch.zeros_like(rows)
        noise = torch.randn(B, A, generator=generator, device=self.means.device, dtype=self.means.dtype)
        return self.means[rows, c] + self.stds[rows, c] * noise

    def mode(self):
        """Mean of the most likely component (B, A). The mixture's own mean can fall between
        its modes; with one component the two coincide."""
        best = self.log_weights.argmax(-1)
        return self.means[torch.arange(self.means.shape[0], device=self.means.device), best]

    def entropy_lower_bound(self):
        """sum_k w_k H(component k) (B,): a lower bound on the mixture entropy, exact for K = 1."""
        return (self.weights * gaussian_entropy(self.stds).sum(-1)).sum(-1)
