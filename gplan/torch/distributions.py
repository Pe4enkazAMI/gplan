"""The planner's step distribution in PyTorch, for sampling at evaluation (twin of the sampling part of
`gplan.jax.distributions`, where training also scores plans).

    DiagGaussianMixture    sum_k w_k N(mean_k, diag std_k^2): one plan step of the policy

Sampling takes an optional `torch.Generator`, so a solver can own its random stream.
"""

import torch


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
