"""The probability module, in both frameworks.

The PyTorch version is checked against torch.distributions (an independent implementation) and
the JAX version against the PyTorch one, so a mistake shared by our two twins would still fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from gplan.jax import distributions as jd
from gplan.torch import distributions as td

B, K, A = 6, 3, 4


def random_mixture(n_components=K, seed=0, logit_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    logits = logit_scale * torch.randn(B, n_components, generator=g)
    means = 2 * torch.randn(B, n_components, A, generator=g)
    stds = torch.exp(0.5 * torch.randn(B, n_components, A, generator=g))
    x = torch.randn(B, A, generator=g)
    return logits, means, stds, x


def reference(logits, means, stds):
    return MixtureSameFamily(Categorical(logits=logits), Independent(Normal(means, stds), 1))


def jax_mixture(logits, means, stds):
    return jd.DiagGaussianMixture.from_logits(*(jnp.asarray(t.numpy()) for t in (logits, means, stds)))


@pytest.mark.parametrize("n_components", [1, K])
def test_torch_log_prob_matches_torch_distributions(n_components):
    logits, means, stds, x = random_mixture(n_components)
    ours = td.DiagGaussianMixture.from_logits(logits, means, stds)
    torch.testing.assert_close(ours.log_prob(x), reference(logits, means, stds).log_prob(x))
    torch.testing.assert_close(ours.weights, torch.softmax(logits, -1))


@pytest.mark.parametrize("n_components", [1, K])
def test_jax_matches_torch(n_components):
    logits, means, stds, x = random_mixture(n_components)
    ours, theirs = jax_mixture(logits, means, stds), td.DiagGaussianMixture.from_logits(logits, means, stds)
    for name in ("log_weights", "means", "stds"):
        np.testing.assert_allclose(np.asarray(getattr(ours, name)), getattr(theirs, name).numpy(), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ours.log_prob(jnp.asarray(x.numpy()))), theirs.log_prob(x).numpy(),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ours.mode()), theirs.mode().numpy())
    np.testing.assert_allclose(np.asarray(ours.entropy_lower_bound()), theirs.entropy_lower_bound().numpy(), rtol=1e-6)
    # one distribution at a time (how the policy uses it under vmap) gives the same numbers
    single = jax.tree.map(lambda a: a[0], ours)
    np.testing.assert_allclose(float(single.log_prob(jnp.asarray(x[0].numpy()))), theirs.log_prob(x)[0].item(),
                               rtol=1e-6)


def test_one_component_is_the_diagonal_gaussian():
    logits, means, stds, x = random_mixture(1)
    mix = td.DiagGaussianMixture.from_logits(logits, means, stds)
    gauss = Independent(Normal(means[:, 0], stds[:, 0]), 1)
    torch.testing.assert_close(mix.log_prob(x), gauss.log_prob(x))
    torch.testing.assert_close(mix.entropy_lower_bound(), gauss.entropy())
    torch.testing.assert_close(mix.mode(), means[:, 0])


def test_entropy_lower_bound_is_below_the_mixture_entropy():
    """Monte Carlo entropy of the mixture vs sum_k w_k H_k."""
    logits, means, stds, _ = random_mixture()
    ref = reference(logits, means, stds)
    mc_entropy = -ref.log_prob(ref.sample((20000,))).mean(0)
    bound = td.DiagGaussianMixture.from_logits(logits, means, stds).entropy_lower_bound()
    assert torch.all(bound <= mc_entropy + 0.05)


def test_mode_is_the_mean_of_the_heaviest_component():
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    means = torch.tensor([[[-1.0], [5.0], [2.0]]])
    stds = torch.ones(1, 3, 1)
    torch.testing.assert_close(td.DiagGaussianMixture.from_logits(logits, means, stds).mode(), torch.tensor([[5.0]]))
    np.testing.assert_allclose(np.asarray(jax_mixture(logits, means, stds).mode()), [[5.0]])


def sample_stats(samples, means):
    """Share of samples nearest each (well separated) component and their per-component spread."""
    nearest = np.argmin(np.abs(samples[:, None, 0] - means[None, :, 0]), axis=1)
    return np.bincount(nearest, minlength=len(means)) / len(samples), nearest


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_samples_follow_the_weights_and_components(framework):
    weights = np.array([0.6, 0.3, 0.1])
    means = np.array([[-10.0, 1.0], [0.0, 2.0], [10.0, 3.0]], dtype=np.float32)
    stds = np.array([[0.5, 1.0], [1.0, 0.5], [0.3, 2.0]], dtype=np.float32)
    n = 40000
    if framework == "torch":
        dist = td.DiagGaussianMixture(torch.log(torch.tensor(weights, dtype=torch.float32)).expand(n, 3),
                                      torch.tensor(means).expand(n, 3, 2), torch.tensor(stds).expand(n, 3, 2))
        samples = dist.sample(torch.Generator().manual_seed(0)).numpy()
    else:
        dist = jd.DiagGaussianMixture(jnp.log(jnp.asarray(weights, dtype=jnp.float32)), jnp.asarray(means),
                                      jnp.asarray(stds))
        samples = np.asarray(jax.vmap(dist.sample)(jax.random.split(jax.random.key(0), n)))
    share, nearest = sample_stats(samples, means)
    np.testing.assert_allclose(share, weights, atol=0.01)
    for k in range(3):
        mine = samples[nearest == k]
        np.testing.assert_allclose(mine.mean(0), means[k], atol=0.1)
        np.testing.assert_allclose(mine.std(0), stds[k], rtol=0.05)


def test_torch_sampling_is_reproducible_with_a_generator():
    logits, means, stds, _ = random_mixture()
    dist = td.DiagGaussianMixture.from_logits(logits, means, stds)
    a = dist.sample(torch.Generator().manual_seed(7))
    b = dist.sample(torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, dist.sample(torch.Generator().manual_seed(8)))


def test_extreme_weights_stay_finite():
    """A component with weight ~0 (logit -1e4) or a narrow std must not produce NaN / inf."""
    logits, means, stds, x = random_mixture(logit_scale=1.0)
    logits[:, 1] = -1e4
    stds[:, 2] = 1e-3
    for dist in (td.DiagGaussianMixture.from_logits(logits, means, stds), jax_mixture(logits, means, stds)):
        log_p = np.asarray(dist.log_prob(x if isinstance(dist, td.DiagGaussianMixture) else jnp.asarray(x.numpy())))
        assert np.all(np.isfinite(log_p))
        assert np.all(np.isfinite(np.asarray(dist.entropy_lower_bound())))


def test_reference_log_prob_is_the_isotropic_gaussian():
    actions = torch.randn(B, 10)
    ref = Normal(0.0, 2.0 ** 0.5).log_prob(actions).sum(-1)
    torch.testing.assert_close(td.reference_log_prob(actions, 2.0), ref)
    np.testing.assert_allclose(np.asarray(jd.reference_log_prob(jnp.asarray(actions.numpy()), 2.0)), ref.numpy(),
                               rtol=1e-6)
