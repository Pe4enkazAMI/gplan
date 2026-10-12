"""Closed-form costs for JAX tests: they stand in for LeWM's J(A, c) where the optimum is known."""

import equinox as eqx
import jax.numpy as jnp


class QuadraticCost(eqx.Module):
    """J(A) = ||A||^2: with the N(0, s^2 I) reference the target is N(0, 1 / (2 beta + 1 / s^2)) per action."""

    def cost(self, z_start, z_goal, plans):
        return jnp.sum(plans ** 2, axis=(1, 2))


class DoubleWell(eqx.Module):
    """One scalar action, J(a) = (a^2 - 4)^2: two equally likely modes at a = -2 and a = +2."""

    def cost(self, z_start, z_goal, plans):
        return (plans[:, 0, 0] ** 2 - 4.0) ** 2
