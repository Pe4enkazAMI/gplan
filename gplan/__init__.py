"""GPlan: GFlowNet samplers as amortized planners for LeWorldModel (LeWM).

The code is split by framework, which is also a split by role:

    gplan.data    NumPy dataset helpers shared by both sides (STABLEWM_HOME, valid start rows)
    gplan.torch   PyTorch: everything that needs the original LeWM. Precomputing latents,
                  the PyTorch policy used at evaluation, solvers, the evaluation protocol
    gplan.jax     JAX (Equinox + Optax): training on precomputed latents, the LeWM predictor
                  ported to Equinox, losses, MALA, the headroom probe, checkpoints, and the
                  weight conversion PyTorch <-> Equinox

The subpackages are named after the frameworks they use; inside them `import torch` /
`import jax` still mean the real libraries (absolute imports). Run code from the repository
root (scripts/, pytest), not from inside gplan/.
"""
