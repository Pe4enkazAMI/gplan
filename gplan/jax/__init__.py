"""JAX side (Equinox + Optax): training on precomputed LeWM latents.

Modules
    lewm        LeWM predictor in Equinox and the planning cost J(A, c)
    policy      GPlaner, sampling, teacher-forced log-probs
    losses      the Trajectory Balance loss + diagnostics
    mala        MALA local search on plans (off-policy training samples)
    latents     (start, goal) latent pairs from precomputed embeddings
    train       optimizer, jitted train step, training loop
    probe       headroom probe: gradient refinement of sampled plans through LeWM
    convert     weight conversion PyTorch <-> Equinox (imports torch)
    checkpoint  save / load JAX checkpoints, export to the PyTorch format evaluation loads
"""
