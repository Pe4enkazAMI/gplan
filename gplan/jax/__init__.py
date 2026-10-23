"""JAX side (Equinox + Optax): training on precomputed LeWM latents.

Modules
    lewm           LeWM predictor in Equinox and the planning cost J(A, c)
    distributions  the policy's step distribution (Gaussian mixture) and reference log-densities
    policy         GPlanner, sampling, teacher-forced log-probs
    losses         the Trajectory Balance loss + diagnostics
    mala           MALA local search on plans (off-policy training samples)
    latents        (start, goal) latent pairs from precomputed embeddings, expert plans between them
    train          optimizer, jitted train step, training loop, step logging, random streams
    baselines      behavior cloning and amortized MAP planning with the same network
    ppo            PPO baseline: the same network as an RL policy, with a learned critic
    probe          headroom probe: gradient refinement of sampled plans through LeWM
    convert        weight conversion PyTorch <-> Equinox (imports torch)
    checkpoint     save / load JAX checkpoints, export to the PyTorch format evaluation loads
"""
