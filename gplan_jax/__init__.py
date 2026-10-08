"""JAX (Equinox + Optax) training for GPlan.

Modules
    lewm      LeWM predictor in Equinox and the planning cost J(A, c)
    policy    GPlaner, sampling, teacher-forced log-probs
    losses    Trajectory Balance and VarGrad losses + diagnostics
    data      (start, goal) latent pairs from precomputed embeddings
    train     optimizer, jitted train step, training loop
    convert   weight conversion PyTorch <-> Equinox (imports torch)
    checkpoint  save / load JAX checkpoints, export to the PyTorch format used by evaluation
"""
