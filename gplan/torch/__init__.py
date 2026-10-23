"""PyTorch side: precompute and evaluation (needs the original LeWM and stable_worldmodel).

Modules
    lewm           frozen LeWorldModel: loading, encoding frames, the planning cost J(A, c)
    precompute     encode every frame of a dataset into latents for training
    distributions  the policy's step distribution (sampling)
    policy         GPlanner (autoregressive Gaussian / Gaussian-mixture policy), Sampler, checkpoint I/O
    solvers        stable_worldmodel solvers: GFlowNet best-of-N (+ test-time refinement), cost logging
    evaluation     the LeWM evaluation protocol (TwoRoom, PushT, Reacher, Cube): tasks, normalizers, start
                   sampling, and the checks a planner's training protocol must pass
"""
