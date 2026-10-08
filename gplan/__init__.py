"""GPlan, PyTorch side: precompute and evaluation (training lives in `gplan_jax`).

Modules
    lewm        frozen LeWorldModel: loading, encoding, the reference planning cost J(A, c)
    data        NumPy dataset helpers shared with gplan_jax (valid start rows, STABLEWM_HOME)
    precompute  encode every frame of a dataset into latents
    policy      GPlaner (autoregressive Gaussian policy), Sampler, checkpoint I/O
    solvers     stable_worldmodel solvers: GFlowNet best-of-N, cost logging wrapper
    evaluation  LeWM evaluation protocol on TwoRoom (needs stable_worldmodel)
"""
