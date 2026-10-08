"""GPlan: GFlowNet samplers as amortized planners for LeWorldModel.

Modules
    lewm        frozen LeWorldModel: loading, encoding, the planning cost J(A, c)
    data        (start, goal) frame pairs from the LeWM .h5 datasets
    policy      GPlaner (autoregressive Gaussian policy), Sampler, checkpoint I/O
    losses      Trajectory Balance and VarGrad losses + their diagnostics
    trainer     the optimization loop
    solvers     stable_worldmodel solvers: GFlowNet best-of-N, cost logging wrapper
    evaluation  LeWM evaluation protocol on TwoRoom (needs stable_worldmodel)
"""
