# gplan

GFlowNet samplers as amortized planners for [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

A goal-conditioned autoregressive policy, `GPlaner`, learns to sample action plans from

    P*(A | z_start, z_goal)  ∝  exp(-β · J(A)) · N(A; 0, s² I)

where J is the frozen LeWM planning cost: the squared distance between the latent that LeWM
predicts after executing A and the goal latent. Each plan step is a diagonal Gaussian or a mixture
of them. The policy is trained with Trajectory Balance or VarGrad (optionally with MALA local
search as off-policy data) and used as an MPC planner (best-of-N under the LeWM cost), compared
against LeWM's CEM planner on the LeWM evaluation protocol.

Training runs in **JAX** (Equinox + Optax) on precomputed LeWM embeddings; precompute and
evaluation run in **PyTorch** (the original LeWM and `stable_worldmodel`).

## Layout

```
gplan/
  data.py            NumPy helpers shared by both sides (STABLEWM_HOME, valid start rows)
  torch/             PyTorch: everything that needs the original LeWM
    lewm.py            load, encode frames, planning cost J (reference implementation)
    precompute.py      encode every dataset frame into latents for training
    distributions.py   DiagGaussianMixture (twin of gplan/jax/distributions.py)
    policy.py          GPlaner + Sampler used at evaluation, checkpoint I/O
    solvers.py         GFlowNet best-of-N solver (+ test-time refinement), cost logging
    evaluation.py      the LeWM protocol: TwoRoom / PushT presets, normalizers, start sampling
  jax/               JAX: training
    lewm.py            the LeWM predictor in Equinox and J (checked against PyTorch)
    distributions.py   DiagGaussianMixture, normal / reference log-probs
    policy.py          GPlaner, sampling, teacher-forced step log-probs
    losses.py          TB and VarGrad on xi = log P_F + β'J − log N(A), diagnostics
    mala.py            MALA local search on plans (off-policy training samples)
    latents.py         (start, goal = start + k) pairs from the precomputed latents
    train.py           optimizer, jitted train step, training loop, random streams
    probe.py           headroom probe: gradient refinement through LeWM, expert plans
    convert.py         weights PyTorch <-> Equinox (LeWM predictor, GPlaner)
    checkpoint.py      save / load .eqx checkpoints, export the PyTorch .pt
scripts/             convert_lewm, encode_dataset, train, export_planner, evaluate, probe_headroom
tests/               test_torch_* / test_jax_* per module (JAX checked against PyTorch), test_distributions,
                     test_eval_e2e (evaluate.py on a tiny real swm setup), test_scripts (every CLI)
le-wm/               LeWM sources (git submodule)
```

The two subpackages are named after the framework they use; inside them `import torch` /
`import jax` still mean the real libraries. Run code from the repository root.

## Setup

```bash
git clone --recurse-submodules https://github.com/Pe4enkazAMI/gplan.git
cd gplan                                   # existing clone: git submodule update --init le-wm
pip install -e ".[eval,convert,test,jax-cuda]"   # on a CPU-only / macOS machine: .[...,jax]
export STABLEWM_HOME=~/.stable_worldmodel  # datasets (*.h5), LeWM checkpoints and latents live here
```

`le-wm` is a git submodule. Its `jepa.py` and `module.py` must be importable under those names
because LeWM checkpoints are pickled `jepa.JEPA` objects; importing `gplan.torch.lewm` handles that.

### DGX Spark / GB10

The GB10 is aarch64 with CUDA 13; `jax[cuda13]` ships wheels for it (driver ≥ 580). CPU and GPU
share memory, and JAX preallocates 75% of it by default, so cap it:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
python -c "import jax; print(jax.devices())"   # should list a CUDA device
```

Run precompute and evaluation (PyTorch) and training (JAX) as separate processes.

## Workflow

```bash
# 1. LeWM weights from the Hugging Face Hub -> $STABLEWM_HOME/tworoom/lewm_object.ckpt
python scripts/convert_lewm.py --repo quentinll/lewm-tworooms --name tworoom/lewm

# 2. Encode every dataset frame once (PyTorch, GPU) -> $STABLEWM_HOME/tworoom_latents.{npy,json} + _meta.npz
python scripts/encode_dataset.py --dataset tworoom

# 3. Train (JAX). β is per embedding dim: the raw cost is scaled by β / 192.
#    Writes outputs/tb-beta120.eqx (JAX weights), .json (config) and .pt (PyTorch copy for eval)
python scripts/train.py --beta 120 --wandb-name tb-beta120 --out outputs/tb-beta120

# 4. Evaluate against CEM on the same 50 TwoRoom episodes (PyTorch); episodes are drawn
#    exactly as le-wm/eval.py draws them, so CEM numbers are comparable to the LeWM paper
python scripts/evaluate.py --policy gflow cem --planner outputs/tb-beta120.pt

# PushT: the same steps with its own checkpoint and dataset
python scripts/convert_lewm.py --repo quentinll/lewm-pusht --name pusht/lewm
python scripts/encode_dataset.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt
python scripts/train.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt --beta 192 \
    --beta-warmup 10000 --n-batches 40000 --out outputs/pusht
python scripts/evaluate.py --task pusht --policy gflow cem --planner outputs/pusht.pt
```

### Evaluation

Every policy has its own settings, so CEM always runs the LeWM paper configuration
(`--cem-samples 300 --cem-steps 30 --cem-topk 30`) whatever GFlow checkpoint or flags sit next to it.

| Flag | Meaning |
| --- | --- |
| `--policy gflow gflow-refine cem random` | policies, run in this order on the same episodes |
| `--gflow-samples N` (64) | plans drawn from the GPlaner per replanning step |
| `--gflow-select min-cost \| sample \| mean` | execute the cheapest of N, one sample, or the most likely mode |
| `--refine-steps / --refine-lr / --refine-objective` | `gflow-refine`: Adam steps on the actions through LeWM before picking the cheapest (a diagnostic, not the amortized planner) |
| `--goal-offset 25 --eval-budget 50 --num-eval 50 --seed 42` | the LeWM protocol |
| `--results-json PATH` | every policy's metrics and per-episode successes, for sweeps over seeds |
| `--deterministic` | deterministic cuDNN / cuBLAS kernels: bit-identical GPU reruns (slower) |

The table reports success, the LeWM cost of the executed plans (recomputed the same way for every
solver; per replan in wandb / json), the largest executed |action| in z-scored units (values well
above ~3 are outside LeWM's training data), and compute per replanning step as world-model passes:
forward rollouts (`fwd/replan`) and backward passes (`bwd/replan`, `gflow-refine` only, each ~2x a
forward). The pairwise lines count episodes solved by both / only one / neither policy.

### Experiments

```bash
# mixture policy (K Gaussians per plan step) + MALA local search as off-policy data:
# every --ls-every steps from --ls-start on, --ls-steps MALA steps from each on-policy plan toward the
# target (gradients through LeWM, step size adapted to ~57% acceptance). wandb: policy/weight_max (-> 1:
# components collapsed), ls/accept, ls/cost_gain (-> 0 when the sampler matches its target), ls/move
python scripts/train.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt --beta 192 \
    --n-components 4 --ls-steps 10 --ls-every 4 --ls-start 10000 --out outputs/pusht-mog-mala

# test-time refinement next to best-of-N and CEM
python scripts/evaluate.py --task pusht --policy gflow gflow-refine cem --planner outputs/pusht.pt

# headroom probe (no env): LeWM cost of best-of-64 samples before / after gradient refinement,
# next to prior shooting and the dataset's expert actions
python scripts/probe_headroom.py outputs/pusht

# re-export a JAX checkpoint to PyTorch, if needed
python scripts/export_planner.py outputs/pusht
```

## Reproducibility

- **Seeds.** `train.py --seed` derives two independent streams: one for the initial weights, one
  for everything sampled in training (plans, MALA). `evaluate.py --seed` picks the episodes and
  seeds CEM's and GFlow's own random generators, so a policy's result does not depend on the
  others evaluated with it.
- **GPU determinism.** On CPU, reruns are bit-identical. On GPU use `--deterministic` in
  `train.py` (`--xla_gpu_deterministic_ops`) and `evaluate.py` (deterministic cuDNN / cuBLAS).
- **Precision.** `train.py --matmul-precision highest` (default) keeps float32 matmuls exact,
  matching the PyTorch cost to round-off; `default` lets the GPU use TF32 (faster, J changes
  slightly). PyTorch runs the LeWM encoder's patch convolution in TF32 on recent GPUs by default,
  both in precompute and in evaluation, so latents and eval-time embeddings are consistent.
- The first training step includes JIT compilation (~10 s).

## Tests

```bash
pytest                    # everything, ~7 min on CPU
pytest -m "not slow"      # skips end-to-end runs and training to convergence, ~1.5 min
ruff check .              # pyflakes + import order
```

JAX code is checked against the PyTorch code on identical weights (cost, policy, losses and
gradients), the PyTorch probability code against `torch.distributions`, training against
closed-form optima, and `scripts/evaluate.py` end to end on a tiny real `stable_worldmodel` setup
(CEM identical whatever GFlow checkpoint runs next to it, every policy reproducible).

The losses take the plan costs as data, and `step_log_probs` scores any plan, so new plan
sources (expert segments, replay, local search) plug into `gplan/jax/train.py` without touching
the policy or the losses.
