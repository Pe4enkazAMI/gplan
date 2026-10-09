# gplan

GFlowNet samplers as amortized planners for [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

A goal-conditioned autoregressive policy, `GPlaner`, learns to sample action plans from

    P*(A | z_start, z_goal)  ∝  exp(-β · J(A)) · N(A; 0, s² I)

where J is the frozen LeWM planning cost: the squared distance between the latent that LeWM
predicts after executing A and the goal latent. It is trained with Trajectory Balance or VarGrad
and used as an MPC planner (best-of-N under the LeWM cost), compared against LeWM's CEM planner.

On this branch training runs in **JAX** (Equinox + Optax) on precomputed LeWM embeddings;
precompute and evaluation stay in **PyTorch** (the LeWM encoder and `stable_worldmodel`).

## Setup

```bash
git clone --recurse-submodules https://github.com/Pe4enkazAMI/gplan.git
cd gplan                                   # existing clone: git submodule update --init le-wm
pip install -e ".[eval,convert,test,jax-cuda]"   # on a CPU-only / macOS machine: .[...,jax]
export STABLEWM_HOME=~/.stable_worldmodel  # datasets (*.h5), LeWM checkpoints and latents live here
```

`le-wm` is a git submodule. Its `jepa.py` and `module.py` must be importable under those names
because LeWM checkpoints are pickled `jepa.JEPA` objects; importing `gplan.lewm` handles that.

### DGX Spark / GB10

The GB10 is aarch64 with CUDA 13; `jax[cuda13]` ships wheels for it (driver ≥ 580). CPU and GPU
share memory, and JAX preallocates 75% of it by default, so cap it:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
python -c "import jax; print(jax.devices())"   # should list a CUDA device
```

Run precompute and evaluation (PyTorch) and training (JAX) as separate processes.

## Usage

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

# PushT: same steps with its own checkpoint and dataset
python scripts/convert_lewm.py --repo quentinll/lewm-pusht --name pusht/lewm
python scripts/encode_dataset.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt
python scripts/train.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt --beta 90 --out outputs/pusht
python scripts/evaluate.py --task pusht --policy gflow cem --planner outputs/pusht.pt

# test-time refinement diagnostic (env): best-of-64 vs the same samples after 20 Adam steps through LeWM
python scripts/evaluate.py --task pusht --policy gflow gflow-refine cem --planner outputs/pusht.pt

# headroom probe (no env): LeWM cost of best-of-64 samples before / after gradient refinement,
# next to prior shooting and the dataset's expert actions
python scripts/probe_headroom.py outputs/tb-beta120

# re-export a JAX checkpoint to PyTorch, if needed
python scripts/export_planner.py outputs/tb-beta120

# tests (CPU, ~1-2 min)
pytest
```

`--matmul-precision highest` (default) keeps float32 matmuls exact, matching the PyTorch cost to
round-off; `default` lets the GPU use TF32, which is faster but changes J slightly. The first
training step includes JIT compilation (~10 s).

## Layout

| Path | Contents |
| --- | --- |
| `gplan_jax/lewm.py` | LeWM predictor in Equinox (action encoder, AdaLN transformer, projector) and the cost J(A, c) |
| `gplan_jax/policy.py` | `GPlaner`, `rollout`, teacher-forced `step_log_probs`, reference log-prob |
| `gplan_jax/losses.py` | TB and VarGrad on `xi = log P_F + βJ − log N(A)`, diagnostics |
| `gplan_jax/train.py` | Optimizer (clip + Adam, separate lr for log Z), jitted train step, training loop |
| `gplan_jax/data.py` | (start, goal = start + 25) pairs from the precomputed latents |
| `gplan_jax/convert.py` | Weights PyTorch ↔ Equinox (LeWM predictor, GPlaner) |
| `gplan_jax/checkpoint.py` | Save / load `.eqx` checkpoints, export the PyTorch `.pt` |
| `gplan_jax/probe.py` | Headroom probe: gradient refinement through LeWM, expert plans from dataset actions |
| `gplan/lewm.py` | PyTorch LeWM: load, `encode`, `lewm_cost` (reference implementation) |
| `gplan/precompute.py` | Encode a dataset's frames into latents |
| `gplan/policy.py` | PyTorch `GPlaner` + `Sampler`, used by evaluation |
| `gplan/solvers.py`, `gplan/evaluation.py` | `stable_worldmodel` solvers and the LeWM evaluation protocol; `TASKS` holds the TwoRoom / PushT presets |
| `gplan/data.py` | NumPy helpers shared by both sides (valid start rows, `STABLEWM_HOME`) |
| `scripts/` | `encode_dataset.py`, `train.py`, `export_planner.py`, `evaluate.py`, `probe_headroom.py`, `convert_lewm.py` |
| `tests/` | JAX vs PyTorch parity (cost, policy, losses and gradients), closed-form TB/VarGrad optimum, end to end |

The losses take the plan costs as data, and `step_log_probs` scores any plan, so new plan
sources (expert segments, replay, local search) plug into `gplan_jax/train.py` without touching
the policy or the losses.
