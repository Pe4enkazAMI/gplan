# gplan

GFlowNet samplers as amortized planners for [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

A goal-conditioned autoregressive policy, `GPlaner`, learns to sample action plans from

    P*(A | z_start, z_goal)  ∝  exp(-β · J(A)) · N(A; 0, s² I)

where J is the frozen LeWM planning cost: the squared distance between the latent that LeWM
predicts after executing A and the goal latent. Each plan step is a diagonal Gaussian or a mixture
of them. The policy is trained with Trajectory Balance (optionally with MALA local
search as off-policy data) and used as an MPC planner (best-of-N under the LeWM cost), compared
against LeWM's CEM planner on the LeWM evaluation protocol.

Training runs in **JAX** (Equinox + Optax) on precomputed LeWM embeddings; precompute and
evaluation run in **PyTorch** (the original LeWM and `stable_worldmodel`).

## Layout

```
gplan/
  data.py            NumPy helpers shared by both sides (STABLEWM_HOME, valid start rows, episode split)
  results.py         pool evaluate.py result files: Wilson intervals, paired McNemar tests
  torch/             PyTorch: everything that needs the original LeWM
    lewm.py            load, encode frames, planning cost J (reference implementation)
    precompute.py      encode every dataset frame into latents for training
    distributions.py   DiagGaussianMixture (twin of gplan/jax/distributions.py)
    policy.py          GPlaner + Sampler used at evaluation, PriorSampler (baselines), checkpoint I/O
    solvers.py         GFlowNet best-of-N solver (+ test-time refinement), cost logging
    evaluation.py      the LeWM protocol: TwoRoom / PushT / Reacher / Cube presets, normalizers, start sampling
  jax/               JAX: training
    lewm.py            the LeWM predictor in Equinox and J (checked against PyTorch)
    distributions.py   DiagGaussianMixture, normal / reference log-probs
    policy.py          GPlaner, sampling, teacher-forced step log-probs
    losses.py          Trajectory Balance on xi = log P_F + β'J − log N(A), diagnostics
    mala.py            MALA local search on plans (off-policy training samples)
    baselines.py       behavior cloning and amortized MAP planning with the same network
    ppo.py             PPO baseline: the same network as an RL policy, PPO-clip + GAE + a learned critic
    latents.py         (start, goal) pairs from the precomputed latents, expert plans between them
    train.py           optimizer, jitted train step, training loop, random streams
    probe.py           headroom probe: gradient refinement of sampled plans through LeWM
    convert.py         weights PyTorch <-> Equinox (LeWM predictor, GPlaner)
    checkpoint.py      save / load .eqx checkpoints, export the PyTorch .pt
scripts/             convert_lewm, download_data, encode_dataset, train, export_planner, evaluate,
                     aggregate_results, probe_headroom
experiments/         train.sh, eval.sh, tables.sh: the paper's runs and tables for one task (config.sh)
tests/               test_torch_* / test_jax_* per module (JAX checked against PyTorch), test_distributions,
                     test_jax_optimum / test_jax_baselines / test_jax_ppo (closed-form optima), test_results,
                     test_eval_e2e (evaluate.py on tiny datasets recorded with the real envs, all four
                     tasks), test_scripts (every CLI), test_download_data
le-wm/               LeWM sources (git submodule)
```

The two subpackages are named after the framework they use; inside them `import torch` /
`import jax` still mean the real libraries. Run code from the repository root.

## Setup

```bash
git clone --recurse-submodules https://github.com/Pe4enkazAMI/gplan.git
cd gplan                                   # existing clone: git submodule update --init le-wm
pip install -e ".[eval,convert,test,jax-cuda]"   # on a CPU-only / macOS machine: .[...,jax]
pip install -e ".[mujoco]"                        # Reacher and Cube (MuJoCo, DMControl, OGBench)
export STABLEWM_HOME=~/.stable_worldmodel  # datasets (*.h5), LeWM checkpoints and latents live here
```

`le-wm` is a git submodule. Its `jepa.py` and `module.py` must be importable under those names
because LeWM checkpoints are pickled `jepa.JEPA` objects; importing `gplan.torch.lewm` handles that.

Reacher and Cube are MuJoCo envs, rendered offscreen. On Linux evaluation sets `MUJOCO_GL=egl`
(as `le-wm/eval.py` does), which needs the NVIDIA EGL driver (present on a GPU machine); set
`MUJOCO_GL=osmesa` yourself for CPU-only rendering. `dm_control` depends on `labmaze`, which has no
wheel for Python >= 3.13 and is not used here: on 3.13 install with
`pip install mujoco==3.16.0 && pip install --no-deps dm_control==1.0.49 ogbench==1.2.1 && pip install absl-py dm-env dm-tree glfw lxml pyopengl pyparsing scipy`.

### DGX Spark / GB10

The GB10 is aarch64 with CUDA 13; `jax[cuda13]` ships wheels for it (driver ≥ 580). CPU and GPU
share memory, and JAX preallocates 75% of it by default, so cap it:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
python -c "import jax; print(jax.devices())"   # should list a CUDA device
```

Run precompute and evaluation (PyTorch) and training (JAX) as separate processes. Each JAX
process reserves its fraction of the shared memory when it starts, so two JAX processes at 0.5
take all of it (the OS included) and can freeze the machine; to run several at once, use
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand) or smaller fractions.

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

### Reacher and Cube

Same steps; `download_data.py` fetches the dataset archive from the LeWM collection on the Hugging
Face Hub, unpacks it to where the presets expect it and checks its columns (Reacher 24 GB, Cube 46 GB
to download, about as much again while unpacking; the archive is deleted afterwards).

| task | env | dataset | LeWM weights | actions (plan step) |
| --- | --- | --- | --- | --- |
| `reacher` | `swm/ReacherDMControl-v0`, joint-angle goal | `dmc/reacher_random` (random policy) | `quentinll/lewm-reacher` | 2-d (10) |
| `cube` | `swm/OGBCube-v0`, one cube, pose goal | `ogbench/cube_single_expert` | `quentinll/lewm-cube` | 5-d (25) |

```bash
python scripts/convert_lewm.py --repo quentinll/lewm-reacher --name reacher/lewm
python scripts/download_data.py --task reacher            # -> $STABLEWM_HOME/dmc/reacher_random.h5
python scripts/encode_dataset.py --dataset dmc/reacher_random --ckpt $STABLEWM_HOME/reacher/lewm_object.ckpt
python scripts/evaluate.py --task reacher --policy cem --num-eval 2   # quick check of the env and data

python scripts/convert_lewm.py --repo quentinll/lewm-cube --name cube/lewm
python scripts/download_data.py --task cube               # -> $STABLEWM_HOME/ogbench/cube_single_expert.h5
python scripts/encode_dataset.py --dataset ogbench/cube_single_expert --ckpt $STABLEWM_HOME/cube/lewm_object.ckpt
python scripts/evaluate.py --task cube --policy cem --num-eval 2
```

`train.py` reads the plan-step size from the data (`--frameskip` 5 x the dataset's action dim) and
stops if LeWM's action encoder expects another size. The experiment scripts take `reacher` / `cube`
like the other tasks (`bash experiments/train.sh cube`); Reacher starts from TwoRoom's GFlowNet
settings and Cube from PushT's (`experiments/config.sh`).

### Evaluation

Every policy has its own settings, so CEM always runs the LeWM paper configuration
(`--cem-samples 300 --cem-steps 30 --cem-topk 30`) whatever GFlow checkpoint or flags sit next to it.

| Flag | Meaning |
| --- | --- |
| `--policy gflow gflow-refine cem prior prior-refine random` | policies, run in this order on the same episodes |
| `--gflow-samples N` (64) | plans drawn from the GPlaner per replanning step |
| `--gflow-select min-cost \| sample \| mean` | execute the cheapest of N, one sample, or the most likely mode |
| `--refine-steps / --refine-lr / --refine-objective` | `gflow-refine` / `prior-refine`: Adam steps on the actions through LeWM before picking the cheapest |
| `--prior-samples N` (64) `--prior-var` (1) | `prior`: best-of-N random shooting from N(0, s² I); `prior-refine`: the same plans refined like `gflow-refine` (the control for it) |
| `--holdout-frac f --split-seed s` | draw episodes only from the episodes `train.py` held out with the same flags; a planner that trained on them is refused (default 0: every episode, the LeWM protocol) |
| `--label NAME` | name of the planner config in the results file; `aggregate_results.py` pools runs with the same label |
| `--goal-offset 25 --eval-budget 50 --num-eval 50 --seed 42` | the LeWM protocol |
| `--results-json PATH` | every policy's metrics and per-episode successes, for sweeps over seeds |
| `--deterministic` | deterministic cuDNN / cuBLAS kernels: bit-identical GPU reruns (slower) |

The table reports success, the LeWM cost of the executed plans (recomputed the same way for every
solver; per replan in wandb / json), the largest executed |action| in z-scored units (values well
above ~3 are outside LeWM's training data), and compute per replanning step as world-model passes:
forward rollouts (`fwd/replan`) and backward passes (`bwd/replan`, refine policies only, each ~2x a
forward). The pairwise lines count episodes solved by both / only one / neither policy.

### Paper protocol: `experiments/`

Three scripts run everything the paper needs for one task. Training uses the 90% of episodes not held
out, constant learning rates, 3 seeds per model (ablations: 2); evaluation uses 4 eval seeds x 50
episodes. Both scripts skip what already exists, so they can be rerun after an interruption. The models,
MALA settings and ablations of each task are in `experiments/config.sh`.

```bash
bash experiments/train.sh tworoom    # -> outputs/tworoom/<model>-s<seed>.{eqx,json,pt}; also pusht, reacher, cube
bash experiments/eval.sh tworoom     # -> results/tworoom/<experiment>/*.json
bash experiments/tables.sh tworoom   # -> results/tworoom/tables/<experiment>.md / .csv, also printed
```

| Model | Trained as |
| --- | --- |
| `gfn` | the GFlowNet (TB) |
| `gfn-mala` | the GFlowNet + MALA plans as off-policy data (from the end of the beta warmup) |
| `bc` | behavior cloning on the expert actions (same network) |
| `map` | amortized MAP planning through LeWM (same network, one plan per condition) |
| `ppo` | PPO (clipped surrogate, GAE, learned critic) maximizing the plan return −βJ + log N(A): a reward maximizer |
| `ppo-maxent` | the same with the entropy term −log π in the reward (`--ent-coef 1`): its optimum is the GFlowNet target |
| ablations | TwoRoom, Reacher: `gfn` at beta x 0.5 / x 2; PushT, Cube: `gfn-mala` at beta x 0.5 / x 2, one Gaussian instead of 7 (`gfn-k1`, `gfn-mala-k1`); every task: `ppo-cost` / `ppo-cost-maxent`, PPO on −βJ alone (classic / with the entropy term) |

| Table | Content |
| --- | --- |
| `heldout` | the main comparison on held-out episodes: CEM (LeWM paper config), prior (best of 64 random plans), prior-refine, and per model `gflow` (best of 64 samples; map: its single plan) and `gflow-refine` (+ 20 gradient steps through LeWM); every refine policy twice, on βJ + ‖A‖²/2 (`obj=target`) and on J alone (`obj=cost`) |
| `alldata` | the same on every episode (the LeWM paper's protocol, comparable with its CEM numbers) |
| `compute_curve` | success vs world-model passes: GFlow N = 1 ... 256, refinement N x steps = 16 x 20, 64 x 5 / 20 / 50 (gfn, gfn-mala and prior starts), CEM samples x iterations |
| `ablations` | the ablations next to `gfn` and `gfn-mala` (`ABLATIONS=0` skips them) |

Every row pools training seeds (same label) and evaluation seeds: success with a Wilson 95% interval
and the spread across training seeds, plan cost, compute per replan, and a paired comparison with the
paper's CEM on the same episodes (exact McNemar p).

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
pytest                    # everything, ~17 min on CPU (Reacher / Cube tests need the [mujoco] extra, else skipped)
pytest -m "not slow"      # skips end-to-end runs and training to convergence, ~2.5 min
ruff check .              # pyflakes + import order
```

The tests run on CPU, also on a GPU machine: `tests/conftest.py` hides the GPU from JAX and
PyTorch for the test session and every script it starts, and turns off JAX's memory
pre-allocation. (The script and end-to-end tests start several processes; with pre-allocation each
JAX process would reserve 75% of GPU memory, which on the GB10's shared memory freezes the machine.)
`GPLAN_TESTS_ON_GPU=1 pytest` runs on the GPU anyway, allocating on demand. The largest test
process peaks at about 2.6 GB.

JAX code is checked against the PyTorch code on identical weights (cost, policy, losses and
gradients), the PyTorch probability code against `torch.distributions`, the losses against the
closed-form optimum (exactly 0 there for any plan, `tests/test_jax_optimum.py`), training against
that optimum from a random init (TB, with and without MALA), and `scripts/evaluate.py` end to end on a tiny real `stable_worldmodel` setup
(CEM identical whatever GFlow checkpoint runs next to it, every policy reproducible).

The losses take the plan costs as data, and `step_log_probs` scores any plan, so new plan
sources (expert segments, replay, local search) plug into `gplan/jax/train.py` without touching
the policy or the losses.
