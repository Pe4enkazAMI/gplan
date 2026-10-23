# gplan

GFlowNet samplers as amortized planners for [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

A goal-conditioned autoregressive policy, `GPlanner`, learns to sample action plans from

    P*(A | z_start, z_goal)  ∝  exp(-β · J(A)) · N(A; 0, s² I)

where J is the frozen LeWM planning cost: the squared distance between the latent that LeWM
predicts after executing A and the goal latent. Each plan step is a diagonal Gaussian or a mixture
of them. The policy is trained with Trajectory Balance (optionally with MALA local
search as off-policy data) and used as an MPC planner (best-of-N under the LeWM cost), compared
against LeWM's CEM planner on the LeWM evaluation protocol.

Training runs in **JAX** (Equinox + Optax) on precomputed LeWM embeddings; precompute and
evaluation run in **PyTorch** (the original LeWM and `stable_worldmodel`).

More: [docs/experiments.md](docs/experiments.md) (the paper's runs, models, tables, configuration),
[docs/evaluation.md](docs/evaluation.md) (the evaluation protocol, policies and metrics).

## Layout

```
gplan/
  config.py          settings of train.py / evaluate.py (types, defaults), composed with Hydra from configs/
  data.py            NumPy helpers shared by both sides (STABLEWM_HOME, valid start rows, episode split)
  results.py         pool evaluate.py result files: Wilson intervals, paired McNemar tests
  torch/             PyTorch: everything that needs the original LeWM (precompute, evaluation)
    lewm.py            load, encode frames, planning cost J (reference implementation)
    precompute.py      encode every dataset frame into latents for training
    distributions.py   DiagGaussianMixture sampling (the policy's step distribution at evaluation)
    policy.py          GPlanner + Sampler used at evaluation, PriorSampler (baselines), checkpoint I/O
    solvers.py         GFlowNet best-of-N solver (+ test-time refinement), cost logging
    evaluation.py      the LeWM protocol: TwoRoom / PushT / Reacher / Cube presets, normalizers, start sampling,
                       the checks a planner's training protocol must pass
  jax/               JAX: training
    lewm.py            the LeWM predictor in Equinox and J (checked against PyTorch)
    distributions.py   DiagGaussianMixture, normal / reference log-probs
    policy.py          GPlanner, sampling, teacher-forced step log-probs
    losses.py          Trajectory Balance on xi = log P_F + β'J − log N(A), diagnostics
    mala.py            MALA local search on plans (off-policy training samples)
    baselines.py       behavior cloning and amortized MAP planning with the same network
    ppo.py             PPO baseline: the same network as an RL policy, PPO-clip + GAE + a learned critic
    latents.py         (start, goal) pairs from the precomputed latents, expert plans between them
    train.py           optimizer, jitted train step, training loop, step logging, random streams
    probe.py           headroom probe: gradient refinement of sampled plans through LeWM
    convert.py         weights PyTorch <-> Equinox (LeWM predictor, GPlanner)
    checkpoint.py      save / load .eqx checkpoints, export the PyTorch .pt
configs/             Hydra configs: train.yaml, evaluate.yaml, run.yaml; task/ (per task), model/ (per
                     training method), eval/ (the paper's evaluations)
scripts/             train, evaluate, run (the paper's runs for one task); utilities: convert_lewm,
                     download_data, encode_dataset, aggregate_results, export_planner, probe_headroom
tests/               test_torch_* / test_jax_* per module (JAX checked against PyTorch), test_distributions,
                     test_jax_optimum / test_jax_baselines / test_jax_ppo (closed-form optima), test_config,
                     test_results, test_eval_e2e (evaluate.py on tiny datasets recorded with the real envs, all
                     four tasks), test_scripts (every CLI), test_download_data
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
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand) or smaller fractions. `scripts/run.py`
sets it for the processes it starts.

## Settings

`train.py`, `evaluate.py` and `run.py` take [Hydra](https://hydra.cc) overrides, `name=value`. Every
setting, its type and default is in `gplan/config.py` (`TrainArgs`, `EvalArgs`); `configs/` composes
them: `task=<name>` picks `configs/task/<name>.yaml` (dataset, LeWM checkpoint and the paper's
settings for the task) and, for training, `model=<name>` a training method with those settings
(`configs/model`). Lists are written `"policy=[gflow,cem]"` (quoted for the shell). `--help` prints a
script's settings as composed with the given overrides:

```bash
python scripts/train.py task=pusht model=gfn-mala --help
```

A setting that does not exist, or a value of the wrong type, is an error. Hydra is used for
composition only: no output folders, no change of the working directory or of logging. Training
writes its composed config next to the checkpoint (`<out>.yaml`); an evaluation's settings are in
its results file (`args`). The utilities (`encode_dataset.py`, `download_data.py`, ...) keep plain
`--flags`.

## Workflow

```bash
# 1. LeWM weights from the Hugging Face Hub -> $STABLEWM_HOME/tworoom/lewm_object.ckpt
python scripts/convert_lewm.py --repo quentinll/lewm-tworooms --name tworoom/lewm

# 2. Encode every dataset frame once (PyTorch, GPU) -> $STABLEWM_HOME/tworoom_latents.{npy,json} + _meta.npz
python scripts/encode_dataset.py --dataset tworoom

# 3. Train (JAX). β is per embedding dim: the raw cost is scaled by β / 192.
#    Writes outputs/tb-beta120.eqx (JAX weights), .json (config), .pt (PyTorch copy for eval), .yaml
python scripts/train.py beta=120 wandb_name=tb-beta120 out=outputs/tb-beta120

# 4. Evaluate against CEM on the same 50 TwoRoom episodes (PyTorch); episodes are drawn
#    exactly as le-wm/eval.py draws them, so CEM numbers are comparable to the LeWM paper
python scripts/evaluate.py "policy=[gflow,cem]" planner=outputs/tb-beta120.pt

# PushT: the same steps with its own checkpoint and dataset; task=pusht picks both, and model=gfn the
# paper's GFlowNet settings for PushT (configs/task/pusht.yaml, configs/model/gfn.yaml)
python scripts/convert_lewm.py --repo quentinll/lewm-pusht --name pusht/lewm
python scripts/encode_dataset.py --dataset pusht_expert_train --ckpt $STABLEWM_HOME/pusht/lewm_object.ckpt
python scripts/train.py task=pusht model=gfn out=outputs/pusht
python scripts/evaluate.py task=pusht "policy=[gflow,cem]" planner=outputs/pusht.pt

# The paper's runs for one task: every model x 3 seeds, the ablations, every evaluation, the tables
python scripts/run.py task=pusht
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
python scripts/evaluate.py task=reacher "policy=[cem]" num_eval=2   # quick check of the env and data

python scripts/convert_lewm.py --repo quentinll/lewm-cube --name cube/lewm
python scripts/download_data.py --task cube               # -> $STABLEWM_HOME/ogbench/cube_single_expert.h5
python scripts/encode_dataset.py --dataset ogbench/cube_single_expert --ckpt $STABLEWM_HOME/cube/lewm_object.ckpt
python scripts/evaluate.py task=cube "policy=[cem]" num_eval=2
```

`train.py` reads the plan-step size from the data (`frameskip` 5 x the dataset's action dim) and
stops if LeWM's action encoder expects another size. Reacher starts from TwoRoom's GFlowNet
settings and Cube from PushT's (`configs/task`).

### More experiments

```bash
# mixture policy (K Gaussians per plan step) + MALA local search as off-policy data:
# every ls_every steps from ls_start on, ls_steps MALA steps from each on-policy plan toward the
# target (gradients through LeWM, step size adapted to ~57% acceptance). wandb: policy/weight_max (-> 1:
# components collapsed), ls/accept, ls/cost_gain (-> 0 when the sampler matches its target), ls/move
python scripts/train.py task=pusht model=gfn n_components=4 ls_steps=10 ls_every=4 ls_start=10000 \
    out=outputs/pusht-mog-mala

# test-time refinement next to best-of-N and CEM
python scripts/evaluate.py task=pusht "policy=[gflow,gflow-refine,cem]" planner=outputs/pusht.pt

# headroom probe (no env): LeWM cost of best-of-64 samples before / after gradient refinement,
# next to prior shooting and the dataset's expert actions
python scripts/probe_headroom.py outputs/pusht

# re-export a JAX checkpoint to PyTorch, if needed
python scripts/export_planner.py outputs/pusht
```

## Tests

```bash
pytest                    # everything, ~17 min on CPU (Reacher / Cube tests need the [mujoco] extra, else skipped)
pytest -m "not slow"      # skips end-to-end runs and training to convergence, ~3 min
ruff check .              # pyflakes + import order
```

The tests run on CPU, also on a GPU machine: `tests/conftest.py` hides the GPU from JAX and
PyTorch for the test session and every script it starts, and turns off JAX's memory
pre-allocation. (The script and end-to-end tests start several processes; with pre-allocation each
JAX process would reserve 75% of GPU memory, which on the GB10's shared memory freezes the machine.)
`GPLAN_TESTS_ON_GPU=1 pytest` runs on the GPU anyway, allocating on demand. The largest test
process peaks at about 2.6 GB.

JAX code is checked against the PyTorch code on identical weights (cost, policy, losses and
gradients), the JAX densities against `torch.distributions`, the losses against the
closed-form optimum (exactly 0 there for any plan, `tests/test_jax_optimum.py`), training against
that optimum from a random init (TB, with and without MALA), and `scripts/evaluate.py` end to end on
a tiny real `stable_worldmodel` setup (CEM identical whatever GFlow checkpoint runs next to it, every
policy reproducible).

The losses take the plan costs as data, and `step_log_probs` scores any plan, so new plan
sources (expert segments, replay, local search) plug into `gplan/jax/train.py` without touching
the policy or the losses.
