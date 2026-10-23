# The paper's experiments

`scripts/run.py` runs everything the paper needs for one task: it trains every model and ablation,
evaluates them and builds the tables. Training uses the 90% of episodes not held out, constant
learning rates, 3 seeds per model (ablations: 2); evaluation uses 4 eval seeds x 50 episodes.
Finished runs are skipped (a checkpoint's `.pt` is written last, a results file atomically), so it
can be rerun after an interruption.

```bash
python scripts/run.py task=tworoom                  # also pusht, reacher, cube
python scripts/run.py task=tworoom stage=train      # one stage: train, eval or tables
python scripts/run.py task=tworoom dry_run=true     # print the commands, run nothing
python scripts/run.py task=pusht "only=[ppo,ppo-maxent,ppo-cost,ppo-cost-maxent]"   # just these models
python scripts/run.py task=pusht "eval_overrides={device:cuda}" "seeds=[0]"
```

-> `outputs/<task>/<model>-s<seed>.{eqx,json,pt,yaml}`, `results/<task>/<evaluation>/*.json`,
`results/<task>/tables/<table>.{md,csv}` (also printed). Each run is a fresh `scripts/train.py` /
`scripts/evaluate.py` process; `dry_run=true` shows their overrides, which reproduce a single run by
hand.

## Configuration

| File | Holds |
| --- | --- |
| `configs/run.yaml` | the runner: models, seeds, ablations on/off, episode split, eval seeds and episodes, wandb mode, the two refinement objectives, extra settings for every run (`train_overrides`, `eval_overrides`) |
| `configs/task/<task>.yaml` | dataset, LeWM checkpoint, β, the GFlowNet schedule (`gfn`), what MALA adds (`mala`), PPO's length and policy (`ppo`), the baselines' length, the ablations (each a list of `train.py` overrides) |
| `configs/model/<model>.yaml` | a training method and its settings, taken from the task: `gfn`, `gfn-mala`, `bc`, `map`, `ppo`, `ppo-maxent` |
| `configs/eval/*.yaml` | the evaluations: their runs (results file name, models, policies, refinement objective, settings), episodes and table |
| `gplan/config.py` | every setting of `train.py` / `evaluate.py`, its type and default |

A task's settings can be changed for one invocation, e.g.
`python scripts/run.py task=pusht task.beta=120 out=outputs/pusht-b120 results=results/pusht-b120`
(the `task.*` overrides are passed on to `train.py`).

## Models

| Model | Trained as |
| --- | --- |
| `gfn` | the GFlowNet (TB) |
| `gfn-mala` | the GFlowNet + MALA plans as off-policy data (from the end of the beta warmup) |
| `bc` | behavior cloning on the expert actions (same network) |
| `map` | amortized MAP planning through LeWM (same network, one plan per condition) |
| `ppo` | PPO (clipped surrogate, GAE, learned critic) maximizing the plan return −βJ + log N(A): a reward maximizer |
| `ppo-maxent` | the same with the entropy term −log π in the reward (`ent_coef=1`): its optimum is the GFlowNet target |
| ablations | TwoRoom, Reacher: `gfn` at beta x 0.5 / x 2; PushT, Cube: `gfn-mala` at beta x 0.5 / x 2, one Gaussian instead of 7 (`gfn-k1`, `gfn-mala-k1`); every task: `ppo-cost` / `ppo-cost-maxent`, PPO on −βJ alone (classic / with the entropy term) |

TwoRoom and Reacher: β 90, warmup 1000, 20k batches, one Gaussian; PushT and Cube: β 192, warmup
30k, 60k batches, 7 components. bc and map: 20k batches.

## Tables

| Table | Content |
| --- | --- |
| `heldout` | the main comparison on held-out episodes: CEM (LeWM paper config), prior (best of 64 random plans), prior-refine, and per model `gflow` (best of 64 samples; map: its single plan) and `gflow-refine` (+ 20 gradient steps through LeWM); every refine policy twice, on βJ + ‖A‖²/2 (`obj=target`) and on J alone (`obj=cost`) |
| `alldata` | the same on every episode (the LeWM paper's protocol, comparable with its CEM numbers) |
| `compute_curve` | success vs world-model passes: GFlow N = 1 ... 256, refinement N x steps = 16 x 20, 64 x 5 / 20 / 50 (gfn, gfn-mala and prior starts), CEM samples x iterations |
| `ablations` | the ablations next to `gfn` and `gfn-mala` (`ablations=false` skips training and evaluating them) |

Every row pools training seeds (same label) and evaluation seeds: success with a Wilson 95% interval
and the spread across training seeds, plan cost, compute per replan, and a paired comparison with the
paper's CEM on the same episodes (exact McNemar p). A table takes its results files glob by glob, each
sorted.

## Reproducibility

- **Seeds.** `train.py seed=` derives independent random streams: the initial weights, the
  on-policy plans, MALA's proposals and accept decisions, PPO's critic weights (`gplan/jax/train.py`).
  `evaluate.py seed=` picks the episodes and seeds CEM's and GFlow's own random generators, so a
  policy's result does not depend on the others evaluated with it.
- **GPU determinism.** On CPU, reruns are bit-identical. On GPU use `deterministic=true` in
  `train.py` (`--xla_gpu_deterministic_ops`) and `evaluate.py` (deterministic cuDNN / cuBLAS).
- **Precision.** `train.py matmul_precision=highest` (default) keeps float32 matmuls exact,
  matching the PyTorch cost to round-off; `default` lets the GPU use TF32 (faster, J changes
  slightly). PyTorch runs the LeWM encoder's patch convolution in TF32 on recent GPUs by default,
  both in precompute and in evaluation, so latents and eval-time embeddings are consistent.
- The first training step includes JIT compilation (~10 s).
