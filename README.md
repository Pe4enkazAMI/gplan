# gplan

GFlowNet samplers as amortized planners for [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

A goal-conditioned autoregressive policy, `GPlaner`, learns to sample action plans from

    P*(A | z_start, z_goal)  ∝  exp(-β · J(A)) · N(A; 0, s² I)

where J is the frozen LeWM planning cost: the squared distance between the latent that LeWM
predicts after executing A and the goal latent. It is trained with Trajectory Balance or VarGrad
and used as an MPC planner (best-of-N under the LeWM cost), compared against LeWM's CEM planner.

## Setup

```bash
git clone --recurse-submodules https://github.com/Pe4enkazAMI/gplan.git
cd gplan                                   # existing clone: git submodule update --init le-wm
pip install -e ".[eval,convert,test]"
export STABLEWM_HOME=~/.stable_worldmodel  # datasets (*.h5) and LeWM checkpoints live here
```

`le-wm` is a git submodule. Its `jepa.py` and `module.py` must be importable under those names
because LeWM checkpoints are pickled `jepa.JEPA` objects; importing `gplan.lewm` handles that.

## Usage

```bash
# 1. LeWM weights from the Hugging Face Hub -> $STABLEWM_HOME/tworoom/lewm_object.ckpt
python scripts/convert_lewm.py --repo quentinll/lewm-tworooms --name tworoom/lewm

# 2. Train the planner (β is per embedding dim: the raw cost is scaled by β / 192)
python scripts/train.py --beta 120 --wandb-name tb-beta120 --out outputs/tb-beta120.pt

# 3. Evaluate against CEM on the same 50 TwoRoom episodes
python scripts/evaluate.py --policy gflow cem --planner outputs/tb-beta120.pt

# tests (CPU, ~10 s)
pytest
```

## Layout

| Path | Contents |
| --- | --- |
| `gplan/lewm.py` | Load the frozen LeWM, `encode` frames, `lewm_cost` J(A, c) |
| `gplan/data.py` | (start, goal = start + 25) frame pairs from the LeWM `.h5` datasets |
| `gplan/policy.py` | `GPlaner`, `Sampler` (rollout, teacher-forced `log_prob`), checkpoint save/load |
| `gplan/losses.py` | TB and VarGrad losses on `xi = log P_F + βJ − log N(A)`, diagnostics |
| `gplan/trainer.py` | Optimization loop, β schedule, gradient and policy statistics |
| `gplan/solvers.py` | `GFlowSolver` (best-of-N) and `CostLoggingSolver` for `stable_worldmodel` |
| `gplan/evaluation.py` | LeWM TwoRoom evaluation protocol |
| `scripts/` | `train.py`, `evaluate.py`, `convert_lewm.py` command-line entry points |
| `tests/` | Unit tests on a tiny random LeWM, including a closed-form check of the TB optimum |

The losses take any `cost_fn(z_start, z_goal, plans) -> (N,)`, and `Sampler.log_prob` scores any
action buffer, so new plan sources (expert segments, replay, local search) plug into `trainer.py`
without touching the policy or the losses.
