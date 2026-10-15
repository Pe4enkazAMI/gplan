# Settings shared by train.sh, eval.sh and tables.sh (sourced, not run). The ": ${X:=...}" values can be
# overridden from the environment, e.g.:  SEEDS="0" EVAL_SEEDS="42" bash experiments/eval.sh tworoom
set -euo pipefail

TASK=${1:?usage: bash experiments/<train|eval|tables>.sh tworoom|pusht}
cd "$(dirname "${BASH_SOURCE[0]}")/.."                  # run from the repository root

: "${STABLEWM_HOME:=$HOME/.stable_worldmodel}"; export STABLEWM_HOME
: "${OUT:=outputs/$TASK}"                    # checkpoints: $OUT/<model>-s<seed>.{eqx,json,pt}
: "${RESULTS:=results/$TASK}"                # evaluation .json files and tables
: "${SEEDS:=0 1 2}"                          # training seeds of every model
: "${EVAL_SEEDS:=42 43 44 45}"               # episode sets, 50 episodes each (LeWM protocol)
: "${NUM_EVAL:=50}"
: "${SPLIT:=--holdout-frac 0.1 --split-seed 0}"   # 10% of the episodes are never trained on
: "${ABLATIONS:=1}"                          # 0: skip the ablations
: "${ABLATION_SEEDS:=0 1}"
: "${WANDB_MODE:=online}"
: "${EXTRA_EVAL_ARGS:=}"                     # e.g. "--device cuda" or "--deterministic"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several processes in a row on the GB10's shared memory

# Per task: the GFlowNet (GFN_ARGS), what MALA adds to it (MALA_ARGS, from the end of the beta warmup)
# and the ablations, one "name|train.py flags" line each.
case $TASK in
  tworoom)
    : "${DATASET:=tworoom}" "${WM:=tworoom/lewm}" "${BETA:=90}"
    : "${GFN_ARGS:=--beta $BETA --beta-warmup 1000 --n-batches 20000}"
    : "${MALA_ARGS:=--ls-steps 10 --ls-every 10 --ls-start 1000}"
    ablations() {  # temperature: half and twice the main beta
      echo "gfn-beta45|$GFN_ARGS --beta 45"
      echo "gfn-beta180|$GFN_ARGS --beta 180"
    }
    ;;
  pusht)
    : "${DATASET:=pusht_expert_train}" "${WM:=pusht/lewm}" "${BETA:=192}"
    : "${GFN_ARGS:=--beta $BETA --beta-warmup 30000 --n-batches 60000 --n-components 7}"
    : "${MALA_ARGS:=--ls-steps 10 --ls-every 10 --ls-start 30000}"
    ablations() {  # temperature (of gfn-mala), and one Gaussian instead of the 7-component mixture
      echo "gfn-mala-beta96|$GFN_ARGS $MALA_ARGS --beta 96"
      echo "gfn-mala-beta384|$GFN_ARGS $MALA_ARGS --beta 384"
      echo "gfn-k1|$GFN_ARGS --n-components 1"
      echo "gfn-mala-k1|$GFN_ARGS $MALA_ARGS --n-components 1"
    }
    ;;
  *) echo "unknown task: $TASK (tworoom or pusht)" >&2; exit 1 ;;
esac
: "${CKPT:=$STABLEWM_HOME/${WM}_object.ckpt}"
: "${BASELINE_BATCHES:=20000}"               # bc and map converge faster than the GFlowNet
REFINE="--refine-steps 20 --refine-objective target --refine-beta $BETA"   # every *-refine policy

TRAIN_ARGS="--dataset $DATASET --ckpt $CKPT --goal-offset 25 --grad-clip 10 $SPLIT --wandb-mode $WANDB_MODE"
EVAL_ARGS="--task $TASK --dataset $DATASET --wm $WM --num-eval $NUM_EVAL --wandb-mode $WANDB_MODE $EXTRA_EVAL_ARGS"
