# Settings shared by train.sh, eval.sh and tables.sh (sourced, not run). Every value can be overridden
# from the environment, e.g.:  SEEDS="0" EVAL_SEEDS="42" bash experiments/eval.sh pusht
set -euo pipefail

TASK=${1:?usage: bash experiments/<train|eval|tables>.sh tworoom|pusht}
cd "$(dirname "${BASH_SOURCE[0]}")/.."                  # run from the repository root

: "${STABLEWM_HOME:=$HOME/.stable_worldmodel}"; export STABLEWM_HOME
: "${OUT:=outputs/$TASK}"                    # checkpoints: $OUT/<model>-s<seed>.{eqx,json,pt}
: "${RESULTS:=results/$TASK}"                # evaluation .json files and tables
: "${SEEDS:=0 1 2}"                          # training seeds
: "${EVAL_SEEDS:=42 43 44 45}"               # episode sets (LeWM protocol: 50 episodes each)
: "${NUM_EVAL:=50}"
: "${SPLIT:=--holdout-frac 0.1 --split-seed 0}"   # episodes held out of training, used for evaluation
: "${WANDB_MODE:=online}"
: "${EXTRA_EVAL_ARGS:=}"                     # e.g. "--device cuda" or "--deterministic"
: "${PROBE_ARGS:=}"                          # e.g. "--n-pairs 1024" if few held-out pairs
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # several processes in a row on the GB10's shared memory

case $TASK in
  tworoom)
    : "${DATASET:=tworoom}"
    : "${WM:=tworoom/lewm}"
    : "${BETA:=90}"
    : "${GFN_ARGS:=--beta $BETA --beta-warmup 1000 --n-batches 20000}"
    : "${ABLATION_BETAS:=60 120 192}"
    ;;
  pusht)
    : "${DATASET:=pusht_expert_train}"
    : "${WM:=pusht/lewm}"
    : "${BETA:=192}"
    : "${GFN_ARGS:=--beta $BETA --beta-warmup 30000 --n-batches 60000 --n-components 7 --ls-steps 10 --ls-every 10 --ls-start 30000}"
    : "${ABLATION_BETAS:=90 120 300}"
    ;;
  *) echo "unknown task: $TASK (tworoom or pusht)" >&2; exit 1 ;;
esac
: "${CKPT:=$STABLEWM_HOME/${WM}_object.ckpt}"
: "${BASELINE_BATCHES:=20000}"               # bc / map converge faster than the GFlowNet
: "${REFINE:=--refine-steps 20 --refine-objective target --refine-beta $BETA}"   # gflow-refine and prior-refine

# ablations (ABLATIONS=1 in train.sh / eval.sh): 2 seeds, one change to the GFlowNet each;
# ABLATION_BETAS (per task above) are the other betas, the main one is the reference
: "${ABLATION_SEEDS:=0 1}"

TRAIN_ARGS="--dataset $DATASET --ckpt $CKPT --goal-offset 25 --grad-clip 10 --lr-schedule cosine $SPLIT --wandb-mode $WANDB_MODE"
EVAL_ARGS="--task $TASK --dataset $DATASET --wm $WM --num-eval $NUM_EVAL $SPLIT --wandb-mode $WANDB_MODE $EXTRA_EVAL_ARGS"

ablations() {  # "name|extra train.py flags" for every ablation; later flags override GFN_ARGS
  local b
  for b in $ABLATION_BETAS; do echo "gfn-beta$b|--beta $b"; done
  echo "gfn-vargrad|--loss vargrad"
  echo "gfn-constlr|--lr-schedule constant"
  case $TASK in
    pusht)    # main = 7 components + MALA: remove each, and both (with gfn: a 2 x 2 design)
      echo "gfn-k1|--n-components 1"
      echo "gfn-k4|--n-components 4"
      echo "gfn-nomala|--ls-steps 0"
      echo "gfn-k1-nomala|--n-components 1 --ls-steps 0"
      ;;
    tworoom)  # main = 1 Gaussian, no MALA: add each, and both (MALA after the beta warmup)
      echo "gfn-k4|--n-components 4"
      echo "gfn-mala|--ls-steps 10 --ls-every 10 --ls-start 1000"
      echo "gfn-k4-mala|--n-components 4 --ls-steps 10 --ls-every 10 --ls-start 1000"
      ;;
  esac
}
