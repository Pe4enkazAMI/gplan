#!/usr/bin/env bash
# Train and evaluate only the PPO models of one task, to add them to tables made before PPO existed:
#   ppo, ppo-maxent                    3 seeds; held-out and all episodes; best-of-64 + refinement
#                                      (target and cost objectives), exactly as eval.sh evaluates them
#   ppo-cost, ppo-cost-maxent          the PPO ablations of config.sh (ABLATION_SEEDS), as eval.sh's
#                                      ablations (held-out episodes; best-of-64 + target refinement)
# Same settings, file names and labels as train.sh / eval.sh, so the rows join the existing tables
# (paired with the CEM results already there) and a later train.sh / eval.sh skips them. Existing
# checkpoints and results are skipped, so the script can be rerun after an interruption.
#
#   bash experiments/ppo_only.sh tworoom
source "$(dirname "$0")/config.sh"
mkdir -p "$OUT"

train() {  # train <model> <seed> <train.py flags...>
  local name=$1 seed=$2; shift 2
  local out="$OUT/$name-s$seed"
  if [[ -f "$out.pt" ]]; then echo "skip $out (exists)"; return; fi
  echo "=== train $out"
  python scripts/train.py $TRAIN_ARGS --seed "$seed" --out "$out" --wandb-name "$TASK-$name-s$seed" "$@"
}

run() {  # run <results .json> <evaluate.py flags...>
  local json=$1; shift
  if [[ -f "$json" ]]; then echo "skip $json (exists)"; return; fi
  mkdir -p "$(dirname "$json")"
  echo "=== eval $json"
  python scripts/evaluate.py $EVAL_ARGS --results-json "$json" "$@"
}

planner() { echo "--planner $OUT/$1-s$2.pt --label $1"; }

ppo_ablations() { ablations | grep '^ppo'; }  # the PPO lines of config.sh's ablations

# -- train
for s in $SEEDS; do
  train ppo        "$s" $PPO_ARGS
  train ppo-maxent "$s" $PPO_ARGS --ent-coef 1
done
while IFS='|' read -r -u 3 name flags; do
  for s in $ABLATION_SEEDS; do train "$name" "$s" $flags; done
done 3< <(ppo_ablations)

# -- evaluate
for pool in "heldout|$SPLIT" "alldata|--holdout-frac 0"; do
  IFS='|' read -r dir flags <<< "$pool"
  for e in $EVAL_SEEDS; do for s in $SEEDS; do for m in ppo ppo-maxent; do
    run "$RESULTS/$dir/$m-s$s-e$e.json" --seed "$e" $flags $(planner "$m" "$s") --policy gflow gflow-refine $REFINE
    run "$RESULTS/$dir/$m-refine-cost-s$s-e$e.json" --seed "$e" $flags $(planner "$m" "$s") \
        --policy gflow-refine $REFINE_COST
  done; done; done
done
while IFS='|' read -r -u 3 name _; do
  for e in $EVAL_SEEDS; do for s in $ABLATION_SEEDS; do
    run "$RESULTS/ablations/$name-s$s-e$e.json" --seed "$e" $SPLIT $(planner "$name" "$s") \
        --policy gflow gflow-refine $REFINE
  done; done
done 3< <(ppo_ablations)

bash experiments/tables.sh "$TASK"   # every table rebuilt from all results, now with the PPO rows
