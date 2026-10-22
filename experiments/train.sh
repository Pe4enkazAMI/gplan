#!/usr/bin/env bash
# Train every model of one task on the training side of the episode split (config.sh), constant LR:
#   gfn        the GFlowNet                                  (GFN_ARGS)
#   gfn-mala   the GFlowNet + MALA plans as off-policy data  (GFN_ARGS + MALA_ARGS)
#   bc         behavior cloning on the expert actions        (same network)
#   map        amortized MAP planning through LeWM           (same network, one plan per condition)
#   ppo        PPO maximizing -beta J + log N(A)             (same network + a critic; PPO_ARGS)
#   ppo-maxent the same with the entropy term: its optimum is the GFlowNet target (--ent-coef 1)
# 3 seeds each (SEEDS), then the ablations of config.sh (ABLATION_SEEDS; ABLATIONS=0 skips them).
# Existing checkpoints are skipped, so the script can be rerun after an interruption.
#
#   bash experiments/train.sh tworoom
source "$(dirname "$0")/config.sh"
mkdir -p "$OUT"

train() {  # train <model> <seed> <train.py flags...>   ->  $OUT/<model>-s<seed>.{eqx,json,pt}
  local name=$1 seed=$2; shift 2
  local out="$OUT/$name-s$seed"
  if [[ -f "$out.pt" ]]; then echo "skip $out (exists)"; return; fi
  echo "=== train $out"
  python scripts/train.py $TRAIN_ARGS --seed "$seed" --out "$out" --wandb-name "$TASK-$name-s$seed" "$@"
}

for s in $SEEDS; do
  train gfn      "$s" $GFN_ARGS
  train gfn-mala "$s" $GFN_ARGS $MALA_ARGS
  train bc       "$s" --method bc --n-batches "$BASELINE_BATCHES"
  train map      "$s" --method map --beta "$BETA" --n-batches "$BASELINE_BATCHES"
  train ppo        "$s" $PPO_ARGS
  train ppo-maxent "$s" $PPO_ARGS --ent-coef 1
done

if [[ $ABLATIONS == 1 ]]; then
  while IFS='|' read -r -u 3 name flags; do  # fd 3, so nothing in the loop can eat the list
    for s in $ABLATION_SEEDS; do train "$name" "$s" $flags; done
  done 3< <(ablations)
fi
