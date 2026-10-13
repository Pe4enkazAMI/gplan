#!/usr/bin/env bash
# Train every model one task needs, 3 seeds each, on the training side of the episode split:
#   gfn   the GFlowNet (config.sh: GFN_ARGS)
#   bc    behavior cloning on the expert actions (same network)
#   map   amortized MAP planning through LeWM (same network, one plan per condition)
# ABLATIONS=1 also trains the ablations of config.sh (2 seeds each). Existing checkpoints are skipped,
# so the script can be rerun after an interruption.
#
#   bash experiments/train.sh pusht
#   ABLATIONS=1 bash experiments/train.sh pusht
source "$(dirname "$0")/config.sh"
mkdir -p "$OUT"

train() {  # train <model name> <seed> <train.py flags...>
  local name=$1 seed=$2; shift 2
  local out="$OUT/$name-s$seed"
  if [[ -f "$out.pt" ]]; then echo "skip $out (exists)"; return; fi
  echo "=== train $out"
  python scripts/train.py $TRAIN_ARGS --seed "$seed" --out "$out" --wandb-name "$TASK-$name-s$seed" "$@"
}

for s in $SEEDS; do
  train gfn "$s" $GFN_ARGS
  train bc  "$s" --method bc --n-batches "$BASELINE_BATCHES"
  train map "$s" --method map --beta "$BETA" --n-batches "$BASELINE_BATCHES"
done

if [[ "${ABLATIONS:-0}" == 1 ]]; then
  while IFS='|' read -r -u 3 name flags; do  # fd 3, so nothing in the loop can eat the list
    for s in $ABLATION_SEEDS; do train "$name" "$s" $GFN_ARGS $flags; done
  done 3< <(ablations)
fi
