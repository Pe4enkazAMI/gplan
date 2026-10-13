#!/usr/bin/env bash
# Evaluate everything train.sh produced, on the held-out episodes, every eval seed. Writes one
# results .json per run under $RESULTS/<experiment>/ (existing ones are skipped); tables.sh turns
# them into tables. Experiments:
#   main         CEM (paper config), random shooting, gradient planning from random starts, random;
#                GFlowNet best-of-64 / mean plan / + refinement; bc; map           (all training seeds)
#   curve        success vs compute: GFlow N, refine N x steps, prior-refine N x steps, CEM samples x iters
#   goal50       goal 50 steps away instead of 25: beyond the training offset (CEM and GFlow)
#   budget25     one plan, no replanning (eval budget 25 instead of 50)
#   lewm         CEM on every episode, first eval seed (the LeWM paper's protocol: a sanity check of CEM)
#   probe        headroom probe on held-out pairs (text report)
#   ablations    (ABLATIONS=1) each ablation's GFlowNet, best-of-64 and + refinement
#
#   bash experiments/eval.sh pusht
source "$(dirname "$0")/config.sh"

run() {  # run <results .json> <evaluate.py flags...>
  local json=$1; shift
  if [[ -f "$json" ]]; then echo "skip $json (exists)"; return; fi
  mkdir -p "$(dirname "$json")"
  echo "=== eval $json"
  python scripts/evaluate.py $EVAL_ARGS --results-json "$json" "$@"
}

for e in $EVAL_SEEDS; do
  # -- main table
  run "$RESULTS/main/reference-e$e.json" --seed "$e" --policy cem prior prior-refine random $REFINE
  for s in $SEEDS; do
    run "$RESULTS/main/gfn-s$s-e$e.json" --seed "$e" --planner "$OUT/gfn-s$s.pt" --label gfn \
        --policy gflow gflow-refine $REFINE
    run "$RESULTS/main/gfn-mean-s$s-e$e.json" --seed "$e" --planner "$OUT/gfn-s$s.pt" --label gfn \
        --policy gflow --gflow-select mean
    run "$RESULTS/main/bc-s$s-e$e.json" --seed "$e" --planner "$OUT/bc-s$s.pt" --label bc \
        --policy gflow gflow-refine $REFINE
    run "$RESULTS/main/map-s$s-e$e.json" --seed "$e" --planner "$OUT/map-s$s.pt" --label map \
        --policy gflow --gflow-select mean
  done

  # -- success vs compute (CEM 300 x 30 and GFlow N = 64 come from main/)
  for s in $SEEDS; do
    for n in 1 4 16 256; do
      run "$RESULTS/curve/gfn-n$n-s$s-e$e.json" --seed "$e" --planner "$OUT/gfn-s$s.pt" --label gfn \
          --policy gflow --gflow-samples "$n"
    done
    for cfg in 16,20 64,5 64,50; do  # samples,steps (64,20 is in main/)
      IFS=, read -r n k <<< "$cfg"
      run "$RESULTS/curve/gfn-refine-n$n-k$k-s$s-e$e.json" --seed "$e" --planner "$OUT/gfn-s$s.pt" --label gfn \
          --policy gflow-refine --gflow-samples "$n" $REFINE --refine-steps "$k"
    done
  done
  for cfg in 16,20 64,5 64,50; do
    IFS=, read -r n k <<< "$cfg"
    run "$RESULTS/curve/prior-refine-n$n-k$k-e$e.json" --seed "$e" --policy prior-refine --prior-samples "$n" \
        $REFINE --refine-steps "$k"
  done
  for cfg in 30,3,3 30,10,3 100,10,10 100,30,10 300,10,30; do  # samples,iterations,elites (elites = 10%)
    IFS=, read -r n it top <<< "$cfg"
    run "$RESULTS/curve/cem-n$n-i$it-e$e.json" --seed "$e" --policy cem --cem-samples "$n" --cem-steps "$it" \
        --cem-topk "$top"
  done

  # -- goal distance and replanning
  run "$RESULTS/goal50/reference-e$e.json" --seed "$e" --goal-offset 50 --policy cem
  for s in $SEEDS; do
    run "$RESULTS/goal50/gfn-s$s-e$e.json" --seed "$e" --goal-offset 50 --planner "$OUT/gfn-s$s.pt" --label gfn \
        --policy gflow gflow-refine $REFINE
  done
  run "$RESULTS/budget25/reference-e$e.json" --seed "$e" --eval-budget 25 --policy cem
  for s in $SEEDS; do
    run "$RESULTS/budget25/gfn-s$s-e$e.json" --seed "$e" --eval-budget 25 --planner "$OUT/gfn-s$s.pt" --label gfn \
        --policy gflow gflow-refine $REFINE
  done

  # -- ablations
  if [[ "${ABLATIONS:-0}" == 1 ]]; then
    while IFS='|' read -r -u 3 name _; do
      for s in $ABLATION_SEEDS; do
        run "$RESULTS/ablations/$name-s$s-e$e.json" --seed "$e" --planner "$OUT/$name-s$s.pt" --label "$name" \
            --policy gflow gflow-refine $REFINE
      done
    done 3< <(ablations)
  fi
done

# -- the LeWM paper's protocol (every episode), CEM only: one eval seed is enough to check its numbers
e=${EVAL_SEEDS%% *}
run "$RESULTS/lewm/reference-e$e.json" --seed "$e" --policy cem --holdout-frac 0

# -- headroom probe on held-out (start, goal) pairs
for s in $SEEDS; do
  probe="$RESULTS/probe/gfn-s$s"
  if [[ -f "$probe.txt" ]]; then echo "skip $probe (exists)"; continue; fi
  mkdir -p "$RESULTS/probe"
  echo "=== probe $probe"
  python scripts/probe_headroom.py "$OUT/gfn-s$s" --out "$probe.npz" $PROBE_ARGS | tee "$probe.txt"
done
