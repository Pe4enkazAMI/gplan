#!/usr/bin/env bash
# Evaluate everything train.sh produced, every eval seed (EVAL_SEEDS x 50 episodes). One results .json
# per run under $RESULTS/<experiment>/ (existing ones are skipped); tables.sh turns them into tables.
#
#   heldout    the main comparison, on the episodes no model was trained on:
#                cem            the LeWM paper's planner (300 samples x 30 iterations)
#                prior          best of 64 random plans from N(0, I)          (no learning)
#                prior-refine   the same 64 plans + 20 gradient steps through LeWM
#                gflow          best of 64 samples of gfn / gfn-mala / bc, and map's single plan
#                gflow-refine   the same 64 samples + 20 gradient steps       (gfn / gfn-mala / bc)
#              every *-refine policy twice: on beta J + |A|^2/2 (obj=target) and on J alone (obj=cost)
#   alldata    the same on every episode (the LeWM paper's protocol; 90% of them were trained on)
#   curve      success vs compute, held out: GFlow N = 1 ... 256, refinement N x steps, CEM grid
#   ablations  each ablation's GFlowNet, gflow and gflow-refine, held out (ABLATIONS=0 skips them)
#
#   bash experiments/eval.sh tworoom
source "$(dirname "$0")/config.sh"

run() {  # run <results .json> <evaluate.py flags...>
  local json=$1; shift
  if [[ -f "$json" ]]; then echo "skip $json (exists)"; return; fi
  mkdir -p "$(dirname "$json")"
  echo "=== eval $json"
  python scripts/evaluate.py $EVAL_ARGS --results-json "$json" "$@"
}

planner() {  # planner <model> <seed>: evaluate.py flags of one checkpoint
  echo "--planner $OUT/$1-s$2.pt --label $1"
}

compare() {  # compare <dir> <episode pool flags>: every model and baseline on the same episodes
  local dir=$1 e s m; shift
  for e in $EVAL_SEEDS; do
    run "$dir/baselines-e$e.json" --seed "$e" "$@" --policy cem prior prior-refine $REFINE
    run "$dir/baselines-refine-cost-e$e.json" --seed "$e" "$@" --policy prior-refine $REFINE_COST
    for s in $SEEDS; do
      for m in gfn gfn-mala bc; do
        run "$dir/$m-s$s-e$e.json" --seed "$e" "$@" $(planner "$m" "$s") --policy gflow gflow-refine $REFINE
        run "$dir/$m-refine-cost-s$s-e$e.json" --seed "$e" "$@" $(planner "$m" "$s") --policy gflow-refine $REFINE_COST
      done
      run "$dir/map-s$s-e$e.json" --seed "$e" "$@" $(planner map "$s") --policy gflow --gflow-select mean
    done
  done
}

compare "$RESULTS/heldout" $SPLIT
compare "$RESULTS/alldata" --holdout-frac 0

# -- success vs compute (CEM 300 x 30, GFlow N = 64 and refinement 64 x 20 come from heldout/)
for e in $EVAL_SEEDS; do
  for s in $SEEDS; do for m in gfn gfn-mala; do
    for n in 1 4 16 256; do
      run "$RESULTS/curve/$m-n$n-s$s-e$e.json" --seed "$e" $SPLIT $(planner "$m" "$s") \
          --policy gflow --gflow-samples "$n"
    done
    for cfg in 16,20 64,5 64,50; do  # samples,refinement steps
      IFS=, read -r n k <<< "$cfg"
      run "$RESULTS/curve/$m-refine-n$n-k$k-s$s-e$e.json" --seed "$e" $SPLIT $(planner "$m" "$s") \
          --policy gflow-refine --gflow-samples "$n" $REFINE --refine-steps "$k"
    done
  done; done
  for cfg in 16,20 64,5 64,50; do
    IFS=, read -r n k <<< "$cfg"
    run "$RESULTS/curve/prior-refine-n$n-k$k-e$e.json" --seed "$e" $SPLIT \
        --policy prior-refine --prior-samples "$n" $REFINE --refine-steps "$k"
  done
  for cfg in 30,3,3 30,10,3 100,10,10 100,30,10 300,10,30; do  # samples,iterations,elites (10%)
    IFS=, read -r n it top <<< "$cfg"
    run "$RESULTS/curve/cem-n$n-i$it-e$e.json" --seed "$e" $SPLIT \
        --policy cem --cem-samples "$n" --cem-steps "$it" --cem-topk "$top"
  done
done

# -- ablations
if [[ $ABLATIONS == 1 ]]; then
  while IFS='|' read -r -u 3 name _; do
    for e in $EVAL_SEEDS; do for s in $ABLATION_SEEDS; do
      run "$RESULTS/ablations/$name-s$s-e$e.json" --seed "$e" $SPLIT $(planner "$name" "$s") \
          --policy gflow gflow-refine $REFINE
    done; done
  done 3< <(ablations)
fi
