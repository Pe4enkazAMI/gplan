#!/usr/bin/env bash
# Turn eval.sh's results into tables: $RESULTS/tables/<experiment>.md (+ .csv), also printed.
# A row is one policy of one model ("label"), pooled over training seeds and eval seeds: success with a
# Wilson 95% interval, "± x" = std across training seeds, and a paired comparison with CEM on the same
# episodes (solved only by this / only by CEM, exact McNemar p). See scripts/aggregate_results.py.
#
#   bash experiments/tables.sh tworoom
source "$(dirname "$0")/config.sh"
shopt -s nullglob
mkdir -p "$RESULTS/tables"

table() {  # table <name> <results .json files...>
  local name=$1; shift
  if (( $# == 0 )); then echo "(no results for $name yet)"; return; fi
  python scripts/aggregate_results.py "$@" --csv "$RESULTS/tables/$name.csv" > "$RESULTS/tables/$name.md"
  printf '\n## %s: %s\n\n' "$TASK" "$name"
  cat "$RESULTS/tables/$name.md"
}

table heldout "$RESULTS"/heldout/*.json
table alldata "$RESULTS"/alldata/*.json
table compute_curve "$RESULTS"/heldout/{baselines,gfn}-*.json "$RESULTS"/curve/*.json
table ablations "$RESULTS"/heldout/{baselines,gfn}-*.json "$RESULTS"/ablations/*.json
