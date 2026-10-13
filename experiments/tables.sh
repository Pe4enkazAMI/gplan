#!/usr/bin/env bash
# Turn eval.sh's results into readable tables: $RESULTS/tables/<experiment>.md (+ .csv), also printed.
# Every row pools training seeds (same --label) and eval seeds; success has a Wilson 95% interval and
# "± x" is the std across training seeds; "vs cem" pairs each row with the paper's CEM on the same
# episodes (exact McNemar p). See scripts/aggregate_results.py.
#
#   bash experiments/tables.sh pusht
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

table main "$RESULTS"/main/*.json
table compute_curve "$RESULTS"/main/reference-*.json "$RESULTS"/main/gfn-s*.json "$RESULTS"/curve/*.json
table goal_offset_10 "$RESULTS"/goal10/*.json
table goal_offset_50 "$RESULTS"/goal50/*.json
table budget_25 "$RESULTS"/budget25/*.json
table lewm_protocol "$RESULTS"/lewm/*.json
table ablations "$RESULTS"/main/reference-*.json "$RESULTS"/main/gfn-s*.json "$RESULTS"/ablations/*.json
for f in "$RESULTS"/probe/*.txt; do  # the probe's summary: from its table header to the end
  printf '\n## %s: probe %s\n\n```\n' "$TASK" "$(basename "$f" .txt)"
  sed -n '/^plans /,$p' "$f"
  echo '```'
done
