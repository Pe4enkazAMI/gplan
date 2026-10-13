"""Pool `scripts/evaluate.py --results-json` files into one comparison table (see gplan/results.py).

    # 3 training seeds x 4 eval seeds of one config, plus CEM, on held-out PushT episodes
    for s in 0 1 2; do for e in 42 43 44 45; do
      python scripts/evaluate.py --task pusht --policy gflow gflow-refine cem --planner outputs/pusht-s$s.pt \\
          --label pusht-mog7 --seed $e --holdout-frac 0.1 --results-json results/pusht-s$s-e$e.json
    done; done
    python scripts/aggregate_results.py results/*.json --csv results/table.csv

Success is pooled over all episodes of a group (Wilson 95% interval); "± x" is the std of the success
rate across planner checkpoints of one label. "only this / only ref" counts episodes solved by one of
the two policies (paired on the same episodes), with an exact McNemar p-value. Pooling several
checkpoints against one CEM run reuses CEM's episodes, so read p as indicative there.
"""

import argparse
import csv
import sys
from pathlib import Path

from gplan.results import comparison_table, format_markdown, group_runs, load_runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path, help="--results-json files of scripts/evaluate.py")
    p.add_argument("--reference", default="cem", help="policy every other policy is paired against")
    p.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    args = p.parse_args()

    rows = comparison_table(group_runs(load_runs(args.results)), args.reference)
    print(format_markdown(rows, args.reference))
    if args.csv:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}", file=sys.stderr)  # stdout is the table only (redirect it to a .md)


if __name__ == "__main__":
    main()
