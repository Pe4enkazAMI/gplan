"""Result aggregation: pooling and de-duplication of runs, Wilson intervals, paired McNemar tests."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gplan.results import comparison_table, group_runs, load_runs, mcnemar_exact, wilson_interval

REPO = Path(__file__).resolve().parents[1]
ARGS = dict(task="pusht", dataset="pusht_expert_train", wm="pusht/lewm", goal_offset=25, eval_budget=50, num_eval=4,
            holdout_frac=0.1, split_seed=0, horizon=5, action_block=5, img_size=224, gflow_samples=64,
            gflow_select="min-cost", cem_samples=300, cem_steps=30, cem_topk=30, label=None)


def write_run(path, seed, planner, successes, label=None):
    """A results file in the format scripts/evaluate.py writes."""
    run = {"args": {**ARGS, "seed": seed, "planner": planner, "label": label},
           "episodes": [[seed, i] for i in range(4)],  # the episodes depend on the eval seed only
           "planner_train_args": {},
           "policies": {name: {"episode_successes": s, "plan_cost": 1.0, "plans_per_replan": n, "backward_per_replan": 0,
                               "solve_time_s": 2.0} for name, (s, n) in successes.items()}}
    path.write_text(json.dumps(run))
    return path


def test_wilson_interval_and_mcnemar_known_values():
    lo, hi = wilson_interval(46, 50)
    assert math.isclose(lo, 0.8116, abs_tol=1e-4) and math.isclose(hi, 0.9685, abs_tol=1e-4)
    assert mcnemar_exact(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)  # 0.109
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_runs_are_pooled_by_label_and_cem_is_counted_once(tmp_path):
    cem = [True, True, False, True]
    files = []
    for train_seed in (0, 1):  # two checkpoints of one config, each next to the same CEM run
        gflow = [True, False, False, train_seed == 1]
        files.append(write_run(tmp_path / f"s{train_seed}.json", seed=42, planner=f"outputs/pusht-s{train_seed}.pt",
                               successes={"gflow": (gflow, 64), "cem": (cem, 9000)}, label="pusht-mog7"))
    rows = {r["policy"]: r for r in comparison_table(group_runs(load_runs(files)))}
    assert rows["cem"]["episodes"] == 4 and rows["cem"]["success"] == 75.0  # duplicate CEM run dropped
    g = rows["gflow"]
    assert g["label"] == "pusht-mog7" and g["planners"] == 2 and g["episodes"] == 8
    assert g["success"] == 100 * 3 / 8
    assert math.isclose(g["success_std_over_planners"], 100 * np.std([0.25, 0.5], ddof=1))
    # paired on the same episodes: per checkpoint, gflow solves nothing CEM misses except seed 1's last episode
    assert (g["only_this"], g["only_reference"], g["both"]) == (0, 3, 3)


def test_settings_and_eval_seeds_separate_or_pool_groups(tmp_path):
    a = write_run(tmp_path / "a.json", 42, "p.pt", {"cem": ([True] * 4, 9000)})
    b = write_run(tmp_path / "b.json", 43, "p.pt", {"cem": ([False] * 4, 9000)})
    rows = comparison_table(group_runs(load_runs([a, b])))
    assert len(rows) == 1 and rows[0]["episodes"] == 8 and rows[0]["success"] == 50.0  # two eval seeds pooled
    c = tmp_path / "c.json"
    run = json.loads(a.read_text())
    run["args"]["cem_samples"] = 100
    c.write_text(json.dumps(run))
    assert len(group_runs(load_runs([a, c]))) == 2  # different CEM settings are different rows


def test_aggregate_script(tmp_path):
    f = write_run(tmp_path / "r.json", 42, "outputs/x.pt", {"gflow": ([True, True, False, False], 64),
                                                           "cem": ([True, False, False, False], 9000)})
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "aggregate_results.py"), str(f), "--csv",
                          str(tmp_path / "t.csv")], capture_output=True, text=True, cwd=REPO,
                         env={"PYTHONPATH": str(REPO), "PATH": ""}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "| gflow | x | samples=64 select=min-cost |" in out.stdout and "1 / 0" in out.stdout
    assert "wrote" not in out.stdout  # stdout is only the table
    assert (tmp_path / "t.csv").read_text().startswith("policy,")


def test_cem_with_the_most_compute_is_the_reference(tmp_path):
    """In a CEM compute sweep, other policies are paired with the paper's CEM (most passes per replan),
    and the cheaper CEM settings are themselves paired with it."""
    weak, strong = [False, False, True, False], [True, True, True, False]
    a = write_run(tmp_path / "weak.json", 42, None, {"cem": (weak, 90)})
    run = json.loads(a.read_text())
    run["args"]["cem_samples"] = 30
    a.write_text(json.dumps(run))
    b = write_run(tmp_path / "strong.json", 42, "p.pt", {"cem": (strong, 9000), "gflow": ([True] * 4, 64)})
    rows = comparison_table(group_runs(load_runs([a, b])))
    by = {(r["policy"], r["plans_per_replan"]): r for r in rows}
    assert (by[("gflow", 64)]["only_this"], by[("gflow", 64)]["only_reference"]) == (1, 0)
    assert (by[("cem", 90)]["only_this"], by[("cem", 90)]["only_reference"]) == (0, 2)
    assert "only_this" not in by[("cem", 9000)]


def test_rows_are_ordered_reference_first_then_by_label_and_compute(tmp_path):
    a = write_run(tmp_path / "a.json", 42, "outputs/zeta.pt", {"random": ([False] * 4, 0), "gflow": ([True] * 4, 64),
                                                               "cem": ([True] * 4, 9000)})
    run = json.loads(a.read_text())
    run["args"]["gflow_samples"] = 1
    run["policies"]["gflow"]["plans_per_replan"] = 1
    b = tmp_path / "b.json"
    b.write_text(json.dumps(run))
    rows = comparison_table(group_runs(load_runs([a, b])))
    assert [(r["policy"], r["plans_per_replan"]) for r in rows] == [("cem", 9000), ("gflow", 1), ("gflow", 64),
                                                                     ("random", 0)]
