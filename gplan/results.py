"""Aggregate `scripts/evaluate.py results_json=...` files into paper tables (NumPy only).

A results file is one evaluation run: one `seed` (one set of episodes) and one planner checkpoint,
with any number of policies. Runs are pooled per (policy, its settings, label):
    - the same policy + settings + eval seed (+ planner for GFlow policies) is one result; duplicates
      (e.g. CEM evaluated next to several checkpoints) are counted once;
    - GFlow results with the same `label` (e.g. three training seeds of one config) are pooled.
Success rates get Wilson 95% intervals over the pooled episodes; comparisons with a reference policy
(CEM by default) are paired on identical episodes, with an exact McNemar test.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# evaluation settings shared by every policy of a run: results are only pooled / paired if they match
COMMON = ("task", "dataset", "wm", "goal_offset", "eval_budget", "num_eval", "holdout_frac", "split_seed",
          "horizon", "action_block", "img_size")
REFINE = ("refine_steps", "refine_lr", "refine_objective", "refine_beta")
POLICY_ARGS = {  # the flags that change what each policy does
    "gflow": ("gflow_samples", "gflow_select"),
    "gflow-refine": ("gflow_samples",) + REFINE,
    "prior": ("prior_samples", "prior_var"),
    "prior-refine": ("prior_samples", "prior_var") + REFINE,
    "cem": ("cem_samples", "cem_steps", "cem_topk"),
    "random": (),
}
LEARNED = ("gflow", "gflow-refine")  # results depend on the planner checkpoint


@dataclass
class Group:
    policy: str
    label: str | None
    settings: dict
    common: dict
    results: list = field(default_factory=list)  # one dict per (eval seed, planner)


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval of a binomial proportion (well behaved near 0 and 1)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar test: p-value of the discordant counts under a fair coin."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def policy_settings(policy, args):
    """The flags that change what `policy` did in a run. Recorded but unused flags are left out:
    gflow_samples when GFlow executes one plan (gflow_select sample / mean), and refine_beta
    when refinement minimizes the cost alone (refine_objective cost)."""
    settings = {k: args.get(k) for k in POLICY_ARGS.get(policy, ())}
    if policy == "gflow" and settings.get("gflow_select") in ("sample", "mean"):
        settings["gflow_samples"] = None
    if settings.get("refine_objective") == "cost":
        settings["refine_beta"] = None
    return settings


def load_runs(paths):
    """Read results files; returns one record per (file, policy)."""
    records = []
    for path in paths:
        run = json.loads(Path(path).read_text())
        args = run["args"]
        for policy, metrics in run["policies"].items():
            learned = policy in LEARNED
            records.append({
                "file": str(path), "policy": policy, "seed": args["seed"],
                "label": (args.get("label") or Path(str(args.get("planner"))).stem) if learned else None,
                "planner": args.get("planner") if learned else None,
                "settings": policy_settings(policy, args),
                "common": {k: args.get(k) for k in COMMON},
                "episodes": [tuple(e) for e in run.get("episodes", [])],
                "successes": np.asarray(metrics["episode_successes"], dtype=bool),
                "metrics": metrics,
            })
    return records


def _key(rec, with_label=True):
    return (rec["policy"], json.dumps(rec["settings"], sort_keys=True), json.dumps(rec["common"], sort_keys=True),
            rec["label"] if with_label else None)


def group_runs(records):
    """Pool records into Groups, counting each (policy, settings, eval seed, planner) once."""
    groups, seen = {}, set()
    for rec in records:
        unit = _key(rec) + (rec["seed"], rec["planner"])
        if unit in seen:
            continue
        seen.add(unit)
        k = _key(rec)
        if k not in groups:
            groups[k] = Group(rec["policy"], rec["label"], rec["settings"], rec["common"])
        groups[k].results.append(rec)
    return list(groups.values())


def summarize(group):
    succ = np.concatenate([r["successes"] for r in group.results])
    lo, hi = wilson_interval(int(succ.sum()), len(succ))
    by_planner = defaultdict(list)
    for r in group.results:
        by_planner[r["planner"]].append(r["successes"])
    per_planner = [np.concatenate(v).mean() for v in by_planner.values()]
    mean_of = lambda k: float(np.mean([r["metrics"].get(k, np.nan) for r in group.results]))
    return {
        "policy": group.policy, "label": group.label or "", "settings": settings_text(group.settings),
        "runs": len(group.results), "planners": len(by_planner),
        "episodes": len(succ), "success": 100 * succ.mean(), "ci_low": 100 * lo, "ci_high": 100 * hi,
        "success_std_over_planners": 100 * float(np.std(per_planner, ddof=1)) if len(per_planner) > 1 else float("nan"),
        "plan_cost": mean_of("plan_cost"), "plans_per_replan": mean_of("plans_per_replan"),
        "backward_per_replan": mean_of("backward_per_replan"),
        "solve_s_per_episode": mean_of("solve_time_s") / group.common["num_eval"] if group.common["num_eval"] else np.nan,
    }


def paired_with(group, reference):
    """Discordant counts of `group` vs `reference` over episodes both evaluated (same eval seed, so the
    same episodes), pooled over runs, and the exact McNemar p-value. Returns None if nothing pairs."""
    ref_by_seed = {r["seed"]: r for r in reference.results}
    only_a = only_b = both = neither = 0
    for r in group.results:
        ref = ref_by_seed.get(r["seed"])
        if ref is None:
            continue
        assert r["episodes"] == ref["episodes"], "same seed and settings must mean the same episodes"
        a, b = r["successes"], ref["successes"]
        only_a += int((a & ~b).sum())
        only_b += int((~a & b).sum())
        both += int((a & b).sum())
        neither += int((~a & ~b).sum())
    if only_a + only_b + both + neither == 0:
        return None
    return {"both": both, "only_this": only_a, "only_reference": only_b, "neither": neither,
            "p_mcnemar": mcnemar_exact(only_a, only_b)}


def settings_text(settings):
    """Short description of a policy's settings for tables, e.g. "samples=64 select=mean"."""
    short = {"gflow_samples": "samples", "prior_samples": "samples", "cem_samples": "samples", "gflow_select": "select",
             "refine_steps": "steps", "refine_lr": "lr", "refine_objective": "obj", "refine_beta": "beta",
             "prior_var": "var", "cem_steps": "iters", "cem_topk": "topk"}
    return " ".join(f"{short.get(k, k)}={v}" for k, v in settings.items() if v is not None)


def comparison_table(groups, reference_policy="cem"):
    """One row per group: summary + paired comparison with the reference policy run on the same
    evaluation settings (if any). With several settings of the reference policy (e.g. a CEM compute
    sweep), the one with the most world-model passes per replan (CEM: the paper's 300 x 30) is used."""
    rows = []
    for g in groups:
        row = summarize(g)
        refs = [r for r in groups if r.policy == reference_policy and r.common == g.common]
        refs.sort(key=lambda r: -summarize(r)["plans_per_replan"])
        pair = paired_with(g, refs[0]) if refs and g is not refs[0] else None
        row.update(pair or {})
        rows.append(row)
    # reference first, then learned planners by label, then the planner-free baselines; cheapest first
    order = [reference_policy, "gflow", "gflow-refine", "prior", "prior-refine", "random"]
    rank = lambda r: (order.index(r["policy"]) if r["policy"] in order else len(order), r["label"],
                      r["plans_per_replan"] + 2 * r["backward_per_replan"], r["settings"])
    return sorted(rows, key=rank)


def format_markdown(rows, reference_policy="cem"):
    header = ["policy", "label", "settings", "planners", "episodes", "success %", "95% CI", "plan cost", "fwd/replan",
              "bwd/replan", "solve s/ep", f"vs {reference_policy}: only this / only ref", "McNemar p"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        std = r["success_std_over_planners"]
        success = f"{r['success']:.1f}" + (f" ± {std:.1f}" if not math.isnan(std) else "")
        versus = f"{r['only_this']} / {r['only_reference']}" if "only_this" in r else ""
        p = f"{r['p_mcnemar']:.3g}" if "p_mcnemar" in r else ""
        lines.append("| " + " | ".join([
            r["policy"], r["label"], r["settings"], str(r["planners"]), str(r["episodes"]), success,
            f"{r['ci_low']:.1f}-{r['ci_high']:.1f}", f"{r['plan_cost']:.2f}", f"{r['plans_per_replan']:.0f}",
            f"{r['backward_per_replan']:.0f}", f"{r['solve_s_per_episode']:.3f}", versus, p]) + " |")
    return "\n".join(lines)
