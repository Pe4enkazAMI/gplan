"""The paper's experiments for one task: train every model and ablation, evaluate them, build the tables.
Settings: configs/run.yaml, with the task (configs/task), the models (configs/model) and the evaluations
(configs/eval); Hydra overrides on the command line.

    python scripts/run.py task=pusht                    # train, evaluate, tables
    python scripts/run.py task=pusht stage=eval         # one stage: train, eval or tables
    python scripts/run.py task=pusht "only=[ppo,ppo-maxent,ppo-cost,ppo-cost-maxent]"   # only these models
    python scripts/run.py task=pusht "seeds=[0]" "eval_overrides={device:cuda}"
    python scripts/run.py task=pusht dry_run=true       # print the commands, run nothing
    python scripts/run.py task=pusht task.beta=120 out=outputs/pusht-b120 results=results/pusht-b120

Every run is a fresh `python scripts/train.py ...` / `scripts/evaluate.py ...` process with Hydra overrides
(task.* overrides are passed on to train.py). A run whose output exists is skipped (train: <out>.pt,
written last; evaluation: its results .json, written atomically), so the script can be rerun after an
interruption.
    train    configs/run.yaml models x seeds -> <out>/<model>-s<seed>.{eqx,json,pt,yaml}, then the task's
             ablations x ablation_seeds
    eval     every run of configs/eval/* -> <results>/<eval dir>/<name>.json
    tables   scripts/aggregate_results.py on each evaluation's table files -> <results>/tables/<table>.{md,csv}
"""

import os
import shlex
import subprocess
import sys
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

from gplan.config import compose_config, script_overrides
from gplan.data import STABLEWM_HOME

REPO = Path(__file__).resolve().parents[1]


def override_value(v):
    """A setting's value as Hydra override text: true / false, null, [a,b], or the value itself."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return f"[{','.join(override_value(x) for x in v)}]"
    return str(v)


def as_overrides(settings):
    """{name: value} (or null) -> ["name=value", ...] (train_overrides, eval_overrides)."""
    return [] if settings is None else [f"{k}={override_value(v)}" for k, v in OmegaConf.to_container(settings).items()]


def wanted(cfg, model):
    return cfg.only is None or model in cfg.only


def train_runs(cfg, task_overrides):
    """(checkpoint prefix, train.py overrides) of every model and ablation to train, in order."""
    runs = [(m, s, [f"model={m}"]) for s in cfg.seeds for m in cfg.models]
    if cfg.ablations:
        runs += [(name, s, list(overrides)) for name, overrides in cfg.task.ablations.items() for s in cfg.ablation_seeds]
    task = cfg.task.name
    return [(f"{cfg.out}/{label}-s{seed}", [
        *task_overrides, *model, f"seed={seed}", f"out={cfg.out}/{label}-s{seed}", f"wandb_name={task}-{label}-s{seed}",
        f"holdout_frac={cfg.holdout_frac}", f"split_seed={cfg.split_seed}", f"wandb_mode={cfg.wandb_mode}",
        *as_overrides(cfg.train_overrides)]) for label, seed, model in runs if wanted(cfg, label)]


def eval_runs(cfg):
    """(results .json, evaluate.py overrides) of every evaluation run, in order. Later overrides of a
    setting win (e.g. a run's refine_steps over its refine objective's)."""
    base = [f"task={cfg.task.name}", f"dataset={cfg.task.dataset}", f"wm={cfg.task.wm}", f"num_eval={cfg.num_eval}",
            f"wandb_mode={cfg.wandb_mode}", *as_overrides(cfg.eval_overrides)]
    runs = []
    for evaluation in cfg.evals.values():
        if not evaluation.get("enabled", True):
            continue
        for eval_seed in cfg.eval_seeds:
            for run in evaluation.runs:
                if "models" in run:
                    planners = [(m, s) for s in run.get("seeds", cfg.seeds) for m in run.models if wanted(cfg, m)]
                elif cfg.only is None:
                    planners = [(None, None)]  # a planner-free run (baselines)
                else:
                    planners = []
                for model, seed in planners:
                    path = f"{cfg.results}/{evaluation.dir}/{run.name.format(model=model, seed=seed, eval_seed=eval_seed)}.json"
                    planner = [] if model is None else [f"planner={cfg.out}/{model}-s{seed}.pt", f"label={model}"]
                    refine = cfg.refine[run.refine] if run.get("refine") else []
                    runs.append((path, [*base, f"results_json={path}", f"seed={eval_seed}", *evaluation.episodes, *planner,
                                        f"policy=[{','.join(run.policy)}]", *refine, *run.get("args", [])]))
    return runs


def tables(cfg):
    """(name, results files) of every evaluation's table: its glob patterns, each sorted, in order."""
    return [(ev.table.name, [f for pattern in ev.table.files for f in sorted(glob(f"{cfg.results}/{pattern}"))])
            for ev in cfg.evals.values()]


def execute(cmd, dry_run, **kwargs):
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, **kwargs)


def main():
    overrides = script_overrides(sys.argv[1:], __doc__, "run")
    cfg = compose_config("run", overrides)
    if cfg.stage not in ("train", "eval", "tables", "all"):
        raise SystemExit(f"stage={cfg.stage!r} is not one of train, eval, tables, all")
    task_overrides = [o for o in overrides if o.lstrip("+~").split("=")[0].split(".")[0] == "task"]
    os.chdir(REPO)  # outputs and results are relative to the repository root, as the scripts' paths
    os.environ.setdefault("STABLEWM_HOME", str(STABLEWM_HOME))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # processes in a row on the GB10's shared memory

    if cfg.stage in ("train", "all"):
        for out, train_overrides in train_runs(cfg, task_overrides):
            if Path(f"{out}.pt").exists():
                print(f"skip {out} (exists)")
                continue
            print(f"=== train {out}", flush=True)
            execute([sys.executable, "scripts/train.py", *train_overrides], cfg.dry_run)
    if cfg.stage in ("eval", "all"):
        for path, eval_overrides in eval_runs(cfg):
            if Path(path).exists():
                print(f"skip {path} (exists)")
                continue
            print(f"=== eval {path}", flush=True)
            if not cfg.dry_run:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            execute([sys.executable, "scripts/evaluate.py", *eval_overrides], cfg.dry_run)
    if cfg.stage in ("tables", "all"):
        for name, files in tables(cfg):
            if not files:
                print(f"(no results for {name} yet)")
                continue
            md, csv = f"{cfg.results}/tables/{name}.md", f"{cfg.results}/tables/{name}.csv"
            print(f"=== table {md}", flush=True)
            if cfg.dry_run:
                print(shlex.join([sys.executable, "scripts/aggregate_results.py", *files, "--csv", csv]))
                continue
            Path(md).parent.mkdir(parents=True, exist_ok=True)
            with open(md, "w") as f:
                subprocess.run([sys.executable, "scripts/aggregate_results.py", *files, "--csv", csv], stdout=f,
                               check=True)
            print(f"\n## {cfg.task.name}: {name}\n\n{Path(md).read_text()}")


if __name__ == "__main__":
    main()
