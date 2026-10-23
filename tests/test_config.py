"""Settings composed with Hydra (gplan/config.py, configs/): the scripts' defaults, types and checks, the
paper's model settings per task, and the runner's run lists (scripts/run.py)."""

from pathlib import Path

import pytest
from conftest import load_script

from gplan.config import compose_config, to_args, TrainArgs
from gplan.data import STABLEWM_HOME

# scripts/train.py with no settings: the defaults of its former argparse parser, key for key (the
# checkpoints record these as train_args)
TRAIN_DEFAULTS = {
    "method": "gfn", "ckpt": STABLEWM_HOME / "tworoom/lewm_object.ckpt", "dataset": "tworoom", "latents": None,
    "goal_offset": 25, "holdout_frac": 0.0, "split_seed": 0, "n_steps": 5, "frameskip": 5, "n_components": 1,
    "n_samples": 8, "beta": 1.0, "beta_start": 0.0, "beta_warmup": 0, "lr": 0.0003, "lr_z": 0.003, "grad_clip": 10.0,
    "ls_steps": 0, "ls_every": 1, "ls_start": 0, "ls_step_size": 0.001, "ls_target_accept": 0.574,
    "ppo_reward": "target", "ent_coef": 0.0, "ppo_epochs": 4, "ppo_minibatches": 4, "clip_eps": 0.2,
    "gae_lambda": 0.95, "vf_coef": 0.5, "target_kl": 0.0, "ppo_grad_clip": 0.5, "batch_size": 64, "n_batches": 1000,
    "seed": 0, "ref_var": 1.0, "matmul_precision": "highest", "deterministic": False, "log_every": 10,
    "out": Path("outputs/gplaner"), "wandb_project": "gplan", "wandb_mode": "online", "wandb_name": None}


def train_args(*overrides):
    return vars(to_args(compose_config("train", overrides), TrainArgs))


def test_train_defaults_are_the_former_flags_defaults():
    args = train_args()
    assert list(args) == list(TRAIN_DEFAULTS)  # same keys, same order as before
    assert args == TRAIN_DEFAULTS
    assert all(type(args[k]) is type(v) for k, v in TRAIN_DEFAULTS.items())


def test_settings_are_typed_and_checked():
    args = train_args("beta=90", "n_batches=20", "latents=a/b", "deterministic=true", "out=x/y")
    assert args["beta"] == 90.0 and type(args["beta"]) is float  # recorded as 90.0, as with --beta 90
    assert args["latents"] == Path("a/b") and args["out"] == Path("x/y") and args["deterministic"] is True
    with pytest.raises(SystemExit, match="method='sac' is not one of"):
        train_args("method=sac")
    with pytest.raises(Exception, match="n_batches"):
        train_args("n_batches=1.5")
    with pytest.raises(Exception, match="bogus"):
        train_args("bogus=1")
    with pytest.raises(SystemExit, match="Hydra overrides"):
        load_script("train").parse_args(["--beta", "90"])


@pytest.mark.parametrize("task, model, expected", [
    ("tworoom", "gfn", dict(method="gfn", dataset="tworoom", beta=90.0, beta_warmup=1000, n_batches=20000,
                            n_components=1, ls_steps=0)),
    ("pusht", "gfn-mala", dict(method="gfn", dataset="pusht_expert_train", beta=192.0, beta_warmup=30000,
                               n_batches=60000, n_components=7, ls_steps=10, ls_every=10, ls_start=30000)),
    ("cube", "ppo-maxent", dict(method="ppo", dataset="ogbench/cube_single_expert", beta=192.0, beta_warmup=0,
                                n_batches=60000, n_components=7, ent_coef=1.0)),
    ("reacher", "map", dict(method="map", dataset="dmc/reacher_random", beta=90.0, n_batches=20000, n_components=1)),
    ("pusht", "bc", dict(method="bc", beta=1.0, n_batches=20000, n_components=1)),
])
def test_models_take_the_tasks_settings(task, model, expected):
    args = train_args(f"task={task}", f"model={model}")
    assert {k: args[k] for k in expected} == expected
    wm = compose_config("train", [f"task={task}"]).task.wm
    assert args["ckpt"] == STABLEWM_HOME / f"{wm}_object.ckpt"


def test_task_configs_match_the_evaluation_presets():
    pytest.importorskip("stable_worldmodel")
    from gplan.torch.evaluation import TASKS
    for name, preset in TASKS.items():
        task = compose_config("train", [f"task={name}"]).task
        assert (task.name, task.dataset, task.wm) == (name, preset.dataset, preset.wm)


def runner_cfg(*overrides):
    return compose_config("run", overrides)


def test_runner_trains_every_model_and_ablation():
    runner = load_script("run")
    runs = runner.train_runs(runner_cfg("task=pusht", "seeds=[0,1]", "ablation_seeds=[3]"), ["task=pusht"])
    outs = [out for out, _ in runs]
    assert outs[:6] == [f"outputs/pusht/{m}-s0" for m in ("gfn", "gfn-mala", "bc", "map", "ppo", "ppo-maxent")]
    assert len(outs) == 12 + 6 and outs[-1] == "outputs/pusht/ppo-cost-maxent-s3"
    out, overrides = runs[-1]
    assert overrides == ["task=pusht", "model=ppo", "ppo_reward=cost", "ent_coef=1", "seed=3", f"out={out}",
                         "wandb_name=pusht-ppo-cost-maxent-s3", "holdout_frac=0.1", "split_seed=0", "wandb_mode=online"]
    assert len(runner.train_runs(runner_cfg("ablations=false"), [])) == 3 * 6


def test_runner_evaluations_and_the_only_filter():
    runner = load_script("run")
    runs = dict(runner.eval_runs(runner_cfg("task=tworoom", "seeds=[0]", "eval_seeds=[42]", "ablation_seeds=[0]")))
    # heldout / alldata: 2 baselines + 5 models x 2 + map; curve: 2 x 7 + 3 + 5; ablations: 4
    assert len(runs) == 2 * 13 + 22 + 4
    assert runs["results/tworoom/heldout/gfn-mala-s0-e42.json"] == [
        "task=tworoom", "dataset=tworoom", "wm=tworoom/lewm", "num_eval=50", "wandb_mode=online",
        "results_json=results/tworoom/heldout/gfn-mala-s0-e42.json", "seed=42", "holdout_frac=0.1", "split_seed=0",
        "planner=outputs/tworoom/gfn-mala-s0.pt", "label=gfn-mala", "policy=[gflow,gflow-refine]",
        "refine_steps=20", "refine_objective=target", "refine_beta=90"]
    assert runs["results/tworoom/alldata/map-s0-e42.json"][-3:] == ["label=map", "policy=[gflow]", "gflow_select=mean"]
    assert runs["results/tworoom/curve/gfn-refine-n64-k5-s0-e42.json"][-3:] == ["refine_beta=90", "gflow_samples=64",
                                                                                 "refine_steps=5"]  # the run's steps win
    only = dict(runner.eval_runs(runner_cfg("seeds=[0]", "eval_seeds=[42]", "ablation_seeds=[0]",
                                             "only=[ppo,ppo-cost]")))
    assert sorted(Path(p).name for p in only) == sorted(["ppo-s0-e42.json", "ppo-refine-cost-s0-e42.json"] * 2
                                                        + ["ppo-cost-s0-e42.json"])


def test_runner_tables_take_their_files_in_glob_order(tmp_path):
    runner = load_script("run")
    for name in ("heldout/gfn-s0-e42", "heldout/baselines-e42", "heldout/bc-s0-e42", "curve/cem-n30-i3-e42"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"{name}.json").touch()
    tables = dict(runner.tables(runner_cfg(f"results={tmp_path}")))
    assert [Path(f).name for f in tables["heldout"]] == ["baselines-e42.json", "bc-s0-e42.json", "gfn-s0-e42.json"]
    assert [Path(f).name for f in tables["compute_curve"]] == ["baselines-e42.json", "gfn-s0-e42.json",
                                                               "cem-n30-i3-e42.json"]
    assert tables["alldata"] == []


def test_runner_passes_extra_settings_to_every_run():
    runner = load_script("run")
    cfg = runner_cfg("seeds=[0]", "ablations=false", "train_overrides={log_every:100}",
                     "eval_overrides={device:cuda,deterministic:true}")
    assert all(overrides[-1] == "log_every=100" for _, overrides in runner.train_runs(cfg, []))
    for _, overrides in runner.eval_runs(cfg):
        assert overrides[5:7] == ["device=cuda", "deterministic=true"]  # after the task's settings, before the run's
