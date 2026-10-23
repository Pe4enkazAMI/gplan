"""Settings of scripts/train.py and scripts/evaluate.py: every one, its type and default (`TrainArgs`,
`EvalArgs`), composed with Hydra from configs/ and command-line overrides into the flat namespace the
scripts run with.

    python scripts/train.py task=pusht model=gfn-mala seed=1 out=outputs/pusht/gfn-mala-s1
    python scripts/evaluate.py task=pusht "policy=[gflow,cem]" planner=outputs/pusht/gfn-mala-s1.pt

configs/train.yaml picks a task (configs/task: its dataset, LeWM checkpoint and the paper's settings)
and optionally a model (configs/model: a training method and its settings for that task); any field
below can then be overridden with `name=value`. Hydra is used for composition only (its compose API):
no output folders, no change of working directory or logging.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from gplan.data import STABLEWM_HOME

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class TrainArgs:
    """scripts/train.py. Fields [ppo] are read by method=ppo only; MALA and the beta warmup by gfn only."""
    task: Any = None   # the composed configs/task/<name>.yaml (not a training setting itself)
    method: str = "gfn"  # gfn: the GFlowNet; bc: behavior cloning on expert plans; map: amortized MAP planning
    # (gplan/jax/baselines.py); ppo: PPO with a critic (gplan/jax/ppo.py)
    ckpt: Path = MISSING  # LeWM *_object.ckpt (train.yaml: the task's, $STABLEWM_HOME/<task.wm>_object.ckpt)
    dataset: str = MISSING  # dataset name under $STABLEWM_HOME (train.yaml: the task's)
    latents: Optional[Path] = None  # prefix of the precomputed embeddings (default: $STABLEWM_HOME/<dataset>_latents)
    goal_offset: int = 25  # goal = start + this many dataset steps
    holdout_frac: float = 0.0  # hold this fraction of episodes out of training (gplan.data.split_episodes), so
    # evaluate.py holdout_frac / split_seed can test on unseen episodes (0 = train on all)
    split_seed: int = 0  # seed of the episode split
    n_steps: int = 5  # planning horizon T
    frameskip: int = 5  # env steps per plan step: a plan step holds frameskip x the dataset's action dim values,
    # which must be the input size of LeWM's action encoder
    n_components: int = 1  # EXP mixture policy: Gaussians per plan step (1 = a single diagonal Gaussian)
    n_samples: int = 8  # plans sampled per (start, goal)
    beta: float = 1.0  # final cost temperature, per embedding dim (the raw cost is scaled by beta / embed_dim)
    beta_start: float = 0.0  # initial beta when annealing, same units as beta
    beta_warmup: int = 0  # steps to anneal beta linearly from beta_start to beta (0 = constant beta)
    lr: float = 3e-4
    lr_z: float = 3e-3  # learning rate of the log Z head (usually 10-100x lr)
    grad_clip: float = 10.0  # max global grad L2 norm; <= 0 disables
    ls_steps: int = 0  # EXP MALA local search: MALA steps from each on-policy plan; the results are added to
    # the batch as off-policy plans (0 = off)
    ls_every: int = 1  # EXP MALA: run local search every this many steps
    ls_start: int = 0  # EXP MALA: first step with local search (e.g. after beta_warmup)
    ls_step_size: float = 1e-3  # EXP MALA: initial step size eta
    ls_target_accept: float = 0.574  # EXP MALA: eta is adapted toward this acceptance rate (0 = keep eta fixed)
    ppo_reward: str = "target"  # [ppo] plan return: -beta J + log N(A; 0, s^2 I) (the GFlowNet's log target) or
    # "cost": -beta J
    ent_coef: float = 0.0  # [ppo] c in the per-step reward -c log pi(a_t | s_t): 0 = classic PPO, 1 = max-ent
    # (optimum: the GFlowNet target)
    ppo_epochs: int = 4  # [ppo] passes over each rollout
    ppo_minibatches: int = 4  # [ppo] minibatches per pass
    clip_eps: float = 0.2  # [ppo] probability-ratio clipping
    gae_lambda: float = 0.95  # [ppo] GAE lambda (gamma = 1: 5-step episodes)
    vf_coef: float = 0.5  # [ppo] weight of the critic's loss
    target_kl: float = 0.0  # [ppo] skip an iteration's remaining epochs once an epoch's approx KL exceeds this (0 = off)
    ppo_grad_clip: float = 0.5  # [ppo] max global grad norm (policy + critic)
    batch_size: int = 64
    n_batches: int = 1000
    seed: int = 0
    ref_var: float = 1.0  # variance s^2 of the N(0, s^2 I) reference measure
    matmul_precision: str = "highest"  # float32 matmul precision on GPU; "default" allows TF32 (faster, less exact)
    deterministic: bool = False  # deterministic XLA GPU kernels (--xla_gpu_deterministic_ops): bit-identical reruns
    log_every: int = 10
    out: Path = Path("outputs/gplaner")  # checkpoint prefix: <out>.eqx / .json / .pt, and the config <out>.yaml
    wandb_project: str = "gplan"
    wandb_mode: str = "online"
    wandb_name: Optional[str] = None  # run name (default: wandb's random name)


@dataclass
class EvalArgs:
    """scripts/evaluate.py. [gflow] etc.: read by those policies only."""
    policy: Optional[List[str]] = None  # policies to evaluate on the same episodes, in this order (default:
    # [gflow, cem]): gflow, gflow-refine, cem, prior, prior-refine, random
    planner: Optional[str] = None  # [gflow] scripts/train.py checkpoint (.pt)
    gflow_select: str = "min-cost"  # [gflow] execute the cheapest of gflow_samples plans ("min-cost"), a single
    # sample ("sample"), or the mean plan ("mean")
    prior_samples: int = 64  # [prior, prior-refine] plans drawn from N(0, prior_var I) per replanning step
    prior_var: float = 1.0  # [prior, prior-refine] variance of the prior plans
    refine_steps: int = 20  # [*-refine] Adam steps on the actions
    refine_lr: float = 0.05  # [*-refine] Adam step size (z-scored units)
    refine_objective: str = "target"  # [*-refine] minimize beta*J - log N(A) ("target") or J alone ("cost")
    refine_beta: Optional[float] = None  # [*-refine] per-dim beta of the target objective (default: the planner's
    # training beta)
    task: Any = None  # the task (configs/task/<name>.yaml); the run's setting is its name, the preset
    # gplan.torch.evaluation.TASKS[name] (env, dataset, LeWM checkpoint, resets)
    wm: Optional[str] = None  # LeWM ckpt name relative to $STABLEWM_HOME (default: from task)
    dataset: Optional[str] = None  # dataset name (default: from task)
    env: Optional[str] = None  # gymnasium env id (default: from task)
    num_eval: int = 50
    holdout_frac: float = 0.0  # draw episodes only from the held-out side of this episode split (train.py
    # holdout_frac); 0 = all episodes, the LeWM protocol
    split_seed: int = 0  # seed of the episode split (train.py split_seed)
    goal_offset: int = 25
    eval_budget: int = 50
    horizon: int = 5
    action_block: int = 5  # frameskip
    gflow_samples: int = 64  # [gflow, gflow-refine] plans sampled from the GPlanner per replanning step
    cem_samples: int = 300  # [cem] samples per iteration (LeWM paper: 300)
    cem_steps: int = 30  # [cem] optimization iterations
    cem_topk: int = 30  # [cem] elites kept per iteration
    img_size: int = 224
    seed: int = 42
    device: Optional[str] = None  # default: cuda if available, else cpu
    deterministic: bool = False  # deterministic cuDNN / cuBLAS kernels: bit-identical GPU reruns (slower)
    allow_mismatch: bool = False  # evaluate a planner trained on another dataset / LeWM checkpoint / goal offset /
    # plan layout (an error otherwise)
    video_dir: Optional[str] = None  # save rollout videos under <dir>/<policy>/ (off by default)
    label: Optional[str] = None  # name of the planner config in results_json (scripts/aggregate_results.py pools
    # runs with the same label, e.g. several training seeds; default: the planner file's name)
    results_json: Optional[Path] = None  # also write every policy's metrics and per-episode successes here
    wandb_project: str = "gplan"
    wandb_mode: str = "online"
    wandb_name: Optional[str] = None


CHOICES = {  # the allowed values of string settings
    "method": ("gfn", "bc", "map", "ppo"),
    "ppo_reward": ("target", "cost"),
    "matmul_precision": ("highest", "high", "default"),
    "wandb_mode": ("online", "offline", "disabled"),
    "policy": ("gflow", "gflow-refine", "cem", "prior", "prior-refine", "random"),
    "gflow_select": ("min-cost", "sample", "mean"),
    "refine_objective": ("target", "cost"),
}
NOT_ARGS = {TrainArgs: ("task",)}  # TrainArgs.task: the task's config, not a setting of the run

ConfigStore.instance().store(name="train_args", node=TrainArgs)
ConfigStore.instance().store(name="eval_args", node=EvalArgs)
if not OmegaConf.has_resolver("stablewm_home"):
    OmegaConf.register_new_resolver("stablewm_home", lambda: str(STABLEWM_HOME))


def compose_config(name, overrides=()):
    """configs/<name>.yaml with Hydra overrides ("key=value", "task=pusht", ...), resolved."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def to_args(cfg, schema):
    """The flat namespace a script runs with: every field of `schema` but NOT_ARGS, path settings as
    pathlib.Path, string settings checked against CHOICES."""
    args = {}
    for f in fields(schema):
        if f.name in NOT_ARGS.get(schema, ()):
            continue
        value = cfg[f.name]
        if f.name == "task":  # EvalArgs: the composed configs/task/<name>.yaml -> its name
            value = value.name
        elif OmegaConf.is_config(value):
            value = OmegaConf.to_container(value)
        if value is not None and f.type in (Path, Optional[Path]):
            value = Path(value)
        allowed = CHOICES.get(f.name)
        for v in value if isinstance(value, list) else [value]:
            if allowed is not None and v is not None and v not in allowed:
                raise SystemExit(f"{f.name}={v!r} is not one of {list(allowed)}")
        args[f.name] = value
    return SimpleNamespace(**args)


def script_overrides(argv, doc, name):
    """A script's command line as Hydra overrides. `--help` prints the script's docstring and its composed
    config; an old-style `--flag` stops with a hint."""
    if "--help" in argv or "-h" in argv:
        overrides = [a for a in argv if a not in ("--help", "-h")]
        print(f"{doc}\nSettings (gplan/config.py describes each), composed from configs/{name}.yaml"
              f"{' with ' + ' '.join(overrides) if overrides else ''}:\n\n"
              f"{OmegaConf.to_yaml(compose_config(name, overrides))}")
        raise SystemExit(0)
    flags = [a for a in argv if a.startswith("--")]
    if flags:
        raise SystemExit(f"the settings are Hydra overrides, name=value (e.g. beta=90, not --beta 90): got {flags}")
    return list(argv)


def write_config(path, overrides, cfg):
    """The composed config of a run, next to its output, with the overrides that produced it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    tmp.write_text(f"# overrides: {' '.join(overrides) if overrides else '(none)'}\n{OmegaConf.to_yaml(cfg)}")
    tmp.replace(path)
