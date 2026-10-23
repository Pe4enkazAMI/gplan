# Evaluation

`scripts/evaluate.py` runs the LeWM evaluation protocol (`le-wm/eval.py` with
`le-wm/config/eval/<task>.yaml`; details in `gplan/torch/evaluation.py`): start states and goals come
from the dataset (the goal is the state `goal_offset` steps ahead), the agent has `eval_budget` env
steps to reach it, and an episode is a success if the env terminates there. Every policy of a run is
evaluated on the *same* (episode, start step) pairs and env seeds, drawn exactly as `le-wm/eval.py`
draws them, so CEM's numbers are comparable to the LeWM paper's.

Every policy has its own settings, so CEM always runs the LeWM paper configuration
(`cem_samples=300 cem_steps=30 cem_topk=30`) whatever GFlow checkpoint or settings sit next to it.

| Setting | Meaning |
| --- | --- |
| `"policy=[gflow,gflow-refine,cem,prior,prior-refine,random]"` | policies, run in this order on the same episodes (default `[gflow,cem]`) |
| `planner=<.pt>` | the `train.py` checkpoint the `gflow*` policies sample |
| `gflow_samples=N` (64) | plans drawn from the GPlanner per replanning step |
| `gflow_select=min-cost \| sample \| mean` | execute the cheapest of N, one sample, or the most likely mode |
| `refine_steps / refine_lr / refine_objective / refine_beta` | `gflow-refine` / `prior-refine`: Adam steps on the actions through LeWM before picking the cheapest, on βJ + ‖A‖²/2 (`target`; β defaults to the planner's) or on J (`cost`) |
| `prior_samples=N` (64) `prior_var` (1) | `prior`: best-of-N random shooting from N(0, s² I); `prior-refine`: the same plans refined like `gflow-refine` (the control for it) |
| `holdout_frac=f split_seed=s` | draw episodes only from the episodes `train.py` held out with the same settings; a planner that trained on them is refused (default 0: every episode, the LeWM protocol) |
| `allow_mismatch=true` | evaluate a planner trained on another dataset, LeWM checkpoint, goal offset or plan layout (an error otherwise) |
| `label=NAME` | name of the planner config in the results file; `aggregate_results.py` pools runs with the same label |
| `goal_offset=25 eval_budget=50 num_eval=50 seed=42` | the LeWM protocol |
| `results_json=PATH` | every policy's metrics and per-episode successes, for sweeps over seeds |
| `deterministic=true` | deterministic cuDNN / cuBLAS kernels: bit-identical GPU reruns (slower) |

The table reports success, the LeWM cost of the executed plans (recomputed the same way for every
solver; per replan in wandb / json), the largest executed |action| in z-scored units (values well
above ~3 are outside LeWM's training data), and compute per replanning step as world-model passes:
forward rollouts (`fwd/replan`) and backward passes (`bwd/replan`, refine policies only, each ~2x a
forward). The pairwise lines count episodes solved by both / only one / neither policy.

`scripts/aggregate_results.py` pools results files into a table: success with a Wilson 95% interval
over the pooled episodes, the spread across training seeds of one label, plan cost, compute per
replan, and a paired comparison with CEM on the same episodes (exact McNemar p; with several
checkpoints pooled against one CEM run, read p as indicative).
