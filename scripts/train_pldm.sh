#!/usr/bin/env bash
# Train PLDM for one task with stable-worldmodel's own recipe (its scripts/train/pldm.py and config: LeWM's
# architecture and data pipeline, the VICReg-style PLDM loss), then convert the result to
# $STABLEWM_HOME/<task>/pldm_object.ckpt and encode the task's dataset with it (<dataset>_pldm_latents).
#
#   SWM_REPO=~/stable-worldmodel SWM_PYTHON=~/venv-swm/bin/python bash scripts/train_pldm.sh tworoom
#   ... bash scripts/train_pldm.sh pusht wandb.enabled=true wandb.config.entity=<you> wandb.config.project=gplan
#
# SWM_REPO: a clone of github.com/galilai-group/stable-worldmodel (checked with commit 21446f1); SWM_PYTHON: the
# Python of a separate environment with `pip install -e "$SWM_REPO[train,format]"` (it needs a newer
# stable-worldmodel than evaluation's 0.0.6, and transformers 5). Run from the gplan repository with its own
# environment active (conversion and encoding use it). Extra arguments are Hydra overrides of pldm.py.
# Weights are saved every 5 epochs to $STABLEWM_HOME/checkpoints/<task>/pldm/weights_epoch_<n>.pt.
set -euo pipefail
TASK=${1:?usage: bash scripts/train_pldm.sh tworoom|pusht|reacher|cube [pldm.py overrides...]}; shift
: "${SWM_REPO:?set SWM_REPO to the stable-worldmodel clone}" "${SWM_PYTHON:?set SWM_PYTHON to the python of its environment}"
export STABLEWM_HOME=${STABLEWM_HOME:-$HOME/.stable_worldmodel}

case $TASK in  # stable-worldmodel's data config, and our file of that dataset
  tworoom) DATA=tworoom DATASET=tworoom ;;
  pusht)   DATA=pusht   DATASET=pusht_expert_train ;;
  reacher) DATA=dmc     DATASET=dmc/reacher_random ;;
  cube)    DATA=ogb     DATASET=ogbench/cube_single_expert ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac
RUN=$STABLEWM_HOME/checkpoints/$TASK/pldm

(cd "$SWM_REPO" && "$SWM_PYTHON" scripts/train/pldm.py data=$DATA "data.dataset.name=$STABLEWM_HOME/$DATASET.h5" \
   output_model_name=$TASK/pldm subdir=$TASK-pldm "$@")
python scripts/convert_lewm.py --run-dir "$RUN" --name $TASK/pldm
python scripts/encode_dataset.py --dataset $DATASET --ckpt "$STABLEWM_HOME/$TASK/pldm_object.ckpt"
