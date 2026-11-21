#!/bin/bash
# Evaluate finished runs on PACE-ICE: one array task per run dir matching RUNS (sorted).
# Submit from the repo root:
#     RUNS='runs_final/*/' EVAL_ARGS='--aligned-test --skip-clean' sbatch --array=0-N scripts/pace/eval.sbatch
# N = number of matching run dirs minus 1:  ls -d runs_final/*/ | wc -l
# Every run is evaluated on the 8 test_*.jsonl manifests with --save-preds; EVAL_ARGS adds the rest.
#SBATCH --job-name=aidet-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
VENV="${VENV:-$SCRATCH_ROOT/venvs/aidet}"
MANIFESTS="${MANIFESTS:-$SCRATCH_ROOT/genimage_subset/manifests}"
RUNS="${RUNS:-runs_final/*/}"
EVAL_ARGS="${EVAL_ARGS:-}"
export TORCH_HOME="$SCRATCH_ROOT/.cache/torch"
source "$VENV/bin/activate"

shopt -s nullglob
RUN_DIRS=($RUNS)
RUN="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]:?no run dir for task $SLURM_ARRAY_TASK_ID (RUNS=$RUNS matched ${#RUN_DIRS[@]})}"

echo "task $SLURM_ARRAY_TASK_ID -> $RUN on $(hostname)"
# shellcheck disable=SC2086  # EVAL_ARGS is split into flags on purpose
python eval.py --run-dir "$RUN" --manifests "$MANIFESTS"/test_*.jsonl \
    --seen test_sdv14 --near test_sdv15 test_wukong \
    --far test_adm test_glide test_vqdm test_midjourney test_biggan \
    --num-workers 8 --save-preds $EVAL_ARGS
