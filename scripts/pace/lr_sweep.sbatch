#!/bin/bash
# M5 LR sweep on PACE-ICE: one array task per config in configs/lr_sweep/ (sorted).
# Submit from the repo root after scripts/pace/setup_pace.sh:
#     sbatch --array=0-N scripts/pace/lr_sweep.sbatch   (setup_pace.sh prints the exact N)
# Rerun a single config:  sbatch --array=3 scripts/pace/lr_sweep.sbatch (replace 3 with whatever corresponding config file you want.)
#SBATCH --job-name=aidet-lr-sweep
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8            # matches num_workers=8; JPEG decoding is the bottleneck
#SBATCH --mem=32G
#SBATCH --time=02:00:00              # a 12k-image x 10-epoch run should take well under an hour
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
VENV="${VENV:-$SCRATCH_ROOT/venvs/aidet}"
export HF_HOME="$SCRATCH_ROOT/.cache/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/.cache/torch"
export WANDB_DIR="$SCRATCH_ROOT/.cache/wandb"
WANDB_MODE="${WANDB_MODE:-online}"   # submit with WANDB_MODE=offline if nodes can't reach W&B, then `wandb sync`
source "$VENV/bin/activate"

CONFIGS=(configs/lr_sweep/*.yaml)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CFG" .yaml)"
RUN_DIR="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")/$NAME"

if [ -f "$RUN_DIR/metrics.json" ]; then
    echo "$NAME already finished ($RUN_DIR/metrics.json); skipping."
    exit 0
fi

echo "task $SLURM_ARRAY_TASK_ID -> $CFG on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python train.py --config "$CFG" --wandb-mode "$WANDB_MODE"
