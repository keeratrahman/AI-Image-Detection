#!/bin/bash
# Local LR sweep. Runs configs/lr_sweep/<pattern>.yaml in order, skipping finished runs.
#   caffeinate -i bash scripts/run_lr_sweep.sh "*resnet50*"    # one model
#   caffeinate -i bash scripts/run_lr_sweep.sh                 # everything
# Safe to rerun after an interruption; a failed run does not stop the rest.
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p logs
for CFG in configs/lr_sweep/${1:-*}.yaml; do
  NAME="$(basename "$CFG" .yaml)"
  RUN_DIR="$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")/$NAME"
  if [ -f "$RUN_DIR/metrics.json" ]; then echo "$NAME already finished; skipping."; continue; fi
  echo "=== $NAME ==="
  python -u train.py --config "$CFG" --wandb-mode "$WANDB_MODE" 2>&1 | tee "logs/$NAME.log" || echo "FAILED: $NAME"
#   python train.py --config "$CFG" --wandb-mode "$WANDB_MODE" 2>&1 | tee "logs/$NAME.log" || echo "FAILED: $NAME"
done