#!/bin/bash
# One-time local setup. Run from anywhere:  bash scripts/setup_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-../genimage_subset}"         # folder that contains manifests/ and the images
RUNS_DIR="${RUNS_DIR:-runs}"
DEIT_LRS="${DEIT_LRS:-3e-4 1e-4 3e-5}" #midpoint lrs
# DEIT_LRS="${DEIT_LRS:-3e-4 1e-4 5e-5 3e-5 1e-5}"   # use 3 values if the team agrees on a 3-LR grid

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt timm

python - <<'EOF'
import torchvision.models as tv, timm
tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
timm.create_model("deit_small_patch16_224", pretrained=True)
print("pretrained weights cached")
EOF

for f in A_train A_val; do
  [ -f "$DATA_ROOT/manifests/$f.jsonl" ] || { echo "Missing $DATA_ROOT/manifests/$f.jsonl" >&2; exit 1; }
done

rm -f configs/lr_sweep/*.yaml              # avoid stale configs if the LR grid changed
COMMON=(--protocol A --init in1k --preps jpeg-crop --seeds 0 --num-workers 8
        --manifest-dir "$DATA_ROOT/manifests" --output-dir "$RUNS_DIR" --out configs/lr_sweep)
python make_configs.py "${COMMON[@]}" --models resnet50 resnet18 --lrs 3e-4 1e-4 3e-5
python make_configs.py "${COMMON[@]}" --models deit-s --lrs $DEIT_LRS

mkdir -p logs
echo
echo "Ready: $(ls configs/lr_sweep/*.yaml | wc -l) configs. Next: wandb login, then"
echo "  caffeinate -i bash scripts/run_lr_sweep.sh \"*resnet50*\""