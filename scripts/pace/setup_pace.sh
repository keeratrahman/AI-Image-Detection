#!/bin/bash
# One-time setup on a PACE-ICE login node (login-ice.pace.gatech.edu). Run from the repo root:
#     bash scripts/pace/setup_pace.sh
# Creates the venv, prefetches pretrained weights (in case compute nodes can't download),
# and writes the M5 LR-sweep configs into configs/lr_sweep/.
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"           # home quota is small; keep big things here
VENV="${VENV:-$SCRATCH_ROOT/venvs/aidet}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/genimage_subset}"  # copy of the shared subset (see docs/DATA.md)
RUNS_DIR="${RUNS_DIR:-$SCRATCH_ROOT/aidet_runs}"
export UV_CACHE_DIR="$SCRATCH_ROOT/.cache/uv"
export HF_HOME="$SCRATCH_ROOT/.cache/huggingface"       # timm weights
export TORCH_HOME="$SCRATCH_ROOT/.cache/torch"          # torchvision weights

if [ ! -f "$DATA_ROOT/manifests/A_train.jsonl" ]; then
    echo "Missing $DATA_ROOT/manifests/A_train.jsonl. Copy the subset first, e.g. from your laptop:" >&2
    echo "  rsync -avP genimage_subset/ <gt-user>@login-ice.pace.gatech.edu:scratch/genimage_subset/" >&2
    exit 1
fi

# uv ships its own Python and the torch wheels bundle CUDA, so no `module load` is needed.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.11
source "$VENV/bin/activate"
uv pip install -r requirements.txt

python - <<'EOF'
import torchvision.models as tv, timm
tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
timm.create_model("deit_small_patch16_224", pretrained=True)
print("pretrained weights cached")
EOF

# M5: Protocol A / jpeg-crop / seed 0, each model on its own 3-LR grid (PLAN.md §1.4).
# resnet18 needs a best LR for M6.
COMMON=(--protocol A --init in1k --preps jpeg-crop --seeds 0 --num-workers 8
        --manifest-dir "$DATA_ROOT/manifests" --output-dir "$RUNS_DIR" --out configs/lr_sweep)
python make_configs.py "${COMMON[@]}" --models resnet50 resnet18 --lrs 3e-4 1e-4 3e-5
# deit-s: its plan grid (1e-4 5e-5 1e-5) plus make_configs' shared DEFAULT_LRS (3e-4 1e-4 3e-5),
# in case the team wants one grid for every model. The ResNet grid already equals DEFAULT_LRS.
python make_configs.py "${COMMON[@]}" --models deit-s            --lrs 3e-4 1e-4 5e-5 3e-5 1e-5

mkdir -p logs   # Slurm will not create the --output directory
N=$(ls configs/lr_sweep/*.yaml | wc -l)
echo
echo "Ready. Log in to W&B once (wandb login), then submit:"
echo "sbatch --array=0-$((N - 1)) scripts/pace/lr_sweep.sbatch"
