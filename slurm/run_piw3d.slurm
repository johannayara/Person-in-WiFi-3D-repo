#!/usr/bin/env bash
# =============================================================================
# Person-in-WiFi 3D -- SLURM run script (training + evaluation)
#
# Requires the environment provisioned by slurm/setup_piw3d.slurm. Submit from
# the repository root (SLURM_SUBMIT_DIR locates the project):
#
#     cd /path/to/piw3d
#     sbatch slurm/setup_piw3d.slurm     # once
#     sbatch slurm/run_piw3d.slurm       # then run
#
# Defaults are a fast smoke test: 1 epoch over 64 frames, then evaluation.
#
# Full training as in the paper (450 epochs, all frames):
#     EPOCHS=450 LIMIT=0 sbatch slurm/run_piw3d.slurm
#
# Evaluate an existing checkpoint only:
#     MODE=test CHECKPOINT=/path/to/epoch_41.pth sbatch slurm/run_piw3d.slurm
#
# Multi-GPU (pass the matching --gres at submit time too):
#     GPUS=4 sbatch --gres=gpu:4 slurm/run_piw3d.slurm
#
# Knobs (all optional):
#   MODE=train|test|train_test   GPUS=1   EPOCHS=1   LIMIT=64
#   CONFIG=configs/wifi/petr_wifi.py   CHECKPOINT=   EVAL=mpjpe
#   SAMPLES_PER_GPU=32   VENV=$PWD/.venv
#   CUDA_MODULE=cuda/13.0   PYTHON_MODULE=python/3.10
#   PROVISION=auto|1   (1 provisions here instead of requiring setup first)
# =============================================================================
#SBATCH --job-name=piw3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# #SBATCH --partition=<your-partition>
# #SBATCH --account=<your-account>

set -euo pipefail

MODE="${MODE:-train_test}"
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-1}"
LIMIT="${LIMIT:-64}"
CONFIG="${CONFIG:-configs/wifi/petr_wifi.py}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL="${EVAL:-mpjpe}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-32}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/slurm/piw3d_env.sh"
cd "$PROJECT_DIR"

log "=== run: mode=$MODE config=$CONFIG gpus=$GPUS epochs=$EPOCHS limit=$LIMIT ==="
load_cluster_modules
require_uv
require_docs_env
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
verify_env

# --- dataset root ------------------------------------------------------------
CFG_OPTIONS=(data.samples_per_gpu="$SAMPLES_PER_GPU" runner.max_epochs="$EPOCHS")
DATA_ROOT="$PROJECT_DIR/data/wifipose"

if [ "$LIMIT" -gt 0 ]; then
    SMOKE_ROOT="$PROJECT_DIR/result/_smoke_data"
    log "preparing truncated dataset ($LIMIT frames/split) in $SMOKE_ROOT"
    rm -rf "$SMOKE_ROOT"
    mkdir -p "$SMOKE_ROOT"
    for split in train_data test_data; do
        mkdir -p "$SMOKE_ROOT/$split"
        for sub in csi keypoint smpl; do
            if [ -d "$DATA_ROOT/$split/$sub" ]; then
                ln -s "$DATA_ROOT/$split/$sub" "$SMOKE_ROOT/$split/$sub"
            fi
        done
        src_list="$DATA_ROOT/$split/${split%_data}_data_list.txt"
        head -n "$LIMIT" "$src_list" > "$SMOKE_ROOT/$split/${split%_data}_data_list.txt"
    done
    DATA_ROOT="$SMOKE_ROOT"
    CFG_OPTIONS+=(
        data_root="$DATA_ROOT"
        data.train.dataset_root="$DATA_ROOT/train_data"
        data.val.dataset_root="$DATA_ROOT/test_data"
        data.test.dataset_root="$DATA_ROOT/test_data"
    )
fi

# --- run ---------------------------------------------------------------------
WORK_DIR="$PROJECT_DIR/result/wifipose"

run_train() {
    local args=("$CONFIG" --work-dir "$WORK_DIR" --cfg-options "${CFG_OPTIONS[@]}")
    if [ "$GPUS" -gt 1 ]; then
        log "training: gpus=$GPUS"
        "$VENV/bin/torchrun" --standalone --nproc_per_node="$GPUS" \
            tools/train.py "${args[@]}" --launcher pytorch
    else
        log "training: gpus=1"
        "$VENV/bin/python" tools/train.py "${args[@]}"
    fi
}

run_test() {
    local ckpt="$1"
    if [ ! -f "$ckpt" ]; then
        log "ERROR: checkpoint not found: $ckpt"
        exit 1
    fi
    local args=("$CONFIG" "$ckpt" --eval "$EVAL" --work-dir "$WORK_DIR/eval"
                --cfg-options "${CFG_OPTIONS[@]}")
    if [ "$GPUS" -gt 1 ]; then
        log "evaluating: $ckpt (metric=$EVAL, gpus=$GPUS)"
        "$VENV/bin/torchrun" --standalone --nproc_per_node="$GPUS" \
            tools/test.py "${args[@]}" --launcher pytorch
    else
        log "evaluating: $ckpt (metric=$EVAL, gpus=1)"
        "$VENV/bin/python" tools/test.py "${args[@]}"
    fi
}

case "$MODE" in
    train)      run_train ;;
    test)       run_test "${CHECKPOINT:-$WORK_DIR/latest.pth}" ;;
    train_test) run_train; run_test "${CHECKPOINT:-$WORK_DIR/latest.pth}" ;;
    *)          log "ERROR: unknown MODE=$MODE (train|test|train_test)"; exit 2 ;;
esac

log "done"
