#!/usr/bin/env bash
# =============================================================================
# Person-in-WiFi 3D -- SLURM run script (training + evaluation)
#
# Runs inside a single conda environment (default: piw3d) that already has the
# project's dependencies installed (torch, mmcv with ops, mmdet, opera). Build
# it once, e.g.:
#
#     conda create -n piw3d python=3.10
#     conda activate piw3d
#     # then follow docs/get_started.md (mmcv ops, mmdet, opera)
#
# Submit from the repository root (SLURM_SUBMIT_DIR locates the project):
#
#     cd /path/to/piw3d
#     sbatch slurm/run_piw3d.slurm
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
#   SAMPLES_PER_GPU=32
#   CONDA_ENV=piw3d   CONDA_EXE=/path/to/conda   (auto-detected when unset)
#   CUDA_MODULE=cuda/13.0   (use "none" to skip; not required when the env
#                            already ships torch's bundled CUDA runtime)
# =============================================================================
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:2

set -euo pipefail

MODE="${MODE:-train_test}"
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-1}"
LIMIT="${LIMIT:-64}"
CONFIG="${CONFIG:-configs/wifi/petr_wifi.py}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL="${EVAL:-mpjpe}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-32}"
CONDA_ENV="${CONDA_ENV:-piw3d}"
CONDA_EXE="${CONDA_EXE:-}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

log() { echo "[$(date '+%F %T')] $*"; }

# Optional cluster toolkit module (GPU runtime/driver). Skipped when the conda
# env already bundles what torch needs, or when the cluster exposes none.
load_cluster_modules() {
    if [ "$CUDA_MODULE" = "none" ]; then
        return 0
    fi
    if ! command -v module >/dev/null 2>&1; then
        log "WARN: no 'module' command; using the current environment"
        return 0
    fi
    module purge || true
    module load "$CUDA_MODULE" \
        || log "WARN: 'module load $CUDA_MODULE' failed; run 'module avail 2>&1 | grep -i cuda' and set CUDA_MODULE"
}

# Put the conda env's bin/ first on PATH. Sources conda.sh (rather than relying
# on an interactive shell hook) so `conda activate` works under sbatch.
activate_conda_env() {
    local conda="$CONDA_EXE" base
    if [ -z "$conda" ]; then
        if [ -n "${CONDA_EXE:-}" ]; then
            conda="$CONDA_EXE"
        elif command -v conda >/dev/null 2>&1; then
            conda="$(command -v conda)"
        elif [ -n "${CONDA_PREFIX:-}" ]; then
            conda="$CONDA_PREFIX/bin/conda"
        fi
    fi
    if [ -z "$conda" ]; then
        log "ERROR: could not find 'conda'. Set CONDA_EXE=/path/to/conda."
        return 1
    fi
    base="$("$conda" info --base 2>/dev/null || true)"
    if [ -z "$base" ] || [ ! -f "$base/etc/profile.d/conda.sh" ]; then
        log "ERROR: could not locate conda.sh via '$conda info --base' (got '$base')."
        log "       Set CONDA_EXE to the conda installation that owns '$CONDA_ENV'."
        return 1
    fi
    # shellcheck disable=SC1090
    source "$base/etc/profile.d/conda.sh"
    if ! conda activate "$CONDA_ENV"; then
        log "ERROR: 'conda activate $CONDA_ENV' failed; create the env first or set CONDA_ENV."
        return 1
    fi
    log "conda env: $CONDA_PREFIX (python: $(command -v python))"
}

# Prove the conda env actually carries the project's dependencies.
verify_env() {
    python - <<'PY'
import os
import sys
import torch, mmcv, mmdet, opera
expected = os.path.join(os.environ["PROJECT_DIR"], "third_party", "mmcv")
assert os.path.realpath(mmcv.__file__).startswith(os.path.realpath(expected)), mmcv.__file__
import mmcv._ext  # noqa: F401  (C++/CUDA ops)
print("env OK: python=%s torch=%s mmcv=%s mmdet=%s opera=%s cuda=%s"
      % (sys.executable, torch.__version__, mmcv.__version__,
         mmdet.__version__, opera.__version__, torch.version.cuda))
PY
}

cd "$PROJECT_DIR"

log "=== run: mode=$MODE config=$CONFIG gpus=$GPUS epochs=$EPOCHS limit=$LIMIT ==="
load_cluster_modules
activate_conda_env
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
        torchrun --standalone --nproc_per_node="$GPUS" \
            tools/train.py "${args[@]}" --launcher pytorch
    else
        log "training: gpus=1"
        python tools/train.py "${args[@]}"
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
        torchrun --standalone --nproc_per_node="$GPUS" \
            tools/test.py "${args[@]}" --launcher pytorch
    else
        log "evaluating: $ckpt (metric=$EVAL, gpus=1)"
        python tools/test.py "${args[@]}"
    fi
}

case "$MODE" in
    train)      run_train ;;
    test)       run_test "${CHECKPOINT:-$WORK_DIR/latest.pth}" ;;
    train_test) run_train; run_test "${CHECKPOINT:-$WORK_DIR/latest.pth}" ;;
    *)          log "ERROR: unknown MODE=$MODE (train|test|train_test)"; exit 2 ;;
esac

log "done"
