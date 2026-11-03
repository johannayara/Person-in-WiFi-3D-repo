#!/usr/bin/env bash
# =============================================================================
# Person-in-WiFi 3D -- SLURM environment setup
#
# Provisions the uv venv exactly as docs/get_started.md describes (mmcv with
# ops, mmdet, opera, requirements) without touching pyproject.toml or
# requirements.txt. Run this ONCE before slurm/run_piw3d.slurm.
#
# Submit from the repository root (SLURM_SUBMIT_DIR locates the project):
#
#     cd /path/to/piw3d
#     sbatch slurm/setup_piw3d.slurm
#
# Needs network access (uv sync + pip installs), so submit it to a login/CPU
# node that has internet. Compiling mmcv's CUDA ops needs nvcc: load a CUDA
# module (default CUDA_MODULE=cuda/13.0) or set FORCE_CUDA=1. A GPU device is
# not required to compile.
#
# Can also be run interactively on a login node:
#     bash slurm/setup_piw3d.slurm
#
# Knobs (all optional):
#   VENV=$PWD/.venv   PY_VERSION=3.10   UV=uv
#   CUDA_MODULE=cuda/13.0   PYTHON_MODULE=python/3.10
#   BUILD_MMCV_OPS=auto|1|0   FORCE_CUDA=auto|1|0   SYNC=auto|0
#   PROVISION=auto|0   (0 = only uv sync, skip the docs install steps)
# =============================================================================
#SBATCH --job-name=piw3d-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# #SBATCH --partition=<your-partition>
# #SBATCH --account=<your-account>

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/slurm/piw3d_env.sh"
cd "$PROJECT_DIR"

log "=== setup: $PROJECT_DIR (venv=$VENV) ==="
load_cluster_modules
require_uv
ensure_base_env
setup_docs_env
if [ "$PROVISION" != "0" ]; then
    verify_env
fi

log "setup complete -- now submit: sbatch slurm/run_piw3d.slurm"
