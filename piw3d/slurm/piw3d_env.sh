#!/usr/bin/env bash
# =============================================================================
# Shared environment bootstrap for the Person-in-WiFi 3D SLURM scripts.
# Sourced (never executed) by slurm/setup_piw3d.slurm and slurm/run_piw3d.slurm.
#
# It provisions the uv venv exactly as docs/get_started.md prescribes, without
# modifying pyproject.toml or requirements.txt:
#   1. mmcv, editable, with C++/CUDA ops        (MMCV_WITH_OPS=1)
#   2. mmdet, editable
#   3. opera: `pip install -r requirements.txt` + editable install
#   plus the unit-test dependencies from requirements/{tests,optional}.txt.
#
# Shared knobs (all optional):
#   VENV=$PWD/.venv        PY_VERSION=3.10        UV=uv
#   CUDA_MODULE=cuda/13.0  PYTHON_MODULE=python/3.10
#   BUILD_MMCV_OPS=auto|1|0    FORCE_CUDA=auto|1|0
#   SYNC=auto|0            PROVISION=auto|1
# =============================================================================

PY_VERSION="${PY_VERSION:-3.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"
BUILD_MMCV_OPS="${BUILD_MMCV_OPS:-auto}"
FORCE_CUDA="${FORCE_CUDA:-auto}"
SYNC="${SYNC:-auto}"
PROVISION="${PROVISION:-auto}"
UV="${UV:-uv}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
DOCS_ENV_STAMP="$VENV/.piw3d-docs-env"
export PROJECT_DIR VENV

log() { echo "[$(date '+%F %T')] $*"; }

load_cluster_modules() {
    if command -v module >/dev/null 2>&1; then
        module purge || true
        module load "$CUDA_MODULE" || log "WARN: 'module load $CUDA_MODULE' failed"
        module load "$PYTHON_MODULE" || log "WARN: 'module load $PYTHON_MODULE' failed"
    else
        log "WARN: no 'module' command; using the current environment"
    fi
}

require_uv() {
    if ! command -v "$UV" >/dev/null 2>&1; then
        log "ERROR: '$UV' not found (install it with: curl -LsSf https://astral.sh/uv/install.sh | sh)"
        return 1
    fi
}

uv_pip() { "$UV" pip install --python "$VENV/bin/python" "$@"; }

# Create the venv and install the uv-locked base (torch + every pinned dep).
ensure_base_env() {
    if [ ! -x "$VENV/bin/python" ]; then
        log "creating venv at $VENV"
        "$UV" venv --python "$PY_VERSION" "$VENV" || return 1
    fi
    export VIRTUAL_ENV="$VENV"
    export PATH="$VENV/bin:$PATH"

    if [ "$SYNC" != "0" ] && ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
        log "uv sync --frozen (torch + locked deps; may need network access)"
        if ! (cd "$PROJECT_DIR" && UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen); then
            log "ERROR: 'uv sync' failed. Run the setup job on a login node with network,"
            log "       then submit the run job against the same \$VENV."
            return 1
        fi
    fi
    if ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
        log "ERROR: torch is not available in $VENV; run slurm/setup_piw3d.slurm first"
        return 1
    fi
}

# docs/get_started.md steps 1-3 + unit-test deps. Idempotent: each part is only
# redone when it is actually missing.
provision_docs_env() {
    # mmcv/mmdet setup.py import pkg_resources, removed in setuptools >= 82;
    # wheel is needed for --no-build-isolation editable builds.
    uv_pip "setuptools<82" wheel

    # step 1 -- mmcv, editable, with C++/CUDA ops.
    # MMCV_WITH_OPS=1 names the distribution 'mmcv-full' and enables the
    # extension; --no-build-isolation forwards that env var (and the venv's
    # torch) into setup.py, which uv's isolated build would otherwise drop.
    if "$VENV/bin/python" -c "import mmcv._ext" >/dev/null 2>&1; then
        log "mmcv._ext already present"
    else
        if [ "$BUILD_MMCV_OPS" = "0" ]; then
            log "ERROR: mmcv._ext is missing and BUILD_MMCV_OPS=0"
            return 1
        fi
        local build_cuda=0
        case "$FORCE_CUDA" in
            1) build_cuda=1 ;;
            0) build_cuda=0 ;;
            *) if command -v nvcc >/dev/null 2>&1; then build_cuda=1; fi ;;
        esac
        if [ "$build_cuda" = "0" ]; then
            log "WARN: building mmcv ops WITHOUT CUDA; GPU training will fail."
            log "      Load a CUDA module (nvcc) or set FORCE_CUDA=1."
        fi
        log "installing mmcv (editable) with ops (CUDA=$build_cuda)"
        "$UV" pip uninstall --python "$VENV/bin/python" -y mmcv mmcv-full >/dev/null 2>&1 || true
        (
            cd third_party/mmcv
            MMCV_WITH_OPS=1 FORCE_CUDA="$build_cuda" \
                "$UV" pip install --python "$VENV/bin/python" --no-build-isolation -e .
        ) || return 1
    fi

    # step 2 -- mmdet, editable.
    local mmdet_path
    mmdet_path="$("$VENV/bin/python" -c 'import os,mmdet;print(os.path.realpath(mmdet.__file__))' 2>/dev/null || true)"
    if [[ "$mmdet_path" == "$PROJECT_DIR"/third_party/mmdet/* ]]; then
        log "mmdet already editable"
    else
        log "installing mmdet (editable)"
        "$UV" pip uninstall --python "$VENV/bin/python" -y mmdet >/dev/null 2>&1 || true
        (cd third_party/mmdet && "$UV" pip install --python "$VENV/bin/python" --no-build-isolation -e .) || return 1
    fi

    # step 3 -- opera requirements + opera, editable.
    # NOTE: pyproject.toml declares the project as 'piw3d' and shadows setup.py,
    # so `pip install -e .` would install piw3d rather than the `opera` package
    # the docs target. Run setuptools' develop command directly instead.
    log "installing requirements.txt"
    uv_pip -r requirements.txt || return 1
    if "$UV" pip show --python "$VENV/bin/python" opera >/dev/null 2>&1; then
        log "opera already installed"
    else
        log "installing opera (editable)"
        (cd "$PROJECT_DIR" && "$VENV/bin/python" setup.py develop --no-deps) || return 1
    fi

    # docs/get_started.md unit-test dependencies.
    log "installing tests/optional requirements"
    uv_pip -r requirements/tests.txt -r requirements/optional.txt || return 1

    touch "$DOCS_ENV_STAMP"
}

# Used by the setup job: provision unless explicitly disabled.
setup_docs_env() {
    if [ "$PROVISION" = "0" ]; then
        log "PROVISION=0: skipping the docs/get_started.md install steps"
        return 0
    fi
    provision_docs_env
}

# Used by the run job: require a provisioned environment (PROVISION=1 forces a
# fresh provisioning pass instead).
require_docs_env() {
    if [ "$PROVISION" = "1" ]; then
        ensure_base_env || return 1
        provision_docs_env
        return $?
    fi
    if [ ! -f "$DOCS_ENV_STAMP" ]; then
        log "ERROR: environment not provisioned ($DOCS_ENV_STAMP missing)."
        log "       Submit slurm/setup_piw3d.slurm first, or set PROVISION=1."
        return 1
    fi
    log "using provisioned environment ($DOCS_ENV_STAMP)"
}

# Prove the docs' install steps took effect.
verify_env() {
    "$VENV/bin/python" - <<'PY'
import os
import torch, mmcv, mmdet, opera
expected = os.path.join(os.environ["PROJECT_DIR"], "third_party", "mmcv")
assert os.path.realpath(mmcv.__file__).startswith(os.path.realpath(expected)), mmcv.__file__
import mmcv._ext  # noqa: F401
print("docs env OK: torch=%s mmcv=%s mmdet=%s opera=%s"
      % (torch.__version__, mmcv.__version__, mmdet.__version__, opera.__version__))
PY
}
