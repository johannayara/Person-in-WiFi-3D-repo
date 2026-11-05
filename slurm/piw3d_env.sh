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
#   UV_MIN_VERSION=0.13.6  (uv is installed/upgraded if older)
#   PYTHON=python3  (interpreter used for the pip uv fallback)
#   CUDA_MODULE=cuda/13.0  PYTHON_MODULE=python/3.10  (use "none" to skip)
#   CUDA_HOME=/path/to/cuda-13.0  (nvcc used for the mmcv CUDA ops)
#   CUDA_TOOLKIT=auto|pip|none  (auto: fall back to a PyPI toolkit matching
#                                torch when no matching cluster nvcc is found;
#                                pip: always install it)
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
UV_MIN_VERSION="${UV_MIN_VERSION:-0.13.6}"
CUDA_HOME="${CUDA_HOME:-}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-auto}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
DOCS_ENV_STAMP="$VENV/.piw3d-docs-env"
export PROJECT_DIR VENV

# Prefer a user-local uv (standalone or pip --user) over a stale module/system
# one that would otherwise shadow it in PATH.
export PATH="$HOME/.local/bin:$PATH"
if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME
fi

log() { echo "[$(date '+%F %T')] $*"; }

load_cluster_modules() {
    if ! command -v module >/dev/null 2>&1; then
        log "WARN: no 'module' command; using the current environment"
        return 0
    fi
    module purge || true
    if [ -n "$CUDA_MODULE" ] && [ "$CUDA_MODULE" != "none" ]; then
        module load "$CUDA_MODULE" \
            || log "WARN: 'module load $CUDA_MODULE' failed; run 'module avail 2>&1 | grep -i cuda' and set CUDA_MODULE"
    fi
    if [ -n "$PYTHON_MODULE" ] && [ "$PYTHON_MODULE" != "none" ]; then
        module load "$PYTHON_MODULE" \
            || log "WARN: 'module load $PYTHON_MODULE' failed; run 'module avail 2>&1 | grep -i python' and set PYTHON_MODULE"
    fi
}

require_uv() {
    # Kept as a thin alias so callers read naturally.
    ensure_uv
}

# uv.lock is revision 5 and pyproject.toml uses `[tool.uv] extra-build-dependencies`
# plus pyproject-less workspace members; both need a recent uv. Older uv fails
# with "unknown field `extra-build-dependencies`" / "Workspace member ... is
# missing a pyproject.toml". Install/upgrade uv locally when that happens.
uv_at_least() {
    local cur
    cur="$("$1" --version 2>/dev/null | awk '{print $2}')" || return 1
    [ -n "$cur" ] || return 1
    [ "$(printf '%s\n%s\n' "$2" "$cur" | sort -V | head -n1)" = "$2" ]
}

ensure_uv() {
    # `module load` above may have re-prepended the cluster software/bin, so
    # re-assert the user-local dir before resolving.
    export PATH="$HOME/.local/bin:$PATH"
    hash -r 2>/dev/null || true

    if command -v "$UV" >/dev/null 2>&1 && uv_at_least "$UV" "$UV_MIN_VERSION"; then
        UV="$(command -v "$UV")"
        log "uv $("$UV" --version | awk '{print $2}') (>= $UV_MIN_VERSION) at $UV"
        return 0
    fi

    log "installing uv >= $UV_MIN_VERSION (current: $(command -v "$UV" >/dev/null 2>&1 && "$UV" --version 2>/dev/null || echo none))"
    mkdir -p "$HOME/.local/bin"

    # 1) `uv self update <version>` pins the version, but only works for uv
    #    binaries installed by the standalone installer.
    if ! { command -v "$UV" >/dev/null 2>&1 && "$UV" self update "$UV_MIN_VERSION" >/dev/null 2>&1; }; then
        # 2) uv installed via pip (or a module): upgrade through pip --user.
        if ! "${PYTHON:-python3}" -m pip install --user --quiet --upgrade "uv==$UV_MIN_VERSION" >/dev/null 2>&1; then
            # 3) last resort: standalone installer (always fetches the latest
            #    release and ignores version env vars).
            curl -LsSf https://astral.sh/uv/install.sh | \
                env UV_INSTALL_DIR="$HOME/.local/bin" sh >/dev/null 2>&1 || true
        fi
    fi
    hash -r 2>/dev/null || true

    # Pin the absolute path so later `module`/PATH changes cannot re-shadow it.
    local installed
    installed="$(command -v uv 2>/dev/null || true)"
    if [ -n "$installed" ] && uv_at_least "$installed" "$UV_MIN_VERSION"; then
        UV="$installed"
        log "uv $("$UV" --version | awk '{print $2}') installed at $UV"
        return 0
    fi

    log "ERROR: could not install uv >= $UV_MIN_VERSION (tried 'uv self update',"
    log "       'pip install --user uv==$UV_MIN_VERSION', and the curl installer)."
    log "       Install it manually, then pass UV=/abs/path/to/uv."
    log "       curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\"\$HOME/.local/bin\" sh"
    return 1
}

uv_pip() { "$UV" pip install --python "$VENV/bin/python" "$@"; }

# CUDA version (major.minor) the installed torch was compiled against.
torch_cuda_version() {
    "$VENV/bin/python" -c 'import torch;print(torch.version.cuda or "")' 2>/dev/null
}

# CUDA version (major.minor) of the nvcc that torch's build would use.
nvcc_cuda_version() {
    local nvcc
    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        nvcc="$CUDA_HOME/bin/nvcc"
    else
        nvcc="$(command -v nvcc 2>/dev/null || true)"
    fi
    [ -n "$nvcc" ] || return 1
    "$nvcc" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1
}

cuda_major_minor() { echo "$1" | awk -F. '{print $1"."$2}'; }

# True when the nvcc torch's build would use reports the same CUDA major.minor
# as the installed torch. Empty torch version (CPU-only torch) counts as a match
# so an explicit cluster nvcc is still used.
cuda_toolkit_matches_torch() {
    local torch_cuda="$1" nvcc_cuda
    nvcc_cuda="$(nvcc_cuda_version || true)"
    [ -n "$nvcc_cuda" ] || return 1
    [ -z "$torch_cuda" ] || [ "$(cuda_major_minor "$nvcc_cuda")" = "$(cuda_major_minor "$torch_cuda")" ]
}

# Install a CUDA toolkit from PyPI matching the torch build and point CUDA_HOME
# at it. `nvidia-cuda-nvcc` provides nvidia/cu13/bin/nvcc; `nvidia-cuda-runtime`
# provides nvidia/cu13/include/{cuda.h,cuda_runtime.h} and lib/libcudart.so.13.
setup_pip_cuda_toolkit() {
    local tc="$1"
    if [ -z "$tc" ]; then
        log "ERROR: CUDA_TOOLKIT=pip but torch reports no CUDA version"
        return 1
    fi
    local spec="cuda-toolkit[all]==${tc}.*"
    log "installing pip CUDA toolkit to match torch CUDA $tc ($spec)"
    uv_pip "$spec" || return 1

    local pyver sp cand
    pyver="$("$VENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
    sp="$VENV/lib/python$pyver/site-packages"
    for cand in "$sp/nvidia/cu${tc%%.*}" "$sp/nvidia/cu12" "$sp/nvidia/cu13"; do
        [ -x "$cand/bin/nvcc" ] || continue
        # torch's cpp_extension looks in both lib and lib64; the pip layout
        # only ships lib.
        if [ ! -e "$cand/lib64" ] && [ -d "$cand/lib" ]; then
            ln -sfn lib "$cand/lib64" 2>/dev/null || true
        fi
        CUDA_HOME="$cand"
        export CUDA_HOME
        log "using pip CUDA toolkit at $CUDA_HOME"
        return 0
    done
    log "ERROR: installed $spec but found no nvcc under $sp/nvidia"
    return 1
}

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
        local build_cuda=0 torch_cuda nvcc_cuda
        torch_cuda="$(torch_cuda_version)"
        case "$FORCE_CUDA" in
            1) build_cuda=1 ;;
            0) build_cuda=0 ;;
            *) if [ "$CUDA_TOOLKIT" != "none" ] && \
                  { [ -n "$torch_cuda" ] || [ -x "${CUDA_HOME:-}/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1; }; then
                   build_cuda=1
               fi ;;
        esac
        if [ "$CUDA_TOOLKIT" = "none" ]; then
            build_cuda=0
        fi

        if [ "$build_cuda" = "1" ]; then
            # torch aborts the mmcv build when nvcc != the CUDA torch was built
            # with. Clusters typically expose a CUDA module older than the torch
            # wheel (or none at all), so `auto` installs a matching toolkit from
            # PyPI instead of failing; `pip` forces that, `none` disables CUDA.
            if [ "$CUDA_TOOLKIT" = "pip" ] || \
               { [ "$CUDA_TOOLKIT" = "auto" ] && ! cuda_toolkit_matches_torch "$torch_cuda"; }; then
                log "no cluster nvcc matches torch CUDA ${torch_cuda:-?} (nvcc: $(nvcc_cuda_version || echo none)); installing a matching toolkit from PyPI"
                setup_pip_cuda_toolkit "$torch_cuda" || return 1
            fi
            nvcc_cuda="$(nvcc_cuda_version || true)"
            if [ -z "$nvcc_cuda" ]; then
                log "ERROR: CUDA build requested but no nvcc found."
                log "       Load a CUDA toolkit module (CUDA_MODULE), set CUDA_HOME,"
                log "       use CUDA_TOOLKIT=pip, or set FORCE_CUDA=0 for CPU-only ops."
                return 1
            fi
            if [ -n "$torch_cuda" ] && \
               [ "$(cuda_major_minor "$nvcc_cuda")" != "$(cuda_major_minor "$torch_cuda")" ]; then
                log "ERROR: nvcc CUDA $nvcc_cuda mismatches torch CUDA $torch_cuda."
                log "       torch was built for CUDA $torch_cuda, so mmcv's CUDA ops need"
                log "       the same toolkit. Load it (e.g. CUDA_MODULE=cuda/$torch_cuda),"
                log "       point CUDA_HOME at it, use CUDA_TOOLKIT=pip (installs a"
                log "       matching toolkit from PyPI), or FORCE_CUDA=0 for CPU-only ops."
                return 1
            fi
            log "CUDA build: nvcc $nvcc_cuda matches torch $torch_cuda"
        else
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
