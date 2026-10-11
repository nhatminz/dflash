#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DFLASH_PYTHON_VERSION="${DFLASH_PYTHON_VERSION:-3.13.14}"
offline="${DFLASH_OFFLINE:-${UV_OFFLINE:-false}}"
validate_runtime=true
local_packages=false
for arg in "$@"; do
  case "$arg" in
    --offline) offline=true ;;
    --local-packages) local_packages=true ;;
    --skip-runtime-check) validate_runtime=false ;;
    *) echo "Usage: $0 [--offline | --local-packages] [--skip-runtime-check]" >&2; exit 2 ;;
  esac
done
case "${offline,,}" in
  true|1|yes)
    offline=true
    export DFLASH_OFFLINE=true UV_OFFLINE=true UV_PYTHON_DOWNLOADS=never
    # Build tools launched underneath uv must also use only supplied artifacts.
    export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    ;;
  false|0|no|"") offline=false ;;
  *) echo 'DFLASH_OFFLINE must be true or false' >&2; exit 2 ;;
esac
if [[ "$local_packages" == true ]]; then
  if [[ "$offline" == true ]]; then
    echo '--local-packages requires access to the pip index; choose --offline for a fully cached install.' >&2
    exit 2
  fi
  args=()
  [[ "$validate_runtime" == false ]] && args+=(--skip-runtime-check)
  exec "${CONFIG_PYTHON_BIN:-python3}" "$PROJECT_DIR/scripts/install_local_packages.py" "${args[@]}"
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 2; }
if [[ "$offline" == true ]]; then
  python3 "$PROJECT_DIR/scripts/fetch_sources.py" --offline
else
  python3 "$PROJECT_DIR/scripts/fetch_sources.py"
  # Exact gitlinks, never the branches named in .gitmodules.
  git -C "$PROJECT_DIR/third_party/nemo-rl" submodule update --init --recursive
fi
python3 "$PROJECT_DIR/scripts/patch_upstream.py"
if [[ "$offline" == true ]]; then
  trap 'echo "Offline setup stopped. Supply the missing cached packages/Python/build artifacts from a compatible connected host; see OFFLINE.md. Nothing will be downloaded automatically." >&2' ERR
fi
for backend in base mcore vllm; do
  args=(sync --locked --project "$PROJECT_DIR/third_party/nemo-rl" --python "$DFLASH_PYTHON_VERSION")
  if [[ "$backend" == base ]]; then
    export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv"
  else
    export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv-$backend"
    args+=(--extra "$backend")
  fi
  uv "${args[@]}"
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -r "$PROJECT_DIR/requirements.txt"
done
"$PROJECT_DIR/.venv-vllm/bin/python" "$PROJECT_DIR/scripts/patch_upstream.py" --vllm-installed
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$validate_runtime" == true ]]; then
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/validate_environment.py" --require-runtime
else
  echo 'Installed packages only; GPU/runtime qualification was explicitly skipped.'
fi
