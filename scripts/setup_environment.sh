#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DFLASH_PYTHON_VERSION="${DFLASH_PYTHON_VERSION:-3.13.14}"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 2; }
python3 "$PROJECT_DIR/scripts/fetch_sources.py"
# Exact gitlinks, never the branches named in .gitmodules.
git -C "$PROJECT_DIR/third_party/nemo-rl" submodule update --init --recursive
python3 "$PROJECT_DIR/scripts/patch_upstream.py"
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
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/validate_environment.py" --require-runtime
