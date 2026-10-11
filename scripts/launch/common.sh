#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
export DFLASH_MCORE_PYTHON="${DFLASH_MCORE_PYTHON:-$PROJECT_DIR/.venv-mcore/bin/python}"
export DFLASH_VLLM_PYTHON="${DFLASH_VLLM_PYTHON:-$PROJECT_DIR/.venv-vllm/bin/python}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
export DRY_RUN="${DRY_RUN:-false}"
case "$DATASET" in
  simplelr) export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
  gsm8k) export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}" ;;
  dapo) export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
  *) echo 'DATASET must be simplelr, gsm8k, or dapo' >&2; exit 2 ;;
esac

resolve_run() {
  local kind="$1"
  local root="$OUTPUT_ROOT/$kind/$MODEL_KEY"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  export RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${kind}__${METHOD}__${DATASET}__seed${TRAIN_SUBSET_SEED}__${stamp}__$$}"
  export RUN_DIR="${RUN_DIR:-$root/$RUN_NAME}"
  export DFLASH_RESUMING=0
  local active="$root/active_run"
  if [[ "$kind" == train && -e "$root/active_run_${METHOD}_${DATASET}" ]]; then
    active="$root/active_run_${METHOD}_${DATASET}"
  fi
  if [[ "$RESUME" == auto && -e "$active" ]]; then
    export RUN_DIR="$(readlink -f "$active")"
    export DFLASH_RESUMING=1
  elif [[ -n "$RESUME" && "$RESUME" != auto ]]; then
    [[ -d "$RESUME" && -d "$RESUME/checkpoints" ]] || { echo 'RESUME must be auto or a run directory' >&2; exit 2; }
    export RUN_DIR="$(readlink -f "$RESUME")"
    export RESUME=auto DFLASH_RESUMING=1
  fi
  export DFLASH_RUN_DIR="$RUN_DIR"
}

prepare_run() {
  local kind="$1"
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN; see ENVIRONMENT.md" >&2; exit 2; }
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_run.py" "$kind"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --require-runtime
  mkdir -p "$OUTPUT_ROOT/$kind/$MODEL_KEY"
  ln -sfn "$RUN_DIR" "$OUTPUT_ROOT/$kind/$MODEL_KEY/active_run"
  if [[ "$kind" == train ]]; then
    ln -sfn "$RUN_DIR" "$OUTPUT_ROOT/$kind/$MODEL_KEY/active_run_${METHOD}_${DATASET}"
  fi
}

print_command() {
  printf 'Run directory: %s\n' "$RUN_DIR"
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}
