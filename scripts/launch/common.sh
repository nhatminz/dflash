#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
export RUNTIME_CHECK="${RUNTIME_CHECK:-auto}"
# MODEL remains overridable for checkpoints stored under custom paths. Reject
# a known model directory from another wrapper before creating any run files.
_dflash_model_name="${MODEL%/}"
_dflash_model_name="${_dflash_model_name##*/}"
_dflash_named_key=""
case "${_dflash_model_name,,}" in
  qwen2.5-1.5b|qwen2.5-1.5b-instruct) _dflash_named_key=qwen25_1p5b ;;
  qwen2.5-3b|qwen2.5-3b-instruct) _dflash_named_key=qwen25_3b ;;
  qwen2.5-7b|qwen2.5-7b-instruct) _dflash_named_key=qwen25_7b ;;
  qwen2.5-14b|qwen2.5-14b-instruct) _dflash_named_key=qwen25_14b ;;
  qwen3-1.7b) _dflash_named_key=qwen3_1p7b ;;
  qwen3-4b) _dflash_named_key=qwen3_4b ;;
  qwen3-8b) _dflash_named_key=qwen3_8b ;;
esac
if [[ -n "$_dflash_named_key" && "$_dflash_named_key" != "$MODEL_KEY" ]]; then
  printf 'Model mismatch: wrapper MODEL_KEY=%s, but MODEL=%s selects %s.\n' \
    "$MODEL_KEY" "$MODEL" "$_dflash_named_key" >&2
  echo 'Unset a stale MODEL or pass MODEL=/path/to/the/correct/model; also check MODEL_ENV.' >&2
  exit 2
fi
unset _dflash_model_name _dflash_named_key
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
export DFLASH_MCORE_PYTHON="${DFLASH_MCORE_PYTHON:-$PROJECT_DIR/.venv-mcore/bin/python}"
export DFLASH_VLLM_PYTHON="${DFLASH_VLLM_PYTHON:-$PROJECT_DIR/.venv-vllm/bin/python}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export PYTHONUNBUFFERED=1
# Keep native CPU pools small in each of the independent GPU jobs. Explicit
# user settings remain supported; Ray CPU resources alone do not cap threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
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

startup_checks() {
  local kind="$1"
  [[ -x "$DFLASH_MCORE_PYTHON" ]] || { echo "Missing MCore Python: $DFLASH_MCORE_PYTHON" >&2; return 2; }
  if [[ "$kind" == train ]]; then
    [[ -x "$DFLASH_VLLM_PYTHON" ]] || { echo "Missing vLLM Python: $DFLASH_VLLM_PYTHON" >&2; return 2; }
  fi
  case "$RUNTIME_CHECK" in
    skip) echo 'Runtime check: skip (explicit; model/checkpoint/resume checks remain enabled).' ;;
    auto|full)
      "$PYTHON_BIN" "$PROJECT_DIR/scripts/runtime_checks.py" --mode "$RUNTIME_CHECK" || return "$?"
      ;;
    *) echo 'RUNTIME_CHECK must be auto, full, or skip' >&2; return 2 ;;
  esac
}

prepare_run() {
  local kind="$1"
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN; see ENVIRONMENT.md" >&2; exit 2; }
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_run.py" "$kind"
  startup_checks "$kind"
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
