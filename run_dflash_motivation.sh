#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export DATASET="${DATASET:-simplelr}"
export SEED="${SEED:-2026}"
export CHECKPOINTS="${CHECKPOINTS:-250,600}"
export EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-20}"
export PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-8}"
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
export BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
export METHOD=dflash OPD_ENABLED=0 METRICS_MODE=production OPD_PROFILE=0
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/nemo-rl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export RUNTIME_CHECK="${RUNTIME_CHECK:-auto}"
if [[ "$MODEL_KEY" != qwen25_3b || "$DATASET" != simplelr ]]; then
  echo 'This frozen experiment requires MODEL_KEY=qwen25_3b and DATASET=simplelr.' >&2
  exit 2
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv-local/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv-local/bin/python"
  elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_dflash_motivation.py" "$@"
