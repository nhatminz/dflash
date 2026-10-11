#!/usr/bin/env bash
set -euo pipefail
export MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
export METHOD="${METHOD:-dflash_reflex}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_DIR/scripts/launch/pretrain_model.sh" "$@"
