#!/usr/bin/env bash
set -euo pipefail
export MODEL_KEY="qwen25_3b"
export METHOD="${METHOD:-dflash}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_DIR/scripts/launch/train_model.sh" "$@"
