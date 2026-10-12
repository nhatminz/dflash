#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
explicit_draft="${DRAFT_CHECKPOINT:-}"
export DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY/latest_checkpoint}"
resolve_run train
if [[ "$DFLASH_RESUMING" == 1 && -z "$explicit_draft" && -f "$RUN_DIR/launch_config.json" ]]; then
  export DRAFT_CHECKPOINT="$("${CONFIG_PYTHON_BIN:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"]["draft"]["model_name"])' "$RUN_DIR/launch_config.json")"
fi
cmd=("$PYTHON_BIN" -m dflash_reflex.train --config "$RUN_DIR/resolved.yaml")
cmd+=("$@")
print_command "${cmd[@]}"
if [[ "$DRY_RUN" == true ]]; then
  # stdlib-only resolver; works even before installing the GPU environment.
  "${CONFIG_PYTHON_BIN:-python3}" "$PROJECT_DIR/scripts/prepare_run.py" train --dry-run
  exit 0
fi
export DRAFT_CHECKPOINT="$(readlink -f "$DRAFT_CHECKPOINT")"
prepare_run train
"$DFLASH_MCORE_PYTHON" "$PROJECT_DIR/scripts/check_dataset_helpers.py"
"${cmd[@]}" 2>&1 | tee -a "$RUN_DIR/logs/console.log"
ln -sfn "$RUN_DIR" "$OUTPUT_ROOT/train/$MODEL_KEY/latest_run"
