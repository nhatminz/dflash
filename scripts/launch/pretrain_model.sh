#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "$PRETRAIN_DATASET" == sharegpt ]] || { echo 'PRETRAIN_DATASET must be sharegpt' >&2; exit 2; }
resolve_run pretrain
cmd=("$DFLASH_MCORE_PYTHON" -m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE"
  -m dflash_reflex.pretrain --model "$MODEL" --data "$PRETRAIN_DATASET_PATH"
  --run-dir "$RUN_DIR" --model-key "$MODEL_KEY" --epochs "$PRETRAIN_EPOCHS"
  --lr "$PRETRAIN_LR" --batch-size "$PRETRAIN_BATCH_SIZE" --accumulation "$PRETRAIN_ACCUMULATION_STEPS"
  --max-length "$PRETRAIN_MAX_LENGTH" --max-samples "$PRETRAIN_MAX_SAMPLES" --max-steps "$PRETRAIN_MAX_STEPS"
  --save-interval "$PRETRAIN_SAVE_INTERVAL" --workers "$PRETRAIN_DATALOADER_WORKERS"
  --block-size "$DFLASH_BLOCK_SIZE" --layers "$DFLASH_LAYERS" --anchors "$DFLASH_ANCHORS_PER_SEQ"
  --mask-id "$DFLASH_MASK_TOKEN_ID" --loss-chunk-size "$DFLASH_LOSS_CHUNK_SIZE" --seed "$TRAIN_SUBSET_SEED")
if [[ "$DFLASH_RESUMING" == 1 ]]; then cmd+=(--resume auto); fi
if [[ -n "$PRETRAIN_WARM_START" ]]; then cmd+=(--warm-start "$PRETRAIN_WARM_START"); fi
cmd+=("$@")
print_command "${cmd[@]}"
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
prepare_run pretrain
"${cmd[@]}" 2>&1 | tee -a "$RUN_DIR/logs/console.log"
ln -sfn "$RUN_DIR" "$OUTPUT_ROOT/pretrain/$MODEL_KEY/latest_run"
