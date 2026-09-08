#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_JSONL:?set an absolute reflection SFT JSONL path}"
: "${BASE_MODEL:?set an absolute base model path}"
: "${OUTPUT_DIR:?set an absolute checkpoint output path}"

for path in "$TRAIN_JSONL" "$BASE_MODEL" "$OUTPUT_DIR"; do
  case "$path" in
    /*) ;;
    *) echo "all paths must be absolute" >&2; exit 2 ;;
  esac
done

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" train.py \
  --deepspeed ./scripts/zero3.json \
  --do_train true \
  --overwrite_output_dir true \
  --output_dir "$OUTPUT_DIR" \
  --run_name reflection-sft \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_annotation_paths "$TRAIN_JSONL" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 25 \
  --logging_steps 1 \
  --report_to tensorboard \
  --text_sink 512 \
  --text_sliding_window 32768

echo "RL handoff: actor_rollout_ref.model.path=$OUTPUT_DIR"
