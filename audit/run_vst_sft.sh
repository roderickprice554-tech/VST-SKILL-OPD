#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
model=/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
manifest_root="$root/data_manifests/vst_no_ego4d_aaef152e"
validator="$root/audit/validate_vst_sft_launch.py"
zero="$root/VST-SFT/scripts/zero3-vst-1xa100-offload.json"
log_root="$root/logs/vst_download_orchestrator"
audit_marker="$root/logs/vst_download_orchestrator/vst_audit.complete"
smoke_marker="$root/logs/vst_download_orchestrator/smoke.complete"
marker="$log_root/sft.complete"

[[ -f "$marker" ]] && exit 0
[[ -f "$audit_marker" && -f "$smoke_marker" ]] || { echo "SFT gates are incomplete" >&2; exit 75; }
/home/bujunru/.conda/envs/vst-sft311/bin/python "$validator" --validate-zero "$zero"

mapfile -t train_files < <(find "$manifest_root" -maxdepth 1 -type f -name '*_train_with_seeks.jsonl' | sort)
mapfile -t valid_files < <(find "$manifest_root" -maxdepth 1 -type f -name '*_valid_with_seeks.jsonl' | sort)

run_id="vst_3b_no_ego4d_single_gpu_optimizer_smoke_$(date +%Y%m%d_%H%M%S)"
output_dir="$root/checkpoints/vst_full/sft_runs/$run_id"
mkdir -p "$output_dir" "$log_root"
printf '{"base_model":"%s","epochs":1,"world_size":1,"per_device_batch":1,"gradient_accumulation":128,"effective_global_batch":128,"zero3_source":"zero3-vst-2xa100.json + cpu parameter/optimizer offload","started_at":"%s"}\n' \
    "$model" "$(date --iso-8601=seconds)" > "$output_dir/provenance.json"

export DATASET_PATH="$root/data/vst-training-media-no-ego4d"
export EVAL_DATASET_PATH="$DATASET_PATH"
export DECORD_EOF_RETRY_MAX=40960
export VIDEO_MIN_PIXELS=78400
export VIDEO_MAX_PIXELS=19267584
export FPS_MAX_FRAMES=384
export TEXT_SLIDING_WINDOW=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PATH=/home/bujunru/.conda/envs/vst-sft311/bin:$PATH
export CPATH="$root/.deps/libaio/root/usr/include"
export LIBRARY_PATH="$root/.deps/libaio/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
mkdir -p "$root/checkpoints/vst_full/nvme_offload_v6"

cd "$root/VST-SFT"
/home/bujunru/.conda/envs/vst-sft311/bin/torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29531 \
    train.py \
    --deepspeed "$zero" \
    --overwrite_output_dir False \
    --output_dir "$output_dir" \
    --run_name "$run_id" \
    --save_on_each_node True \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path "$model" \
    --train_annotation_paths "${train_files[@]}" \
    --dataloader_num_workers 0 \
    --use_liger_kernel True \
    --report_to tensorboard \
    --ignore_data_skip False \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 100 \
    --load_best_model_at_end False \
    --prediction_loss_only True \
    --eval_steps 50 \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${valid_files[@]}" \
    --text_sink 512 \
    --text_sliding_window 32768

printf '{"run_id":"%s","completed_at":"%s"}\n' "$run_id" "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
