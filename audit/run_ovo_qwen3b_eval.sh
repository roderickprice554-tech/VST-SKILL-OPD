#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
model=/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
python=/home/bujunru/.conda/envs/vst-eval/bin/python
validator="$root/audit/validate_ovobench.py"
log_root="$root/logs/vst_download_orchestrator"
prepare_marker="$log_root/ovo_prepare.complete"
sft_marker="$log_root/sft.complete"
marker="$log_root/ovo_eval.complete"

[[ -f "$marker" ]] && exit 0
[[ -f "$prepare_marker" && -f "$sft_marker" ]] || { echo "OVO evaluation gates are incomplete" >&2; exit 75; }
report=$($python "$validator")
gpu=$($python "$validator" --field selected_gpu)
media=$($python "$validator" --field media_root)

eval_root="$root/data/ovo-eval-root-fec29e3"
mkdir -p "$eval_root/OVO-Bench"
if [[ ! -e "$eval_root/OVO-Bench/chunked_videos" ]]; then
    ln -s "$media" "$eval_root/OVO-Bench/chunked_videos"
fi

run_id="qwen2_5_vl_3b_ovobench_$(date +%Y%m%d_%H%M%S)"
output="$root/results/ovo_full_qwen3b_base/$run_id"
mkdir -p "$output"
printf '%s\n' "$report" > "$output/validation.json"

export RLSD_EVAL_DATA_ROOT="$eval_root"
export RLSD_EVAL_ANNO_ROOT="$root/eval/eval_data/anno/eval"
export CUDA_VISIBLE_DEVICES="$gpu"
cd "$root/eval"
$python eval_entry.py \
    --eval-method vllm \
    --model "$model" \
    --tasks ovobench \
    --gpus 0 \
    --output-dir "$output" \
    --chunking-mode vst_original \
    --sample-concurrency 1 \
    --max-num-frames 384 \
    --gpu-memory-utilization 0.85

printf '{"run_id":"%s","completed_at":"%s"}\n' "$run_id" "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
