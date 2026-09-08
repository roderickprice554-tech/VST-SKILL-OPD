#!/usr/bin/env bash
set -euo pipefail

worktree=/home/bujunru/vlm-repro/VST-skill-opd
project=${worktree}/VST-RL
python=/home/bujunru/.conda/envs/vision-se/bin/python
model=/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct
train_data=/home/bujunru/vlm-repro/VST-full-reproduction/data/VST-Training-Data-official-5647583491c2/vst_rl_data/train.parquet
output_root=/home/bujunru/vlm-repro/skill-opd-smoke
skill_opd_enable=${SKILL_OPD_ENABLE:-true}
config_only=${SMOKE_CONFIG_ONLY:-false}
hydra_args=()
if [[ "${config_only}" == "true" ]]; then
  hydra_args=(--cfg job)
fi
if [[ "${skill_opd_enable}" == "true" ]]; then
  run_name=enabled
else
  run_name=disabled
fi
metrics=${output_root}/${run_name}-metrics.json
log=${output_root}/${run_name}.log

mkdir -p "${output_root}"
cd "${project}"
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

"${python}" -m verl.trainer.main_ppo \
  recurrent.enable=video_memory \
  recurrent.video_memory.path="${project}/recurrent/impls/video_memory.py" \
  recurrent.video_memory.config.video_clip_token_size=200 \
  recurrent.video_memory.config.max_video_frame=12 \
  recurrent.video_memory.config.max_video_clips=3 \
  recurrent.video_memory.config.video_root=/home/bujunru/vlm-repro/VST-full-reproduction/data/vst-training-media-no-ego4d \
  recurrent.video_memory.config.max_prompt_length=256 \
  recurrent.video_memory.config.max_memorization_length=128 \
  recurrent.video_memory.config.max_final_response_length=64 \
  recurrent.video_memory.config.prompt_type=type2 \
  skill_opd.enable="${skill_opd_enable}" \
  skill_opd.mode=global_episode \
  skill_opd.top_k=100 \
  skill_opd.teacher_temperature=1.0 \
  skill_opd.lambda_opd=0.01 \
  skill_opd.max_key_transitions=3 \
  skill_opd.reflection.max_tokens=256 \
  skill_opd.reflection.max_context_tokens=32768 \
  skill_opd.smoke_metrics_path="${metrics}" \
  algorithm.adv_estimator=grpo \
  algorithm.grpo_use_adv=false \
  data.train_files="${train_data}" \
  data.val_files="${train_data}" \
  data.shuffle=false \
  data.filter_overlong_prompts=false \
  data.train_batch_size=1 \
  data.truncation=center \
  +data.context_key=context \
  data.max_prompt_length=600 \
  data.max_response_length=256 \
  reward_model.reward_manager=thread \
  actor_rollout_ref.model.path="${model}" \
  actor_rollout_ref.model.freeze_vision_tower=true \
  actor_rollout_ref.model.enable_embed_cache=false \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  trainer.logger='[console]' \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${output_root}/checkpoint" \
  "${hydra_args[@]}" \
  2>&1 | tee "${log}"

if [[ "${config_only}" == "true" ]]; then
  exit 0
fi

if [[ "${skill_opd_enable}" == "true" ]]; then
  "${python}" "${project}/scripts/audit_skill_opd_smoke.py" "${metrics}"
else
  "${python}" "${project}/scripts/audit_skill_opd_smoke.py" --disabled "${metrics}"
fi
