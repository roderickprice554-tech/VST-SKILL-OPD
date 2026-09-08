# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty, grad_acc_mode
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                if key != 'second_per_grid_ts':
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                        
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_embed(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        video_embed = []
        if "multi_modal_embeds" in micro_batch:
            for embed in micro_batch["multi_modal_embeds"]:
                video_embed.append(embed['video'])
            video_embed = torch.cat(video_embed, dim=0).to(torch.bfloat16)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    precomputed_video_embeds=video_embed,
                    use_cache=False,
                )  # prevent model thinks we are generating
                        
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                breakpoint()
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    precomputed_video_embeds=video_embed,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_opd_teacher_topk(self, data, top_k: int, temperature: float):
        """Forward the current Actor and immediately compress response logits to teacher top-k."""
        from verl.trainer.ppo.skill_opd_loss import build_teacher_topk

        response_length = data["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in data:
            for key in data["multi_modal_inputs"][0].keys():
                if key != "second_per_grid_ts":
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in data["multi_modal_inputs"]], dim=0
                    )

        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        position_ids = data["position_ids"]
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            if not self.use_remove_padding:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                response_logits = output.logits[:, -response_length - 1:-1, :]
                return build_teacher_topk(response_logits, top_k, temperature)

            if self.use_ulysses_sp:
                raise NotImplementedError("Skill OPD teacher cache does not support Ulysses SP")
            batch_size, sequence_length = input_ids.shape
            input_ids_rmpad, unpad_indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            if position_ids.dim() == 3:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids, "c b s ... -> (b s) c ..."), unpad_indices
                ).transpose(0, 1).unsqueeze(1)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    unpad_indices,
                ).transpose(0, 1)
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            indices, log_probs, retained_mass = build_teacher_topk(
                output.logits.squeeze(0), top_k, temperature
            )
            full_indices = pad_input(
                indices, unpad_indices, batch_size, sequence_length
            )[:, -response_length - 1:-1]
            full_log_probs = pad_input(
                log_probs, unpad_indices, batch_size, sequence_length
            )[:, -response_length - 1:-1]
            full_retained_mass = pad_input(
                retained_mass.unsqueeze(-1), unpad_indices, batch_size, sequence_length
            ).squeeze(-1)[:, -response_length - 1:-1]
            return full_indices, full_log_probs, full_retained_mass

    def compute_opd_teacher_cache(self, data: DataProto, top_k: int, temperature: float):
        was_training = self.actor_module.training
        self.actor_module.eval()
        try:
            model_data = {
                **data.batch,
                **data.non_tensor_batch,
            }
            indices, log_probs, retained_mass = self._forward_opd_teacher_topk(
                model_data, top_k, temperature
            )
        finally:
            self.actor_module.train(was_training)
        return {
            "opd_topk_indices": indices.detach(),
            "opd_teacher_topk_log_probs": log_probs.detach().to(torch.bfloat16),
            "opd_retained_mass": retained_mass.detach(),
            "opd_valid_token_mask": data.batch["opd_valid_token_mask"].detach(),
        }

    def _forward_opd_student_selected(self, data, topk_indices):
        """Gather trainable student logits only on the cached teacher support."""
        response_length = data["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in data:
            for key in data["multi_modal_inputs"][0].keys():
                if key != "second_per_grid_ts":
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in data["multi_modal_inputs"]], dim=0
                    )
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        position_ids = data["position_ids"]
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if not self.use_remove_padding:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                response_logits = output.logits[:, -response_length - 1:-1, :]
                return torch.gather(response_logits, -1, topk_indices)

            if self.use_ulysses_sp:
                raise NotImplementedError("Skill OPD student loss does not support Ulysses SP")
            batch_size, sequence_length = input_ids.shape
            input_ids_rmpad, unpad_indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            if position_ids.dim() == 3:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids, "c b s ... -> (b s) c ..."), unpad_indices
                ).transpose(0, 1).unsqueeze(1)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    unpad_indices,
                ).transpose(0, 1)
            full_support = torch.zeros(
                batch_size,
                sequence_length,
                topk_indices.shape[-1],
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
            full_support[:, -response_length - 1:-1] = topk_indices
            support_rmpad = index_first_axis(
                rearrange(full_support, "b s k -> (b s) k"), unpad_indices
            )
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            selected = torch.gather(output.logits.squeeze(0), -1, support_rmpad)
            return pad_input(
                selected, unpad_indices, batch_size, sequence_length
            )[:, -response_length - 1:-1]

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()
        
        # ============================================================
        # Step 1: 解析 meta_info
        # ============================================================
        micro_batch_size = data.meta_info.get("micro_batch_size") or 1
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        
        # ============================================================
        # Step 2: 选择需要的 keys
        # ============================================================
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch
        has_multi_modal_embeds = "multi_modal_embeds" in data.non_tensor_batch
        if has_multi_modal_embeds:
            non_tensor_select_keys = ["multi_modal_embeds"]
        elif has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []
        
        proto = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        # ============================================================
        # Step 3: 分割成 micro_batches
        # ============================================================
        revert_indices = None  # 用于 dynamic_bsz 恢复原始顺序
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            
            if has_multi_modal_inputs or has_multi_modal_embeds:
                # 多模态 + dynamic_bsz：复用 rearrange_micro_batches，额外处理 non_tensor
                tensor_micro_batches, micro_bsz_idx = rearrange_micro_batches(
                    batch=proto.batch,
                    max_token_len=max_token_len
                )
                from recurrent.utils import slice_non_tensor_batch
                non_tensor_micro_batches = slice_non_tensor_batch(
                    proto.non_tensor_batch,
                    micro_bsz_idx
                )
                micro_batches = list(zip(tensor_micro_batches, non_tensor_micro_batches))
            else:
                # 纯文本 + dynamic_bsz
                tensor_micro_batches, micro_bsz_idx = rearrange_micro_batches(
                    batch=proto.batch,
                    max_token_len=max_token_len
                )
                micro_batches = tensor_micro_batches
            
            # 计算恢复索引
            indices = list(itertools.chain.from_iterable(micro_bsz_idx))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
        else:
            # 固定 micro_batch_size
            if has_multi_modal_inputs or has_multi_modal_embeds:
                num_micro_batches = -(-proto.batch.batch_size[0] // micro_batch_size)
                micro_batches = proto.chunk(num_micro_batches)
            else:
                micro_batches = proto.batch.split(micro_batch_size)
        
        # ============================================================
        # Step 4: 前向计算
        # ============================================================
        log_probs_lst = []
        entropy_lst = []
        
        for micro_batch in micro_batches:
            # 统一数据格式
            if (has_multi_modal_inputs or has_multi_modal_embeds) and use_dynamic_bsz:
                # dynamic_bsz 多模态：micro_batch 是 (tensor_batch, non_tensor_dict) 元组
                tensor_batch, non_tensor_dict = micro_batch
                micro_batch_data = {**tensor_batch, **non_tensor_dict}
            elif isinstance(micro_batch, DataProto):
                # 非 dynamic_bsz 多模态
                micro_batch_data = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            else:
                # 纯文本
                micro_batch_data = micro_batch
            
            with torch.no_grad():

                if has_multi_modal_embeds:
                    entropy, log_probs = self._forward_micro_batch_embed(
                        micro_batch_data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy
                    )
                else:
                    entropy, log_probs = self._forward_micro_batch(
                        micro_batch_data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy
                    )
            
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        
        # ============================================================
        # Step 5: 合并结果 & 恢复顺序
        # ============================================================
        log_probs = torch.cat(log_probs_lst, dim=0)
        entropys = torch.cat(entropy_lst, dim=0) if calculate_entropy else None
        
        if revert_indices is not None:
            assert len(revert_indices) == log_probs.size(0), f"{len(revert_indices)} vs. {log_probs.size()}"
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
        
        return log_probs, entropys


    # @GPUMemoryLogger(role="dp actor", logger=logger)
    # def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
    #     """Compute the log probability of the responses given input_ids, attention_mask and position_ids

    #     Args:
    #         data (DataProto): a DataProto containing keys

    #             ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
    #             concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

    #             ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

    #     Returns:
    #         torch.Tensor: the log_prob tensor
    #     """
    #     # set to eval
    #     self.actor_module.eval()
    #     micro_batch_size = data.meta_info["micro_batch_size"]
    #     if micro_batch_size is None:
    #         micro_batch_size = 1
    #     temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
    #     use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
    #     use_dyn_b = False

    #     select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    #     batch = data.select(batch_keys=select_keys).batch
    #     has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

    #     if has_multi_modal_inputs:
    #         num_micro_batches = data.batch.batch_size[0] // micro_batch_size
    #         non_tensor_select_keys = ["multi_modal_inputs"]
    #         micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
    #     elif use_dynamic_bsz:
    #         use_dyn_b = True
    #         # split using dynamic bsz
    #         max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
    #         micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
    #     else:
    #         micro_batches = batch.split(micro_batch_size)

    #     log_probs_lst = []
    #     entropy_lst = []
    #     for micro_batch in micro_batches:
    #         if isinstance(micro_batch, DataProto):
    #             micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    #         with torch.no_grad():
    #             entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
    #         log_probs_lst.append(log_probs)
    #         if calculate_entropy:
    #             entropy_lst.append(entropy)

    #     log_probs = torch.concat(log_probs_lst, dim=0)
    #     entropys = None
    #     if calculate_entropy:
    #         entropys = torch.concat(entropy_lst, dim=0)
    #     if use_dyn_b:
    #         indices = list(itertools.chain.from_iterable(indices))
    #         assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
    #         revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
    #         log_probs = log_probs[revert_indices]

    #     return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        skill_opd_enabled = data.meta_info.get("skill_opd", {}).get("enable", False)
        skill_opd_config = data.meta_info.get("skill_opd", {})

        # ============================================================
        # Step 1: 确定需要选择的 keys
        # ============================================================
        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "response_mask"
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if skill_opd_enabled:
            select_keys.extend([
                "opd_topk_indices",
                "opd_teacher_topk_log_probs",
                "opd_memory_mask",
                "opd_episode_mask",
                "opd_reflection_mask",
                "opd_metadata_mask",
            ])

        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch
        has_multi_modal_embeds = 'multi_modal_embeds' in data.non_tensor_batch
        if has_multi_modal_embeds:
            non_tensor_select_keys = ["multi_modal_embeds"]
        elif has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # ============================================================
        # Step 2: 选择数据 & 过滤空样本（如果 padded）
        # ============================================================
        padded = 'no_padding_mask' in data.batch

        proto = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if padded:
            from recurrent.utils import indexing_proto
            # 过滤空样本，indexing_proto 同时处理 batch 和 non_tensor_batch
            proto = indexing_proto(proto, data.batch['no_padding_mask'])

        # ============================================================
        # Step 3: 分割成 mini-batches
        # ============================================================
        if padded:
            # padded 模式：实际样本数可能不等于原始 batch_size，使用 train_batch_size 计算
            num_mini_batches = self.config.train_batch_size // self.config.ppo_mini_batch_size
        else:
            num_mini_batches = proto.batch.batch_size[0] // self.config.ppo_mini_batch_size

        if has_multi_modal_inputs or has_multi_modal_embeds:
            # 多模态：需要同时分割 tensor 和 non_tensor
            from recurrent.utils import proto_split
            dataloader = proto_split(proto, num_mini_batches)
        elif padded:
            # 纯文本 + padded：只需分割 TensorDict
            from recurrent.utils import td_split
            dataloader = td_split(proto.batch, num_mini_batches)
        else:
            # 纯文本 + 非 padded：直接用原生 split
            dataloader = proto.batch.split(self.config.ppo_mini_batch_size)


        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    
                    if has_multi_modal_inputs or has_multi_modal_embeds:
                        # 多模态 + dynamic_bsz：复用 rearrange_micro_batches，额外处理 non_tensor
                        tensor_micro_batches, micro_bsz_idx = rearrange_micro_batches(
                            batch=mini_batch.batch,
                            max_token_len=max_token_len
                        )
                        from recurrent.utils import slice_non_tensor_batch
                        non_tensor_micro_batches = slice_non_tensor_batch(
                            mini_batch.non_tensor_batch, 
                            micro_bsz_idx
                        )
                        # 组装成 (tensor_batch, non_tensor_batch) 的元组列表
                        micro_batches = list(zip(tensor_micro_batches, non_tensor_micro_batches))
                    else:
                        # 纯文本 + dynamic_bsz
                        batch_for_rearrange = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                        micro_batches, _ = rearrange_micro_batches(
                            batch=batch_for_rearrange,
                            max_token_len=max_token_len
                        )
                else:
                    # 固定 micro_batch_size
                    if has_multi_modal_inputs or has_multi_modal_embeds:
                        from recurrent.utils import proto_split
                        num_micro_batches = -(-mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu)
                        micro_batches = proto_split(mini_batch, num_micro_batches)
                    elif padded:
                        from recurrent.utils import td_split
                        num_micro_batches = -(-len(mini_batch) // self.config.ppo_micro_batch_size_per_gpu)
                        micro_batches = td_split(mini_batch, num_micro_batches)
                    else:
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                #######
                # ADD: For unbias grad_gcc, see MODIFY in below for more info.
                #######
                if not self.config.use_dynamic_bsz:
                    from warnings import warn
                    warn("Using dynamic bsz is highly recommended for multiturn since there will be padding samples")

                if isinstance(mini_batch, DataProto):
                    mini_batch_token_nums = mini_batch.batch['response_mask'].sum()
                    mini_batch_seq_nums = mini_batch.batch.batch_size[0]
                else:
                    mini_batch_token_nums = mini_batch['response_mask'].sum()
                    mini_batch_seq_nums = len(mini_batch)

                for micro_batch in micro_batches:
                    # Support all hardwares
                    if (has_multi_modal_inputs or has_multi_modal_embeds) and self.config.use_dynamic_bsz:
                        # dynamic_bsz 多模态：micro_batch 是 (tensor_batch, non_tensor_dict) 元组
                        tensor_batch, non_tensor_dict = micro_batch
                        data = {**tensor_batch.to(torch.cuda.current_device()), **non_tensor_dict}
                        micro_batch_seq_nums = tensor_batch.shape[0]
                    elif isinstance(micro_batch, DataProto):
                        # 非 dynamic_bsz 多模态：micro_batch 是 DataProto
                        data = {
                            **micro_batch.batch.to(torch.cuda.current_device()),
                            **micro_batch.non_tensor_batch
                        }
                        micro_batch_seq_nums = micro_batch.batch.batch_size[0]
                    elif isinstance(micro_batch, TensorDict):
                        data = micro_batch.to(torch.cuda.current_device())
                        micro_batch_seq_nums = micro_batch.batch_size[0]
                    else:
                        # torch.Tensor (from torch.cat in rearrange_micro_batches)
                        data = micro_batch.to(torch.cuda.current_device())
                        micro_batch_seq_nums = micro_batch.shape[0]

                    #######
                    # MODIFIED: use loss_mask directly
                    #######
                    response_mask = data['response_mask']
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    if has_multi_modal_embeds:
                        entropy, log_prob = self._forward_micro_batch_embed(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    else:
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    vst_rl_loss = policy_loss
                    if skill_opd_enabled:
                        from verl.trainer.ppo.skill_opd_loss import (
                            combine_vst_rl_and_opd_loss,
                            skill_conditioned_topk_opd_loss,
                        )

                        selected_student_logits = self._forward_opd_student_selected(
                            data, data["opd_topk_indices"]
                        )
                        lopd_loss, lopd_metrics = skill_conditioned_topk_opd_loss(
                            selected_student_logits,
                            torch.arange(
                                selected_student_logits.shape[-1],
                                device=selected_student_logits.device,
                            ).expand_as(data["opd_topk_indices"]),
                            data["opd_teacher_topk_log_probs"],
                            response_mask,
                            data["opd_memory_mask"],
                            data["opd_episode_mask"],
                            data["opd_reflection_mask"],
                            data["opd_metadata_mask"],
                        )
                        policy_loss = combine_vst_rl_and_opd_loss(
                            vst_rl_loss,
                            lopd_loss,
                            enabled=True,
                            lambda_opd=skill_opd_config.get("lambda_opd", 0.01),
                        )
                        metrics["actor/vst_rl_loss"] = vst_rl_loss.detach().item()
                        metrics["actor/lopd_loss"] = lopd_loss.detach().item()
                        metrics["actor/weighted_lopd_loss"] = (
                            lopd_loss.detach().item()
                            * skill_opd_config.get("lambda_opd", 0.01)
                        )
                        metrics["actor/total_loss"] = policy_loss.detach().item()
                        for metric_key, metric_value in lopd_metrics.items():
                            metrics[metric_key] = metric_value

                    ######
                    # MODIFY: we have to fix grad_acc computation: weighted averaging by token num in stead of len(data)
                    #         See 
                    #         If we use Dr. GRPO algorithm（unbias_length_enable）, then this fix is no 
                    #           more needed since policy averaging there is sequence-level.
                    #         Since we have a variant of batchsize, we also remove self.gradient_accumulation
                    ######
                    acc_grad_mode = grad_acc_mode(loss_agg_mode)
                    if acc_grad_mode == "seq":
                        loss = policy_loss * (micro_batch_seq_nums / mini_batch_seq_nums)
                    elif acc_grad_mode == "token":
                        loss = policy_loss * (response_mask.sum().item() / mini_batch_token_nums.item())
                    else:
                        raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")
                    # if acc_grad_mode == "seq":
                    #     loss = policy_loss * (len(data) / len(mini_batch)) # self.gradient_accumulation
                    # elif acc_grad_mode == "token":
                    #     # weights by token nums, note that we want to apply a simple scalar, or the compute-graph will be extremely large.
                    #     loss = policy_loss * (response_mask.sum().item() / mini_batch_token_nums.item())
                    # else:
                    #     raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    # @GPUMemoryLogger(role="dp actor", logger=logger)
    # def update_policy(self, data: DataProto):
    #     self.actor_module.train()
    #     temperature = data.meta_info["temperature"]

    #     # ============================================================
    #     # Step 1: 确定需要选择的 keys
    #     # ============================================================
    #     select_keys = [
    #         "responses", "input_ids", "attention_mask", "position_ids",
    #         "old_log_probs", "advantages", "response_mask"
    #     ]
    #     if self.config.use_kl_loss:
    #         select_keys.append('ref_log_prob')

    #     has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch
    #     non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

    #     # ============================================================
    #     # Step 2: 选择数据 & Unpad (还原真实数据)
    #     # ============================================================
    #     # 先只选出需要的字段，减少显存占用
    #     proto = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

    #     # 检查是否有 padding mask (这是 Driver 端 graceful_padding 加上去的)
    #     if 'no_padding_mask' in data.batch:
    #         from recurrent.utils import indexing_proto
    #         # 直接过滤掉 Padding 的数据，剩下的就是纯净的训练数据
    #         # 假设 Driver 端的 Padding 策略保证了 Unpad 后的数据量是 ppo_mini_batch_size 的整数倍
    #         proto = indexing_proto(proto, data.batch['no_padding_mask'])

    #     # ============================================================
    #     # Step 3: 分割成 Mini-Batches (PPO Updates)
    #     # ============================================================
    #     # 计算当前这块数据包含多少个 mini_batch
    #     # 例如：收到 104 条数据 (Pad过)，Unpad 后剩 100 条。ppo_mini_batch_size=50。
    #     # 那么 num_mini_batches = 2。我们将执行 2 次 optimizer.step()
    #     total_samples = len(proto)
    #     target_mini_bsz = self.config.ppo_mini_batch_size
        
    #     # 确保至少有 1 个 batch (防止数据过少)
    #     num_mini_batches = max(1, total_samples // target_mini_bsz)

    #     if has_multi_modal_inputs:
    #         from recurrent.utils import proto_split
    #         dataloader = proto_split(proto, num_mini_batches)
    #     else:
    #         # 纯文本模式
    #         from recurrent.utils import td_split
    #         dataloader = td_split(proto.batch, num_mini_batches)

    #     metrics = {}

    #     # ============================================================
    #     # Step 4: PPO Epoch Loop
    #     # ============================================================
    #     for epoch in range(self.config.ppo_epochs):
    #         # 这里的 dataloader 里的每一个 item 就是一个标准的 ppo_mini_batch
    #         for batch_idx, mini_batch in enumerate(dataloader):
                
    #             # ============================================================
    #             # Step 5: Micro-Batch 切分 (梯度累积)
    #             # ============================================================
    #             if self.config.use_dynamic_bsz:
    #                 max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    
    #                 if has_multi_modal_inputs:
    #                     # Dynamic BSZ + Multi-modal
    #                     tensor_micro_batches, micro_bsz_idx = rearrange_micro_batches(
    #                         batch=mini_batch.batch,
    #                         max_token_len=max_token_len
    #                     )
    #                     from recurrent.utils import slice_non_tensor_batch
    #                     non_tensor_micro_batches = slice_non_tensor_batch(
    #                         mini_batch.non_tensor_batch, 
    #                         micro_bsz_idx
    #                     )
    #                     micro_batches = list(zip(tensor_micro_batches, non_tensor_micro_batches))
    #                 else:
    #                     # Dynamic BSZ + Text
    #                     batch_for_rearrange = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
    #                     micro_batches, _ = rearrange_micro_batches(
    #                         batch=batch_for_rearrange,
    #                         max_token_len=max_token_len
    #                     )
    #             else:
    #                 # Fixed Micro Batch Size
    #                 if has_multi_modal_inputs:
    #                     from recurrent.utils import proto_split
    #                     num_micro_batches = -(-mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu)
    #                     micro_batches = proto_split(mini_batch, num_micro_batches)
    #                 else:
    #                     # 这里的 mini_batch 可能是 DataProto 也可能是 TensorDict (取决于 Step 3 的 split 结果)
    #                     # 为了通用性，统一处理
    #                     current_batch = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
    #                     from recurrent.utils import td_split
    #                     num_micro_batches = -(-len(current_batch) // self.config.ppo_micro_batch_size_per_gpu)
    #                     micro_batches = td_split(current_batch, num_micro_batches)

    #             # ============================================================
    #             # Step 6: Forward & Backward (Accumulation)
    #             # ============================================================
    #             self.actor_optimizer.zero_grad()

    #             if not self.config.use_dynamic_bsz:
    #                 from warnings import warn
    #                 warn("Using dynamic bsz is highly recommended for multiturn since there will be padding samples")

    #             # 获取当前 Mini-Batch 的统计信息用于 Loss Normalization
    #             if isinstance(mini_batch, DataProto):
    #                 mini_batch_token_nums = mini_batch.batch['response_mask'].sum()
    #                 mini_batch_seq_nums = mini_batch.batch.batch_size[0]
    #             else:
    #                 mini_batch_token_nums = mini_batch['response_mask'].sum()
    #                 mini_batch_seq_nums = len(mini_batch)

    #             for micro_batch in micro_batches:
    #                 # 统一数据格式
    #                 if has_multi_modal_inputs and self.config.use_dynamic_bsz:
    #                     tensor_batch, non_tensor_dict = micro_batch
    #                     data = {**tensor_batch.to(torch.cuda.current_device()), **non_tensor_dict}
    #                     micro_batch_seq_nums = tensor_batch.shape[0]
    #                 elif isinstance(micro_batch, DataProto):
    #                     data = {
    #                         **micro_batch.batch.to(torch.cuda.current_device()),
    #                         **micro_batch.non_tensor_batch
    #                     }
    #                     micro_batch_seq_nums = micro_batch.batch.batch_size[0]
    #                 elif isinstance(micro_batch, TensorDict):
    #                     data = micro_batch.to(torch.cuda.current_device())
    #                     micro_batch_seq_nums = micro_batch.batch_size[0]
    #                 else:
    #                     data = micro_batch.to(torch.cuda.current_device())
    #                     micro_batch_seq_nums = micro_batch.shape[0]

    #                 # --- Loss Calculation ---
    #                 response_mask = data['response_mask']
    #                 old_log_prob = data["old_log_probs"]
    #                 advantages = data["advantages"]

    #                 # Configs
    #                 clip_ratio = self.config.clip_ratio
    #                 clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
    #                 clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
    #                 clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
    #                 entropy_coeff = self.config.entropy_coeff
    #                 loss_agg_mode = self.config.loss_agg_mode

    #                 # Forward
    #                 calculate_entropy = (entropy_coeff != 0)
    #                 entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

    #                 pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
    #                     old_log_prob=old_log_prob,
    #                     log_prob=log_prob,
    #                     advantages=advantages,
    #                     response_mask=response_mask,
    #                     cliprange=clip_ratio,
    #                     cliprange_low=clip_ratio_low,
    #                     cliprange_high=clip_ratio_high,
    #                     clip_ratio_c=clip_ratio_c,
    #                     loss_agg_mode=loss_agg_mode,
    #                 )

    #                 if entropy_coeff != 0:
    #                     entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    #                     policy_loss = pg_loss - entropy_loss * entropy_coeff
    #                 else:
    #                     policy_loss = pg_loss

    #                 if self.config.use_kl_loss:
    #                     ref_log_prob = data["ref_log_prob"]
    #                     kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
    #                     kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
    #                     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    #                     metrics["actor/kl_loss"] = kl_loss.detach().item()

    #                 # --- Gradient Normalization (Fix) ---
    #                 # 这里的关键是：分母必须是当前 Mini-Batch 的总数，而不是 Global Batch
    #                 acc_grad_mode = grad_acc_mode(loss_agg_mode)
    #                 if acc_grad_mode == "seq":
    #                     loss = policy_loss * (micro_batch_seq_nums / mini_batch_seq_nums)
    #                 elif acc_grad_mode == "token":
    #                     loss = policy_loss * (response_mask.sum().item() / mini_batch_token_nums.item())
    #                 else:
    #                     raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")
                    
    #                 loss.backward()

    #                 # Metrics logging
    #                 data_metrics = {
    #                     "actor/pg_loss": pg_loss.detach().item(),
    #                     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #                     "actor/ppo_kl": ppo_kl.detach().item(),
    #                     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    #                 }
    #                 append_to_dict(metrics, data_metrics)

    #             # ============================================================
    #             # Step 7: Optimizer Step (Per Mini-Batch)
    #             # ============================================================
    #             grad_norm = self._optimizer_step()
    #             append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

    #     self.actor_optimizer.zero_grad()
    #     return metrics
