from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from recurrent.interface import aggregate_trajectories
from verl.protocol import DataProto


@dataclass(frozen=True)
class MemoryTransition:
    row_index: int
    transition_index: int
    previous_memory_tokens: tuple[int, ...]
    current_chunk_boundary: Any
    generated_y_t_tokens: tuple[int, ...]
    updated_memory_tokens: tuple[int, ...]


@dataclass(frozen=True)
class ReflectionTrajectory:
    group_uid: str
    trajectory_uid: str
    policy_version: int
    sample_index: int
    transitions: tuple[MemoryTransition, ...]
    transition_rows: tuple[int, ...]
    final_row: int
    query_tokens: tuple[int, ...]
    query_text: str
    prediction_tokens: tuple[int, ...]
    prediction_text: str
    reward: float
    observed_video: Any

    def to_analyzer_input(self) -> dict[str, Any]:
        """Construct the analyzer payload from an explicit non-label allowlist."""
        return {
            "trajectory_uid": self.trajectory_uid,
            "policy_version": self.policy_version,
            "observed_video": self.observed_video,
            "transitions": [
                {
                    "transition_index": transition.transition_index,
                    "previous_memory_tokens": list(transition.previous_memory_tokens),
                    "current_chunk_boundary": transition.current_chunk_boundary,
                    "generated_y_t_tokens": list(transition.generated_y_t_tokens),
                    "updated_memory_tokens": list(transition.updated_memory_tokens),
                }
                for transition in self.transitions
            ],
            "query": self.query_text,
            "prediction": self.prediction_text,
            "reward": self.reward,
        }


def _required_lookup(mapping: Mapping, key, name: str):
    if key not in mapping:
        raise ValueError(f"missing {name} for sample/row {key}")
    return mapping[key]


def _token_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(int(token) for token in value)


def assemble_reflection_trajectories(
    *,
    output: DataProto,
    final_mask: torch.Tensor,
    sample_index: torch.Tensor,
    rewards_by_trajectory: Mapping[str, float],
    query_tokens_by_sample: Mapping[int, list[int]],
    query_text_by_sample: Mapping[int, str],
    prediction_text_by_final_row: Mapping[int, str],
    observed_video_by_sample: Mapping[int, Any],
) -> list[ReflectionTrajectory]:
    """Assemble validated, reward-bearing trajectories without copying label fields."""
    rows_by_trajectory = aggregate_trajectories(output, final_mask, sample_index)
    row_samples = sample_index.detach().cpu().tolist()
    response_masks = output.batch["response_mask"].detach().cpu()
    responses = output.batch["responses"].detach().cpu()
    assembled = []

    for trajectory_uid, rows in rows_by_trajectory.items():
        memory_rows = rows[:-1]
        final_row = rows[-1]
        sample = int(row_samples[final_row])
        if trajectory_uid not in rewards_by_trajectory:
            raise ValueError(f"missing reward for trajectory_uid {trajectory_uid!r}")

        query_tokens = _required_lookup(query_tokens_by_sample, sample, "query tokens")
        query_text = _required_lookup(query_text_by_sample, sample, "query text")
        observed_video = _required_lookup(observed_video_by_sample, sample, "observed video")
        prediction_text = _required_lookup(
            prediction_text_by_final_row, final_row, "prediction text"
        )
        policy_versions = {
            int(output.non_tensor_batch["policy_version"][row]) for row in rows
        }
        if len(policy_versions) != 1:
            raise ValueError(f"policy_version mismatch in trajectory_uid {trajectory_uid!r}")

        transitions = tuple(
            MemoryTransition(
                row_index=row,
                transition_index=int(output.non_tensor_batch["transition_index"][row]),
                previous_memory_tokens=_token_tuple(
                    output.non_tensor_batch["previous_memory_tokens"][row]
                ),
                current_chunk_boundary=output.non_tensor_batch["current_chunk_boundary"][row],
                generated_y_t_tokens=_token_tuple(
                    output.non_tensor_batch["generated_y_t_tokens"][row]
                ),
                updated_memory_tokens=_token_tuple(
                    output.non_tensor_batch["updated_memory_tokens"][row]
                ),
            )
            for row in memory_rows
        )
        prediction_tokens = responses[final_row][response_masks[final_row].bool()].tolist()
        assembled.append(
            ReflectionTrajectory(
                group_uid=str(output.non_tensor_batch["group_uid"][final_row]),
                trajectory_uid=str(trajectory_uid),
                policy_version=policy_versions.pop(),
                sample_index=sample,
                transitions=transitions,
                transition_rows=tuple(memory_rows),
                final_row=final_row,
                query_tokens=_token_tuple(query_tokens),
                query_text=str(query_text),
                prediction_tokens=_token_tuple(prediction_tokens),
                prediction_text=str(prediction_text),
                reward=float(rewards_by_trajectory[trajectory_uid]),
                observed_video=observed_video,
            )
        )

    return assembled


def build_opd_annotations(output, trajectories, reflections):
    """Map validated reflection decisions to row-aligned masks and skill strings."""
    if len(trajectories) != len(reflections):
        raise ValueError("trajectory and reflection counts must match")
    response_shape = output.batch["response_mask"].shape
    device = output.batch["response_mask"].device
    key_mask = torch.zeros(response_shape, dtype=torch.bool, device=device)
    reflection_mask = torch.zeros_like(key_mask)
    metadata_mask = torch.zeros_like(key_mask)
    final_rows = torch.as_tensor(
        np.asarray(output.non_tensor_batch["final_mask"].tolist(), dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    )
    memory_mask = (~final_rows).unsqueeze(-1).expand(response_shape).clone()
    episode_skills = np.empty(len(output), dtype=object)
    step_skills = np.empty(len(output), dtype=object)
    episode_skills[:] = None
    step_skills[:] = None

    for trajectory, reflection in zip(trajectories, reflections):
        metadata_valid = (
            reflection.trajectory_uid == trajectory.trajectory_uid
            and reflection.policy_version == trajectory.policy_version
        )
        if not (reflection.reflection_valid and reflection.apply_opd and metadata_valid):
            continue
        rows_by_transition = {
            transition.transition_index: transition.row_index
            for transition in trajectory.transitions
        }
        for key_transition in reflection.key_transitions:
            row = rows_by_transition[key_transition.transition_index]
            key_mask[row] = output.batch["response_mask"][row].bool()
            reflection_mask[row] = True
            metadata_mask[row] = True
            episode_skills[row] = reflection.episode_skill
            step_skills[row] = key_transition.step_skill

    valid_token_mask = (
        output.batch["response_mask"].bool()
        & memory_mask
        & key_mask
        & reflection_mask
        & metadata_mask
    )
    return {
        "opd_memory_mask": memory_mask,
        "opd_key_mask": key_mask,
        "opd_reflection_mask": reflection_mask,
        "opd_metadata_mask": metadata_mask,
        "opd_valid_token_mask": valid_token_mask,
        "opd_episode_skill": episode_skills,
        "opd_step_skill": step_skills,
    }


def augment_teacher_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    step_skills,
    tokenizer,
    *,
    max_context_tokens: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert query-independent skill tokens before the original response tokens."""
    if len(input_ids) != len(step_skills):
        raise ValueError("step skills must be row-aligned")
    response_length = responses.shape[-1]
    prompt_length = input_ids.shape[-1] - response_length
    prompt_rows = []
    for row, skill in enumerate(step_skills):
        prompt = input_ids[row, :prompt_length][attention_mask[row, :prompt_length].bool()]
        if skill is not None:
            skill_tokens = tokenizer.encode(
                f"\n[Query-independent memory skill]\n{skill}\n",
                add_special_tokens=False,
            )
            prompt = torch.cat(
                [prompt, torch.as_tensor(skill_tokens, dtype=input_ids.dtype, device=input_ids.device)]
            )
        if len(prompt) + response_length > max_context_tokens:
            raise ValueError("skill-augmented teacher context exceeds context limit")
        prompt_rows.append(prompt)

    augmented_prompt_length = max(len(prompt) for prompt in prompt_rows)
    padded_prompts = torch.full(
        (len(prompt_rows), augmented_prompt_length),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    prompt_attention = torch.zeros_like(padded_prompts)
    for row, prompt in enumerate(prompt_rows):
        padded_prompts[row, -len(prompt):] = prompt
        prompt_attention[row, -len(prompt):] = 1
    augmented_ids = torch.cat([padded_prompts, responses.to(input_ids.device)], dim=-1)
    augmented_attention = torch.cat(
        [prompt_attention, attention_mask[:, -response_length:]], dim=-1
    )
    return augmented_ids, augmented_attention


def validate_opd_teacher_cache(rollout: DataProto, cache: DataProto, top_k: int) -> None:
    required_tensors = {
        "opd_topk_indices",
        "opd_teacher_topk_log_probs",
        "opd_retained_mass",
        "opd_valid_token_mask",
    }
    missing = required_tensors - set(cache.batch.keys())
    if missing:
        raise ValueError(f"teacher cache missing tensors: {sorted(missing)}")
    expected_prefix = rollout.batch["responses"].shape
    indices = cache.batch["opd_topk_indices"]
    log_probs = cache.batch["opd_teacher_topk_log_probs"]
    if indices.shape != log_probs.shape or indices.shape != (*expected_prefix, top_k):
        raise ValueError("teacher cache shape/top_k mismatch")
    if cache.batch["opd_retained_mass"].shape != expected_prefix:
        raise ValueError("teacher retained-mass shape mismatch")
    if cache.batch["opd_valid_token_mask"].shape != expected_prefix:
        raise ValueError("teacher valid-token mask shape mismatch")
    if any(value.requires_grad for value in cache.batch.values()):
        raise ValueError("teacher cache must be detached")
    if not torch.isfinite(log_probs).all() or not torch.isfinite(
        cache.batch["opd_retained_mass"]
    ).all():
        raise ValueError("teacher cache must be finite")
    valid = cache.batch["opd_valid_token_mask"].bool()
    if valid.any():
        normalization = log_probs.float().logsumexp(dim=-1)[valid]
        if not torch.allclose(normalization, torch.zeros_like(normalization), atol=2e-2):
            raise ValueError("teacher top-k probabilities are not normalized")
    for key in ("trajectory_uid", "policy_version", "transition_index"):
        if key not in cache.non_tensor_batch:
            raise ValueError(f"teacher cache missing {key}")
        if not np.array_equal(
            cache.non_tensor_batch[key], rollout.non_tensor_batch[key]
        ):
            raise ValueError(f"teacher cache {key} mismatch")


class SkillOPDManager:
    def __init__(
        self,
        tokenizer,
        processor,
        *,
        max_key_transitions: int = 3,
        max_output_tokens: int = 256,
        max_context_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_key_transitions = max_key_transitions
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def _visual_token_count(observed_video) -> int:
        if not isinstance(observed_video, dict) or not observed_video.get("video"):
            return 0
        video = observed_video["video"][0]
        time, _, height, width = video.shape
        return int((time + 1) // 2 * (height // 28) * (width // 28))

    def _build_reflection_batch(self, trajectories):
        if self.processor is None:
            raise ValueError("a multimodal processor is required for reflection generation")
        from recurrent.reflection import build_reflection_prompt
        from recurrent.utils import (
            create_attention_mask,
            create_position_ids_vl,
            pad_tensor_list_to_length,
        )

        input_rows = []
        video_inputs = []
        multi_modal_data = []
        for trajectory in trajectories:
            prompt = build_reflection_prompt(
                trajectory,
                self.max_key_transitions,
                tokenizer=self.tokenizer,
                visual_token_count=self._visual_token_count(trajectory.observed_video),
                max_output_tokens=self.max_output_tokens,
                max_context_tokens=self.max_context_tokens,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            processed = self.processor(
                text=[rendered],
                videos=trajectory.observed_video["video"],
                return_tensors="pt",
            )
            if processed["input_ids"].shape[-1] + self.max_output_tokens > self.max_context_tokens:
                raise ValueError("processed reflection input exceeds shared context limit")
            input_rows.append(processed["input_ids"][0])
            video_inputs.append(
                {
                    "video_grid_thw": processed["video_grid_thw"],
                    "second_per_grid_ts": processed.get("second_per_grid_ts", [1.0]),
                }
            )
            multi_modal_data.append(trajectory.observed_video)

        input_ids = pad_tensor_list_to_length(
            input_rows,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=max(len(row) for row in input_rows),
            left_pad=True,
        )
        attention_mask = create_attention_mask(input_ids, self.tokenizer.pad_token_id)
        position_ids = create_position_ids_vl(
            attention_mask, self.processor, video_inputs, input_ids
        )
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={
                "uid": np.asarray([item.trajectory_uid for item in trajectories], dtype=object),
                "trajectory_uid": np.asarray(
                    [item.trajectory_uid for item in trajectories], dtype=object
                ),
                "multi_modal_data": np.asarray(multi_modal_data, dtype=object),
            },
            meta_info={
                "do_sample": False,
                "pad_to": self.max_output_tokens,
                "generation_kwargs": {
                    "max_tokens": self.max_output_tokens,
                    "n": 1,
                    "temperature": 0,
                    "top_p": 1.0,
                },
            },
        )

    def generate_reflections(self, trajectories, actor_rollout_wg):
        from recurrent.reflection import parse_and_validate_reflection

        if not trajectories:
            return []
        batch = self._build_reflection_batch(trajectories)
        output = actor_rollout_wg.generate_sequences(batch)
        if len(output) != len(trajectories):
            raise ValueError("reflection output count does not match trajectory count")
        reflections = []
        for row, trajectory in enumerate(trajectories):
            tokens = output.batch["responses"][row].detach().cpu().tolist()
            tokens = [
                token
                for token in tokens
                if token not in {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
            ]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            reflections.append(
                parse_and_validate_reflection(
                    text,
                    trajectory,
                    policy_version=trajectory.policy_version,
                    max_key_transitions=self.max_key_transitions,
                )
            )
        return reflections
