from dataclasses import dataclass
from typing import Any, Mapping

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
