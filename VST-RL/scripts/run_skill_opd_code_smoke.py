#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from recurrent.reflection import parse_and_validate_reflection
from recurrent.skill_opd import assemble_reflection_trajectories, build_opd_annotations
from verl.protocol import DataProto
from verl.trainer.ppo.skill_opd_loss import (
    build_teacher_topk,
    combine_vst_rl_and_opd_loss,
    localized_topk_opd_loss,
)


def _rollout_output() -> tuple[DataProto, torch.Tensor, torch.Tensor]:
    final_mask = torch.tensor([False, False, True])
    sample_index = torch.tensor([0, 0, 0])
    output = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 0], [12, 13], [21, 0]]),
            "response_mask": torch.tensor(
                [[True, False], [True, True], [True, False]]
            ),
        },
        non_tensors={
            "group_uid": np.array(["group-a"] * 3, dtype=object),
            "trajectory_uid": np.array(["group-a:rollout-0"] * 3, dtype=object),
            "sample_index": np.array([0, 0, 0], dtype=np.int64),
            "transition_index": np.array([0, 1, None], dtype=object),
            "previous_memory_tokens": np.array([[], [11], None], dtype=object),
            "current_chunk_boundary": np.array(
                [
                    {"frames": [0, 4], "seconds": [0.0, 2.0]},
                    {"frames": [4, 8], "seconds": [2.0, 4.0]},
                    {"frames": [8, 12], "seconds": [4.0, 6.0]},
                ],
                dtype=object,
            ),
            "generated_y_t_tokens": np.array([[11], [12, 13], None], dtype=object),
            "updated_memory_tokens": np.array([[11], [11, 12, 13], None], dtype=object),
            "policy_version": np.array([5, 5, 5], dtype=np.int64),
            "final_mask": np.array([False, False, True], dtype=np.bool_),
        },
    )
    return output, final_mask, sample_index


def run_code_smoke(report_path: Path) -> dict:
    torch.manual_seed(7)
    output, final_mask, sample_index = _rollout_output()
    trajectories = assemble_reflection_trajectories(
        output=output,
        final_mask=final_mask,
        sample_index=sample_index,
        rewards_by_trajectory={"group-a:rollout-0": 1.0},
        query_tokens_by_sample={0: [101, 102]},
        query_text_by_sample={0: "Where does the object move?\nA. Left side\nB. Right side"},
        prediction_text_by_final_row={2: "A"},
        observed_video_by_sample={0: "cpu-code-smoke-video-marker"},
    )
    reflection_text = json.dumps(
        {
            "apply_opd": True,
            "episode_skill": "Track stable entities across successive observations.",
            "key_transitions": [
                {
                    "transition_index": 1,
                    "kind": "preserve",
                    "memory_attribute": "entity_identity",
                    "step_skill": "Preserve entity identity while adding newly observed motion.",
                }
            ],
        }
    )
    reflection = parse_and_validate_reflection(
        reflection_text, trajectories[0], policy_version=5
    )
    annotations = build_opd_annotations(output, trajectories, [reflection])

    teacher_logits = torch.randn(3, 2, 128, requires_grad=True)
    teacher_indices, teacher_log_probs, retained_mass = build_teacher_topk(
        teacher_logits, top_k=100, temperature=1.0
    )
    student_logits = torch.nn.Parameter(torch.randn(3, 2, 128))
    optimizer = torch.optim.SGD([student_logits], lr=0.05)
    before = student_logits.detach().clone()
    lopd_loss, _ = localized_topk_opd_loss(
        student_logits,
        teacher_indices,
        teacher_log_probs,
        output.batch["response_mask"],
        annotations["opd_memory_mask"],
        annotations["opd_key_mask"],
        annotations["opd_reflection_mask"],
        annotations["opd_metadata_mask"],
    )
    rl_loss = student_logits.square().mean()
    total_loss = combine_vst_rl_and_opd_loss(
        rl_loss, lopd_loss, enabled=True, lambda_opd=0.01
    )
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    parameter_change = (student_logits.detach() - before).abs().max().item()
    disabled_rl_loss = rl_loss.detach()
    disabled_loss = combine_vst_rl_and_opd_loss(
        disabled_rl_loss, None, enabled=False, lambda_opd=0.01
    )
    valid_tokens = annotations["opd_valid_token_mask"]
    final_tokens = valid_tokens & final_mask.unsqueeze(-1)
    cache_tensors = (teacher_indices, teacher_log_probs, retained_mass, valid_tokens)
    report = {
        "smoke_type": "cpu_code",
        "reflection_source": "fixture",
        "memory_transition_count": sum(
            len(trajectory.transitions) for trajectory in trajectories
        ),
        "final_turn_count": int(final_mask.sum().item()),
        "trajectory_count": len(trajectories),
        "reward_mapped": trajectories[0].reward == 1.0,
        "reflection_valid": int(reflection.reflection_valid),
        "reflection_applied": int(reflection.apply_opd),
        "key_transition_count": len(reflection.key_transitions),
        "query_leakage_count": int(
            bool(reflection.rejection_reason)
            and "leakage" in reflection.rejection_reason
        ),
        "top_k": int(teacher_indices.shape[-1]),
        "valid_token_count": int(valid_tokens.sum().item()),
        "final_token_count": int(final_tokens.sum().item()),
        "teacher_detached": teacher_logits.grad is None
        and all(not tensor.requires_grad for tensor in cache_tensors[:3]),
        "rl_loss": float(rl_loss.detach().item()),
        "lopd_loss": float(lopd_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "cache_bytes": sum(tensor.numel() * tensor.element_size() for tensor in cache_tensors),
        "optimizer_completed": True,
        "trainable_param_max_change": float(parameter_change),
        "disabled_path_equivalent": disabled_loss is disabled_rl_loss,
        "retained_mass": float(retained_mass[valid_tokens].mean().item()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    run_code_smoke(args.report)
    print(args.report)


if __name__ == "__main__":
    main()
