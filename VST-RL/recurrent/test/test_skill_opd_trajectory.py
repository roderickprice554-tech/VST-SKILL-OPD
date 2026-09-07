import numpy as np
import pytest
import torch

from recurrent.reflection import parse_and_validate_reflection
from recurrent.skill_opd import (
    SkillOPDManager,
    assemble_reflection_trajectories,
    build_opd_annotations,
)
from verl.protocol import DataProto


def _output():
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 0], [21, 0], [12, 0], [22, 23]]),
            "response_mask": torch.tensor(
                [[True, False], [True, False], [True, False], [True, True]]
            ),
        },
        non_tensors={
            "group_uid": np.array(["group", "group", "group", "group"], dtype=object),
            "trajectory_uid": np.array(
                ["group:rollout-0", "group:rollout-1", "group:rollout-0", "group:rollout-1"],
                dtype=object,
            ),
            "sample_index": np.array([0, 1, 0, 1], dtype=np.int64),
            "transition_index": np.array([0, 0, None, None], dtype=object),
            "previous_memory_tokens": np.array([[], [], None, None], dtype=object),
            "current_chunk_boundary": np.array(
                [
                    {"frames": [0, 2]},
                    {"frames": [0, 2]},
                    {"frames": [2, 4]},
                    {"frames": [2, 4]},
                ],
                dtype=object,
            ),
            "generated_y_t_tokens": np.array([[11], [21], None, None], dtype=object),
            "updated_memory_tokens": np.array([[11], [21], None, None], dtype=object),
            "policy_version": np.array([9, 9, 9, 9], dtype=np.int64),
            "final_mask": np.array([False, False, True, True], dtype=np.bool_),
            "ground_truth": np.array(["A", "B", "A", "B"], dtype=object),
            "correct_answer": np.array(["A", "B", "A", "B"], dtype=object),
            "answer_key": np.array(["A", "B", "A", "B"], dtype=object),
        },
    )


def _assemble(output=None, rewards=None):
    return assemble_reflection_trajectories(
        output=output or _output(),
        final_mask=torch.tensor([False, False, True, True]),
        sample_index=torch.tensor([0, 1, 0, 1]),
        rewards_by_trajectory=rewards
        or {"group:rollout-0": 1.0, "group:rollout-1": -1.0},
        query_tokens_by_sample={0: [101, 102], 1: [201, 202]},
        query_text_by_sample={0: "Question zero? A. left B. right", 1: "Question one? A. up B. down"},
        prediction_text_by_final_row={2: "A", 3: "B"},
        observed_video_by_sample={0: "video-zero", 1: "video-one"},
    )


def test_assembly_keeps_same_group_rollout_answers_and_rewards_separate():
    first, second = _assemble()

    assert first.group_uid == second.group_uid == "group"
    assert first.trajectory_uid != second.trajectory_uid
    assert (first.reward, second.reward) == (1.0, -1.0)
    assert (first.prediction_text, second.prediction_text) == ("A", "B")
    assert first.query_text.startswith("Question zero")
    assert second.query_text.startswith("Question one")
    assert (first.transition_rows, first.final_row) == ((0,), 2)
    assert (second.transition_rows, second.final_row) == ((1,), 3)


def test_assembly_rejects_missing_reward_instead_of_defaulting_to_zero():
    with pytest.raises(ValueError, match="missing reward"):
        _assemble(rewards={"group:rollout-0": 1.0})


def test_analyzer_input_is_allowlisted_and_contains_no_labels():
    analyzer_input = _assemble()[0].to_analyzer_input()

    assert analyzer_input["trajectory_uid"] == "group:rollout-0"
    assert analyzer_input["observed_video"] == "video-zero"
    assert analyzer_input["prediction"] == "A"
    assert not ({"ground_truth", "correct_answer", "answer_key"} & analyzer_input.keys())


def test_assembly_rejects_policy_version_mismatch():
    output = _output()
    output.non_tensor_batch["policy_version"][3] = 10

    with pytest.raises(ValueError, match="policy_version"):
        _assemble(output=output)


def test_assembly_rejects_query_lookup_that_does_not_cover_final_sample():
    with pytest.raises(ValueError, match="query"):
        assemble_reflection_trajectories(
            output=_output(),
            final_mask=torch.tensor([False, False, True, True]),
            sample_index=torch.tensor([0, 1, 0, 1]),
            rewards_by_trajectory={"group:rollout-0": 1.0, "group:rollout-1": -1.0},
            query_tokens_by_sample={0: [101]},
            query_text_by_sample={0: "Question zero?"},
            prediction_text_by_final_row={2: "A", 3: "B"},
            observed_video_by_sample={0: "video-zero", 1: "video-one"},
        )


class _ReflectionTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, tokens, skip_special_tokens=True):
        return (
            '{"apply_opd":true,"episode_skill":"Track stable identities across views.",'
            '"key_transitions":[{"transition_index":0,"kind":"correct",'
            '"memory_attribute":"entity_identity","step_skill":"Preserve object identity across cuts."}]}'
        )


class _RecordingWorker:
    world_size = 1

    def __init__(self, events):
        self.events = events

    def generate_sequences(self, batch):
        self.events.append("reflection")
        assert "ground_truth" not in batch.non_tensor_batch
        assert "correct_answer" not in batch.non_tensor_batch
        return DataProto.from_dict(tensors={"responses": torch.tensor([[7, 99]])})


def test_reflection_generation_occurs_after_reward_and_before_actor_update(monkeypatch):
    events = ["rollout", "reward"]
    manager = SkillOPDManager(_ReflectionTokenizer(), processor=None)
    monkeypatch.setattr(
        manager,
        "_build_reflection_batch",
        lambda trajectories: DataProto.from_dict(
            tensors={"input_ids": torch.tensor([[1]])},
            non_tensors={"trajectory_uid": np.array([trajectories[0].trajectory_uid], dtype=object)},
        ),
    )

    reflections = manager.generate_reflections([_assemble()[0]], _RecordingWorker(events))
    events.append("update_actor")

    assert events == ["rollout", "reward", "reflection", "update_actor"]
    assert reflections[0].reflection_valid is True


def test_invalid_reflection_produces_zero_opd_masks_without_changing_rl_mask():
    output = _output()
    trajectories = _assemble(output=output)
    valid = parse_and_validate_reflection(
        _ReflectionTokenizer().decode([]), trajectories[0], policy_version=9
    )
    invalid = parse_and_validate_reflection("not json", trajectories[1], policy_version=9)
    original_response_mask = output.batch["response_mask"].clone()

    annotations = build_opd_annotations(output, trajectories, [valid, invalid])

    assert torch.equal(output.batch["response_mask"], original_response_mask)
    assert annotations["opd_key_mask"].tolist() == [[True, False], [False, False], [False, False], [False, False]]
    assert annotations["opd_reflection_mask"].tolist() == [[True, True], [False, False], [False, False], [False, False]]
    assert annotations["opd_memory_mask"][2:].sum().item() == 0
    assert annotations["opd_step_skill"].tolist() == [
        "Preserve object identity across cuts.", None, None, None
    ]


def test_default_config_keeps_skill_opd_disabled():
    config_text = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "verl"
        / "trainer"
        / "config"
        / "ppo_trainer.yaml"
    ).read_text(encoding="utf-8")

    assert "skill_opd:\n  enable: false" in config_text
    assert "top_k: 100" in config_text
    assert "lambda_opd: 0.01" in config_text
