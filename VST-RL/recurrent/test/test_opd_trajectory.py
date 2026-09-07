import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.modules.setdefault("lmdb", SimpleNamespace())

from recurrent.interface import (
    aggregate_trajectories,
    make_repeated_rollout_ids,
    propagate_trajectory_reward,
)
from recurrent.impls.video_memory import TEMPLATE_TYPE_2, VideoMemoryAgent
from recurrent.generation_manager import LLMGenerationManager
from verl.protocol import DataProto


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_provenance_manifest():
    manifest_path = REPO_ROOT / "manifests" / "opd-foundation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_commit"] == "26f31d36eb8bcc0b480b43eaaed1277b33a10488"
    assert manifest["python"] == "/home/bujunru/.conda/envs/vision-se/bin/python"
    assert manifest["assets_copied"] is False
    assert manifest["environment"] == {
        "path": "/home/bujunru/.conda/envs/vision-se",
        "manifest": "VST-RL/requirements.txt",
        "manifest_sha256": "529e571ffeae13ffe8f2bc53b993d6c3c94695fcd0218fe6ca7323a848875029",
    }
    assert manifest["models"]
    assert manifest["datasets"]

    for entry in manifest["models"] + manifest["datasets"]:
        assert Path(entry["path"]).is_absolute()
        assert entry["access"] == "read-only"
        assert len(entry["manifest_sha256"]) == 64
        int(entry["manifest_sha256"], 16)

    for relative_path, expected_hash in manifest["tracked_file_sha256"].items():
        actual_hash = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def _synthetic_recurrent_output():
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 0], [20, 0], [11, 12], [30, 0], [40, 0]]),
            "response_mask": torch.tensor(
                [[True, False], [True, False], [True, True], [True, False], [True, False]]
            ),
        },
        non_tensors={
            "group_uid": np.array(["group-a", "group-b", "group-a", "group-a", "group-b"], dtype=object),
            "trajectory_uid": np.array(["a", "b", "a", "a", "b"], dtype=object),
            "sample_index": np.array([0, 1, 0, 0, 1], dtype=np.int64),
            "transition_index": np.array([0, 0, 1, None, None], dtype=object),
            "previous_memory_tokens": np.array([[], [], [10], None, None], dtype=object),
            "current_chunk_boundary": np.array(
                [
                    {"frames": [0, 2], "seconds": [0.0, 1.0]},
                    {"frames": [0, 2], "seconds": [0.0, 1.0]},
                    {"frames": [2, 4], "seconds": [1.0, 2.0]},
                    {"frames": [4, 6], "seconds": [2.0, 3.0]},
                    {"frames": [2, 4], "seconds": [1.0, 2.0]},
                ],
                dtype=object,
            ),
            "generated_y_t_tokens": np.array([[10], [20], [11, 12], None, None], dtype=object),
            "updated_memory_tokens": np.array([[10], [20], [10, 11, 12], None, None], dtype=object),
            "policy_version": np.array([7, 7, 7, 7, 7], dtype=np.int64),
            "final_mask": np.array([False, False, False, True, True], dtype=np.bool_),
        },
    )


def test_aggregate_trajectories_orders_transitions_before_final():
    output = _synthetic_recurrent_output()
    final_mask = torch.tensor([False, False, False, True, True])
    sample_index = torch.tensor([0, 1, 0, 0, 1])

    assert aggregate_trajectories(output, final_mask, sample_index) == {
        "a": [0, 2, 3],
        "b": [1, 4],
    }


@pytest.mark.parametrize(
    ("final_mask", "transition_index", "message"),
    [
        ([False, False, False, False, True], [0, 0, 1, 2, None], "exactly one final"),
        ([False, False, False, True, True], [0, 0, 3, None, None], "contiguous"),
    ],
)
def test_aggregate_trajectories_rejects_malformed_turns(final_mask, transition_index, message):
    output = _synthetic_recurrent_output()
    output.non_tensor_batch["final_mask"] = np.array(final_mask, dtype=np.bool_)
    output.non_tensor_batch["transition_index"] = np.array(transition_index, dtype=object)

    with pytest.raises(ValueError, match=message):
        aggregate_trajectories(
            output,
            torch.tensor(final_mask),
            torch.tensor([0, 1, 0, 0, 1]),
        )


def test_aggregate_trajectories_rejects_misaligned_final_mask_metadata():
    output = _synthetic_recurrent_output()

    with pytest.raises(ValueError, match="metadata final_mask"):
        aggregate_trajectories(
            output,
            torch.tensor([False, False, False, False, True]),
            torch.tensor([0, 1, 0, 0, 1]),
        )


def test_propagate_trajectory_reward_maps_final_reward_to_every_turn():
    output = _synthetic_recurrent_output()
    final_mask = torch.tensor([False, False, False, True, True])
    sample_index = torch.tensor([0, 1, 0, 0, 1])

    expanded = propagate_trajectory_reward(
        torch.tensor([[1.0], [3.0]]), output, final_mask, sample_index
    )

    assert expanded.squeeze(-1).tolist() == [1.0, 3.0, 1.0, 1.0, 3.0]


def test_propagate_trajectory_reward_rejects_uid_sample_index_mismatch():
    output = _synthetic_recurrent_output()
    output.non_tensor_batch["sample_index"][2] = 1
    sample_index = torch.tensor([0, 1, 1, 0, 1])

    with pytest.raises(ValueError, match="trajectory_uid"):
        propagate_trajectory_reward(
            torch.tensor([[1.0], [3.0]]),
            output,
            torch.tensor([False, False, False, True, True]),
            sample_index,
        )


class _QueryGuard(dict):
    def __getitem__(self, key):
        if key in {"prompt_ids", "question_ids"}:
            raise AssertionError(f"non-final turn accessed {key}")
        return super().__getitem__(key)


class _CaptureTemplate:
    def format(self, **kwargs):
        return kwargs


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, value, add_special_tokens=False):
        return [len(value)]

    def decode(self, value):
        return str(value)


def _video_agent_for_action(step, guarded):
    agent = VideoMemoryAgent.__new__(VideoMemoryAgent)
    agent.config = SimpleNamespace(video_clip_token_size=2, gen_pad_to=4)
    agent.tokenizer = _TinyTokenizer()
    agent.token_message_template = _CaptureTemplate()
    agent.token_final_message_template = _CaptureTemplate()
    agent.max_input_length = 32
    agent.NO_MEMORY_TOKENS = []
    agent.ctx_length = torch.tensor([6])
    agent.tokens_per_frame = torch.tensor([1])
    agent.num_frames = torch.tensor([12])
    agent.memory = np.empty(1, dtype=object)
    agent.memory[0] = [91]
    agent.bsz = 1
    agent.step = step
    data = {
        "video_duration": np.array([6.0]),
        "multi_modal_data": np.array(
            [{"video": [np.zeros((12, 3, 28, 28), dtype=np.uint8)]}], dtype=object
        ),
        "multi_modal_inputs": np.array(
            [{"video_grid_thw": torch.tensor([[12, 1, 1]]), "second_per_grid_ts": [1.0]}],
            dtype=object,
        ),
        "uid": np.array(["trajectory-a"], dtype=object),
        "group_uid": np.array(["group-a"], dtype=object),
        "trajectory_uid": np.array(["group-a:rollout-0"], dtype=object),
        "prompt_ids": np.array([[71, 72]], dtype=object),
        "question_ids": np.array([[61]], dtype=object),
    }
    agent.gen_batch = SimpleNamespace(
        non_tensor_batch=_QueryGuard(data) if guarded else data
    )
    agent.final_mask_list = []
    agent.sample_index_list = []
    return agent


def test_type2_memory_template_is_query_independent():
    assert "{prompt}" not in TEMPLATE_TYPE_2
    assert "<problem>" not in TEMPLATE_TYPE_2


def test_nonfinal_action_does_not_access_query_fields():
    agent = _video_agent_for_action(step=0, guarded=True)

    messages, video_messages, _, _, _ = agent.action()

    assert messages[0]["memory"] == [91]
    assert video_messages[0]["video"][0].shape[0] == 4


def test_final_action_includes_memory_final_chunk_and_prompt():
    agent = _video_agent_for_action(step=2, guarded=False)

    messages, video_messages, _, _, _ = agent.action()

    assert messages[0]["memory"] == [91]
    assert list(messages[0]["PromptFinal"]) == [71, 72]
    assert video_messages[0]["video"][0].shape[0] == 4


def _generation_output(response_token):
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1462, 44, 151652]]),
            "responses": torch.tensor([[response_token, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
    )


def test_three_chunks_emit_two_memory_transitions_and_one_final_row():
    agent = _video_agent_for_action(step=0, guarded=False)
    outputs = []

    for response_token in (5, 6, 7):
        agent.action()
        outputs.append(agent.update(_generation_output(response_token)))

    assert torch.cat(agent.final_mask_list).tolist() == [False, False, True]
    assert [output.non_tensor_batch["group_uid"][0] for output in outputs] == ["group-a"] * 3
    assert [output.non_tensor_batch["trajectory_uid"][0] for output in outputs] == [
        "group-a:rollout-0"
    ] * 3
    assert [output.non_tensor_batch["final_mask"][0] for output in outputs] == [False, False, True]
    assert [output.non_tensor_batch["transition_index"][0] for output in outputs] == [0, 1, None]
    assert outputs[0].non_tensor_batch["previous_memory_tokens"][0] == [91]
    assert outputs[0].non_tensor_batch["generated_y_t_tokens"][0] == [5]
    first_updated = outputs[0].non_tensor_batch["updated_memory_tokens"][0]
    assert outputs[1].non_tensor_batch["previous_memory_tokens"][0] == first_updated
    assert first_updated == [91, 1462, 44, 5, 198]
    assert outputs[1].non_tensor_batch["updated_memory_tokens"][0] == [
        91, 1462, 44, 5, 198, 1462, 44, 6, 198
    ]
    assert outputs[0].non_tensor_batch["current_chunk_boundary"][0]["frames"] == [0, 4]
    assert outputs[1].non_tensor_batch["current_chunk_boundary"][0]["frames"] == [4, 8]
    assert outputs[2].non_tensor_batch["current_chunk_boundary"][0]["frames"] == [8, 12]
    for key in (
        "transition_index",
        "previous_memory_tokens",
        "generated_y_t_tokens",
        "updated_memory_tokens",
    ):
        assert outputs[2].non_tensor_batch[key][0] is None


def test_manager_aligns_response_mask_policy_version_and_metadata():
    output = _synthetic_recurrent_output()
    output.batch["attention_mask"] = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ]
    )
    output.batch.pop("response_mask")
    output.non_tensor_batch.pop("policy_version")
    turns = output.chunk(2)

    for turn in turns:
        LLMGenerationManager._annotate_turn_output(turn, policy_version=23)
    combined = LLMGenerationManager._concat_and_validate(
        turns,
        torch.tensor([False, False, False, True, True]),
        torch.tensor([0, 1, 0, 0, 1]),
    )

    assert combined.batch["response_mask"].tolist() == [
        [True, False],
        [True, False],
        [True, True],
        [True, False],
        [True, False],
    ]
    assert combined.batch["final_mask"].tolist() == [False, False, False, True, True]
    assert combined.non_tensor_batch["policy_version"].tolist() == [23, 23, 23, 23, 23]


def test_manager_requires_integer_policy_version():
    output = _synthetic_recurrent_output()
    output.non_tensor_batch.pop("policy_version")

    with pytest.raises(ValueError, match="policy_version"):
        LLMGenerationManager._annotate_turn_output(output, policy_version=None)


def test_manager_preserves_legacy_agents_without_transition_metadata():
    output = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[8, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        },
        non_tensors={"uid": np.array(["legacy-a"], dtype=object)},
    )
    LLMGenerationManager._annotate_turn_output(output, policy_version=9)

    combined = LLMGenerationManager._concat_and_validate(
        [output], torch.tensor([True]), torch.tensor([0])
    )

    assert combined.batch["final_mask"].tolist() == [True]


def test_recurrent_trainer_passes_policy_version_and_uses_uid_checked_reward():
    trainer_source = (REPO_ROOT / "VST-RL" / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text(
        encoding="utf-8"
    )

    assert trainer_source.count("policy_version=self.global_steps") == 2
    assert "propagate_trajectory_reward(" in trainer_source
    assert "batch.batch['trajectory_reward'] = trajectory_reward" in trainer_source
    assert "batch.batch['token_level_scores'] = batch.batch['trajectory_reward']" in trainer_source


def test_make_repeated_rollout_ids_separates_group_and_trajectory_identity():
    group_uid, trajectory_uid = make_repeated_rollout_ids(
        np.array(["prompt-a", "prompt-b"], dtype=object), repeat_times=2
    )

    assert group_uid.tolist() == ["prompt-a", "prompt-a", "prompt-b", "prompt-b"]
    assert trajectory_uid.tolist() == [
        "prompt-a:rollout-0",
        "prompt-a:rollout-1",
        "prompt-b:rollout-0",
        "prompt-b:rollout-1",
    ]
    assert len(set(trajectory_uid)) == 4
    assert all(group != trajectory for group, trajectory in zip(group_uid, trajectory_uid))
