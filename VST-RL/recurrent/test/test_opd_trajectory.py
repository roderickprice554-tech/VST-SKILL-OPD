import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent.interface import aggregate_trajectories, propagate_trajectory_reward
from verl.protocol import DataProto


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_provenance_manifest():
    manifest_path = REPO_ROOT / "manifests" / "opd-foundation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_commit"] == "26f31d36eb8bcc0b480b43eaaed1277b33a10488"
    assert manifest["python"] == "/home/bujunru/.conda/envs/vision-se/bin/python"
    assert manifest["assets_copied"] is False
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
    output.non_tensor_batch["transition_index"] = np.array(transition_index, dtype=object)

    with pytest.raises(ValueError, match=message):
        aggregate_trajectories(
            output,
            torch.tensor(final_mask),
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
