import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from recurrent.reflection_sft import (
    build_teacher_request,
    generate_with_client,
    offline_record_to_trajectory,
    to_vst_sft_record,
    validate_teacher_response,
    write_jsonl_with_seeks,
)


def _load_command():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_reflection_sft_data.py"
    spec = importlib.util.spec_from_file_location("build_reflection_sft_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transition(index=0):
    return {
        "transition_index": index,
        "previous_memory_tokens": [] if index == 0 else [10],
        "current_chunk_boundary": {"frames": [index * 2, index * 2 + 2]},
        "generated_y_t_tokens": [10 + index],
        "updated_memory_tokens": [10, 10 + index],
    }


def _record(**overrides):
    record = {
        "trajectory_uid": "sample-0:rollout-0",
        "policy_version": 12,
        "observed_video": "/readonly/video.mp4",
        "transitions": [_transition(0), _transition(1)],
        "query": "What moved?\nA. red box\nB. blue ball",
        "prediction": "B",
        "final_reward": 0.0,
    }
    record.update(overrides)
    return record


def _reflection(step_skill="Preserve temporal order across adjacent chunks."):
    return json.dumps(
        {
            "apply_opd": True,
            "episode_skill": "Track persistent entities before compressing memory.",
            "key_transitions": [
                {
                    "transition_index": 1,
                    "kind": "correct",
                    "memory_attribute": "temporal_order",
                    "step_skill": step_skill,
                }
            ],
        },
        separators=(",", ":"),
    )


def test_offline_record_converts_reward_before_building_allowlisted_request():
    trajectory = offline_record_to_trajectory(_record())
    request = build_teacher_request(trajectory)

    assert trajectory.is_correct is False
    assert request.payload["is_correct"] is False
    assert set(request.payload) == {
        "trajectory_uid",
        "policy_version",
        "observed_video",
        "transitions",
        "query",
        "prediction",
        "is_correct",
    }
    serialized = json.dumps(request.payload).casefold()
    for forbidden in ("final_reward", "ground_truth", "correct_answer", "answer_key", "solution"):
        assert forbidden not in serialized


def test_offline_record_rejects_extra_label_field_and_non_finite_reward():
    with pytest.raises(ValueError, match="exactly"):
        offline_record_to_trajectory(_record(ground_truth="A"))
    with pytest.raises(ValueError, match="finite"):
        offline_record_to_trajectory(_record(final_reward=float("nan")))


def test_valid_fixture_becomes_existing_vst_sft_conversation_shape():
    trajectory = offline_record_to_trajectory(_record())
    raw = _reflection()
    accepted = validate_teacher_response(trajectory, raw, source="fixture")
    conversation = to_vst_sft_record(trajectory, accepted)

    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0] == {
        "type": "video",
        "video": "/readonly/video.mp4",
    }
    assert conversation[0]["content"][1]["type"] == "text"
    assert conversation[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": raw}],
    }
    assert accepted.source == "fixture"


def test_invalid_or_leaking_fixture_is_rejected():
    trajectory = offline_record_to_trajectory(_record())
    with pytest.raises(ValueError, match="json"):
        validate_teacher_response(trajectory, "not-json", source="fixture")
    with pytest.raises(ValueError, match="leakage"):
        validate_teacher_response(
            trajectory,
            _reflection(step_skill="The correct answer is option A."),
            source="fixture",
        )


def test_jsonl_writer_emits_byte_offsets_expected_by_vst_sft(tmp_path):
    output = tmp_path / "reflection_train.jsonl"
    records = [[{"role": "user", "content": []}], [{"role": "assistant", "content": []}]]

    seeks_path = write_jsonl_with_seeks(records, output)

    seeks = json.loads(seeks_path.read_text(encoding="utf-8"))
    assert seeks[0] == 0
    with output.open("rb") as stream:
        for seek, expected in zip(seeks, records):
            stream.seek(seek)
            assert json.loads(stream.readline()) == expected


class _RecordingClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return self.response


def test_injected_external_client_is_called_and_response_is_validated():
    trajectory = offline_record_to_trajectory(
        _record(observed_video="https://assets.example/video.mp4")
    )
    client = _RecordingClient(_reflection())

    accepted = asyncio.run(generate_with_client(trajectory, client))

    assert len(client.requests) == 1
    assert accepted.source == "external"


def test_cli_export_and_import_do_not_construct_external_client(tmp_path, monkeypatch):
    command = _load_command()

    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    requests_path = tmp_path / "requests.jsonl"

    class FailOnInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("external client must not be constructed")

    monkeypatch.setattr(command, "OpenAICompatibleReflectionTeacher", FailOnInit)
    assert command.main(
        [
            "export",
            "--trajectory-jsonl",
            str(trajectory_path),
            "--requests-jsonl",
            str(requests_path),
        ]
    ) == 0

    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        json.dumps({"trajectory_uid": "sample-0:rollout-0", "response": _reflection()})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "train.jsonl"
    assert command.main(
        [
            "import",
            "--trajectory-jsonl",
            str(trajectory_path),
            "--responses-jsonl",
            str(responses_path),
            "--output-jsonl",
            str(output_path),
        ]
    ) == 0
    assert output_path.exists()
    assert output_path.with_name("train_seeks.jsonl").exists()


def test_cli_call_requires_explicit_credentials(tmp_path):
    main = _load_command().main

    with pytest.raises(SystemExit):
        main(
            [
                "call",
                "--trajectory-jsonl",
                str(tmp_path / "trajectories.jsonl"),
                "--output-jsonl",
                str(tmp_path / "train.jsonl"),
            ]
        )


def test_reflection_sft_launcher_has_explicit_stage_boundary():
    launcher = Path(__file__).resolve().parents[2] / "VST-SFT" / "run_reflection_sft.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "${TRAIN_JSONL:?" in text
    assert "${BASE_MODEL:?" in text
    assert "${OUTPUT_DIR:?" in text
    assert '--train_annotation_paths "$TRAIN_JSONL"' in text
    assert '--pretrained_model_name_or_path "$BASE_MODEL"' in text
    assert "actor_rollout_ref.model.path=$OUTPUT_DIR" in text
