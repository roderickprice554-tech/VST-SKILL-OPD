import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from recurrent.reflection import (
    ReflectionEnvelope,
    build_reflection_prompt,
    parse_and_validate_reflection,
)
from recurrent.skill_opd import (
    MemoryTransition,
    ReflectionTrajectory,
    reward_to_is_correct,
)


@dataclass(frozen=True)
class TeacherRequest:
    payload: dict[str, Any]
    analyzer_prompt: str


@dataclass(frozen=True)
class AcceptedTeacherResponse:
    raw_response: str
    parsed: ReflectionEnvelope
    source: str


class ReflectionTeacherClient(Protocol):
    async def generate(self, request: TeacherRequest) -> str:
        raise NotImplementedError


class OpenAICompatibleReflectionTeacher:
    """Optional external client. Tests and CPU smoke never instantiate it."""

    def __init__(self, *, model: str, api_key: str, base_url: str | None = None):
        if not model or not api_key:
            raise ValueError("call mode requires model and API key")
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: TeacherRequest) -> str:
        video_url = str(request.payload["observed_video"])
        if urlparse(video_url).scheme not in {"http", "https", "data"}:
            raise ValueError("call mode requires an externally accessible video URL")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": request.analyzer_prompt},
                    ],
                }
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("external teacher returned an empty response")
        return content


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields must match exactly: {sorted(fields)}")


def offline_record_to_trajectory(record: Mapping[str, Any]) -> ReflectionTrajectory:
    fields = {
        "trajectory_uid",
        "policy_version",
        "observed_video",
        "transitions",
        "query",
        "prediction",
        "final_reward",
    }
    _require_exact_fields(record, fields, "offline trajectory")
    reward = float(record["final_reward"])
    if not math.isfinite(reward):
        raise ValueError("final reward must be finite")

    transition_fields = {
        "transition_index",
        "previous_memory_tokens",
        "current_chunk_boundary",
        "generated_y_t_tokens",
        "updated_memory_tokens",
    }
    transitions = []
    for row_index, item in enumerate(record["transitions"]):
        _require_exact_fields(item, transition_fields, "memory transition")
        transitions.append(
            MemoryTransition(
                row_index=row_index,
                transition_index=int(item["transition_index"]),
                previous_memory_tokens=tuple(int(token) for token in item["previous_memory_tokens"]),
                current_chunk_boundary=item["current_chunk_boundary"],
                generated_y_t_tokens=tuple(int(token) for token in item["generated_y_t_tokens"]),
                updated_memory_tokens=tuple(int(token) for token in item["updated_memory_tokens"]),
            )
        )
    if not transitions:
        raise ValueError("offline trajectory requires at least one memory transition")

    trajectory_uid = str(record["trajectory_uid"])
    return ReflectionTrajectory(
        group_uid=trajectory_uid.split(":rollout-", 1)[0],
        trajectory_uid=trajectory_uid,
        policy_version=int(record["policy_version"]),
        sample_index=0,
        transitions=tuple(transitions),
        transition_rows=tuple(range(len(transitions))),
        final_row=len(transitions),
        query_tokens=(),
        query_text=str(record["query"]),
        prediction_tokens=(),
        prediction_text=str(record["prediction"]),
        is_correct=reward_to_is_correct(reward),
        observed_video=record["observed_video"],
    )


def build_teacher_request(trajectory: ReflectionTrajectory) -> TeacherRequest:
    return TeacherRequest(
        payload=trajectory.to_analyzer_input(),
        analyzer_prompt=build_reflection_prompt(trajectory),
    )


def validate_teacher_response(
    trajectory: ReflectionTrajectory,
    raw_response: str,
    *,
    source: str,
) -> AcceptedTeacherResponse:
    if source not in {"fixture", "external"}:
        raise ValueError("teacher source must be fixture or external")
    parsed = parse_and_validate_reflection(
        raw_response,
        trajectory,
        policy_version=trajectory.policy_version,
    )
    if not parsed.reflection_valid:
        raise ValueError(parsed.rejection_reason or "invalid teacher reflection")
    return AcceptedTeacherResponse(raw_response, parsed, source)


async def generate_with_client(
    trajectory: ReflectionTrajectory,
    client: ReflectionTeacherClient,
) -> AcceptedTeacherResponse:
    raw_response = await client.generate(build_teacher_request(trajectory))
    return validate_teacher_response(trajectory, raw_response, source="external")


def to_vst_sft_record(
    trajectory: ReflectionTrajectory,
    accepted: AcceptedTeacherResponse,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.fspath(trajectory.observed_video)},
                {"type": "text", "text": build_reflection_prompt(trajectory)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": accepted.raw_response}],
        },
    ]


def write_jsonl_with_seeks(records, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeks = []
    with output_path.open("wb") as stream:
        for record in records:
            seeks.append(stream.tell())
            stream.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
    seeks_path = output_path.with_name(output_path.stem + "_seeks.jsonl")
    seeks_path.write_text(json.dumps(seeks) + "\n", encoding="utf-8")
    return seeks_path
