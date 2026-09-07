import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from recurrent.skill_opd import ReflectionTrajectory


SCHEMA_VERSION = "skill_opd.reflection.v1"
KINDS = {"preserve", "correct"}
MEMORY_ATTRIBUTES = {
    "entity_identity",
    "state_change",
    "temporal_order",
    "event_existence",
    "count",
    "spatial_relation",
    "visible_text",
    "compression",
}
SKIP_REASONS = {
    "memory_cause_uncertain",
    "answer_only_error",
    "insufficient_evidence",
    "invalid_trajectory",
}


@dataclass(frozen=True)
class KeyTransition:
    transition_index: int
    kind: str
    memory_attribute: str
    step_skill: str


@dataclass(frozen=True)
class ReflectionEnvelope:
    schema_version: str
    trajectory_uid: str
    policy_version: int
    apply_opd: bool
    episode_skill: Optional[str]
    key_transitions: tuple[KeyTransition, ...]
    skip_reason: Optional[str]
    reflection_valid: bool
    rejection_reason: Optional[str] = None


def _invalid(trajectory, policy_version: int, reason: str) -> ReflectionEnvelope:
    return ReflectionEnvelope(
        schema_version=SCHEMA_VERSION,
        trajectory_uid=trajectory.trajectory_uid,
        policy_version=policy_version,
        apply_opd=False,
        episode_skill=None,
        key_transitions=(),
        skip_reason="invalid_trajectory",
        reflection_valid=False,
        rejection_reason=reason,
    )


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(value.split())


def _extract_options(query: str) -> list[str]:
    matches = list(re.finditer(r"(?im)(?:^|\n)\s*[a-d][.)]\s*", query))
    options = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        options.append(query[match.end():end].strip())
    return options


def _leakage_reason(texts: list[str], query: str) -> Optional[str]:
    generated = _normalize(" ".join(texts))
    normalized_query = _normalize(query)
    if normalized_query and normalized_query in generated:
        return "query leakage: full query"

    query_words = normalized_query.split()
    for index in range(len(query_words) - 3):
        phrase = " ".join(query_words[index:index + 4])
        if phrase in generated:
            return "query leakage: four-word query n-gram"

    compact_generated = generated.replace(" ", "")
    for option in _extract_options(query):
        compact_option = _normalize(option).replace(" ", "")
        if len(compact_option) < 8:
            continue
        for index in range(len(compact_option) - 7):
            if compact_option[index:index + 8] in compact_generated:
                return "query leakage: option fragment"

    if re.search(r"\b(?:correct\s+answer|answer\s+is|option\s+[a-d]|choice\s+[a-d])\b", generated):
        return "query leakage: answer indicator"

    query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", normalized_query))
    generated_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", generated))
    if query_numbers & generated_numbers:
        return "query leakage: query-specific number"
    return None


def build_reflection_prompt(
    trajectory: ReflectionTrajectory,
    max_key_transitions: int = 3,
    *,
    tokenizer=None,
    visual_token_count: int = 0,
    max_output_tokens: int = 256,
    max_context_tokens: int = 32768,
) -> str:
    transitions = "\n".join(
        (
            f"Transition {item.transition_index}: boundary={item.current_chunk_boundary}; "
            f"previous_memory={list(item.previous_memory_tokens)}; "
            f"Y_{item.transition_index}={list(item.generated_y_t_tokens)}; "
            f"updated_memory={list(item.updated_memory_tokens)}"
        )
        for item in trajectory.transitions
    )
    prompt = f"""Analyze whether a query-independent memory skill should be distilled.
Return exactly one JSON object with no markdown or trailing text. Select at most {max_key_transitions} non-final transitions.
Do not copy the question, options, answer indicators, or question-specific numbers into skill text.

<observed_video>attached observed video reference: {type(trajectory.observed_video).__name__}</observed_video>
{transitions}

Question/options:
{trajectory.query_text}
Prediction: {trajectory.prediction_text}
Reward: {trajectory.reward}

For apply_opd=true use exactly: apply_opd, episode_skill, key_transitions.
Each key transition uses exactly: transition_index, kind, memory_attribute, step_skill.
kind must be preserve or correct.
memory_attribute must be one of: entity_identity, state_change, temporal_order, event_existence, count, spatial_relation, visible_text, compression.
For apply_opd=false set episode_skill=null and key_transitions=[] and include skip_reason.
skip_reason must be one of: memory_cause_uncertain, answer_only_error, insufficient_evidence, invalid_trajectory.
"""
    prompt_tokens = (
        len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokenizer is not None
        else len(prompt.split())
    )
    total_tokens = prompt_tokens + int(visual_token_count) + int(max_output_tokens)
    if total_tokens > max_context_tokens:
        raise ValueError(
            f"reflection context exceeds {max_context_tokens}: "
            f"prompt={prompt_tokens}, visual={visual_token_count}, output={max_output_tokens}"
        )
    return prompt


def parse_and_validate_reflection(
    text: str,
    trajectory: ReflectionTrajectory,
    policy_version: int,
    max_key_transitions: int = 3,
) -> ReflectionEnvelope:
    if policy_version != trajectory.policy_version:
        return _invalid(trajectory, policy_version, "policy_version mismatch")

    source = text.lstrip()
    try:
        payload, end = json.JSONDecoder().raw_decode(source)
    except (json.JSONDecodeError, TypeError):
        return _invalid(trajectory, policy_version, "json decoding failed")
    if source[end:].strip():
        return _invalid(trajectory, policy_version, "trailing text after JSON object")
    if not isinstance(payload, dict):
        return _invalid(trajectory, policy_version, "json root must be an object")
    if type(payload.get("apply_opd")) is not bool:
        return _invalid(trajectory, policy_version, "apply_opd must be a boolean")

    apply_opd = payload["apply_opd"]
    expected_fields = (
        {"apply_opd", "episode_skill", "key_transitions"}
        if apply_opd
        else {"apply_opd", "episode_skill", "key_transitions", "skip_reason"}
    )
    if set(payload) != expected_fields:
        return _invalid(trajectory, policy_version, "payload fields do not exactly match schema")

    if not apply_opd:
        if payload["episode_skill"] is not None or payload["key_transitions"] != []:
            return _invalid(trajectory, policy_version, "conditional fields invalid for skip payload")
        if payload["skip_reason"] not in SKIP_REASONS:
            return _invalid(trajectory, policy_version, "invalid skip_reason")
        return ReflectionEnvelope(
            SCHEMA_VERSION,
            trajectory.trajectory_uid,
            policy_version,
            False,
            None,
            (),
            payload["skip_reason"],
            True,
        )

    episode_skill = payload["episode_skill"]
    raw_transitions = payload["key_transitions"]
    if not isinstance(episode_skill, str) or not episode_skill.strip():
        return _invalid(trajectory, policy_version, "conditional episode_skill must be non-empty")
    if not isinstance(raw_transitions, list) or not 1 <= len(raw_transitions) <= max_key_transitions:
        return _invalid(trajectory, policy_version, "conditional key_transitions count is invalid")

    allowed_indices = {item.transition_index for item in trajectory.transitions}
    parsed = []
    seen_indices = set()
    for item in raw_transitions:
        if not isinstance(item, dict) or set(item) != {
            "transition_index", "kind", "memory_attribute", "step_skill"
        }:
            return _invalid(trajectory, policy_version, "key transition fields are invalid")
        transition_index = item["transition_index"]
        if type(transition_index) is not int or transition_index not in allowed_indices:
            return _invalid(trajectory, policy_version, "transition_index is not a memory transition")
        if transition_index in seen_indices:
            return _invalid(trajectory, policy_version, "duplicate transition_index")
        if item["kind"] not in KINDS:
            return _invalid(trajectory, policy_version, "invalid kind")
        if item["memory_attribute"] not in MEMORY_ATTRIBUTES:
            return _invalid(trajectory, policy_version, "invalid memory_attribute")
        if not isinstance(item["step_skill"], str) or not item["step_skill"].strip():
            return _invalid(trajectory, policy_version, "step_skill must be non-empty")
        seen_indices.add(transition_index)
        parsed.append(
            KeyTransition(
                transition_index,
                item["kind"],
                item["memory_attribute"],
                item["step_skill"].strip(),
            )
        )

    leakage = _leakage_reason(
        [episode_skill] + [item.step_skill for item in parsed], trajectory.query_text
    )
    if leakage:
        return _invalid(trajectory, policy_version, leakage)
    return ReflectionEnvelope(
        SCHEMA_VERSION,
        trajectory.trajectory_uid,
        policy_version,
        True,
        episode_skill.strip(),
        tuple(parsed),
        None,
        True,
    )
