import json

import pytest

from recurrent.reflection import build_reflection_prompt, parse_and_validate_reflection
from recurrent.skill_opd import MemoryTransition, ReflectionTrajectory


def _trajectory():
    transition = MemoryTransition(
        row_index=0,
        transition_index=0,
        previous_memory_tokens=(),
        current_chunk_boundary={"frames": [0, 8], "seconds": [0.0, 4.0]},
        generated_y_t_tokens=(10, 11),
        updated_memory_tokens=(10, 11),
    )
    return ReflectionTrajectory(
        group_uid="group-a",
        trajectory_uid="group-a:rollout-0",
        policy_version=9,
        sample_index=0,
        transitions=(transition,),
        transition_rows=(0,),
        final_row=1,
        query_tokens=(1, 2),
        query_text=(
            "At 12 seconds, what color is the vehicle near the gate?\n"
            "A. Bright crimson coat\nB. Deep blue finish\nC. White\nD. Black"
        ),
        prediction_tokens=(3,),
        prediction_text="B",
        reward=-1.0,
        observed_video="video-ref",
    )


def _valid_apply(**overrides):
    payload = {
        "apply_opd": True,
        "episode_skill": "Track stable visual attributes across changing views.",
        "key_transitions": [
            {
                "transition_index": 0,
                "kind": "correct",
                "memory_attribute": "entity_identity",
                "step_skill": "Keep the same object's identity across camera cuts.",
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_apply_payload_gets_program_owned_envelope():
    result = parse_and_validate_reflection(_valid_apply(), _trajectory(), policy_version=9)

    assert result.reflection_valid is True
    assert result.schema_version == "skill_opd.reflection.v1"
    assert result.trajectory_uid == "group-a:rollout-0"
    assert result.policy_version == 9
    assert result.apply_opd is True
    assert result.key_transitions[0].transition_index == 0


def test_valid_skip_payload():
    text = json.dumps(
        {
            "apply_opd": False,
            "episode_skill": None,
            "key_transitions": [],
            "skip_reason": "memory_cause_uncertain",
        }
    )
    result = parse_and_validate_reflection(text, _trajectory(), policy_version=9)

    assert result.reflection_valid is True
    assert result.apply_opd is False
    assert result.skip_reason == "memory_cause_uncertain"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("```json\n" + _valid_apply() + "\n```", "json"),
        (_valid_apply() + " trailing", "trailing"),
        (_valid_apply(extra=True), "fields"),
        (_valid_apply(key_transitions=[{"transition_index": 0, "kind": "revise", "memory_attribute": "entity_identity", "step_skill": "Keep identities stable."}]), "kind"),
        (_valid_apply(key_transitions=[{"transition_index": 0, "kind": "correct", "memory_attribute": "unknown", "step_skill": "Keep identities stable."}]), "memory_attribute"),
        (json.dumps({"apply_opd": False, "episode_skill": "not null", "key_transitions": [], "skip_reason": "answer_only_error"}), "conditional"),
        (_valid_apply(key_transitions=[]), "conditional"),
        (_valid_apply(key_transitions=[{"transition_index": 0, "kind": "correct", "memory_attribute": "entity_identity", "step_skill": "Stable identity."}, {"transition_index": 0, "kind": "preserve", "memory_attribute": "compression", "step_skill": "Compress facts."}]), "duplicate"),
        (_valid_apply(key_transitions=[{"transition_index": 1, "kind": "correct", "memory_attribute": "entity_identity", "step_skill": "Stable identity."}]), "transition_index"),
    ],
)
def test_hard_schema_rejections(text, reason):
    result = parse_and_validate_reflection(text, _trajectory(), policy_version=9)

    assert result.reflection_valid is False
    assert reason in result.rejection_reason


def test_policy_version_mismatch_is_rejected():
    result = parse_and_validate_reflection(_valid_apply(), _trajectory(), policy_version=10)

    assert result.reflection_valid is False
    assert "policy_version" in result.rejection_reason


@pytest.mark.parametrize(
    "leaked_skill",
    [
        "At 12 seconds, what color is the vehicle near the gate? A. Bright crimson coat B. Deep blue finish C. White D. Black",
        "Focus on what color is the vehicle before compressing.",
        "Remember the bright crimson coat precisely.",
        "The correct answer is A.",
        "Inspect the event at 12 seconds.",
    ],
)
def test_lexical_query_leakage_is_rejected_after_normalization(leaked_skill):
    result = parse_and_validate_reflection(
        _valid_apply(episode_skill=leaked_skill), _trajectory(), policy_version=9
    )

    assert result.reflection_valid is False
    assert "leakage" in result.rejection_reason


def test_reflection_prompt_contains_evidence_but_no_ground_truth():
    prompt = build_reflection_prompt(_trajectory())

    assert "<observed_video>" in prompt
    assert "Y_0" in prompt
    assert _trajectory().query_text in prompt
    assert "Prediction: B" in prompt
    assert "Reward: -1.0" in prompt
    assert "ground_truth" not in prompt.casefold()
    assert "correct_answer" not in prompt.casefold()


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_reflection_prompt_enforces_shared_total_context_budget():
    with pytest.raises(ValueError, match="32768|context"):
        build_reflection_prompt(
            _trajectory(),
            tokenizer=_CharacterTokenizer(),
            visual_token_count=32760,
            max_output_tokens=256,
        )
