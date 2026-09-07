import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_skill_opd_smoke.py"
SPEC = importlib.util.spec_from_file_location("audit_skill_opd_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _valid_report():
    return {
        "memory_transition_count": 2,
        "final_turn_count": 1,
        "trajectory_count": 1,
        "reward_mapped": True,
        "reflection_valid": 1,
        "reflection_applied": 1,
        "key_transition_count": 1,
        "query_leakage_count": 0,
        "top_k": 100,
        "valid_token_count": 5,
        "final_token_count": 0,
        "teacher_detached": True,
        "rl_loss": 0.3,
        "lopd_loss": 0.2,
        "total_loss": 0.302,
        "cache_bytes": 4096,
        "optimizer_completed": True,
        "trainable_param_max_change": 1e-6,
    }


def test_valid_smoke_report_passes():
    MODULE.audit_enabled_smoke(_valid_report())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_transition_count", 1),
        ("final_turn_count", 0),
        ("reward_mapped", False),
        ("reflection_valid", 0),
        ("reflection_applied", 0),
        ("key_transition_count", 0),
        ("query_leakage_count", 1),
        ("top_k", 99),
        ("valid_token_count", 0),
        ("final_token_count", 1),
        ("teacher_detached", False),
        ("rl_loss", float("nan")),
        ("lopd_loss", float("inf")),
        ("total_loss", float("nan")),
        ("cache_bytes", 0),
        ("optimizer_completed", False),
        ("trainable_param_max_change", 0.0),
    ],
)
def test_each_failed_acceptance_gate_is_rejected(field, value):
    report = copy.deepcopy(_valid_report())
    report[field] = value

    with pytest.raises(ValueError):
        MODULE.audit_enabled_smoke(report)


def test_missing_field_is_rejected():
    report = _valid_report()
    report.pop("lopd_loss")

    with pytest.raises(ValueError, match="missing"):
        MODULE.audit_enabled_smoke(report)


def test_disabled_smoke_requires_original_branch_and_optimizer():
    MODULE.audit_disabled_smoke(
        {
            "skill_opd_enabled": False,
            "opd_branch_executed": False,
            "optimizer_completed": True,
            "rl_loss": 0.5,
        }
    )

    with pytest.raises(ValueError, match="OPD branch"):
        MODULE.audit_disabled_smoke(
            {
                "skill_opd_enabled": False,
                "opd_branch_executed": True,
                "optimizer_completed": True,
                "rl_loss": 0.5,
            }
        )


def test_smoke_runner_uses_absolute_read_only_assets_and_one_update():
    runner = (SCRIPT.parent / "run_skill_opd_smoke.sh").read_text(encoding="utf-8")

    assert "/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct" in runner
    assert "/home/bujunru/vlm-repro/VST-full-reproduction/data/" in runner
    assert "trainer.total_training_steps=1" in runner
    assert "skill_opd.top_k=100" in runner
    assert "skill_opd.lambda_opd=0.01" in runner
    assert "SKILL_OPD_ENABLE" in runner
    assert "SMOKE_CONFIG_ONLY" in runner
    assert "recurrent.video_memory.config.prompt_type=type2" in runner
