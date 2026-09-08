import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_skill_opd_smoke.py"
SPEC = importlib.util.spec_from_file_location("audit_skill_opd_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _valid_report():
    return {
        "smoke_type": "cpu_code",
        "reflection_source": "fixture",
        "teacher_source": "fixture",
        "external_api_called": False,
        "device": "cpu",
        "gpu_used": False,
        "memory_transition_count": 2,
        "episode_memory_row_count": 2,
        "key_memory_row_count": 1,
        "non_key_episode_row_count": 1,
        "final_turn_count": 1,
        "trajectory_count": 1,
        "reward_mapped": True,
        "correctness_mapped": True,
        "reflection_valid": 1,
        "reflection_applied": 1,
        "key_transition_count": 1,
        "query_leakage_count": 0,
        "top_k": 100,
        "valid_token_count": 5,
        "final_token_count": 0,
        "final_opd_token_count": 0,
        "teacher_detached": True,
        "sft_target_token_count": 6,
        "sft_loss": 1.2,
        "sft_trainable_param_max_change": 1e-4,
        "rl_loss": 0.3,
        "lopd_loss": 0.2,
        "weighted_lopd_loss": 0.002,
        "total_loss": 0.302,
        "cache_bytes": 4096,
        "optimizer_completed": True,
        "trainable_param_max_change": 1e-6,
        "actor_trainable_param_max_change": 1e-6,
        "disabled_path_equivalent": True,
        "retained_mass": 0.9,
    }


def test_valid_smoke_report_passes():
    MODULE.audit_enabled_smoke(_valid_report())


def test_valid_cpu_code_smoke_report_passes():
    MODULE.audit_cpu_code_smoke(_valid_report())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("smoke_type", "gpu"),
        ("reflection_source", "actor"),
        ("teacher_source", "external"),
        ("external_api_called", True),
        ("device", "cuda"),
        ("gpu_used", True),
        ("disabled_path_equivalent", False),
        ("retained_mass", float("nan")),
        ("retained_mass", 0.0),
        ("retained_mass", 1.1),
    ],
)
def test_cpu_code_smoke_rejects_mislabelled_or_invalid_evidence(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError):
        MODULE.audit_cpu_code_smoke(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_transition_count", 1),
        ("final_turn_count", 0),
        ("reward_mapped", False),
        ("correctness_mapped", False),
        ("reflection_valid", 0),
        ("reflection_applied", 0),
        ("key_transition_count", 0),
        ("query_leakage_count", 1),
        ("top_k", 99),
        ("valid_token_count", 0),
        ("final_token_count", 1),
        ("final_opd_token_count", 1),
        ("teacher_detached", False),
        ("rl_loss", float("nan")),
        ("lopd_loss", float("inf")),
        ("weighted_lopd_loss", float("nan")),
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


def test_enabled_smoke_rejects_inconsistent_joint_loss():
    report = _valid_report()
    report["total_loss"] = report["rl_loss"] + report["weighted_lopd_loss"] + 0.1

    with pytest.raises(ValueError, match="composition"):
        MODULE.audit_enabled_smoke(report)


def test_enabled_training_report_does_not_claim_offline_sft_evidence():
    report = _valid_report()
    report.pop("sft_target_token_count")
    report.pop("sft_loss")
    report.pop("sft_trainable_param_max_change")

    MODULE.audit_enabled_smoke(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sft_target_token_count", 0),
        ("sft_loss", float("nan")),
        ("sft_trainable_param_max_change", 0.0),
        ("actor_trainable_param_max_change", 0.0),
    ],
)
def test_cpu_code_smoke_rejects_missing_sft_or_actor_update(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError):
        MODULE.audit_cpu_code_smoke(report)


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
    assert "skill_opd.mode=global_episode" in runner
    assert "skill_opd.lambda_opd=0.01" in runner
    assert "SKILL_OPD_ENABLE" in runner
    assert "SMOKE_CONFIG_ONLY" in runner
    assert "recurrent.video_memory.config.prompt_type=type2" in runner


def test_training_smoke_report_uses_measured_dtype_and_empty_safe_context_metric():
    trainer = (
        SCRIPT.parents[1] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert '"teacher_log_probs_dtype": str(' in trainer
    assert '"teacher_log_probs_dtype": "bfloat16"' not in trainer
    assert "skill_opd_manager.last_context_tokens, default=0" in trainer


def test_cpu_code_smoke_runs_formal_pipeline_and_passes_audit(tmp_path):
    code_smoke_path = SCRIPT.parent / "run_skill_opd_code_smoke.py"
    spec = importlib.util.spec_from_file_location("run_skill_opd_code_smoke", code_smoke_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_code_smoke(tmp_path / "code-smoke.json")

    MODULE.audit_cpu_code_smoke(report)
    assert report["smoke_type"] == "cpu_code"
    assert report["reflection_source"] == "fixture"
    assert report["teacher_source"] == "fixture"
    assert report["external_api_called"] is False
    assert report["episode_memory_row_count"] == 2
    assert report["key_memory_row_count"] == 1
    assert report["non_key_episode_row_count"] == 1
    assert report["final_opd_token_count"] == 0
    assert report["sft_target_token_count"] > 0
    assert report["sft_trainable_param_max_change"] > 0
    assert report["actor_trainable_param_max_change"] > 0
    assert report["disabled_path_equivalent"] is True


def test_cpu_code_smoke_is_directly_executable(tmp_path):
    code_smoke_path = SCRIPT.parent / "run_skill_opd_code_smoke.py"
    report_path = tmp_path / "direct-code-smoke.json"

    subprocess.run(
        [sys.executable, str(code_smoke_path), str(report_path)],
        cwd=tmp_path,
        check=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    MODULE.audit_cpu_code_smoke(report)
