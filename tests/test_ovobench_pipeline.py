import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "audit"))


def test_full_ovo_validator_uses_official_3035_counts():
    validator_path = ROOT / "audit/validate_ovobench.py"
    launcher_path = ROOT / "audit/run_ovo_qwen3b_eval.sh"
    assert validator_path.is_file(), "OVO validator is missing"
    assert launcher_path.is_file(), "OVO baseline launcher is missing"

    import validate_ovobench as validator

    assert validator.EXPECTED_COUNTS == {
        "backward_tracking.json": 631,
        "real_time_visual_perception.json": 837,
        "forward_active_responding.json": 1567,
    }
    assert sum(validator.EXPECTED_COUNTS.values()) == 3035

    launcher = launcher_path.read_text(encoding="utf-8")
    assert "/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct" in launcher
    assert "--tasks ovobench" in launcher
    assert "--eval-method vllm" in launcher
    assert "ovo_eval.complete" in launcher
