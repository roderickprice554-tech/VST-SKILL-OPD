import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "audit"))


def test_sft_launcher_is_fixed_to_approved_3b_protocol():
    validator_path = ROOT / "audit/validate_vst_sft_launch.py"
    launcher_path = ROOT / "audit/run_vst_sft.sh"
    zero_path = ROOT / "VST-SFT/scripts/zero3-vst-2xa100.json"
    assert validator_path.is_file(), "SFT validator is missing"
    assert launcher_path.is_file(), "SFT launcher is missing"
    assert zero_path.is_file(), "ZeRO-3 config is missing"

    import validate_vst_sft_launch as validator

    assert str(validator.MODEL_PATH).endswith("Qwen2.5-VL-3B-Instruct")
    assert validator.REQUIRED_IDLE_GPUS == (0, 1)
    assert validator.MIN_FREE_BYTES == 2 * 1024**4
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    assert zero["zero_optimization"]["stage"] == 3

    launcher = launcher_path.read_text(encoding="utf-8")
    assert "--nproc_per_node=2" in launcher
    assert "--num_train_epochs 1" in launcher
    assert "--gradient_accumulation_steps 64" in launcher
    assert "--per_device_train_batch_size 1" in launcher
    assert "/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct" in launcher
    assert "logs/vst_download_orchestrator/vst_audit.complete" in launcher
    assert "logs/vst_download_orchestrator/smoke.complete" in launcher


def test_effective_global_batch_is_128():
    world_size = 2
    per_device_batch = 1
    accumulation = 64
    assert world_size * per_device_batch * accumulation == 128
