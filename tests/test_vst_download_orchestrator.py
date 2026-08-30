import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))

import json
import vst_download_orchestrator as orchestrator

from vst_download_orchestrator import (
    Decision,
    Observation,
    atomic_write_json,
    command_for,
    decide,
    inventory_report,
)


def test_live_vst_download_does_not_duplicate():
    decision = decide(Observation(vst_downloader_pid=10))
    assert decision == Decision("none", "vst_downloading")


def test_missing_vst_process_restarts_once():
    decision = decide(Observation())
    assert decision == Decision("restart_vst", "vst_downloading")


def test_completed_snapshot_prepares_vst_before_ovo():
    decision = decide(Observation(vst_snapshot_complete=True))
    assert decision == Decision("prepare_vst", "vst_preparing")


def test_prepared_snapshot_is_audited_before_ovo():
    decision = decide(
        Observation(vst_snapshot_complete=True, vst_prepare_complete=True)
    )
    assert decision == Decision("audit_vst", "vst_media_auditing")


def test_non_ego4d_missing_media_blocks_ovo():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/", "LLaVA-Video-178K/"),
    )
    assert decide(observation) == Decision(
        "block", "blocked_non_ego4d_media"
    )


def test_only_ego4d_missing_media_allows_smoke_before_ovo():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        ovo_downloader_pid=99,
        ovo_stopped=True,
    )
    assert decide(observation) == Decision("run_smoke", "smoke_running")


def test_absent_incomplete_ovo_does_not_block_smoke():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
    )
    assert decide(observation) == Decision("run_smoke", "smoke_running")


def test_complete_ovo_does_not_block_smoke():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        ovo_snapshot_complete=True,
    )
    assert decide(observation) == Decision("run_smoke", "smoke_running")


def test_smoke_runs_after_vst_gates_even_when_ovo_is_prepared():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        ovo_snapshot_complete=True,
        ovo_prepare_complete=True,
    )
    assert decide(observation) == Decision("run_smoke", "smoke_running")


def test_smoke_complete_starts_sft_when_two_gpus_are_idle():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        ovo_snapshot_complete=True,
        ovo_prepare_complete=True,
        smoke_complete=True,
        idle_gpu_ids=(0, 1),
    )
    assert decide(observation) == Decision("run_sft", "sft_running")


def test_smoke_does_not_wait_for_ovo_snapshot():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
    )
    assert decide(observation) == Decision("run_smoke", "smoke_running")


def test_completed_smoke_starts_sft_before_ovo_preparation():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        smoke_complete=True,
        idle_gpu_ids=(0, 1),
    )
    assert decide(observation) == Decision("run_sft", "sft_running")


def test_live_sft_is_not_started_twice():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        smoke_complete=True,
        sft_pid=4242,
        training_gpu_ids=(0, 1),
    )
    assert decide(observation) == Decision("none", "sft_running")


def test_sft_waits_until_two_whole_gpus_are_idle():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        smoke_complete=True,
        idle_gpu_ids=(1,),
    )
    assert decide(observation) == Decision("none", "waiting_for_sft_gpus")


def test_ovo_eval_waits_when_sft_owns_both_gpus():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_complete=True,
        vst_audit_complete=True,
        missing_prefixes=("Ego4D/",),
        smoke_complete=True,
        sft_complete=True,
        ovo_snapshot_complete=True,
        ovo_prepare_complete=True,
        training_gpu_ids=(0, 1),
        idle_gpu_ids=(),
    )
    assert decide(observation) == Decision("none", "waiting_for_ovo_gpu")


def test_stopped_ovo_download_is_maintained_while_vst_is_preparing():
    observation = Observation(
        vst_snapshot_complete=True,
        vst_prepare_pid=2374820,
        ovo_downloader_pid=1162909,
        ovo_stopped=True,
    )
    assert decide(observation) == Decision("none", "vst_preparing")
    assert hasattr(orchestrator, "ovo_download_decision")
    assert orchestrator.ovo_download_decision(observation) == "resume_ovo"


def test_ovo_wrapper_counts_as_live_during_retry_wait():
    assert hasattr(orchestrator, "matches_process")
    decoded = [
        "/usr/bin/bash",
        "/home/bujunru/vlm-repro/VST-full-reproduction/audit/download_ovobench_official.sh",
    ]
    assert orchestrator.matches_process("ovo", decoded) is True


def test_inventory_report_requires_every_exact_file(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    report = inventory_report(
        tmp_path,
        [
            {"path": "a.bin", "size": 3},
            {"path": "missing.bin", "size": 4},
        ],
    )
    assert report == {
        "complete": False,
        "expected_files": 2,
        "valid_files": 1,
        "missing": ["missing.bin"],
        "wrong_size": [],
    }


def test_inventory_report_rejects_wrong_size(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"ab")
    report = inventory_report(tmp_path, [{"path": "a.bin", "size": 3}])
    assert report["complete"] is False
    assert report["wrong_size"] == [
        {"path": "a.bin", "expected": 3, "actual": 2}
    ]


def test_atomic_write_json_replaces_existing_document(tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(path, {"state": "vst_downloading", "bytes": 12})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "state": "vst_downloading",
        "bytes": 12,
    }
    assert not (tmp_path / "status.json.tmp").exists()


def test_restart_vst_command_is_pinned_and_not_training():
    command = command_for("restart_vst")
    rendered = " ".join(command)
    assert "modelscope" in rendered
    assert "aaef152ea68ffa0e9d9f7367ccf871ea2f699693" in rendered
    assert "--max-workers 8" in rendered
    assert "train.py" not in rendered
    assert "VST-RL" not in rendered


def test_restart_ovo_command_is_pinned_and_not_evaluation():
    command = command_for("restart_ovo")
    rendered = " ".join(command)
    assert command[-1].endswith("audit/download_ovobench_official.sh")
    assert "fec29e3" not in rendered  # revision is fixed inside the audited script
    assert "lmms_eval" not in rendered


def test_arbitrary_action_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unsupported action"):
        command_for("run_full_sft")


def test_stage_scripts_exist_and_forbid_full_training():
    for action in ("prepare_vst", "audit_vst", "prepare_ovo", "run_smoke"):
        script = Path(command_for(action)[-1])
        assert script.is_file(), f"missing allow-listed script: {script}"
        text = script.read_text(encoding="utf-8")
        assert "--num_train_epochs 1" not in text
        assert "VST-RL/run.sh" not in text
        assert "sft_selected" not in text
        assert "rl_selected" not in text
