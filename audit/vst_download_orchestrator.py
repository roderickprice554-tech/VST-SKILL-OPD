#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class Observation:
    vst_snapshot_complete: bool = False
    vst_downloader_pid: Optional[int] = None
    vst_prepare_complete: bool = False
    vst_prepare_pid: Optional[int] = None
    vst_audit_complete: bool = False
    vst_audit_pid: Optional[int] = None
    missing_prefixes: Tuple[str, ...] = ()
    ovo_snapshot_complete: bool = False
    ovo_downloader_pid: Optional[int] = None
    ovo_stopped: bool = False
    ovo_prepare_complete: bool = False
    ovo_prepare_pid: Optional[int] = None
    smoke_pid: Optional[int] = None
    smoke_complete: bool = False
    sft_pid: Optional[int] = None
    sft_complete: bool = False
    ovo_eval_pid: Optional[int] = None
    ovo_eval_complete: bool = False
    training_gpu_ids: Tuple[int, ...] = ()
    idle_gpu_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Decision:
    action: str
    state: str


def inventory_report(root: Path, entries: list[dict]) -> dict:
    missing = []
    wrong_size = []
    valid = 0
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        actual = path.stat().st_size
        expected = entry["size"]
        if actual != expected:
            wrong_size.append(
                {"path": entry["path"], "expected": expected, "actual": actual}
            )
            continue
        valid += 1
    return {
        "complete": not missing and not wrong_size,
        "expected_files": len(entries),
        "valid_files": valid,
        "missing": missing,
        "wrong_size": wrong_size,
    }


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def decide(observation: Observation) -> Decision:
    if not observation.vst_snapshot_complete:
        if observation.vst_downloader_pid is not None:
            return Decision("none", "vst_downloading")
        return Decision("restart_vst", "vst_downloading")

    if not observation.vst_prepare_complete:
        if observation.vst_prepare_pid is not None:
            return Decision("none", "vst_preparing")
        return Decision("prepare_vst", "vst_preparing")

    if not observation.vst_audit_complete:
        if observation.vst_audit_pid is not None:
            return Decision("none", "vst_media_auditing")
        return Decision("audit_vst", "vst_media_auditing")

    if any(prefix != "Ego4D/" for prefix in observation.missing_prefixes):
        return Decision("block", "blocked_non_ego4d_media")

    if not observation.smoke_complete:
        if observation.smoke_pid is not None:
            return Decision("none", "smoke_running")
        return Decision("run_smoke", "smoke_running")

    if not observation.sft_complete:
        if observation.sft_pid is not None:
            return Decision("none", "sft_running")
        if len(observation.idle_gpu_ids) < 2:
            return Decision("none", "waiting_for_sft_gpus")
        return Decision("run_sft", "sft_running")

    if not observation.ovo_snapshot_complete:
        if observation.ovo_downloader_pid is None:
            return Decision("restart_ovo", "ovo_downloading")
        if observation.ovo_stopped:
            return Decision("resume_ovo", "ovo_downloading")
        return Decision("none", "ovo_downloading")

    if not observation.ovo_prepare_complete:
        if observation.ovo_prepare_pid is not None:
            return Decision("none", "ovo_preparing")
        return Decision("prepare_ovo", "ovo_preparing")

    if observation.ovo_eval_complete:
        return Decision("none", "ovo_eval_complete")
    if observation.ovo_eval_pid is not None:
        return Decision("none", "ovo_eval_running")
    if not observation.idle_gpu_ids:
        return Decision("none", "waiting_for_ovo_gpu")
    return Decision("run_ovo_eval", "ovo_eval_running")


def ovo_download_decision(observation: Observation) -> Optional[str]:
    if observation.ovo_snapshot_complete:
        return None
    if observation.ovo_downloader_pid is None:
        return "restart_ovo"
    if observation.ovo_stopped:
        return "resume_ovo"
    return None


ROOT = Path("/home/bujunru/vlm-repro/VST-full-reproduction")
VST_ROOT = ROOT / "data/VST-Training-Data-official-5647583491c2"
OVO_ROOT = ROOT / "data/OVO-Bench-official-fec29e3"
LOG_ROOT = ROOT / "logs/vst_download_orchestrator"
STATUS_PATH = LOG_ROOT / "status.json"
EVENTS_PATH = LOG_ROOT / "events.jsonl"
VST_INVENTORY = ROOT / "audit/inventories/vst_modelscope_aaef152e.json"
OVO_INVENTORY = ROOT / "audit/inventories/ovo_hf_fec29e3.json"


def command_for(action: str) -> list[str]:
    commands = {
        "restart_vst": [
            str(ROOT / ".venv-modelscope/bin/modelscope"),
            "download",
            "--repo-type",
            "dataset",
            "--revision",
            "aaef152ea68ffa0e9d9f7367ccf871ea2f699693",
            "--local-dir",
            str(VST_ROOT),
            "--max-workers",
            "8",
            "catalan/VST-Training-Data",
        ],
        "prepare_vst": ["/usr/bin/bash", str(ROOT / "audit/prepare_vst_official.sh")],
        "audit_vst": ["/usr/bin/bash", str(ROOT / "audit/audit_vst_no_ego4d.sh")],
        "restart_ovo": ["/usr/bin/bash", str(ROOT / "audit/download_ovobench_official.sh")],
        "prepare_ovo": ["/usr/bin/bash", str(ROOT / "audit/prepare_ovobench_official.sh")],
        "run_smoke": ["/usr/bin/bash", str(ROOT / "audit/run_vst_smoke.sh")],
        "run_sft": ["/usr/bin/bash", str(ROOT / "audit/run_vst_sft.sh")],
        "run_ovo_eval": ["/usr/bin/bash", str(ROOT / "audit/run_ovo_qwen3b_eval.sh")],
    }
    if action not in commands:
        raise ValueError(f"unsupported action: {action}")
    return commands[action]


def load_inventory(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def find_process(kind: str) -> tuple[Optional[int], bool]:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            decoded = [arg.decode(errors="replace") for arg in args if arg]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if kind == "vst":
            matches = (
                any(arg.endswith("/modelscope") for arg in decoded)
                and "download" in decoded
                and "catalan/VST-Training-Data" in decoded
            )
        elif kind == "ovo":
            matches = (
                any(arg.endswith("/hf") for arg in decoded)
                and "download" in decoded
                and "JoeLeelyf/OVO-Bench" in decoded
            )
        else:
            fragment = str(ROOT / f"audit/{kind}.sh")
            matches = fragment in decoded
        if not matches:
            continue
        try:
            state = (proc / "stat").read_text().split()[2]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        return int(proc.name), state == "T"
    return None, False


def marker_exists(name: str) -> bool:
    return (LOG_ROOT / name).is_file()


def gpu_allocation() -> tuple[Tuple[int, ...], Tuple[int, ...]]:
    try:
        all_gpu_text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
        )
        busy_gpu_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        uuid_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return (), ()
    all_ids = tuple(int(line.strip()) for line in all_gpu_text.splitlines() if line.strip())
    uuid_to_id = {}
    for line in uuid_rows.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        uuid_to_id[uuid] = int(index)
    busy_ids = tuple(
        sorted(
            {
                uuid_to_id[uuid.strip()]
                for uuid in busy_gpu_text.splitlines()
                if uuid.strip() in uuid_to_id
            }
        )
    )
    idle_ids = tuple(index for index in all_ids if index not in busy_ids)
    return busy_ids, idle_ids


def missing_prefixes() -> Tuple[str, ...]:
    path = ROOT / "audit/vst_no_ego4d_media_audit.json"
    if not path.is_file():
        return ()
    value = json.loads(path.read_text(encoding="utf-8"))
    return tuple(sorted(value.get("missing_prefixes", [])))


def observe() -> tuple[Observation, dict]:
    vst_entries = load_inventory(VST_INVENTORY)
    ovo_entries = load_inventory(OVO_INVENTORY)
    vst_report = inventory_report(VST_ROOT, vst_entries) if vst_entries else {
        "complete": False, "expected_files": 0, "valid_files": 0,
        "missing": [], "wrong_size": []
    }
    ovo_report = inventory_report(OVO_ROOT, ovo_entries) if ovo_entries else {
        "complete": False, "expected_files": 0, "valid_files": 0,
        "missing": [], "wrong_size": []
    }
    vst_pid, _ = find_process("vst")
    ovo_pid, ovo_stopped = find_process("ovo")
    prepare_vst_pid, _ = find_process("prepare_vst_official")
    audit_vst_pid, _ = find_process("audit_vst_no_ego4d")
    prepare_ovo_pid, _ = find_process("prepare_ovobench_official")
    smoke_pid, _ = find_process("run_vst_smoke")
    sft_pid, _ = find_process("run_vst_sft")
    ovo_eval_pid, _ = find_process("run_ovo_qwen3b_eval")
    training_gpu_ids, idle_gpu_ids = gpu_allocation()
    observation = Observation(
        vst_snapshot_complete=vst_report["complete"],
        vst_downloader_pid=vst_pid,
        vst_prepare_complete=marker_exists("vst_prepare.complete"),
        vst_prepare_pid=prepare_vst_pid,
        vst_audit_complete=marker_exists("vst_audit.complete"),
        vst_audit_pid=audit_vst_pid,
        missing_prefixes=missing_prefixes(),
        ovo_snapshot_complete=ovo_report["complete"],
        ovo_downloader_pid=ovo_pid,
        ovo_stopped=ovo_stopped,
        ovo_prepare_complete=marker_exists("ovo_prepare.complete"),
        ovo_prepare_pid=prepare_ovo_pid,
        smoke_pid=smoke_pid,
        smoke_complete=marker_exists("smoke.complete"),
        sft_pid=sft_pid,
        sft_complete=marker_exists("sft.complete"),
        ovo_eval_pid=ovo_eval_pid,
        ovo_eval_complete=marker_exists("ovo_eval.complete"),
        training_gpu_ids=training_gpu_ids,
        idle_gpu_ids=idle_gpu_ids,
    )
    details = {
        "vst": {"bytes": directory_bytes(VST_ROOT), **vst_report},
        "ovo": {"bytes": directory_bytes(OVO_ROOT), **ovo_report},
        "pids": {
            "vst": vst_pid,
            "ovo": ovo_pid,
            "sft": sft_pid,
            "ovo_eval": ovo_eval_pid,
        },
        "gpus": {"busy": list(training_gpu_ids), "idle": list(idle_gpu_ids)},
        "ovo_stopped": ovo_stopped,
        "missing_prefixes": list(observation.missing_prefixes),
    }
    return observation, details


def append_event(value: dict) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def start_command(action: str) -> int:
    command = command_for(action)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log = (LOG_ROOT / f"{action}.log").open("ab")
    environment = os.environ.copy()
    if action == "restart_vst":
        environment.update({
            "MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS": "1",
            "MODELSCOPE_DOWNLOAD_MAX_RETRIES": "10",
        })
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def execute(decision: Decision, observation: Observation) -> Optional[int]:
    if decision.action in {"none", "block"}:
        return None
    if decision.action == "resume_ovo":
        if observation.ovo_downloader_pid is None:
            raise RuntimeError("cannot resume OVO without a PID")
        os.kill(observation.ovo_downloader_pid, signal.SIGCONT)
        return observation.ovo_downloader_pid
    return start_command(decision.action)


def read_status() -> dict:
    if not STATUS_PATH.is_file():
        return {}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def run_once(dry_run: bool = False) -> dict:
    observation, details = observe()
    decision = decide(observation)
    maintenance_action = ovo_download_decision(observation)
    maintenance_pid = None
    if maintenance_action and maintenance_action != decision.action and not dry_run:
        maintenance_pid = execute(
            Decision(maintenance_action, "ovo_downloading"), observation
        )
    started_pid = None if dry_run else execute(decision, observation)
    now = datetime.now(timezone.utc).isoformat()
    previous = read_status()
    details["vst"]["delta_bytes"] = details["vst"]["bytes"] - previous.get("vst", {}).get("bytes", details["vst"]["bytes"])
    details["ovo"]["delta_bytes"] = details["ovo"]["bytes"] - previous.get("ovo", {}).get("bytes", details["ovo"]["bytes"])
    status = {
        "updated_at": now,
        "state": decision.state,
        "action": decision.action,
        "dry_run": dry_run,
        "started_pid": started_pid,
        "maintenance_action": maintenance_action,
        "maintenance_pid": maintenance_pid,
        **details,
    }
    if not dry_run:
        atomic_write_json(STATUS_PATH, status)
        append_event(status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--status-field")
    args = parser.parse_args()
    if args.status or args.status_field:
        value = read_status()
        if args.status_field:
            print(value.get(args.status_field, ""))
        else:
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.once:
        parser.error("use --once or --status")
    print(json.dumps(run_once(dry_run=args.dry_run), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
