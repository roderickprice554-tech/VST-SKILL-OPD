"""
progress.py
===========
Small JSON-file based progress reporting for multi-process vLLM evaluation.

Each GPU worker owns one progress JSON file and atomically rewrites it when its
state changes. The engine process polls those files and renders a single
aggregate progress line across all shards.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_progress(path: str | None, state: dict[str, Any]) -> None:
    if not path:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**state, "updated_at": time.time()}
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(progress_path.parent)) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, progress_path)


def read_progress(path: str | Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"
    try:
        seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        return "--:--"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def summarize_progress(states: list[dict[str, Any]], started_at: float) -> dict[str, Any]:
    total = sum(_as_int(s.get("total")) for s in states)
    done = sum(_as_int(s.get("done")) for s in states)
    correct = sum(_as_int(s.get("correct")) for s in states)
    scored = sum(_as_int(s.get("scored")) for s in states)
    elapsed = time.time() - started_at
    throughput = done / elapsed if elapsed > 0 and done > 0 else None
    remaining = max(0, total - done) if total else 0
    eta = remaining / throughput if throughput and remaining else None

    phase_counts: dict[str, int] = {}
    for state in states:
        phase = str(state.get("phase") or "starting")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    phase = max(phase_counts.items(), key=lambda kv: kv[1])[0] if phase_counts else "starting"
    pct = (100.0 * done / total) if total else 0.0
    acc = (100.0 * correct / scored) if scored > 0 else None
    return {
        "done": done,
        "total": total,
        "pct": pct,
        "elapsed": elapsed,
        "eta": eta,
        "throughput": throughput,
        "phase": phase,
        "phase_counts": phase_counts,
        "correct": correct,
        "scored": scored,
        "acc": acc,
    }


def render_progress_line(task_label: str, summary: dict[str, Any]) -> str:
    throughput = summary.get("throughput")
    throughput_text = f"{throughput:.2f} samples/s" if throughput else "-- samples/s"
    acc = summary.get("acc")
    acc_text = f"acc={acc:.1f}% ({summary['correct']}/{summary['scored']})" if acc is not None else "acc=--"
    return (
        f"{task_label} | {summary['done']}/{summary['total']} "
        f"({summary['pct']:.1f}%) | {acc_text} | phase={summary['phase']} | "
        f"elapsed={format_duration(summary['elapsed'])} | eta={format_duration(summary['eta'])} | "
        f"{throughput_text}"
    )


def poll_progress_until_done(
    *,
    procs: list[tuple[Any, Any, int]],
    progress_paths: list[Path],
    task_label: str,
    poll_interval: float = 1.0,
) -> list[int]:
    started_at = time.time()
    failures: list[int] = []
    finished: set[int] = set()
    last_line_len = 0

    while len(finished) < len(procs):
        for idx, (proc, _log_f, gpu_id) in enumerate(procs):
            if idx in finished:
                continue
            ret = proc.poll()
            if ret is not None:
                finished.add(idx)
                if ret != 0:
                    failures.append(gpu_id)

        states = [s for p in progress_paths if (s := read_progress(p)) is not None]
        if states:
            line = render_progress_line(task_label, summarize_progress(states, started_at))
            padding = " " * max(0, last_line_len - len(line))
            print("\r" + line + padding, end="", flush=True)
            last_line_len = len(line)
        time.sleep(poll_interval)

    states = [s for p in progress_paths if (s := read_progress(p)) is not None]
    if states:
        line = render_progress_line(task_label, summarize_progress(states, started_at))
        padding = " " * max(0, last_line_len - len(line))
        print("\r" + line + padding, flush=True)
    elif last_line_len:
        print(file=sys.stderr)
    return failures


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
