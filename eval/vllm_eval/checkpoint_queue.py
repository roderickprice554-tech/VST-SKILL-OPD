"""
checkpoint_queue.py
====================
Reads/writes the checkpoint evaluation queue defined by
`eval_config.CHECKPOINT_QUEUE_CSV`.

Queue CSV columns:
  exp_name     - human-readable experiment name (free text)
  key_config   - key hyperparams, "|"-joined (free text, for eyeballing)
  ckpt_path    - HF model dir OR verl actor dir (same as --model previously)
  created_at   - "YYYY-MM-DD HH:MM:SS", when this row was added
  evaluated    - 0/1 flag; engines flip this to 1 once the row's checkpoint
                 has been fully evaluated across all requested tasks

Concurrency note: multiple engine invocations could race on this file if
launched at the same time. We take a simple file lock (fcntl) around the
read-modify-write of the "mark evaluated" step so two concurrently-running
engines don't stomp on each other's flag updates. Picking which row(s) to
run is still "read whatever is 0 at process start" -- if you truly run two
engine processes in parallel pointed at the same queue, they may both pick
up the same pending row. Recommended usage is one engine process at a time
consuming the queue (it already evaluates every pending row per invocation
and is internally multi-GPU data-parallel), which is what vllm_eval_engine*.py
now do by default.
"""
from __future__ import annotations

import csv
import fcntl
from dataclasses import dataclass
from pathlib import Path

_FIELDNAMES = ["exp_name", "key_config", "ckpt_path", "created_at", "evaluated", "train_steps"]


@dataclass
class QueueRow:
    exp_name: str
    key_config: str
    ckpt_path: str
    created_at: str
    evaluated: int
    row_index: int  # 0-based position in the CSV, used to write the flag back


def _read_all_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_pending(csv_path: str) -> list[QueueRow]:
    """Return all rows with evaluated == 0, in file order."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Checkpoint queue CSV not found: {csv_path}. "
            f"Create it with columns: {','.join(_FIELDNAMES)}"
        )
    rows = _read_all_rows(csv_path)
    pending = []
    for i, r in enumerate(rows):
        if str(r.get("evaluated", "0")).strip() in ("0", "", "False", "false"):
            pending.append(QueueRow(
                exp_name=r.get("exp_name", ""),
                key_config=r.get("key_config", ""),
                ckpt_path=r.get("ckpt_path", ""),
                created_at=r.get("created_at", ""),
                evaluated=0,
                row_index=i,
            ))
    return pending


def mark_evaluated(csv_path: str, row_index: int) -> None:
    """Flip a single row's evaluated flag to 1 and persist, under a file lock
    so concurrent readers/writers don't corrupt the CSV."""
    with open(csv_path, "r+", encoding="utf-8", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            rows = list(csv.DictReader(f))
            if row_index < 0 or row_index >= len(rows):
                raise IndexError(f"Queue row index {row_index} out of range for {csv_path}")
            rows[row_index]["evaluated"] = "1"

            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_row(csv_path: str, exp_name: str, key_config: str, ckpt_path: str, created_at: str, train_steps: str = "") -> None:
    """Append a new pending (evaluated=0) row to the queue CSV, creating the
    file with a header if it doesn't exist yet."""
    exists = Path(csv_path).exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "exp_name": exp_name,
                "key_config": key_config,
                "ckpt_path": ckpt_path,
                "created_at": created_at,
                "evaluated": "0",
                "train_steps": train_steps,
            })
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
