"""
result_store.py
================
Write cumulative RLSD evaluation summaries to:
  - eval_summary.csv: narrow per-dataset/per-overall rows
  - eval_summary_by_dataset.csv: wide acc/ttft rows
  - eval_summary.xlsx: standard Excel workbook with 8 sheets:
      streamingbench, ovobench, videoholmes, longvideobench, videomme,
      overall, acc_by_dataset, ttft_by_dataset

OVO-Bench is treated as one dataset: its three task shards are pooled into a
single `ovobench` row before writing any summary output.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from openpyxl import Workbook

DATASET_SHEETS = ["ovobench", "streamingbench", "videomme", "longvideobench", "videoholmes"]
SHEET_NAMES = DATASET_SHEETS + ["overall", "acc_by_dataset", "ttft_by_dataset"]

_TASK_TO_DATASET = {
    "streamingbench_subset7500": "streamingbench",
    "ovobench_real_time_visual_perception_subset7500": "ovobench",
    "ovobench_backward_tracking_subset7500": "ovobench",
    "ovobench_forward_active_responding_subset7500": "ovobench",
    "videoholmes_subset7500": "videoholmes",
    "longvideobench_subset7500": "longvideobench",
    "videomme_subset7500": "videomme",
    # Full dataset OVO-Bench sub-tasks also map to "ovobench"
    "ovobench_real_time_visual_perception": "ovobench",
    "ovobench_backward_tracking": "ovobench",
    "ovobench_forward_active_responding": "ovobench",
}

_HEADERS = [
    "run_id",
    "sheet",
    "task",
    "engine",
    "exp_name",
    "key_config",
    "ckpt_path",
    "ckpt_created_at",
    "run_output_dir",
    "num_docs",
    "num_gpus",
    "acc",
    "total_correct",
    "total_answered",
    "answer_time_s",
    "num_segments",
    "ttft_s",
    "answer_total_tokens",
    "memory_final_chars",
    "wall_clock_s",
    "samples_per_s",
]

_DATASET_NUM_COLS = [f"n_{ds}" for ds in DATASET_SHEETS]  # n_ovobench, n_streamingbench, ...
_WIDE_HEADERS = ["run_id", "exp_name", "key_config"] + DATASET_SHEETS + ["overall", "样本数"] + _DATASET_NUM_COLS


def task_to_dataset(task_name: str) -> str:
    return _TASK_TO_DATASET.get(task_name, task_name.replace("_subset7500", ""))


def build_dataset_rows(
    *,
    engine: str,
    queue_row: Any,
    task_results: dict[str, dict],
    run_output_dir: str,
    chunking_mode: str | None = None,
    sample_concurrency: int | None = None,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict]] = {}
    for task_name, result in task_results.items():
        by_dataset.setdefault(task_to_dataset(task_name), {})[task_name] = result

    rows = []
    for dataset in DATASET_SHEETS:
        if dataset in by_dataset:
            rows.append(
                _pool_task_results(
                    sheet=dataset,
                    task_name=dataset,
                    engine=engine,
                    queue_row=queue_row,
                    task_results=by_dataset[dataset],
                    run_output_dir=run_output_dir,
                    chunking_mode=chunking_mode,
                    sample_concurrency=sample_concurrency,
                )
            )
    return rows


def build_overall_row(
    *,
    engine: str,
    queue_row: Any,
    task_results: dict[str, dict],
    run_output_dir: str,
    chunking_mode: str | None = None,
    sample_concurrency: int | None = None,
) -> dict[str, Any]:
    return _pool_task_results(
        sheet="overall",
        task_name="overall",
        engine=engine,
        queue_row=queue_row,
        task_results=task_results,
        run_output_dir=run_output_dir,
        chunking_mode=chunking_mode,
        sample_concurrency=sample_concurrency,
    )


def build_wide_rows(
    *,
    engine: str,
    queue_row: Any,
    task_results: dict[str, dict],
    chunking_mode: str | None = None,
    sample_concurrency: int | None = None,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict]] = {}
    for task_name, result in task_results.items():
        by_dataset.setdefault(task_to_dataset(task_name), {})[task_name] = result

    acc_row = {"sheet": "acc_by_dataset"}
    ttft_row = {"sheet": "ttft_by_dataset"}
    total_num_docs = 0
    # For computing overall weighted average
    acc_weighted_sum = 0.0
    acc_total_answered = 0
    ttft_weighted_sum = 0.0
    ttft_total_docs = 0
    for dataset in DATASET_SHEETS:
        n_col = f"n_{dataset}"
        if dataset not in by_dataset:
            acc_row[dataset] = ""
            ttft_row[dataset] = ""
            acc_row[n_col] = ""
            ttft_row[n_col] = ""
            continue
        pooled = _pool_task_results(
            sheet=dataset,
            task_name=dataset,
            engine=engine,
            queue_row=queue_row,
            task_results=by_dataset[dataset],
            run_output_dir="",
            chunking_mode=chunking_mode,
            sample_concurrency=sample_concurrency,
        )
        docs = pooled.get("num_docs") or 0
        acc_row[dataset] = pooled["acc"]
        ttft_row[dataset] = pooled["ttft_s"]
        acc_row[n_col] = docs
        ttft_row[n_col] = docs
        total_num_docs += docs
        # Accumulate for overall
        answered = pooled.get("total_answered") or 0
        correct = pooled.get("total_correct") or 0
        acc_weighted_sum += correct
        acc_total_answered += answered
        if pooled["ttft_s"] is not None and docs:
            ttft_weighted_sum += pooled["ttft_s"] * docs
            ttft_total_docs += docs

    # Compute overall column
    acc_row["overall"] = (100.0 * acc_weighted_sum / acc_total_answered) if acc_total_answered else ""
    ttft_row["overall"] = (ttft_weighted_sum / ttft_total_docs) if ttft_total_docs else ""

    # Add metadata columns
    exp_name = getattr(queue_row, "exp_name", "manual_model")
    key_config = _build_key_config(queue_row, chunking_mode, sample_concurrency)
    acc_row["exp_name"] = exp_name
    acc_row["key_config"] = key_config
    acc_row["样本数"] = total_num_docs
    ttft_row["exp_name"] = exp_name
    ttft_row["key_config"] = key_config
    ttft_row["样本数"] = total_num_docs
    return [acc_row, ttft_row]


def _pool_task_results(
    *,
    sheet: str,
    task_name: str,
    engine: str,
    queue_row: Any,
    task_results: dict[str, dict],
    run_output_dir: str,
    chunking_mode: str | None,
    sample_concurrency: int | None,
) -> dict[str, Any]:
    total_correct = 0
    total_answered = 0
    num_docs = 0
    num_gpus = None
    wall_clock_s = 0.0
    ttft_sum = 0.0
    ttft_n = 0
    sample_total_sum = 0.0
    sample_total_n = 0
    num_segments_sum = 0.0
    num_segments_n = 0
    answer_tokens_sum = 0.0
    answer_tokens_n = 0
    memory_chars_sum = 0.0
    memory_chars_n = 0

    for result in task_results.values():
        metrics = result.get("metrics", {})
        answered = metrics.get("total_answered") or 0
        correct = metrics.get("total_correct") or 0
        docs = result.get("num_docs") or 0
        total_correct += correct
        total_answered += answered
        num_docs += docs
        num_gpus = result.get("num_gpus", num_gpus)
        wall_clock_s += result.get("wall_clock_s") or 0.0

        timing = result.get("timing", {}) if isinstance(result.get("timing"), dict) else {}
        ttft = timing.get("ttft_s", {}) if isinstance(timing.get("ttft_s"), dict) else {}
        sample_total = timing.get("sample_total_s", {}) if isinstance(timing.get("sample_total_s"), dict) else {}
        if ttft.get("mean") is not None and docs:
            ttft_sum += ttft["mean"] * docs
            ttft_n += docs
        if sample_total.get("mean") is not None and docs:
            sample_total_sum += sample_total["mean"] * docs
            sample_total_n += docs

        rollup = _sample_rollup(result.get("sample_log"))
        if rollup["num_segments"] is not None:
            num_segments_sum += rollup["num_segments"] * rollup["num_segments_n"]
            num_segments_n += rollup["num_segments_n"]
        if rollup["answer_total_tokens"] is not None:
            answer_tokens_sum += rollup["answer_total_tokens"] * rollup["answer_total_tokens_n"]
            answer_tokens_n += rollup["answer_total_tokens_n"]
        if rollup["memory_final_chars"] is not None:
            memory_chars_sum += rollup["memory_final_chars"] * rollup["memory_final_chars_n"]
            memory_chars_n += rollup["memory_final_chars_n"]

    return {
        "sheet": sheet,
        "task": task_name,
        "engine": engine,
        "exp_name": getattr(queue_row, "exp_name", "manual_model"),
        "key_config": _build_key_config(queue_row, chunking_mode, sample_concurrency),
        "ckpt_path": getattr(queue_row, "ckpt_path", ""),
        "ckpt_created_at": getattr(queue_row, "created_at", ""),
        "run_output_dir": run_output_dir,
        "num_docs": num_docs,
        "num_gpus": num_gpus,
        "acc": (100.0 * total_correct / total_answered) if total_answered else None,
        "total_correct": total_correct,
        "total_answered": total_answered,
        "answer_time_s": (sample_total_sum / sample_total_n) if sample_total_n else None,
        "num_segments": (num_segments_sum / num_segments_n) if num_segments_n else None,
        "ttft_s": (ttft_sum / ttft_n) if ttft_n else None,
        "answer_total_tokens": (answer_tokens_sum / answer_tokens_n) if answer_tokens_n else None,
        "memory_final_chars": (memory_chars_sum / memory_chars_n) if memory_chars_n else None,
        "wall_clock_s": wall_clock_s,
        "samples_per_s": (num_docs / wall_clock_s) if wall_clock_s > 0 else None,
    }


def _build_key_config(queue_row: Any, chunking_mode: str | None, sample_concurrency: int | None) -> str:
    key_config = getattr(queue_row, "key_config", "")
    if chunking_mode:
        key_config = f"{key_config}|chunking_mode={chunking_mode}" if key_config else f"chunking_mode={chunking_mode}"
    if sample_concurrency is not None:
        key_config = f"{key_config}|sample_concurrency={sample_concurrency}" if key_config else f"sample_concurrency={sample_concurrency}"
    return key_config


def _sample_rollup(sample_log: str | None) -> dict[str, Any]:
    empty = {
        "num_segments": None,
        "num_segments_n": 0,
        "answer_total_tokens": None,
        "answer_total_tokens_n": 0,
        "memory_final_chars": None,
        "memory_final_chars_n": 0,
    }
    if not sample_log or not Path(sample_log).exists():
        return empty

    num_segments_sum = 0.0
    num_segments_n = 0
    answer_tokens_sum = 0.0
    answer_tokens_n = 0
    memory_chars_sum = 0.0
    memory_chars_n = 0
    with open(sample_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            num_chunks = (sample.get("stream_config") or {}).get("num_chunks")
            if num_chunks is not None:
                num_segments_sum += num_chunks
                num_segments_n += 1
            chunks = sample.get("chunks") or []
            final_tokens = None
            if chunks:
                final_tokens = chunks[-1].get("num_output_tokens")
            if final_tokens is not None:
                answer_tokens_sum += final_tokens
                answer_tokens_n += 1
            memory_chars = sample.get("memory_final_chars")
            if memory_chars is not None:
                memory_chars_sum += memory_chars
                memory_chars_n += 1

    return {
        "num_segments": (num_segments_sum / num_segments_n) if num_segments_n else None,
        "num_segments_n": num_segments_n,
        "answer_total_tokens": (answer_tokens_sum / answer_tokens_n) if answer_tokens_n else None,
        "answer_total_tokens_n": answer_tokens_n,
        "memory_final_chars": (memory_chars_sum / memory_chars_n) if memory_chars_n else None,
        "memory_final_chars_n": memory_chars_n,
    }


def _round_floats(row: dict[str, Any], ndigits: int = 3) -> dict[str, Any]:
    """Round all float values in a row dict to `ndigits` decimal places."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out


def append_summary_rows(summary_csv: str, summary_xlsx: str, new_rows: list[dict[str, Any]]) -> None:
    if not new_rows:
        return
    narrow_csv, wide_csv = _narrow_and_wide_paths(summary_csv)
    Path(narrow_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(wide_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_xlsx).parent.mkdir(parents=True, exist_ok=True)

    narrow_rows = _read_csv_rows(narrow_csv)
    wide_rows = _read_csv_rows(wide_csv)
    run_id = max(_next_run_id(narrow_rows), _next_run_id(wide_rows))

    for row in new_rows:
        row = _round_floats(row)
        if row.get("sheet") in ("acc_by_dataset", "ttft_by_dataset"):
            normalized = {h: row.get(h, "") for h in _WIDE_HEADERS}
            normalized["run_id"] = run_id
            normalized["sheet"] = row.get("sheet")
            wide_rows.append(normalized)
        else:
            normalized = {h: row.get(h, "") for h in _HEADERS}
            normalized["run_id"] = run_id
            narrow_rows.append(normalized)

    _write_csv_atomic(narrow_csv, narrow_rows, _HEADERS)
    _write_csv_atomic(wide_csv, wide_rows, ["sheet"] + _WIDE_HEADERS)
    _write_xlsx_atomic(summary_xlsx, narrow_rows, wide_rows)


def reset_summary(summary_csv: str, summary_xlsx: str) -> None:
    narrow_csv, wide_csv = _narrow_and_wide_paths(summary_csv)
    Path(narrow_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(wide_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_xlsx).parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(narrow_csv, [], _HEADERS)
    _write_csv_atomic(wide_csv, [], ["sheet"] + _WIDE_HEADERS)
    _write_xlsx_atomic(summary_xlsx, [], [])


def _narrow_and_wide_paths(csv_path: str) -> tuple[str, str]:
    p = Path(csv_path)
    return str(p), str(p.with_name(p.stem + "_by_dataset" + p.suffix))


def _next_run_id(rows: list[dict[str, Any]]) -> int:
    max_id = 0
    for row in rows:
        try:
            max_id = max(max_id, int(row.get("run_id") or 0))
        except (TypeError, ValueError):
            continue
    return max_id + 1


def _read_csv_rows(path: str) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_atomic(path: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    out_path = Path(path)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_path.parent)) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _write_xlsx_atomic(path: str, narrow_rows: list[dict[str, Any]], wide_rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    with NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent), suffix=".xlsx") as tmp:
        tmp_path = tmp.name
    try:
        wb = Workbook()
        default = wb.active
        wb.remove(default)
        for sheet in SHEET_NAMES:
            ws = wb.create_sheet(title=sheet)
            if sheet in ("acc_by_dataset", "ttft_by_dataset"):
                ws.append(_WIDE_HEADERS)
                for row in wide_rows:
                    if row.get("sheet") == sheet:
                        ws.append([row.get(h, "") for h in _WIDE_HEADERS])
            else:
                ws.append(_HEADERS)
                for row in narrow_rows:
                    if row.get("sheet") == sheet:
                        ws.append([row.get(h, "") for h in _HEADERS])
            _autosize(ws)
        wb.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _autosize(ws: Any) -> None:
    for column_cells in ws.columns:
        max_len = 0
        letter = column_cells[0].column_letter
        for cell in column_cells:
            value = cell.value
            if value is not None:
                max_len = max(max_len, len(str(value)))
        ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 60)
