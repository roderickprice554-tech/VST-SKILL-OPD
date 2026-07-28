#!/usr/bin/env python3
"""
vllm_eval_engine_stream.py
==========================
High-efficiency vLLM StreamThink evaluator for streaming video tasks.

Compared with vllm_eval_engine.py (single-pass batch VQA), this file runs a
stateful streaming-style inference loop:
  1. Split each video into estimated visual-token chunks.
  2. For all early chunks, ask the model for intermediate thinking and append it
     to textual memory.
  3. Submit the final chunk + accumulated memory + official question to generate
     the official answer.
  4. Record TTFT and detailed per-sample logs (stream chunks, memory updates,
     final output, GT, score).

It remains data-parallel across GPUs: one vLLM engine per GPU, each worker owns
a disjoint sample shard. This is deliberate: Qwen3-VL-8B fits in one H800, and
TP was measured to be a net loss for this evaluation workload.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_eval.ckpt_utils import resolve_hf_model_dir  # noqa: E402
from vllm_eval.tasks import TASK_REGISTRY, build_task  # noqa: E402
from vllm_eval.checkpoint_queue import load_pending, mark_evaluated  # noqa: E402
from vllm_eval.progress import poll_progress_until_done  # noqa: E402
from vllm_eval.result_store import append_summary_rows, build_dataset_rows, build_overall_row, build_wide_rows  # noqa: E402

try:
    from eval_config import (
        CHECKPOINT_QUEUE_CSV,
        DEFAULT_ENGINE,
        DEFAULT_GPUS,
        DEFAULT_SAMPLING,
        DEFAULT_STREAM,
        DEFAULT_STREAM_OUTPUT_DIR,
        DEFAULT_VST_ORIGINAL,
        RESULTS_SUMMARY_CSV,
        RESULTS_SUMMARY_XLSX,
    )
except ImportError:
    CHECKPOINT_QUEUE_CSV = None
    DEFAULT_ENGINE = None
    DEFAULT_GPUS = None
    DEFAULT_SAMPLING = None
    DEFAULT_STREAM = None
    DEFAULT_VST_ORIGINAL = None
    DEFAULT_STREAM_OUTPUT_DIR = "./eval_outputs/vllm_stream_run"
    RESULTS_SUMMARY_CSV = "./results/eval_summary.csv"
    RESULTS_SUMMARY_XLSX = "./results/eval_summary.xlsx"


_WORKER_MODULE = "vllm_eval.worker_stream"
_THIS_DIR = Path(__file__).resolve().parent
_TASK_GROUPS = {
    "ovobench": [
        "ovobench_real_time_visual_perception",
        "ovobench_backward_tracking",
        "ovobench_forward_active_responding",
    ],
    "all": [
        "streamingbench",
        "ovobench_real_time_visual_perception",
        "ovobench_backward_tracking",
        "ovobench_forward_active_responding",
        "videoholmes",
        "longvideobench",
        "videomme",
    ],
    "all_no_streamingbench": [
        "ovobench_real_time_visual_perception",
        "ovobench_backward_tracking",
        "ovobench_forward_active_responding",
        "videoholmes",
        "longvideobench",
        "videomme",
    ],
    "all_subset7500": [
        "streamingbench_subset7500",
        "ovobench_real_time_visual_perception_subset7500",
        "ovobench_backward_tracking_subset7500",
        "ovobench_forward_active_responding_subset7500",
        "videoholmes_subset7500",
        "longvideobench_subset7500",
        "videomme_subset7500",
    ],
}


def _expand_tasks(task_names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in task_names:
        expanded.extend(_TASK_GROUPS.get(name, [name]))
    unknown = [name for name in expanded if name not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Available: {sorted(TASK_REGISTRY)} plus groups {sorted(_TASK_GROUPS)}")
    return expanded


def _summarize_timing(rows: list[dict]) -> dict:
    ttfts = [r.get("ttft_s") for r in rows if isinstance(r.get("ttft_s"), (int, float))]
    totals = [r.get("sample_total_s") for r in rows if isinstance(r.get("sample_total_s"), (int, float))]

    def stats(xs):
        if not xs:
            return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
        xs = sorted(xs)
        def pct(p):
            idx = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
            return xs[idx]
        return {
            "count": len(xs),
            "mean": statistics.mean(xs),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "max": max(xs),
        }

    return {"ttft_s": stats(ttfts), "sample_total_s": stats(totals)}


def _terminate_process_group(proc: subprocess.Popen, *, grace_s: float = 8.0) -> None:
    """Terminate a worker and every child it spawned (including vLLM EngineCore)."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _cleanup_worker_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        _terminate_process_group(proc)


# Module-level registry of currently-live worker Popen handles, so a SIGINT
# (Ctrl-C) or SIGTERM sent to *this* engine process can still reap every
# worker's process group -- otherwise killing only the engine PID leaves the
# `python -m vllm_eval.worker_stream` processes (and their vLLM EngineCore
# children / GPU memory) running as orphans, which is exactly what happened
# when a partially-launched run was killed manually before this fix.
_ACTIVE_WORKER_PROCS: list[subprocess.Popen] = []


def _handle_termination_signal(signum, _frame):
    print(f"\n[stream-engine] received signal {signum}, terminating {len(_ACTIVE_WORKER_PROCS)} worker process group(s)...", flush=True)
    _cleanup_worker_processes(list(_ACTIVE_WORKER_PROCS))
    sys.exit(128 + signum)


signal.signal(signal.SIGINT, _handle_termination_signal)
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _run_task_on_gpus(
    *,
    task_name: str,
    model_dir: str,
    gpus: list[int],
    output_dir: Path,
    max_num_frames: int,
    max_pixels: int | None,
    sample_fps: float,
    chunking_mode: str,
    stream_token_budget: int,
    tokens_per_frame: int,
    max_chunks: int,
    max_memory_chars: int,
    stream_think_times: str,
    max_stream_vid_tokens: int,
    max_keep_memory: int,
    think_max_tokens: int,
    use_predecoded_video: bool,
    decode_num_threads: int,
    sample_concurrency: int,
    decode_timeout_s: float,
    max_model_len: int,
    gpu_memory_utilization: float,
    limit: int,
    use_torchvision_resize: bool = False,
    frame_sampling_mode: str = "linspace",
    max_retry_rounds: int = 3,
) -> dict:
    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    num_shards = len(gpus)
    procs = []
    shard_outputs = []
    shard_jsonls = []
    progress_paths = []
    t0 = time.perf_counter()

    for shard_id, gpu_id in enumerate(gpus):
        shard_output = task_output_dir / f"shard_{shard_id}.json"
        shard_jsonl = task_output_dir / f"shard_{shard_id}.samples.jsonl"
        progress_path = task_output_dir / f"shard_{shard_id}.progress.json"
        progress_path.unlink(missing_ok=True)
        shard_outputs.append(shard_output)
        shard_jsonls.append(shard_jsonl)
        progress_paths.append(progress_path)
        cmd = [
            sys.executable, "-m", _WORKER_MODULE,
            "--task", task_name,
            "--model", model_dir,
            "--gpu-id", str(gpu_id),
            "--shard-id", str(shard_id),
            "--num-shards", str(num_shards),
            "--max-num-frames", str(max_num_frames),
            "--max-pixels", str(max_pixels) if max_pixels else "0",
            "--sample-fps", str(sample_fps),
            "--chunking-mode", chunking_mode,
            "--stream-token-budget", str(stream_token_budget),
            "--tokens-per-frame", str(tokens_per_frame),
            "--max-chunks", str(max_chunks),
            "--max-memory-chars", str(max_memory_chars),
            "--stream-think-times", stream_think_times,
            "--max-stream-vid-tokens", str(max_stream_vid_tokens),
            "--max-keep-memory", str(max_keep_memory),
            "--think-max-tokens", str(think_max_tokens),
            "--use-predecoded-video", "1" if use_predecoded_video else "0",
            "--decode-num-threads", str(decode_num_threads),
            "--sample-concurrency", str(sample_concurrency),
            "--decode-timeout-s", str(decode_timeout_s),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--use-torchvision-resize", "1" if use_torchvision_resize else "0",
            "--frame-sampling-mode", frame_sampling_mode,
            "--max-retry-rounds", str(max_retry_rounds),
            "--output", str(shard_output),
            "--jsonl-log", str(shard_jsonl),
            "--progress-file", str(progress_path),
        ]
        if limit > 0:
            cmd += ["--limit", str(limit)]

        log_path = task_output_dir / f"shard_{shard_id}.log"
        print(f"[stream-engine] launching {task_name} shard {shard_id} on GPU {gpu_id} -> {log_path}")
        log_f = open(log_path, "w", encoding="utf-8")
        # start_new_session=True puts each worker in its own process group so
        # we can SIGTERM/SIGKILL the whole group later -- otherwise killing
        # only the worker's own pid leaves vLLM's "VLLM::EngineCore" child
        # process (and its GPU memory) running as an orphan.
        proc = subprocess.Popen(
            cmd, cwd=str(_THIS_DIR.parent), stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((proc, log_f, gpu_id))
        _ACTIVE_WORKER_PROCS.append(proc)

    try:
        failures = poll_progress_until_done(
            procs=procs,
            progress_paths=progress_paths,
            task_label=f"[stream-engine] {task_name}",
        )
    except BaseException:
        _cleanup_worker_processes([proc for proc, _log_f, _gpu_id in procs])
        raise
    finally:
        for proc, log_f, _gpu_id in procs:
            log_f.close()
            if proc in _ACTIVE_WORKER_PROCS:
                _ACTIVE_WORKER_PROCS.remove(proc)
    wall_clock_s = time.perf_counter() - t0

    if failures:
        raise RuntimeError(
            f"Stream task '{task_name}' failed on GPU(s) {failures}. "
            f"See {task_output_dir}/shard_*.log for details."
        )

    rows = []
    total_timeout_count = 0
    total_error_count = 0
    total_retry_recovered = 0
    total_retry_still_failed = 0
    retry_still_failed_samples = []
    for shard_output in shard_outputs:
        with open(shard_output, "r", encoding="utf-8") as f:
            shard_data = json.load(f)
            rows.extend(shard_data["rows"])
            total_timeout_count += shard_data.get("timeout_count", 0)
            total_error_count += shard_data.get("error_count", 0)
            total_retry_recovered += shard_data.get("retry_recovered_count", 0)
            total_retry_still_failed += shard_data.get("retry_still_failed_count", 0)
            retry_still_failed_samples.extend(shard_data.get("retry_still_failed_samples", []))

    task = build_task(task_name)
    metrics = task.aggregate(rows)
    timing = _summarize_timing(rows)

    # Merge detailed JSONL logs into one file for easier downstream analysis.
    merged_jsonl = task_output_dir / "samples.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as out_f:
        for shard_jsonl in shard_jsonls:
            with open(shard_jsonl, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)

    num_docs = len(rows)
    failure_count = total_timeout_count + total_error_count
    result = {
        "task": task_name,
        "num_docs": num_docs,
        "num_gpus": num_shards,
        "wall_clock_s": wall_clock_s,
        "samples_per_s": num_docs / wall_clock_s if wall_clock_s > 0 else None,
        "metrics": metrics,
        "timing": timing,
        "runtime_config": {
            "sample_concurrency": sample_concurrency,
            "decode_timeout_s": decode_timeout_s,
        },
        "timeout_count": total_timeout_count,
        "error_count": total_error_count,
        "failure_rate": failure_count / num_docs if num_docs else 0.0,
        "retry_recovered": total_retry_recovered,
        "retry_still_failed": total_retry_still_failed,
        "retry_still_failed_samples": retry_still_failed_samples,
        "sample_log": str(merged_jsonl),
    }
    if failure_count:
        retry_suffix = f" [retry recovered {total_retry_recovered}]" if total_retry_recovered else ""
        print(
            f"[stream-engine] ⚠️  {task_name}: {total_timeout_count} timeout(s), "
            f"{total_error_count} other error(s) / {num_docs} total "
            f"({100.0 * failure_count / num_docs:.2f}% failure rate)"
            f"{retry_suffix}"
        )
    with open(task_output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({**result, "rows": rows}, f, ensure_ascii=False, indent=2)
    return result


def _manual_queue_row(model_path: str):
    return SimpleNamespace(
        exp_name=Path(model_path).name or "manual_model",
        key_config="manual --model override",
        ckpt_path=model_path,
        created_at="",
        row_index=-1,
    )


def _run_checkpoint(args, queue_row, tasks: list[str], gpus: list[int]) -> dict:
    # Nest under chunking_mode + sample_concurrency so running token_budget vs
    # vst_original, or serial vs concurrent pipelines, for the same exp_name
    # doesn't overwrite each other's samples.jsonl/summary.json.
    output_dir = Path(args.output_dir) / queue_row.exp_name / args.chunking_mode / f"concurrency_{args.sample_concurrency}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stream-engine] Resolving model dir for '{queue_row.ckpt_path}' ...")
    model_dir = resolve_hf_model_dir(queue_row.ckpt_path)
    print(f"[stream-engine] Experiment: {queue_row.exp_name}")
    print(f"[stream-engine] Key config: {queue_row.key_config}")
    print(f"[stream-engine] Using HF model dir: {model_dir}")
    print(f"[stream-engine] GPUs: {gpus} (data parallel)")
    print(
        f"[stream-engine] chunking_mode={args.chunking_mode}, "
        f"stream_token_budget={args.stream_token_budget}, "
        f"tokens_per_frame={args.tokens_per_frame}, max_chunks={args.max_chunks}, "
        f"stream_think_times={args.stream_think_times}"
    )

    all_results = {}
    for task_name in tasks:
        print(f"\n[stream-engine] ==== Running stream task: {task_name} ====")
        result = _run_task_on_gpus(
            task_name=task_name,
            model_dir=model_dir,
            gpus=gpus,
            output_dir=output_dir,
            max_num_frames=args.max_num_frames,
            max_pixels=args.max_pixels,
            sample_fps=args.sample_fps,
            chunking_mode=args.chunking_mode,
            stream_token_budget=args.stream_token_budget,
            tokens_per_frame=args.tokens_per_frame,
            max_chunks=args.max_chunks,
            max_memory_chars=args.max_memory_chars,
            stream_think_times=args.stream_think_times,
            max_stream_vid_tokens=args.max_stream_vid_tokens,
            max_keep_memory=args.max_keep_memory,
            think_max_tokens=args.think_max_tokens,
            use_predecoded_video=bool(args.use_predecoded_video),
            decode_num_threads=args.decode_num_threads,
            sample_concurrency=args.sample_concurrency,
            decode_timeout_s=args.decode_timeout_s,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit=args.limit,
            use_torchvision_resize=getattr(args, 'use_torchvision_resize', False),
            frame_sampling_mode=getattr(args, 'frame_sampling_mode', 'linspace'),
            max_retry_rounds=getattr(args, 'max_retry_rounds', 3),
        )
        all_results[task_name] = result
        m = result["metrics"]
        ttft = result["timing"]["ttft_s"]
        print(
            f"[stream-engine] {task_name}: acc={m['overall_accuracy']:.2f}% "
            f"({m['total_correct']}/{m['total_answered']}), "
            f"ttft_mean={ttft['mean']}, ttft_p95={ttft['p95']}, "
            f"{result['num_docs']} docs in {result['wall_clock_s']:.1f}s "
            f"({result['samples_per_s']:.2f} samples/s)"
        )
        if result.get("timeout_count") or result.get("error_count"):
            retry_info = f", retry_recovered={result.get('retry_recovered', 0)}" if result.get('retry_recovered') else ""
            print(
                f"[stream-engine]   ⚠️  timeout={result.get('timeout_count', 0)}, "
                f"error={result.get('error_count', 0)}, "
                f"failure_rate={100.0 * result.get('failure_rate', 0):.2f}%"
                f"{retry_info}"
            )
        print(f"[stream-engine] detailed samples: {result['sample_log']}")

    # Print overall failure summary across all tasks
    all_timeout = sum(r.get("timeout_count", 0) for r in all_results.values())
    all_error = sum(r.get("error_count", 0) for r in all_results.values())
    all_docs = sum(r.get("num_docs", 0) for r in all_results.values())
    all_retry_recovered = sum(r.get("retry_recovered", 0) for r in all_results.values())
    all_retry_still_failed = sum(r.get("retry_still_failed", 0) for r in all_results.values())
    if all_timeout or all_error or all_retry_recovered:
        print(
            f"\n[stream-engine] ━━━ Failure Summary ━━━\n"
            f"  Total samples: {all_docs}\n"
            f"  Timeouts:      {all_timeout} ({100.0 * all_timeout / all_docs:.2f}%)\n"
            f"  Other errors:  {all_error} ({100.0 * all_error / all_docs:.2f}%)\n"
            f"  Total failed:  {all_timeout + all_error} ({100.0 * (all_timeout + all_error) / all_docs:.2f}%)\n"
            f"  Retry recovered: {all_retry_recovered}\n"
            f"  Retry still failed: {all_retry_still_failed}\n"
            f"  ⚠️  Failed samples (after retry) are scored as 0 and included in accuracy denominator.\n"
            f"  If failure rate > 1%, consider increasing --decode-timeout-s."
        )

    # One row per dataset sheet touched by this run (OVO-Bench's 3 sub-tasks
    # pooled into a single row), plus one `overall` row and the two wide
    # (acc_by_dataset / ttft_by_dataset) rows -- all sharing a single run_id.
    if not getattr(args, "no_write_summary", False):
        summary_rows = build_dataset_rows(
            engine="stream_think_vllm",
            queue_row=queue_row,
            task_results=all_results,
            run_output_dir=str(output_dir),
            chunking_mode=args.chunking_mode,
            sample_concurrency=args.sample_concurrency,
        )
        summary_rows.append(
            build_overall_row(
                engine="stream_think_vllm",
                queue_row=queue_row,
                task_results=all_results,
                run_output_dir=str(output_dir),
                chunking_mode=args.chunking_mode,
                sample_concurrency=args.sample_concurrency,
            )
        )
        summary_rows.extend(
            build_wide_rows(
                engine="stream_think_vllm",
                queue_row=queue_row,
                task_results=all_results,
                chunking_mode=args.chunking_mode,
                sample_concurrency=args.sample_concurrency,
            )
        )
        append_summary_rows(RESULTS_SUMMARY_CSV, RESULTS_SUMMARY_XLSX, summary_rows)

    with open(output_dir / "all_tasks_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU vLLM StreamThink evaluator")
    parser.add_argument("--model", default=None, help="Optional manual HF model dir OR verl actor checkpoint dir. If omitted, scan eval_config.CHECKPOINT_QUEUE_CSV for evaluated=0 rows.")
    parser.add_argument("--checkpoint-csv", default=CHECKPOINT_QUEUE_CSV, help="Checkpoint queue CSV. Used when --model is omitted.")
    parser.add_argument("--tasks", required=True, help=f"Comma-separated task names. Available: {sorted(TASK_REGISTRY)} plus groups {sorted(_TASK_GROUPS)}")
    parser.add_argument(
        "--gpus", default=(",".join(str(g) for g in DEFAULT_GPUS) if DEFAULT_GPUS else None),
        required=DEFAULT_GPUS is None,
        help="Explicit comma-separated GPU ids, e.g. 0,1,2,3,4. Defaults to eval_config.DEFAULT_GPUS if omitted.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. If omitted, auto-generated as: eval_outputs/{tasks}-{ckpt_name}-{limit|all}-{timestamp}")
    parser.add_argument("--limit", type=int, default=-1, help="Debug cap per task before sharding")
    parser.add_argument("--max-num-frames", type=int, default=DEFAULT_SAMPLING.max_num_frames if DEFAULT_SAMPLING else 32, help="Global max sampled frames per video (shared across all chunks, not per-chunk)")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_SAMPLING.max_pixels if DEFAULT_SAMPLING else 0, help="Per-frame pixel cap passed to Qwen-VL mm_processor_kwargs; use 0 to disable")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLING.sample_fps if DEFAULT_SAMPLING else 1.0, help="Candidate sampling fps used to pick the whole video's frame set before chunking")
    parser.add_argument("--chunking-mode", default=getattr(DEFAULT_STREAM, "chunking_mode", "token_budget") if DEFAULT_STREAM else "token_budget", choices=["token_budget", "vst_original"], help="token_budget: split by estimated visual-token budget. vst_original: VST duration-bucket stream_think_times + even split")
    parser.add_argument("--stream-token-budget", type=int, default=DEFAULT_STREAM.stream_token_budget if DEFAULT_STREAM else 5000, help="[token_budget mode] Accumulate estimated visual tokens up to this budget before flushing a chunk into an intermediate-thinking request")
    parser.add_argument("--tokens-per-frame", type=int, default=DEFAULT_STREAM.tokens_per_frame if DEFAULT_STREAM else 256, help="[token_budget mode] Estimated visual tokens per sampled frame; combined with --stream-token-budget to size each chunk")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_STREAM.max_chunks if DEFAULT_STREAM else 8, help="[token_budget mode] Safety ceiling on chunk count; excess token-budget boundaries are folded into the last chunk")
    parser.add_argument("--max-memory-chars", type=int, default=DEFAULT_STREAM.max_memory_chars if DEFAULT_STREAM else 12000, help="[token_budget mode] Memory truncation by character count")
    parser.add_argument("--stream-think-times", default=DEFAULT_VST_ORIGINAL.stream_think_times if DEFAULT_VST_ORIGINAL else "2-3-5-5", help="[vst_original mode] VST duration-bucket segment counts, e.g. 2-3-5-5 or a single int")
    parser.add_argument("--max-stream-vid-tokens", type=int, default=DEFAULT_VST_ORIGINAL.max_stream_vid_tokens if DEFAULT_VST_ORIGINAL else 8192, help="[vst_original mode] VST per-round visual token budget, recorded for parity/logging")
    parser.add_argument("--max-keep-memory", type=int, default=DEFAULT_VST_ORIGINAL.max_keep_memory if DEFAULT_VST_ORIGINAL else 0, help="[vst_original mode] Memory truncation by round count; <=0 disables truncation")
    parser.add_argument("--think-max-tokens", type=int, default=DEFAULT_STREAM.think_max_tokens if DEFAULT_STREAM else 512, help="max_tokens for intermediate memory generation")
    parser.add_argument("--use-predecoded-video", type=int, default=1 if (DEFAULT_STREAM is None or getattr(DEFAULT_STREAM, "use_predecoded_video", True)) else 0, help="1: decode each chunk's frames once via decord and reuse across think/final requests (avoids vLLM re-decoding the whole mp4 per chunk). 0: fall back to file:// video_url path.")
    parser.add_argument("--decode-num-threads", type=int, default=getattr(DEFAULT_STREAM, "decode_num_threads", 4) if DEFAULT_STREAM else 4, help="decord decoder thread count")
    parser.add_argument("--sample-concurrency", type=int, default=1, help="Number of samples kept in flight inside each GPU worker; 1 preserves the serial baseline")
    parser.add_argument("--decode-timeout-s", type=float, default=0.0, help="Optional per-sample probe/decode timeout in seconds; <=0 disables timeout")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_ENGINE.max_model_len if DEFAULT_ENGINE else 20480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_ENGINE.gpu_memory_utilization if DEFAULT_ENGINE else 0.75)
    # Precision-alignment options
    parser.add_argument("--use-torchvision-resize", type=int,
                        default=1 if (DEFAULT_STREAM is not None and getattr(DEFAULT_STREAM, "use_torchvision_resize", False)) else 0,
                        help="1: torchvision resize with antialias (matches official). 0: PIL resize (faster)")
    parser.add_argument("--frame-sampling-mode",
                        default=getattr(DEFAULT_STREAM, "frame_sampling_mode", "linspace") if DEFAULT_STREAM else "linspace",
                        choices=["linspace", "strict_fps"],
                        help="strict_fps: official timestamp-based. linspace: uniform frame-index")
    parser.add_argument("--no-write-summary", action="store_true", default=False,
                        help="Skip writing to eval_summary CSV/XLSX (still writes all_tasks_summary.json). Used by eval_entry.py hybrid mode.")
    parser.add_argument("--max-retry-rounds", type=int, default=3,
                        help="Max retry rounds for failed (decode error/timeout) samples per worker. "
                             "Retries run serially to eliminate IO contention. 0 disables retry.")
    args = parser.parse_args()

    # Convert int flags
    args.use_torchvision_resize = bool(args.use_torchvision_resize)

    gpus = [int(g) for g in args.gpus.split(",")]
    tasks = _expand_tasks([t.strip() for t in args.tasks.split(",")])

    # Auto-generate output_dir if not explicitly specified:
    # Format: eval_outputs/{tasks}-{ckpt_name}-{limit|all}-{timestamp}
    if args.output_dir is None:
        from datetime import datetime
        _task_str = args.tasks.replace(",", "_")
        _ckpt_name = Path(args.model).name if args.model else "queue"
        _limit_str = "all" if args.limit <= 0 else str(args.limit)
        _timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.output_dir = str(
            Path(__file__).resolve().parents[1] / "eval_outputs" / f"{_task_str}-{_ckpt_name}-{_limit_str}-{_timestamp}"
        )

    if args.model:
        queue_rows = [_manual_queue_row(args.model)]
    else:
        if not args.checkpoint_csv:
            raise ValueError("--model omitted but no checkpoint queue CSV is configured")
        queue_rows = load_pending(args.checkpoint_csv)
        if not queue_rows:
            print(f"[stream-engine] No pending checkpoints found in {args.checkpoint_csv} (evaluated=0).")
            return
        print(f"[stream-engine] Found {len(queue_rows)} pending checkpoint(s) in {args.checkpoint_csv}.")

    completed_rows = []
    for idx, queue_row in enumerate(queue_rows, start=1):
        print("\n" + "=" * 80)
        print(f"[stream-engine] Checkpoint {idx}/{len(queue_rows)}: {queue_row.exp_name}")
        print("=" * 80)
        _run_checkpoint(args, queue_row, tasks, gpus)
        if not args.model and queue_row.row_index >= 0:
            mark_evaluated(args.checkpoint_csv, queue_row.row_index)
            completed_rows.append(queue_row.exp_name)
            print(f"[stream-engine] Marked queue row evaluated=1: {queue_row.exp_name}")

    print("\n" + "=" * 60)
    print("STREAM EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Processed checkpoints: {[r.exp_name for r in queue_rows]}")
    if completed_rows:
        print(f"Marked evaluated=1: {completed_rows}")
    print(f"Aggregate CSV: {RESULTS_SUMMARY_CSV}")
    print(f"Aggregate XLSX: {RESULTS_SUMMARY_XLSX}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Belt-and-suspenders: if something above raised or exited without
        # going through _run_task_on_gpus's own cleanup (e.g. an exception
        # between checkpoints, before/after a task's worker loop), make sure
        # no worker process group is left running with GPU memory pinned.
        _cleanup_worker_processes(list(_ACTIVE_WORKER_PROCS))
        raise
