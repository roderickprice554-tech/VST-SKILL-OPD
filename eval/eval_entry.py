#!/usr/bin/env python3
"""
eval_entry.py — 统一 Eval 入口
================================
支持 --eval-method vllm / official / hybrid 三种评估方法，保持统一的命令行风格。

用法:
  # vLLM 方法（默认，4.7x加速，strict_fps + torchvision_resize 对齐精度）
  python eval_entry.py --eval-method vllm --model /path/to/model --tasks all --gpus 0,1,2,3,4,5,6,7

  # Official 方法（lmms-eval HF generate，StreamingBench精度更高）
  python eval_entry.py --eval-method official --model /path/to/model --tasks all --gpus 0,1,2,3,4,5,6,7

  # Hybrid: StreamingBench用official, 其余用vLLM（默认strict_fps + torchvision_resize）
  python eval_entry.py --eval-method hybrid --model /path/to/model --tasks all --gpus 0,1,2,3,4,5,6,7

  # 省略 --model 时自动从 checkpoint_queue.csv 读取 evaluated=0 的行逐个评估
  python eval_entry.py --eval-method hybrid --tasks all --gpus 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent  # RLSD/

# ---------------------------------------------------------------------------
# Signal handling: ensure Ctrl+C kills all child processes (vLLM workers, etc.)
# ---------------------------------------------------------------------------
_ACTIVE_CHILD: subprocess.Popen | None = None


def _sigint_handler(signum, _frame):
    """Forward SIGTERM to the active child process group, then exit."""
    global _ACTIVE_CHILD
    if _ACTIVE_CHILD and _ACTIVE_CHILD.poll() is None:
        try:
            pgid = os.getpgid(_ACTIVE_CHILD.pid)
            print(f"\n[eval_entry] Ctrl+C received, terminating child process group (pgid={pgid})...", flush=True)
            os.killpg(pgid, signal.SIGTERM)
            # Give it a few seconds to clean up
            try:
                _ACTIVE_CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[eval_entry] Child did not exit in 10s, sending SIGKILL...", flush=True)
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    else:
        print("\n[eval_entry] Ctrl+C received, exiting.", flush=True)
    sys.exit(130)


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


def _run_subprocess(cmd: list[str], **kwargs) -> int:
    """Run a subprocess in its own process group so we can kill the whole tree.

    This replaces subprocess.call() throughout eval_entry.py. The child is
    started with start_new_session=True so it gets its own process group.
    On Ctrl+C, _sigint_handler sends SIGTERM to that group, killing the child
    AND all its descendants (vLLM EngineCore, worker_stream processes, etc.).
    """
    global _ACTIVE_CHILD
    proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    _ACTIVE_CHILD = proc
    try:
        ret = proc.wait()
    finally:
        _ACTIVE_CHILD = None
    return ret


def _load_env_local():
    """Source env_local.sh if present, loading machine-specific paths."""
    env_local = _REPO_ROOT / "env_local.sh"
    if not env_local.exists():
        return
    # Parse simple export KEY="VALUE" lines
    import shlex
    with open(env_local) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and not key.startswith("#"):
                    os.environ.setdefault(key, val)


def _get_eval_paths():
    """Resolve eval-related paths from environment."""
    eval_dir = _THIS_DIR
    eval_data_root = os.environ.get("RLSD_EVAL_DATA_ROOT", "/path/to/your/datasets/EVAL")
    eval_anno_root = os.environ.get("RLSD_EVAL_ANNO_ROOT", str(eval_dir / "eval_data" / "anno" / "eval"))

    # Resolve Python/accelerate from RLSD_VENV_ACTIVATE or sys.executable fallback
    venv_activate = os.environ.get("RLSD_VENV_ACTIVATE", "")
    if venv_activate and Path(venv_activate).exists():
        venv_bin = str(Path(venv_activate).parent)
    else:
        # Use the same Python that is running this script
        venv_bin = str(Path(sys.executable).parent)

    python_bin = os.path.join(venv_bin, "python3")
    accelerate_bin = os.path.join(venv_bin, "accelerate")
    return {
        "eval_dir": str(eval_dir),
        "eval_data_root": eval_data_root,
        "eval_anno_root": eval_anno_root,
        "python_bin": python_bin,
        "accelerate_bin": accelerate_bin,
        "venv_bin": venv_bin,
    }


# ============================================================================
# Task mapping for official method
# ============================================================================
_OFFICIAL_TASK_GROUPS = {
    "all": ["streamingbench", "ovobench", "videoholmes", "longvideobench", "videomme"],
}


def _expand_official_tasks(tasks_str: str) -> list[str]:
    tasks = [t.strip() for t in tasks_str.split(",")]
    expanded = []
    for t in tasks:
        expanded.extend(_OFFICIAL_TASK_GROUPS.get(t, [t]))
    return expanded


# ============================================================================
# Helper: collect StreamingBench official result from output jsonl
# ============================================================================
def _collect_official_streamingbench_result(result_dir: str) -> tuple:
    """Parse merged output jsonl from qwen2_5_vl_sf.py and return (acc%, num_docs).

    Uses the LATEST merged file (without _rankN suffix) if available, otherwise
    collects from rank files. Avoids double-counting by only reading one file.
    """
    import re
    output_dir = os.path.join(result_dir, "output")
    if not os.path.isdir(output_dir):
        return None, 0

    # Prefer merged file (no _rankN in name); use only the latest one
    all_jsonl = [f for f in os.listdir(output_dir) if f.endswith(".jsonl")]
    merged_files = sorted([f for f in all_jsonl if "_rank" not in f])
    if merged_files:
        # Take only the latest merged file (lexicographic sort on timestamp in filename)
        target_files = [merged_files[-1]]
    else:
        # No merged file — collect from rank files; use only the latest set
        rank_files = sorted([f for f in all_jsonl if "_rank" in f])
        if not rank_files:
            return None, 0
        # Group by prefix (timestamp) and take the latest group
        latest_prefix = rank_files[-1].rsplit("_rank", 1)[0]
        target_files = [f for f in rank_files if f.startswith(latest_prefix)]

    correct = 0
    total = 0
    for fname in target_files:
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                response = item.get("response", "")
                answer = item.get("answer", "")
                # Extract letter from response
                resp_clean = response.strip()
                for prefix in ["The best answer is", "The correct answer is", "The answer is",
                               "The answer", "The best option is", "The correct option is"]:
                    resp_clean = resp_clean.replace(prefix, "")
                match = re.search(r"[ABCD]", resp_clean)
                if match and match[0] == answer:
                    correct += 1

    if total == 0:
        return None, 0
    return 100.0 * correct / total, total


def _write_hybrid_streamingbench_to_summary(args, sb_acc: float, sb_num_docs: int):
    """DEPRECATED: Use _write_hybrid_combined_summary instead."""
    pass


def _write_hybrid_combined_summary(args, sb_acc: float | None, sb_num_docs: int,
                                    vllm_output_dir: str | None, queue_row=None):
    """Write combined hybrid eval results (StreamingBench + vLLM) as a single run_id.

    Reads vLLM results from all_tasks_summary.json (if available), merges with
    StreamingBench official result, and writes everything in one append_summary_rows
    call so all datasets share a single run_id.
    """
    sys.path.insert(0, str(_THIS_DIR))
    from vllm_eval.result_store import (
        append_summary_rows, build_dataset_rows, build_overall_row, build_wide_rows
    )

    summary_csv = str(_THIS_DIR / "results" / "eval_summary.csv")
    summary_xlsx = str(_THIS_DIR / "results" / "eval_summary.xlsx")

    # Determine exp_name / key_config
    if queue_row is not None:
        exp_name = queue_row.exp_name
        key_config_base = queue_row.key_config
    else:
        exp_name = Path(args.model).name
        key_config_base = ""

    engine = "hybrid"
    key_config = f"{key_config_base}|eval_method=hybrid|chunking_mode={args.chunking_mode}|sample_concurrency={args.sample_concurrency}" if key_config_base else f"eval_method=hybrid|chunking_mode={args.chunking_mode}|sample_concurrency={args.sample_concurrency}"

    # Build a combined queue_row-like object for result_store functions
    combined_queue = SimpleNamespace(
        exp_name=exp_name,
        key_config=key_config,
        ckpt_path=args.model,
        created_at=getattr(queue_row, "created_at", "") if queue_row else "",
    )

    # Collect vLLM task results from all_tasks_summary.json
    vllm_results = {}
    if vllm_output_dir:
        summary_json = os.path.join(vllm_output_dir, "all_tasks_summary.json")
        if os.path.isfile(summary_json):
            with open(summary_json, "r", encoding="utf-8") as f:
                vllm_results = json.load(f)

    # Build a synthetic StreamingBench result dict (matching vLLM result structure)
    all_results = dict(vllm_results)
    if sb_acc is not None:
        sb_correct = int(round(sb_acc * sb_num_docs / 100.0))
        all_results["streamingbench"] = {
            "task": "streamingbench",
            "num_docs": sb_num_docs,
            "num_gpus": len(args.gpus.split(",")),
            "wall_clock_s": 0,
            "samples_per_s": 0,
            "metrics": {
                "overall_accuracy": sb_acc,
                "total_answered": sb_num_docs,
                "total_correct": sb_correct,
            },
            "timing": {
                "ttft_s": {"mean": None, "count": 0},
                "sample_total_s": {"mean": None, "count": 0},
            },
            "sample_log": None,
        }

    if not all_results:
        print("  [Summary] No results to write.")
        return

    # Build rows — per-dataset rows
    run_output_dir = vllm_output_dir or ""
    summary_rows = build_dataset_rows(
        engine=engine,
        queue_row=combined_queue,
        task_results=all_results,
        run_output_dir=run_output_dir,
        chunking_mode=args.chunking_mode,
        sample_concurrency=args.sample_concurrency,
    )
    # Overall row
    summary_rows.append(
        build_overall_row(
            engine=engine,
            queue_row=combined_queue,
            task_results=all_results,
            run_output_dir=run_output_dir,
            chunking_mode=args.chunking_mode,
            sample_concurrency=args.sample_concurrency,
        )
    )
    # Wide rows (acc_by_dataset, ttft_by_dataset)
    summary_rows.extend(
        build_wide_rows(
            engine=engine,
            queue_row=combined_queue,
            task_results=all_results,
            chunking_mode=args.chunking_mode,
            sample_concurrency=args.sample_concurrency,
        )
    )

    append_summary_rows(summary_csv, summary_xlsx, summary_rows)
    # Print summary
    for row in summary_rows:
        if row.get("sheet") == "overall":
            print(f"  [Summary] Hybrid overall acc={row.get('acc', 'N/A'):.2f}% "
                  f"({row.get('total_correct')}/{row.get('total_answered')}) "
                  f"written to {summary_xlsx}")
            break


# ============================================================================
# vLLM branch
# ============================================================================
def _run_vllm(args):
    """Invoke vllm_eval_engine_stream.py via subprocess.

    By default enables strict_fps + torchvision_resize for precision alignment
    with official eval (these are the defaults in eval_config.py's DEFAULT_STREAM).
    """
    engine_script = _THIS_DIR / "vllm_eval" / "vllm_eval_engine_stream.py"
    cmd = [
        sys.executable, str(engine_script),
        "--model", args.model,
        "--tasks", args.tasks,
        "--gpus", args.gpus,
        "--chunking-mode", args.chunking_mode,
        "--stream-think-times", args.stream_think_times,
        "--max-num-frames", str(args.max_num_frames),
        "--max-stream-vid-tokens", str(args.max_stream_vid_tokens),
        "--sample-concurrency", str(args.sample_concurrency),
        "--decode-timeout-s", str(args.decode_timeout_s),
        "--max-retry-rounds", str(args.max_retry_rounds),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--think-max-tokens", str(args.think_max_tokens),
        # Precision alignment: strict_fps + torchvision_resize (matches official eval)
        "--use-torchvision-resize", "1",
        "--frame-sampling-mode", "strict_fps",
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]

    print(f"[eval_entry] Running vLLM eval (strict_fps + torchvision_resize):")
    print(f"  {' '.join(cmd)}")
    print()
    return _run_subprocess(cmd, cwd=str(_THIS_DIR))


# ============================================================================
# Official branch
# ============================================================================
def _run_official(args):
    """Run official lmms-eval based evaluation."""
    paths = _get_eval_paths()
    eval_dir = paths["eval_dir"]
    python_bin = paths["python_bin"]
    accelerate_bin = paths["accelerate_bin"]
    eval_data_root = paths["eval_data_root"]
    eval_anno_root = paths["eval_anno_root"]

    gpus_list = args.gpus.split(",")
    num_gpus = len(gpus_list)
    model_name = Path(args.model).name

    result_dir = os.path.join(eval_dir, "result", f"official_{model_name}")
    if args.output_dir:
        result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)

    # Set CUDA_VISIBLE_DEVICES and PYTHONPATH for lmms-eval
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    lmms_eval_path = os.path.join(eval_dir, "lmms-eval")
    env["PYTHONPATH"] = f"{lmms_eval_path}:{env.get('PYTHONPATH', '')}"
    env["HF_HOME"] = os.path.join(eval_dir, "eval_data")
    env["HF_DATASETS_CACHE"] = os.path.join(eval_dir, "eval_data")
    env["RLSD_EVAL_DATA_ROOT"] = eval_data_root
    env["RLSD_EVAL_ANNO_ROOT"] = eval_anno_root

    tasks = _expand_official_tasks(args.tasks)
    limit_arg = ["--limit", str(args.limit)] if args.limit > 0 else []

    print("=" * 64)
    print(f"  Eval Method: official")
    print(f"  Model:       {args.model}")
    print(f"  GPUs:        {args.gpus} ({num_gpus} GPUs)")
    print(f"  Tasks:       {tasks}")
    print(f"  Limit:       {args.limit if args.limit > 0 else 'full'}")
    print(f"  Output:      {result_dir}")
    print("=" * 64)

    ret = 0

    # Phase 1: StreamingBench (standalone ffmpeg-clip script)
    if "streamingbench" in tasks:
        print("\n[Official][1/4] StreamingBench")
        sb_result_dir = os.path.join(result_dir, "streamingbench")
        os.makedirs(sb_result_dir, exist_ok=True)
        task_json = os.path.join(eval_anno_root, "StreamingBench", "json", "real_time_visual_understanding.json")
        video_dir = os.path.join(eval_data_root, "StreamingBench")
        sf_script = os.path.join(eval_dir, "lmms-eval", "lmms_eval", "models", "simple", "qwen2_5_vl_sf.py")
        cmd = [
            python_bin, sf_script,
            "--run_name", "official_eval",
            "--ckpt_path", args.model,
            "--task_json", task_json,
            "--video_dir", video_dir,
            "--result_dir", sb_result_dir,
            "--world_size", str(num_gpus),
        ] + limit_arg
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, env=env, cwd=eval_dir)
        if rc != 0:
            print(f"  [WARN] StreamingBench exited with code {rc}")
            ret = rc

    # Phase 2: OVOBench + VideoHolmes (stream_think_times=2-3-5-5)
    ovo_holmes_tasks = [t for t in tasks if t in ("ovobench", "videoholmes")]
    if ovo_holmes_tasks:
        print("\n[Official][2/4] OVOBench + VideoHolmes")
        ovo_result_dir = os.path.join(result_dir, "ovo_holmes")
        os.makedirs(ovo_result_dir, exist_ok=True)
        lmms_tasks = "ovobench,video_holmes"
        model_args = (
            f"pretrained={args.model},"
            f"attn_implementation=sdpa,"
            f"stream_think_times={args.stream_think_times},"
            f"max_stream_vid_tokens={args.max_stream_vid_tokens},"
            f"max_num_frames={args.max_num_frames}"
        )
        cmd = [
            accelerate_bin, "launch",
            f"--num_processes={num_gpus}",
            "--main_process_port=12346",
            "-m", "lmms_eval",
            "--model", "qwen3_vl_stream_think",
            f"--model_args={model_args}",
            "--tasks", lmms_tasks,
            "--batch_size", "1",
            "--log_samples",
            "--log_samples_suffix", "official_eval",
            "--output_path", ovo_result_dir,
        ] + limit_arg
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, env=env, cwd=eval_dir)
        if rc != 0:
            print(f"  [WARN] OVOBench+VideoHolmes exited with code {rc}")
            ret = rc

    # Phase 3: LongVideoBench (stream_think_times=1)
    if "longvideobench" in tasks:
        print("\n[Official][3/4] LongVideoBench")
        lvb_result_dir = os.path.join(result_dir, "longvideobench")
        os.makedirs(lvb_result_dir, exist_ok=True)
        model_args = (
            f"pretrained={args.model},"
            f"attn_implementation=sdpa,"
            f"stream_think_times=1,"
            f"max_stream_vid_tokens={args.max_stream_vid_tokens},"
            f"max_num_frames={args.max_num_frames}"
        )
        cmd = [
            accelerate_bin, "launch",
            f"--num_processes={num_gpus}",
            "--main_process_port=12346",
            "-m", "lmms_eval",
            "--model", "qwen3_vl_stream_think",
            f"--model_args={model_args}",
            "--tasks", "longvideobench_val_v",
            "--batch_size", "1",
            "--log_samples",
            "--log_samples_suffix", "official_eval",
            "--output_path", lvb_result_dir,
        ] + limit_arg
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, env=env, cwd=eval_dir)
        if rc != 0:
            print(f"  [WARN] LongVideoBench exited with code {rc}")
            ret = rc

    # Phase 4: VideoMME (stream_think_times=1-1-3-4)
    if "videomme" in tasks:
        print("\n[Official][4/4] VideoMME")
        vmme_result_dir = os.path.join(result_dir, "videomme")
        os.makedirs(vmme_result_dir, exist_ok=True)
        model_args = (
            f"pretrained={args.model},"
            f"attn_implementation=sdpa,"
            f"stream_think_times=1-1-3-4,"
            f"max_stream_vid_tokens={args.max_stream_vid_tokens},"
            f"max_num_frames={args.max_num_frames}"
        )
        cmd = [
            accelerate_bin, "launch",
            f"--num_processes={num_gpus}",
            "--main_process_port=12346",
            "-m", "lmms_eval",
            "--model", "qwen3_vl_stream_think",
            f"--model_args={model_args}",
            "--tasks", "videomme",
            "--batch_size", "1",
            "--log_samples",
            "--log_samples_suffix", "official_eval",
            "--output_path", vmme_result_dir,
        ] + limit_arg
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, env=env, cwd=eval_dir)
        if rc != 0:
            print(f"  [WARN] VideoMME exited with code {rc}")
            ret = rc

    print()
    print("=" * 64)
    print(f"  Official Eval DONE — results in: {result_dir}")
    print("=" * 64)
    return ret


# ============================================================================
# Hybrid branch: StreamingBench uses official, others use vLLM
# ============================================================================
def _run_hybrid(args):
    """
    Hybrid eval: StreamingBench via official (ffmpeg-clip, higher accuracy),
    all other tasks via vLLM (4.7x faster).
    """
    # Expand tasks to determine which need which method
    all_tasks = _expand_official_tasks(args.tasks)
    has_streamingbench = "streamingbench" in all_tasks
    vllm_tasks = [t for t in all_tasks if t != "streamingbench"]

    # Map back to vLLM task names (they use the same names)
    vllm_tasks_str = ",".join(vllm_tasks) if vllm_tasks else ""

    print("=" * 64)
    print(f"  Eval Method: hybrid")
    print(f"  Model:       {args.model}")
    print(f"  GPUs:        {args.gpus}")
    print(f"  Tasks:       {all_tasks}")
    print(f"  StreamingBench: {'official (ffmpeg-clip)' if has_streamingbench else 'skipped'}")
    print(f"  Other tasks:    {'vLLM' if vllm_tasks else 'none'}")
    print(f"  Limit:       {args.limit if args.limit > 0 else 'full'}")
    print("=" * 64)

    ret = 0

    # Phase 1: StreamingBench via official method
    sb_acc = None
    sb_num_docs = 0
    if has_streamingbench:
        print("\n[Hybrid][1/2] StreamingBench → official method")
        paths = _get_eval_paths()
        eval_dir = paths["eval_dir"]
        python_bin = paths["python_bin"]
        eval_data_root = paths["eval_data_root"]
        eval_anno_root = paths["eval_anno_root"]
        model_name = Path(args.model).name

        result_dir = args.output_dir or os.path.join(eval_dir, "result", f"hybrid_{model_name}")
        sb_result_dir = os.path.join(result_dir, "streamingbench_official")
        os.makedirs(sb_result_dir, exist_ok=True)

        task_json = os.path.join(eval_anno_root, "StreamingBench", "json", "real_time_visual_understanding.json")
        video_dir = os.path.join(eval_data_root, "StreamingBench")
        sf_script = os.path.join(eval_dir, "lmms-eval", "lmms_eval", "models", "simple", "qwen2_5_vl_sf.py")

        gpus_list = args.gpus.split(",")
        num_gpus = len(gpus_list)
        limit_arg = ["--limit", str(args.limit)] if args.limit > 0 else []

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        lmms_eval_path = os.path.join(eval_dir, "lmms-eval")
        env["PYTHONPATH"] = f"{lmms_eval_path}:{env.get('PYTHONPATH', '')}"
        env["HF_HOME"] = os.path.join(eval_dir, "eval_data")
        env["HF_DATASETS_CACHE"] = os.path.join(eval_dir, "eval_data")
        env["RLSD_EVAL_DATA_ROOT"] = eval_data_root
        env["RLSD_EVAL_ANNO_ROOT"] = eval_anno_root

        cmd = [
            python_bin, sf_script,
            "--run_name", "hybrid_eval",
            "--ckpt_path", args.model,
            "--task_json", task_json,
            "--video_dir", video_dir,
            "--result_dir", sb_result_dir,
            "--world_size", str(num_gpus),
        ] + limit_arg
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, env=env, cwd=eval_dir)
        if rc != 0:
            print(f"  [WARN] StreamingBench (official) exited with code {rc}")
            ret = rc

        # Collect StreamingBench accuracy from output jsonl
        sb_acc, sb_num_docs = _collect_official_streamingbench_result(sb_result_dir)

    # Phase 2: Other tasks via vLLM (strict_fps + torchvision_resize for precision alignment)
    vllm_output_dir = None
    if vllm_tasks:
        print(f"\n[Hybrid][2/2] {vllm_tasks} → vLLM method (strict_fps + torchvision_resize)")
        engine_script = _THIS_DIR / "vllm_eval" / "vllm_eval_engine_stream.py"
        cmd = [
            sys.executable, str(engine_script),
            "--model", args.model,
            "--tasks", vllm_tasks_str,
            "--gpus", args.gpus,
            "--chunking-mode", args.chunking_mode,
            "--stream-think-times", args.stream_think_times,
            "--max-num-frames", str(args.max_num_frames),
            "--max-stream-vid-tokens", str(args.max_stream_vid_tokens),
            "--sample-concurrency", str(args.sample_concurrency),
            "--decode-timeout-s", str(args.decode_timeout_s),
            "--max-retry-rounds", str(args.max_retry_rounds),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--think-max-tokens", str(args.think_max_tokens),
            # Precision alignment: strict_fps + torchvision_resize (matches official eval)
            "--use-torchvision-resize", "1",
            "--frame-sampling-mode", "strict_fps",
            # Don't let vLLM engine write its own summary — we'll merge & write below
            "--no-write-summary",
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        print(f"  {' '.join(cmd)}")
        rc = _run_subprocess(cmd, cwd=str(_THIS_DIR))
        if rc != 0:
            print(f"  [WARN] vLLM eval exited with code {rc}")
            ret = rc

        # Find the vLLM output dir (contains all_tasks_summary.json)
        # The engine auto-generates: eval_outputs/{tasks}-{model_name}-{limit|all}-{timestamp}/{model_name}/{chunking}/{concurrency}/
        # We need the directory with all_tasks_summary.json
        if args.output_dir:
            _search_base = args.output_dir
        else:
            _search_base = str(_THIS_DIR / "eval_outputs")
        # Find the most recent all_tasks_summary.json
        import glob
        candidates = sorted(
            glob.glob(os.path.join(_search_base, "**", "all_tasks_summary.json"), recursive=True),
            key=os.path.getmtime,
            reverse=True,
        )
        if candidates:
            vllm_output_dir = os.path.dirname(candidates[0])

    print()
    print("=" * 64)
    print(f"  Hybrid Eval DONE")
    print("=" * 64)

    # Write combined summary (StreamingBench + vLLM) as a single run_id
    _write_hybrid_combined_summary(args, sb_acc, sb_num_docs, vllm_output_dir,
                                    queue_row=getattr(args, '_queue_row', None))

    return ret


def _load_queue_rows():
    """Load pending checkpoint rows from checkpoint_queue.csv."""
    sys.path.insert(0, str(_THIS_DIR))
    from vllm_eval.checkpoint_queue import load_pending
    try:
        from eval_config import CHECKPOINT_QUEUE_CSV
    except ImportError:
        CHECKPOINT_QUEUE_CSV = str(_THIS_DIR / "checkpoint_queue.csv")
    return load_pending(CHECKPOINT_QUEUE_CSV), CHECKPOINT_QUEUE_CSV


def _mark_queue_evaluated(csv_path: str, row_index: int):
    """Mark a queue row as evaluated=1."""
    sys.path.insert(0, str(_THIS_DIR))
    from vllm_eval.checkpoint_queue import mark_evaluated
    mark_evaluated(csv_path, row_index)


def main():
    parser = argparse.ArgumentParser(
        description="统一 Eval 入口 — 支持 vllm / official / hybrid 三种评估方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # vLLM 快速评估（默认, strict_fps + torchvision_resize 精度对齐）
  python eval_entry.py --model /path/to/model --tasks all --gpus 0,1,2,3,4,5,6,7

  # Official lmms-eval 评估
  python eval_entry.py --eval-method official --model /path/to/model --tasks streamingbench --gpus 0,1,2,3

  # Hybrid: StreamingBench用official高精度, 其余用vLLM快速推理
  python eval_entry.py --eval-method hybrid --model /path/to/model --tasks all --gpus 0,1,2,3,4,5,6,7

  # 省略 --model: 自动从 checkpoint_queue.csv 读取 evaluated=0 的 checkpoint 逐个评估
  python eval_entry.py --eval-method hybrid --tasks all --gpus 0,1,2,3,4,5,6,7
""",
    )

    # ---- 共享参数 ----
    parser.add_argument("--eval-method", default="vllm", choices=["vllm", "official", "hybrid"],
                        help="评估方法: vllm (默认, 快) 或 official (精度高) 或 hybrid (StreamingBench用official, 其余用vllm)")
    parser.add_argument("--model", default=None,
                        help="模型路径。省略时自动从 checkpoint_queue.csv 读取 evaluated=0 的行逐个评估")
    parser.add_argument("--tasks", default="all",
                        help="任务列表，逗号分隔。支持: all, streamingbench, ovobench, videoholmes, longvideobench, videomme")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU列表，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="每个数据集样本数上限 (0 = 全量)")
    parser.add_argument("--output-dir", default=None, help="输出目录 (可选，自动生成)")

    # ---- vLLM 专有参数 ----
    vllm_group = parser.add_argument_group("vLLM 专有参数")
    vllm_group.add_argument("--chunking-mode", default="vst_original",
                            choices=["token_budget", "vst_original"])
    vllm_group.add_argument("--sample-concurrency", type=int, default=8)
    vllm_group.add_argument("--decode-timeout-s", type=float, default=180)
    vllm_group.add_argument("--max-retry-rounds", type=int, default=3,
                            help="解码失败样本最大重试轮次 (串行重试消除IO竞争, 0=禁用)")
    vllm_group.add_argument("--max-model-len", type=int, default=128000)
    vllm_group.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    vllm_group.add_argument("--think-max-tokens", type=int, default=5000)

    # ---- 共享的模型相关参数（两种方法都用到） ----
    model_group = parser.add_argument_group("模型推理参数")
    model_group.add_argument("--stream-think-times", default="2-3-5-5",
                             help="流式思考次数 (vLLM + Official OVO/Holmes 共用)")
    model_group.add_argument("--max-num-frames", type=int, default=512,
                             help="最大采样帧数")
    model_group.add_argument("--max-stream-vid-tokens", type=int, default=8192,
                             help="每轮视觉token上限")

    args = parser.parse_args()

    # Load machine-specific env
    _load_env_local()

    # If --model is provided, run once with that model
    if args.model:
        args._queue_row = None  # No queue row in manual mode
        if args.eval_method == "vllm":
            sys.exit(_run_vllm(args))
        elif args.eval_method == "official":
            sys.exit(_run_official(args))
        else:
            sys.exit(_run_hybrid(args))
    else:
        # Queue mode: load pending checkpoints from checkpoint_queue.csv
        queue_rows, csv_path = _load_queue_rows()
        if not queue_rows:
            print("[eval_entry] No pending checkpoints (evaluated=0) in checkpoint_queue.csv.")
            sys.exit(0)
        print(f"[eval_entry] Found {len(queue_rows)} pending checkpoint(s) in {csv_path}")
        for idx, qrow in enumerate(queue_rows, start=1):
            print(f"\n{'=' * 72}")
            print(f"[eval_entry] Checkpoint {idx}/{len(queue_rows)}: {qrow.exp_name}")
            print(f"[eval_entry]   ckpt_path: {qrow.ckpt_path}")
            print(f"[eval_entry]   key_config: {qrow.key_config}")
            print(f"{'=' * 72}")
            # Resolve HF model dir (handles verl FSDP checkpoints)
            from vllm_eval.ckpt_utils import resolve_hf_model_dir
            model_path = resolve_hf_model_dir(qrow.ckpt_path)
            args.model = model_path
            args._queue_row = qrow  # Attach queue row for hybrid summary writing

            if args.eval_method == "vllm":
                ret = _run_vllm(args)
            elif args.eval_method == "official":
                ret = _run_official(args)
            else:
                ret = _run_hybrid(args)

            if ret == 0:
                _mark_queue_evaluated(csv_path, qrow.row_index)
                print(f"[eval_entry] ✓ Marked evaluated=1: {qrow.exp_name}")
            else:
                print(f"[eval_entry] ✗ Eval failed (exit={ret}) for {qrow.exp_name}, skipping mark.")
        # Reset model to None after queue processing
        args.model = None
        print(f"\n[eval_entry] Queue processing complete. {len(queue_rows)} checkpoint(s) evaluated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
