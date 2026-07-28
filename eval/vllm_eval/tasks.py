"""
Benchmark task definitions for the vLLM-based streaming-video eval engine.
============================================================================
Each task mirrors the doc_to_text / process_results / aggregate_results
logic from VST/eval/lmms-eval/lmms_eval/tasks/{streamingbench,ovobench,videoholmes}
so that scores stay comparable with the original lmms-eval pipeline.

A "task" here is one JSON annotation file + a video root + a prompt/metric
recipe. Register new tasks in TASK_REGISTRY at the bottom of this file.
"""
from __future__ import annotations

import json
import os
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from eval_config import DEFAULT_PROMPTS, TASK_PATHS
except ImportError:  # fallback for isolated imports/tests
    DEFAULT_PROMPTS = None
    TASK_PATHS = None, Optional


# ============================================================================
# Shared MCQ answer-extraction helpers (ported from lmms-eval tasks/*/utils.py)
# ============================================================================

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in _PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_COMMA_STRIP, in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _normalize_mcq_answer(answer: str) -> str:
    option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
    match = option_regex.match(answer.strip())
    if match:
        return match.group(1).upper()

    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = _process_punctuation(answer)
    answer = answer.strip("'\"()").strip().lower()

    letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return answer


def mcq_acc(gt_letter: str, pred: str) -> int:
    return 1 if _normalize_mcq_answer(pred) == _normalize_mcq_answer(gt_letter) else 0


def _process_ssr_crr(answer: str) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    return answer.strip("'\"()").strip().lower()


def _process_rec(answer: str) -> Optional[int]:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = answer.strip("'\"()").strip().lower()
    numbers = re.findall(r"\b\d+\b", answer)
    if len(numbers) == 1:
        return int(numbers[0])
    return None


def far_acc(gt: str, pred: str, subtask: str) -> int:
    """forward_active_responding scoring (OVOBench-specific, 3 sub-modes)."""
    if subtask in ("SSR", "CRR"):
        pred_n, gt_n = _process_ssr_crr(pred), _process_ssr_crr(gt)
        return 1 if gt_n in pred_n else 0
    pred_n, gt_n = _process_rec(pred), _process_rec(gt)
    return 1 if pred_n == gt_n else 0


def gt_letter_from_candidates(candidates: list[str], answer_text: str) -> Optional[str]:
    letters = string.ascii_uppercase
    for i, cand in enumerate(candidates):
        if cand == answer_text:
            return letters[i]
    return None


def format_mcq_prompt(question: str, candidates: list[str], post_prompt: str) -> str:
    letters = string.ascii_uppercase
    options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(candidates))
    return f"{question}\n{options}\n{post_prompt}"


# ============================================================================
# Generic aggregation (task/subtask/subsubtask breakdown), shared by all MCQ
# style benchmarks (StreamingBench, OVOBench, VideoHolmes all use this shape).
# ============================================================================

def aggregate_task_subtask(rows: list[dict]) -> dict:
    total_answered = 0
    total_correct = 0
    task_stats: dict[str, dict[str, int]] = {}
    subtask_stats: dict[str, dict[str, int]] = {}
    subsubtask_stats: dict[str, dict[str, int]] = {}

    for r in rows:
        if r.get("pred_answer", "") == "":
            continue
        total_answered += 1
        total_correct += r["score"]

        for key, stats_dict in (("task", task_stats), ("subtask", subtask_stats), ("subsubtask", subsubtask_stats)):
            val = r.get(key)
            if val is None:
                continue
            stats_dict.setdefault(val, {"answered": 0, "correct": 0})
            stats_dict[val]["answered"] += 1
            stats_dict[val]["correct"] += r["score"]

    def _finalize(stats_dict):
        out = {}
        for k, v in stats_dict.items():
            acc = 100 * v["correct"] / v["answered"] if v["answered"] else 0.0
            out[k] = {"answered": v["answered"], "correct": v["correct"], "accuracy": acc}
        return out

    overall_acc = 100 * total_correct / total_answered if total_answered else 0.0
    subtask_acc_values = [v["accuracy"] for v in _finalize(subtask_stats).values()]
    subtask_acc_avg = sum(subtask_acc_values) / len(subtask_acc_values) if subtask_acc_values else 0.0

    return {
        "overall_accuracy": overall_acc,
        "total_answered": total_answered,
        "total_correct": total_correct,
        "subtask_acc_avg": subtask_acc_avg,
        "by_task": _finalize(task_stats),
        "by_subtask": _finalize(subtask_stats),
        "by_subsubtask": _finalize(subsubtask_stats),
    }


# ============================================================================
# Task spec: declarative description of one benchmark JSON + video root
# ============================================================================

@dataclass
class TaskSpec:
    name: str
    anno_path: str
    video_root: str
    build_prompt: Callable[[dict], str]
    resolve_video: Callable[[dict, str], str]
    score_one: Callable[[dict, str], dict]  # (doc, pred_text) -> row dict (must include "score", "pred_answer")
    generation_kwargs: dict = field(default_factory=lambda: {"max_new_tokens": 16, "temperature": 0.0})
    aggregate: Callable[[list[dict]], dict] = aggregate_task_subtask
    sampling_override: dict = field(default_factory=dict)
    # sampling_override supported keys:
    #   "clip_to_annotation": bool  -> use doc["start"]/doc["end"] to trim video
    #   "skip_intermediate_thinking": bool  -> single-pass, no intermediate think
    #   "max_num_frames": int  -> override global max frame count
    #   "sample_fps": float  -> override global sample fps
    #   "max_pixels": int  -> override global pixel cap
    #   "system_prompt": str  -> override default system prompt
    #   "stream_think_times": str  -> override global stream_think_times (e.g. "1-1-3-4")

    def load_docs(self) -> list[dict]:
        with open(self.anno_path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# StreamingBench: real_time_visual_understanding
# ---------------------------------------------------------------------------

_STREAMINGBENCH_POST_PROMPT = (
    DEFAULT_PROMPTS.mcq_post_prompt if DEFAULT_PROMPTS is not None
    else "Answer with the option's letter from the given choices directly."
)

_STREAMINGBENCH_SF_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice "
    "question related to the video. Your task is to carefully analyze the video and "
    "provide the best answer to question, choosing from the four options provided. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\nThe best option is:"
)


def _streamingbench_prompt(doc: dict) -> str:
    """Official StreamingBench prompt matching qwen2_5_vl_sf.py format."""
    options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(doc["candidates"]))
    return _STREAMINGBENCH_SF_PROMPT_TEMPLATE.format(question=doc["question"], options=options)


def _streamingbench_resolve_video(doc: dict, video_root: str) -> str:
    # annotation "video" field looks like "videos/sample_348_real.mp4";
    # local layout is <video_root>/sample_348/video.mp4
    m = re.match(r"videos/sample_(\d+)_real\.mp4", doc["video"])
    if m:
        return str(Path(video_root) / f"sample_{m.group(1)}" / "video.mp4")
    return str(Path(video_root) / doc["video"])


def _streamingbench_score(doc: dict, pred: str) -> dict:
    gt_letter = gt_letter_from_candidates(doc["candidates"], doc["answer"])
    score = mcq_acc(gt_letter, pred) if gt_letter is not None else 0
    return {
        "pred_answer": pred,
        "gt_answer": gt_letter,
        "score": score,
        "task": doc.get("task"),
        "subtask": doc.get("subtask"),
        "subsubtask": doc.get("subsubtask"),
    }


def make_streamingbench_task(anno_path: str, video_root: str) -> TaskSpec:
    return TaskSpec(
        name="streamingbench",
        anno_path=anno_path,
        video_root=video_root,
        build_prompt=_streamingbench_prompt,
        resolve_video=_streamingbench_resolve_video,
        score_one=_streamingbench_score,
        generation_kwargs={"max_new_tokens": 128, "temperature": 0.0},
        sampling_override={
            "clip_to_annotation": True,
            "skip_intermediate_thinking": True,
            "max_num_frames": 56,
            "sample_fps": 1.0,
            "max_pixels": 802816,
            "system_prompt": "You are a helpful assistant.",
        },
    )


# ---------------------------------------------------------------------------
# OVOBench: real_time_visual_perception / backward_tracking (MCQ) and
#           forward_active_responding (SSR/CRR/REC, non-MCQ)
# ---------------------------------------------------------------------------

_OVOBENCH_POST_PROMPT = (
    DEFAULT_PROMPTS.mcq_post_prompt if DEFAULT_PROMPTS is not None
    else "Answer with the option's letter from the given choices directly."
)


def _ovobench_prompt(doc: dict) -> str:
    if doc["task"] != "forward_active_responding":
        return format_mcq_prompt(doc["question"], doc["candidates"], _OVOBENCH_POST_PROMPT)
    return doc["question"]


def _ovobench_resolve_video(doc: dict, video_root: str) -> str:
    return str(Path(video_root) / doc["video"])


def _ovobench_score(doc: dict, pred: str) -> dict:
    if doc["task"] != "forward_active_responding":
        gt_letter = gt_letter_from_candidates(doc["candidates"], doc["answer"])
        score = mcq_acc(gt_letter, pred) if gt_letter is not None else 0
        return {
            "pred_answer": pred,
            "gt_answer": gt_letter,
            "score": score,
            "task": doc.get("task"),
            "subtask": doc.get("subtask"),
        }
    score = far_acc(doc["answer"], pred, doc["subtask"])
    return {
        "pred_answer": pred,
        "gt_answer": doc["answer"],
        "score": score,
        "task": doc.get("task"),
        "subtask": doc.get("subtask"),
    }


def make_ovobench_task(anno_path: str, video_root: str) -> TaskSpec:
    return TaskSpec(
        name="ovobench",
        anno_path=anno_path,
        video_root=video_root,
        build_prompt=_ovobench_prompt,
        resolve_video=_ovobench_resolve_video,
        score_one=_ovobench_score,
        generation_kwargs={"max_new_tokens": 16, "temperature": 0.0},
    )


# ---------------------------------------------------------------------------
# VideoHolmes: single-choice + forced <think>/<answer> CoT
# ---------------------------------------------------------------------------

_VIDEOHOLMES_COT_PROMPT = (
    DEFAULT_PROMPTS.videoholmes_cot_prompt if DEFAULT_PROMPTS is not None
    else (
        "Based on the given video, reason and answer the single-choice question."
    )
)
# NOTE: Official stream_think model REMOVES the CoT instruction
# (" Provide your reasoning between the <think> and </think> tags, ...")
# from the prompt at runtime (see qwen2_5_vl_stream_think.py line 554).
# The model generates <think>/<answer> tags on its own without being told to.
# We match that behavior here by not including the instruction.
_VIDEOHOLMES_FORMAT_TEMPLATE = (
    DEFAULT_PROMPTS.videoholmes_format_template if DEFAULT_PROMPTS is not None
    else "The question is: {question}\nThe options are:\n{options}\nYour answer:"
)


def _videoholmes_prompt(doc: dict) -> str:
    options_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(doc.get("candidates", [])))
    formatted = _VIDEOHOLMES_FORMAT_TEMPLATE.format(question=doc["question"], options=options_str)
    return f"{_VIDEOHOLMES_COT_PROMPT} {formatted}"


def _videoholmes_resolve_video(doc: dict, video_root: str) -> str:
    return str(Path(video_root) / doc["video"])


def _videoholmes_score(doc: dict, pred: str) -> dict:
    candidates = doc.get("candidates", [])
    gt_text = doc.get("answer")
    try:
        gt_letter = chr(65 + candidates.index(gt_text))
    except ValueError:
        gt_letter = "UNKNOWN"

    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", pred, re.DOTALL)
    choice = matches[-1].strip() if matches else pred.strip()

    predicted = "WRONG"
    found = False
    for i in range(len(candidates)):
        letter = chr(65 + i)
        if f"{letter} " in choice or f"{letter}:" in choice or f"[{letter}" in choice:
            predicted = letter
            found = True
            break
    if not found:
        for i in range(len(candidates)):
            letter = chr(65 + i)
            if letter in choice:
                predicted = letter
                break

    score = 1 if predicted == gt_letter else 0
    return {
        "pred_answer": predicted,
        "gt_answer": gt_letter,
        "score": score,
        "task": doc.get("task", "video_holmes"),
        "subtask": doc.get("subtask"),
        "raw_pred": pred,
    }


def make_videoholmes_task(anno_path: str, video_root: str) -> TaskSpec:
    return TaskSpec(
        name="videoholmes",
        anno_path=anno_path,
        video_root=video_root,
        build_prompt=_videoholmes_prompt,
        resolve_video=_videoholmes_resolve_video,
        score_one=_videoholmes_score,
        generation_kwargs={"max_new_tokens": 1024, "temperature": 0.0},
    )


# ---------------------------------------------------------------------------
# LongVideoBench / Video-MME: unified MCQ JSON adapters
# ---------------------------------------------------------------------------

_GENERIC_MCQ_POST_PROMPT = (
    DEFAULT_PROMPTS.mcq_post_prompt if DEFAULT_PROMPTS is not None
    else "Answer with the option's letter from the given choices directly."
)


def _generic_mcq_prompt(doc: dict) -> str:
    return format_mcq_prompt(doc["question"], doc["candidates"], _GENERIC_MCQ_POST_PROMPT)


def _generic_mcq_resolve_video(doc: dict, video_root: str) -> str:
    return str(Path(video_root) / doc["video"])


def _generic_mcq_score(doc: dict, pred: str) -> dict:
    # Unified LongVideoBench/Video-MME JSON already stores the GT as a
    # bare letter (e.g. "B"), unlike StreamingBench's "match candidate text"
    # flow -- so score directly against doc["answer"] instead of going
    # through gt_letter_from_candidates().
    gt_letter = str(doc.get("answer", "")).strip().upper()
    score = mcq_acc(gt_letter, pred) if gt_letter else 0
    return {
        "pred_answer": pred,
        "gt_answer": gt_letter,
        "score": score,
        "task": doc.get("task", "generic_mcq"),
        "subtask": doc.get("subtask"),
    }


def make_generic_mcq_task(name: str) -> Callable[[str, str], TaskSpec]:
    # Per-task stream_think_times matching official eval scripts:
    #   VideoMME: 1-1-3-4
    #   LongVideoBench: 1 (= 1-1-1-1, single segment, no intermediate thinking)
    _STREAM_THINK_TIMES_BY_TASK = {
        "videomme": "1-1-3-4",
        "videomme_subset7500": "1-1-3-4",
        "longvideobench": "1",
        "longvideobench_subset7500": "1",
    }

    def _builder(anno_path: str, video_root: str) -> TaskSpec:
        override = {}
        if name in _STREAM_THINK_TIMES_BY_TASK:
            override["stream_think_times"] = _STREAM_THINK_TIMES_BY_TASK[name]
        return TaskSpec(
            name=name,
            anno_path=anno_path,
            video_root=video_root,
            build_prompt=_generic_mcq_prompt,
            resolve_video=_generic_mcq_resolve_video,
            score_one=_generic_mcq_score,
            generation_kwargs={"max_new_tokens": 16, "temperature": 0.0},
            sampling_override=override,
        )

    return _builder


# ============================================================================
# Registry: task name -> (builder_fn, default_anno_path, default_video_root)
# Paths are sourced from eval_config.TASK_PATHS so that dataset relocation
# only requires editing RLSD/eval/eval_config.py, not this file.
# ============================================================================

_EVAL_ROOT = os.environ.get("RLSD_EVAL_DATA_ROOT", "/path/to/your/datasets/EVAL")
_ANNO_ROOT = os.environ.get(
    "RLSD_EVAL_ANNO_ROOT",
    str(Path(__file__).resolve().parents[1] / "eval_data" / "anno" / "eval"),
)

_FALLBACK_TASK_PATHS: dict[str, dict[str, str]] = {
    "streamingbench": {
        "anno_path": f"{_ANNO_ROOT}/StreamingBench/json/real_time_visual_understanding.json",
        "video_root": f"{_EVAL_ROOT}/StreamingBench",
    },
    "ovobench_real_time_visual_perception": {
        "anno_path": f"{_ANNO_ROOT}/OVOBench/json/real_time_visual_perception.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_backward_tracking": {
        "anno_path": f"{_ANNO_ROOT}/OVOBench/json/backward_tracking.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_forward_active_responding": {
        "anno_path": f"{_ANNO_ROOT}/OVOBench/json/forward_active_responding.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "videoholmes": {
        "anno_path": f"{_ANNO_ROOT}/VideoHolmes/test.json",
        "video_root": f"{_EVAL_ROOT}/Video-Holmes",
    },
    "longvideobench": {
        "anno_path": f"{_ANNO_ROOT}/LongVideoBench/json/val.json",
        "video_root": f"{_EVAL_ROOT}/LongVideoBench",
    },
    "videomme": {
        "anno_path": f"{_ANNO_ROOT}/VideoMME/json/test.json",
        "video_root": f"{_EVAL_ROOT}/Video-MME",
    },
    "streamingbench_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/streamingbench.json",
        "video_root": f"{_EVAL_ROOT}/StreamingBench",
    },
    "ovobench_real_time_visual_perception_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/ovobench_real_time_visual_perception.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_backward_tracking_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/ovobench_backward_tracking.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_forward_active_responding_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/ovobench_forward_active_responding.json",
        "video_root": f"{_EVAL_ROOT}/OVO-Bench/chunked_videos",
    },
    "videoholmes_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/videoholmes.json",
        "video_root": f"{_EVAL_ROOT}/Video-Holmes",
    },
    "longvideobench_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/longvideobench.json",
        "video_root": f"{_EVAL_ROOT}/LongVideoBench",
    },
    "videomme_subset7500": {
        "anno_path": f"{_ANNO_ROOT}/subset7500/videomme.json",
        "video_root": f"{_EVAL_ROOT}/Video-MME",
    },
}

_TASK_PATHS = TASK_PATHS if TASK_PATHS is not None else _FALLBACK_TASK_PATHS
_TASK_BUILDERS: dict[str, Callable[[str, str], TaskSpec]] = {
    "streamingbench": make_streamingbench_task,
    "ovobench_real_time_visual_perception": make_ovobench_task,
    "ovobench_backward_tracking": make_ovobench_task,
    "ovobench_forward_active_responding": make_ovobench_task,
    "videoholmes": make_videoholmes_task,
    "longvideobench": make_generic_mcq_task("longvideobench"),
    "videomme": make_generic_mcq_task("videomme"),
    # Proportionally down-sampled 7.5k-sample subset (see
    # eval_data/anno/eval/subset7500/manifest.json for the per-task quotas
    # and random seeds used to build it). Same annotation schema per task,
    # so builders/scorers are reused as-is; only anno_path differs.
    "streamingbench_subset7500": make_streamingbench_task,
    "ovobench_real_time_visual_perception_subset7500": make_ovobench_task,
    "ovobench_backward_tracking_subset7500": make_ovobench_task,
    "ovobench_forward_active_responding_subset7500": make_ovobench_task,
    "videoholmes_subset7500": make_videoholmes_task,
    "longvideobench_subset7500": make_generic_mcq_task("longvideobench_subset7500"),
    "videomme_subset7500": make_generic_mcq_task("videomme_subset7500"),
}

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    name: {"builder": _TASK_BUILDERS[name], **paths}
    for name, paths in _TASK_PATHS.items()
    if name in _TASK_BUILDERS
}



def build_task(name: str, anno_path: Optional[str] = None, video_root: Optional[str] = None) -> TaskSpec:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(TASK_REGISTRY)}")
    entry = TASK_REGISTRY[name]
    return entry["builder"](
        anno_path or entry["anno_path"],
        video_root or entry["video_root"],
    )
