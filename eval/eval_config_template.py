"""
eval_config_template.py
========================
Configuration template for the vLLM-accelerated evaluation pipeline.

Instructions:
  1. Copy this file to eval_config.py:
       cp eval_config_template.py eval_config.py
  2. Edit eval_config.py to set your local paths (dataset root, annotation root, GPUs, etc.)
  3. Alternatively, set environment variables:
       export RLSD_EVAL_DATA_ROOT=/path/to/your/datasets/EVAL
       export RLSD_EVAL_ANNO_ROOT=/path/to/your/annotations

The vLLM eval pipeline reads defaults from eval_config.py; CLI arguments override them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 1. GPU Configuration
# ============================================================================

# GPUs for data-parallel evaluation (one vLLM engine per GPU).
# Override via --gpus CLI argument.
DEFAULT_GPUS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]


# ============================================================================
# 2. Output Directories
# ============================================================================

DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "eval_outputs" / "default_run")
DEFAULT_STREAM_OUTPUT_DIR = str(Path(__file__).resolve().parent / "eval_outputs" / "stream_default_run")

# Checkpoint queue CSV (for batch evaluation of multiple models)
CHECKPOINT_QUEUE_CSV = str(Path(__file__).resolve().parent / "checkpoint_queue.csv")

# Summary results (CSV + XLSX)
RESULTS_DIR = str(Path(__file__).resolve().parent / "results")
RESULTS_SUMMARY_CSV = str(Path(RESULTS_DIR) / "eval_summary.csv")
RESULTS_SUMMARY_XLSX = str(Path(RESULTS_DIR) / "eval_summary.xlsx")


# ============================================================================
# 3. Video Sampling Parameters
# ============================================================================

@dataclass
class SamplingConfig:
    max_num_frames: int = 512       # Global max frames per video
    sample_fps: float = 3.0         # Candidate sampling rate (frames/sec)
    max_pixels: int | None = 800000 # Per-frame pixel cap for Qwen-VL


DEFAULT_SAMPLING = SamplingConfig()


# ============================================================================
# 4. Stream Think Parameters
# ============================================================================

@dataclass
class StreamConfig:
    chunking_mode: str = "vst_original"     # "vst_original" or "token_budget"
    stream_token_budget: int = 20480        # [token_budget mode] flush threshold
    tokens_per_frame: int = 256             # Estimated visual tokens per frame
    max_chunks: int = 32                    # Safety ceiling on chunk count
    max_memory_chars: int = 12000           # Memory truncation limit
    think_max_tokens: int = 5000            # Max tokens for intermediate thinking
    use_predecoded_video: bool = True       # Pre-decode frames via decord
    decode_num_threads: int = 24            # Decord thread count
    use_torchvision_resize: bool = True     # Match official HF resize
    frame_sampling_mode: str = "strict_fps" # "strict_fps" or "linspace"


DEFAULT_STREAM = StreamConfig()


@dataclass
class VSTOriginalConfig:
    stream_think_times: str = "2-3-5-5"    # Duration-bucket segment counts
    max_stream_vid_tokens: int = 8192       # Per-round visual token budget
    max_keep_memory: int = 0                # Memory truncation by round (0=unlimited)


DEFAULT_VST_ORIGINAL = VSTOriginalConfig()


# ============================================================================
# 5. vLLM Engine Parameters
# ============================================================================

@dataclass
class EngineConfig:
    max_model_len: int = 128000             # KV cache max sequence length
    gpu_memory_utilization: float = 0.90    # Fraction of GPU memory for vLLM
    batch_size: int = 64                    # Non-streaming batch size


DEFAULT_ENGINE = EngineConfig()


# ============================================================================
# 6. Prompt Templates
# ============================================================================

@dataclass
class PromptConfig:
    mcq_post_prompt: str = "Answer with the option's letter from the given choices directly."
    videoholmes_cot_prompt: str = "Based on the given video, reason and answer the single-choice question."
    videoholmes_format_template: str = "The question is: {question}\nThe options are:\n{options}\nYour answer:"
    stream_system_prompt: str = "You are a helpful assistant."
    stream_think_prompt_template: str = (
        "Time={start:.1f}-{end:.1f}s\n"
        "Watch this video segment and describe all visual evidence that may be useful "
        "for answering future questions. Be concise but preserve key objects, actions, "
        "counts, text, directions, and temporal events."
    )
    stream_memory_prefix: str = "Here is the accumulated memory from previous video segments:\n"


DEFAULT_PROMPTS = PromptConfig()


# ============================================================================
# 7. Dataset Paths (annotation JSON + video root)
# ============================================================================
# Override defaults via environment variables:
#   RLSD_EVAL_DATA_ROOT — root dir of downloaded videos
#   RLSD_EVAL_ANNO_ROOT — root dir of annotation JSONs (default: eval_data/anno/eval)

import os as _os

# EDIT THIS: point to where you downloaded the evaluation videos
EVAL_DATA_ROOT = _os.environ.get("RLSD_EVAL_DATA_ROOT", "/path/to/your/datasets/EVAL")
ANNO_ROOT = _os.environ.get(
    "RLSD_EVAL_ANNO_ROOT",
    str(Path(__file__).resolve().parent / "eval_data" / "anno" / "eval"),
)

TASK_PATHS: dict[str, dict[str, str]] = {
    "streamingbench": {
        "anno_path": f"{ANNO_ROOT}/StreamingBench/json/real_time_visual_understanding.json",
        "video_root": f"{EVAL_DATA_ROOT}/StreamingBench",
    },
    "ovobench_real_time_visual_perception": {
        "anno_path": f"{ANNO_ROOT}/OVOBench/json/real_time_visual_perception.json",
        "video_root": f"{EVAL_DATA_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_backward_tracking": {
        "anno_path": f"{ANNO_ROOT}/OVOBench/json/backward_tracking.json",
        "video_root": f"{EVAL_DATA_ROOT}/OVO-Bench/chunked_videos",
    },
    "ovobench_forward_active_responding": {
        "anno_path": f"{ANNO_ROOT}/OVOBench/json/forward_active_responding.json",
        "video_root": f"{EVAL_DATA_ROOT}/OVO-Bench/chunked_videos",
    },
    "videoholmes": {
        "anno_path": f"{ANNO_ROOT}/VideoHolmes/test.json",
        "video_root": f"{EVAL_DATA_ROOT}/Video-Holmes",
    },
    "longvideobench": {
        "anno_path": f"{ANNO_ROOT}/LongVideoBench/json/val.json",
        "video_root": f"{EVAL_DATA_ROOT}/LongVideoBench",
    },
    "videomme": {
        "anno_path": f"{ANNO_ROOT}/VideoMME/json/test.json",
        "video_root": f"{EVAL_DATA_ROOT}/Video-MME",
    },
}
