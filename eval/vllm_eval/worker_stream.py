#!/usr/bin/env python3
"""
Single-GPU worker for vLLM stream-think video evaluation.
=========================================================
Implements a vLLM version of the VST StreamThink idea:

  video chunks -> intermediate "thinking" outputs -> textual memory -> final answer

The video's frame budget is decided once per sample, globally: candidate
frames are laid down at `sample_fps` across the full duration, then thinned
to at most `max_num_frames` -- the same total frame budget a single
non-streaming request would use. That fixed frame set is decoded once and
then fed into a token-budget streaming loop: frames are accumulated in
order, and each time the running estimate reaches `stream_token_budget`
visual tokens, we flush a chunk into an intermediate-thinking request.
`max_chunks` is only a safety ceiling for pathological settings; it does not
replace the token-budget rule. The single most important instrumented metric
is TTFT
(time-to-first-token) for the FINAL answer request -- i.e. the latency from
"video input for this sample is fully available to the model" to "the model
emits its first output token of the official answer". Intermediate "thinking"
requests are not counted towards this number; only the final-answer request's
TTFT is what we report, since that's the round-trip a real-time streaming
user would perceive.

Uses vllm_eval.async_chat_client.StreamChatClient (built on vLLM's AsyncLLM)
because, empirically, the sync LLM.generate()/LLM.chat() path in vLLM 0.11.0
does not populate RequestOutput.metrics.first_token_time -- see
async_chat_client.py docstring and VLLM_MIGRATION_ANALYSIS.md for the
investigation trail.

Every sample produces one JSONL log line capturing:
  - which chunks were streamed in, with their frame/time ranges
  - the prompt sent for each intermediate "think" step and its output
  - the running textual memory after each step (and when/why it was truncated)
  - the final prompt (memory + last chunk + question), final output, GT, score
  - TTFT for the final-answer request, plus total per-sample latency
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_eval.tasks import build_task  # noqa: E402
from vllm_eval.progress import write_progress  # noqa: E402
try:
    from eval_config import DEFAULT_PROMPTS, DEFAULT_SAMPLING, DEFAULT_STREAM, DEFAULT_VST_ORIGINAL
except ImportError:  # fallback for isolated imports/tests
    DEFAULT_PROMPTS = None
    DEFAULT_SAMPLING = None
    DEFAULT_STREAM = None
    DEFAULT_VST_ORIGINAL = None

_SYSTEM_PROMPT = (
    DEFAULT_PROMPTS.stream_system_prompt if DEFAULT_PROMPTS is not None
    else "You are a helpful assistant."
)
_THINK_PROMPT_TEMPLATE = (
    DEFAULT_PROMPTS.stream_think_prompt_template if DEFAULT_PROMPTS is not None
    else (
        "Time={start:.1f}-{end:.1f}s\n"
        "Watch this video segment and describe all visual evidence that may be useful "
        "for answering future questions. Be concise but preserve key objects, actions, "
        "counts, text, directions, and temporal events."
    )
)
_MEMORY_PREFIX = (
    DEFAULT_PROMPTS.stream_memory_prefix if DEFAULT_PROMPTS is not None
    else "Here is the accumulated memory from previous video segments:\n"
)


def _load_shard(docs: list[dict], shard_id: int, num_shards: int) -> list[dict]:
    return [d for i, d in enumerate(docs) if i % num_shards == shard_id]


def _probe_video(video_path: str, decode_num_threads: int = 1) -> tuple[int, float, float]:
    import decord
    from decord import cpu
    num_threads = decode_num_threads if decode_num_threads and decode_num_threads > 0 else 0
    vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration_s = total_frames / fps if fps > 0 else 0.0
    return total_frames, fps, duration_s


def _sample_global_frame_indices(
    total_frames: int, fps: float, duration_s: float, sample_fps: float, max_num_frames: int,
) -> list[int]:
    """Decide once, for the whole video, which original frame indices to decode.

    This replaces the old per-chunk `_sample_indices()` call: instead of each
    chunk independently taking up to `max_num_frames` frames (so a sample with
    N chunks could end up decoding close to N * max_num_frames frames total),
    `max_num_frames` now bounds the *entire sample's* visual input, matching
    how a single non-streaming request would sample the same video.

    Algorithm: first lay down candidate frames at `sample_fps` across the full
    duration (e.g. sample_fps=2 -> one candidate every 0.5s), then -- only if
    that candidate set exceeds `max_num_frames` -- uniformly thin it down to
    exactly `max_num_frames`. If `sample_fps <= 0`, skip straight to a uniform
    `max_num_frames` split across the whole video.
    """
    import numpy as np

    if total_frames <= 0:
        return [0]

    # If the video is already shorter than the frame budget, just take every
    # frame -- no need to thin a candidate set that's smaller than the video.
    if max_num_frames > 0 and total_frames <= max_num_frames:
        return list(range(total_frames))

    if fps > 0 and sample_fps > 0:
        num_candidates = max(1, int(round(duration_s * sample_fps)))
    else:
        num_candidates = total_frames

    if max_num_frames > 0:
        num_candidates = min(num_candidates, max_num_frames)
    num_candidates = min(num_candidates, total_frames)
    num_candidates = max(1, num_candidates)

    if num_candidates >= total_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_candidates, dtype=int).tolist()
    return sorted(set(indices))


def _sample_global_frame_indices_strict_fps(
    total_frames: int, fps: float, duration_s: float, sample_fps: float, max_num_frames: int,
) -> list[int]:
    """Official-matching frame sampling: strict_fps=True, drop_last=False.

    Replicates _read_video_decord_plus(strict_fps=True, drop_last=False):
    1. Generate expected timestamps at fixed `sample_fps` intervals.
    2. If count > max_num_frames, uniformly downsample the timestamp array
       (preserving temporal distribution, not frame-index distribution).
    3. Snap each timestamp to the nearest frame index.
    4. Ensure frame count is aligned to FRAME_FACTOR=2.

    This produces slightly different indices than linspace in frame-space,
    especially for long videos, resulting in better temporal coverage.
    """
    import numpy as np

    FRAME_FACTOR = 2  # Match official Qwen-VL constant

    if total_frames <= 0:
        return [0]

    if max_num_frames > 0 and total_frames <= max_num_frames:
        indices = list(range(total_frames))
        # FRAME_FACTOR alignment
        while len(indices) % FRAME_FACTOR != 0:
            indices.append(indices[-1])
        return indices

    # Generate timestamps at sample_fps (default 2.0 fps)
    clip_pts = np.arange(total_frames) / fps  # all frame timestamps
    start_t = clip_pts[0]
    end_t = clip_pts[-1]

    expected_timestamps = np.arange(start_t, end_t + 1e-6, 1.0 / sample_fps)

    # If too many, uniformly downsample timestamps (drop_last=False style)
    if max_num_frames > 0 and len(expected_timestamps) > max_num_frames:
        subsample_idx = np.linspace(
            0, len(expected_timestamps) - 1, max_num_frames
        ).round().astype(int)
        expected_timestamps = expected_timestamps[subsample_idx]

    # Snap each timestamp to nearest frame index
    frame_indices = np.searchsorted(clip_pts, expected_timestamps, side="right") - 1
    frame_indices = np.clip(frame_indices, 0, total_frames - 1).tolist()

    # FRAME_FACTOR alignment (pad last frame if needed)
    while len(frame_indices) % FRAME_FACTOR != 0:
        frame_indices.append(frame_indices[-1])

    return frame_indices


def _compute_dynamic_resolution(
    nframes: int,
    stream_think_times_count: int,
    max_stream_vid_tokens: int = 8192,
    min_pixels: int = 200704,  # 256 * 28 * 28
    frame_factor: int = 2,
    video_max_pixels: int = 602112,  # 768 * 28 * 28
) -> tuple[int, int]:
    """Replicate official _spatial_resize_video's resolution logic.

    Official flow: total pixel budget = max_stream_vid_tokens * 28² * stream_think_times.
    Then pixels_per_frame = (total_budget * FRAME_FACTOR) / nframes.
    If pixels_per_frame < min_pixels, reduce nframes so each frame gets at least min_pixels.
    Finally, per-frame max_pixels = min(VIDEO_MAX_PIXELS, actual_pixels_per_frame).

    Returns:
        (new_nframes, dynamic_max_pixels): possibly reduced frame count and
        the per-frame pixel cap to use for resizing.
    """
    if nframes <= 0:
        return max(nframes, 1), min_pixels

    video_total_pixels = max_stream_vid_tokens * (28 * 28) * stream_think_times_count
    pixels_per_frame = (video_total_pixels * frame_factor) / nframes

    actual_nframes = nframes
    if pixels_per_frame < min_pixels:
        # Reduce frame count so each frame gets at least min_pixels
        new_nframes = int((video_total_pixels * frame_factor) / min_pixels)
        new_nframes = max(1, new_nframes)
        if new_nframes < nframes:
            actual_nframes = new_nframes

    # Recompute per-frame pixels after possible nframes reduction
    current_pixels_per_frame = (video_total_pixels * frame_factor) / actual_nframes
    dynamic_max_pixels = max(
        min(video_max_pixels, int(current_pixels_per_frame)),
        int(min_pixels * 1.05),
    )
    return actual_nframes, dynamic_max_pixels


def _smart_resize_size(height: int, width: int, max_pixels: int, factor: int = 28, min_pixels: int = 3136) -> tuple[int, int]:
    """Qwen-style aspect-preserving resize size, rounded to `factor`.

    vLLM does not reliably apply request-level max_pixels to predecoded ndarray
    videos, so we resize the ndarray before handing it to the processor.
    """
    import math

    def round_by_factor(number: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(number: float) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: float) -> int:
        return math.floor(number / factor) * factor

    resized_h = max(factor, round_by_factor(height))
    resized_w = max(factor, round_by_factor(width))
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, floor_by_factor(height / beta))
        resized_w = max(factor, floor_by_factor(width / beta))
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = ceil_by_factor(height * beta)
        resized_w = ceil_by_factor(width * beta)
    return int(resized_h), int(resized_w)


def _resize_video_frames(frames: Any, max_pixels: int | None) -> Any:
    """Resize video frames to fit within max_pixels budget.

    Uses PIL BICUBIC (legacy mode). See _resize_video_frames_torchvision for
    the official-matching antialias version.
    """
    if not max_pixels or frames.size == 0:
        return frames
    height, width = frames.shape[1], frames.shape[2]
    if height * width <= max_pixels:
        return frames

    from PIL import Image

    resized_h, resized_w = _smart_resize_size(height, width, max_pixels)
    resized = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)
        resized.append(np.asarray(image, dtype=frames.dtype))
    return np.stack(resized, axis=0)


def _resize_video_frames_torchvision(frames: Any, max_pixels: int | None) -> Any:
    """Resize video frames using torchvision with antialias — matches official.

    Official path uses torchvision.transforms.functional.resize with:
      - InterpolationMode.BICUBIC
      - antialias=True
    This preserves more high-frequency detail (text, edges) than PIL resize
    and produces visual tokens closer to the official HF generate path.

    Args:
        frames: numpy array (T, H, W, C) uint8
        max_pixels: per-frame pixel budget

    Returns:
        numpy array (T, H, W, C) uint8, resized
    """
    if not max_pixels or frames.size == 0:
        return frames
    height, width = frames.shape[1], frames.shape[2]
    if height * width <= max_pixels:
        return frames

    import torch
    from torchvision.transforms.functional import resize
    from torchvision.transforms import InterpolationMode

    resized_h, resized_w = _smart_resize_size(height, width, max_pixels)

    # Convert (T, H, W, C) uint8 numpy → (T, C, H, W) torch tensor
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

    # Batch resize: torchvision resize handles (T, C, H, W) directly
    video_resized = resize(
        video_tensor,
        [resized_h, resized_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Convert back to (T, H, W, C) uint8 numpy
    result = video_resized.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    return result


def _decode_global_video_once(
    video_path: str,
    frame_indices: list[int],
    decode_num_threads: int,
    max_pixels: int | None = None,
    use_torchvision_resize: bool = False,
) -> Any:
    """Decode the whole sample's sampled frames in a single decord call.

    This is the only decode for the sample: every chunk (intermediate-think
    and final-answer requests alike) slices its frames out of this one
    already-decoded ndarray instead of re-sampling/re-decoding from the mp4.
    """
    import decord
    from decord import cpu

    num_threads = decode_num_threads if decode_num_threads and decode_num_threads > 0 else 0
    vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    frames = vr.get_batch(frame_indices).asnumpy()
    if use_torchvision_resize:
        return _resize_video_frames_torchvision(frames, max_pixels)
    return _resize_video_frames(frames, max_pixels)


def _build_frame_chunks(
    frame_indices: list[int], fps: float, tokens_per_frame: int, stream_token_budget: int, max_chunks: int,
) -> list[dict]:
    """Split the already-decided global frame set into streaming
    think/final-answer segments by accumulating estimated visual tokens
    until they cross `stream_token_budget` -- the "buffer until ~5000
    tokens, then think" rule -- then cutting a chunk boundary there.

    Unlike the old per-chunk sampling, this never re-derives *which* frames
    to look at (that's already fixed by `_sample_global_frame_indices()`);
    it only decides how the fixed, already-sampled frame set is grouped into
    requests. `max_chunks` is kept purely as a safety ceiling: if the
    token-budget accumulation would produce more segments than that (e.g.
    tokens_per_frame set too low for a long video), the remaining frames are
    folded into the last chunk instead of spawning more requests.
    """
    num_frames = len(frame_indices)
    if num_frames == 0:
        return [{
            "chunk_id": 0, "sampled_start_idx": 0, "sampled_end_idx": 0,
            "num_input_frames": 0, "frame_indices": [],
            "start_frame": 0, "end_frame": 0, "start_s": 0.0, "end_s": 0.0,
            "estimated_visual_tokens": 0,
        }]

    frames_per_budget = max(1, stream_token_budget // max(tokens_per_frame, 1))

    boundaries = list(range(0, num_frames, frames_per_budget))
    if boundaries[-1] != num_frames:
        boundaries.append(num_frames)
    # Safety ceiling: never spawn more than max_chunks requests, regardless
    # of how the token-budget accumulation split things up -- fold any
    # excess boundaries into the last chunk.
    if max_chunks > 0 and len(boundaries) - 1 > max_chunks:
        boundaries = boundaries[:max_chunks] + [num_frames]

    chunks = []
    for chunk_id in range(len(boundaries) - 1):
        lo, hi = boundaries[chunk_id], boundaries[chunk_id + 1]
        chunk_frame_indices = frame_indices[lo:hi]
        start_frame = chunk_frame_indices[0] if chunk_frame_indices else (frame_indices[-1] if frame_indices else 0)
        end_frame = chunk_frame_indices[-1] + 1 if chunk_frame_indices else start_frame + 1
        chunks.append({
            "chunk_id": chunk_id,
            "sampled_start_idx": lo,
            "sampled_end_idx": hi,
            "num_input_frames": len(chunk_frame_indices),
            "frame_indices": chunk_frame_indices,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_frame / fps if fps > 0 else 0.0,
            "end_s": end_frame / fps if fps > 0 else 0.0,
            "estimated_visual_tokens": len(chunk_frame_indices) * tokens_per_frame,
        })
    return chunks


def _resolve_stream_think_times(stream_think_times_str: str) -> list[int]:
    """Parse a VST-style `stream_think_times` spec into a 4-element list.

    Mirrors qwen2_5_vl_stream_think.py's own parsing: a single integer
    (e.g. "1") expands to `[n, n, n, n]`; a "-"-separated spec (e.g.
    "2-3-5-5") is split into exactly 4 ints, one per duration bucket.
    """
    spec = stream_think_times_str.strip()
    if "-" in spec:
        parts = [int(p) for p in spec.split("-")]
        if len(parts) != 4:
            raise ValueError(f"invalid stream_think_times spec: {stream_think_times_str!r}, expected 4 '-'-separated ints")
        return parts
    n = int(spec)
    return [n] * 4


def _pick_stream_think_times(duration_s: float, stream_think_times: list[int]) -> int:
    """Duration-bucket lookup, matching qwen2_5_vl_stream_think.py exactly:
    <=0.5min / 0.5-4min / 4-30min / >=30min -> stream_think_times[0..3].
    """
    duration_min = duration_s / 60.0
    if duration_min <= 0.5:
        return stream_think_times[0]
    if duration_min < 4:
        return stream_think_times[1]
    if duration_min < 30:
        return stream_think_times[2]
    return stream_think_times[3]


def _build_frame_chunks_vst_original(
    frame_indices: list[int], fps: float, duration_s: float, stream_think_times: list[int],
) -> list[dict]:
    """VST-original chunking: split the already-sampled global frame set
    into N evenly-sized segments, where N is looked up from video duration
    (see `_pick_stream_think_times`) rather than accumulated from a token
    budget. Mirrors qwen2_5_vl_stream_think.py's
    `segment_length = total_frames // stream_think_times` (last segment
    absorbs the remainder). Returns the same chunk dict shape as
    `_build_frame_chunks()` so the rest of `_eval_one_sample()` is unchanged.
    """
    num_frames = len(frame_indices)
    if num_frames == 0:
        return [{
            "chunk_id": 0, "sampled_start_idx": 0, "sampled_end_idx": 0,
            "num_input_frames": 0, "frame_indices": [],
            "start_frame": 0, "end_frame": 0, "start_s": 0.0, "end_s": 0.0,
            "estimated_visual_tokens": 0,
        }]

    num_segments = max(1, min(_pick_stream_think_times(duration_s, stream_think_times), num_frames))
    segment_length = max(num_frames // num_segments, 1)

    boundaries = [i * segment_length for i in range(num_segments)]
    boundaries.append(num_frames)  # last segment absorbs the remainder frames
    # Drop any boundary that would start beyond the frame set (only possible
    # if segment_length * num_segments overshoots num_frames slightly).
    boundaries = sorted(set(b for b in boundaries if b <= num_frames))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    chunks = []
    for chunk_id in range(len(boundaries) - 1):
        lo, hi = boundaries[chunk_id], boundaries[chunk_id + 1]
        if chunk_id == len(boundaries) - 2:
            hi = num_frames  # last segment absorbs remainder
        chunk_frame_indices = frame_indices[lo:hi]
        if not chunk_frame_indices:
            continue
        start_frame = chunk_frame_indices[0]
        end_frame = chunk_frame_indices[-1] + 1
        chunks.append({
            "chunk_id": len(chunks),
            "sampled_start_idx": lo,
            "sampled_end_idx": hi,
            "num_input_frames": len(chunk_frame_indices),
            "frame_indices": chunk_frame_indices,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_frame / fps if fps > 0 else 0.0,
            "end_s": end_frame / fps if fps > 0 else 0.0,
            "estimated_visual_tokens": None,  # not a driver of chunking in this mode
        })
    return chunks


def _slice_predecoded_chunks(decoded_frames: Any, chunks: list[dict]) -> list[tuple[Any, dict[str, Any]]]:
    """Slice the one globally-decoded frame ndarray into a per-chunk video
    item (frames, metadata) ready to hand to chat_with_predecoded_video_ttft().
    """
    items = []
    for chunk in chunks:
        frames = decoded_frames[chunk["sampled_start_idx"]:chunk["sampled_end_idx"]]
        metadata = {
            "total_num_frames": len(frames),
            "fps": max(1e-6, len(frames) / max(1e-6, chunk["end_s"] - chunk["start_s"])),
            "duration": chunk["end_s"] - chunk["start_s"],
            "video_backend": "decord_global_predecoded",
            "frames_indices": list(range(len(frames))),
            "do_sample_frames": False,
        }
        items.append((frames, metadata))
    return items


def _build_segment_messages(video_path: str, segment_text: str, memory: str) -> list[dict]:
    """Build messages matching official VST stream_think structure:
    message[0] = {"role": "system", "content": system_prompt}
    message[1] = {"role": "previous text", "content": memory}
    message[2] = {"role": "user", "content": [{"text": time_str}, {"video_url": ...}]}
    """
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "previous text", "content": memory},
    ]
    user_content: list[dict] = [
        {"type": "text", "text": segment_text},
        {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
    ]
    messages.append({"role": "user", "content": user_content})
    return messages


async def _eval_one_sample(
    *,
    client,
    task,
    doc: dict,
    sampling_params_think,
    sampling_params_final,
    max_num_frames: int,
    max_pixels: int | None,
    sample_fps: float,
    chunking_mode: str,
    stream_token_budget: int,
    tokens_per_frame: int,
    max_chunks: int,
    max_memory_chars: int,
    stream_think_times: list[int],
    max_stream_vid_tokens: int,
    max_keep_memory: int,
    use_predecoded_video: bool,
    decode_num_threads: int,
    decode_timeout_s: float = 0.0,
    use_torchvision_resize: bool = False,
    frame_sampling_mode: str = "linspace",
    progress_file: str | None = None,
    local_idx: int = 0,
    total_docs: int = 0,
) -> dict:
    mm_processor_kwargs = {"max_pixels": max_pixels} if max_pixels else None

    def _report(phase: str, **extra):
        write_progress(progress_file, {
            "phase": phase,
            "done": local_idx,
            "total": total_docs,
            **extra,
        })

    _report("probing_video")
    video_path = task.resolve_video(doc, task.video_root)
    if decode_timeout_s and decode_timeout_s > 0:
        total_frames, fps, duration_s = await asyncio.wait_for(
            asyncio.to_thread(_probe_video, video_path, decode_num_threads=decode_num_threads),
            timeout=decode_timeout_s,
        )
    else:
        total_frames, fps, duration_s = await asyncio.to_thread(
            _probe_video, video_path, decode_num_threads=decode_num_threads
        )

    # Decide the whole sample's frame budget once (sample_fps candidate
    # density, thinned to at most max_num_frames), then split that fixed set
    # into chunks. The chunking rule itself is mode-dependent:
    #   token_budget  -- accumulate estimated visual tokens to stream_token_budget
    #   vst_original  -- duration-bucket lookup (stream_think_times) + even split
    # Either way, max_num_frames bounds the same global frame set, so results
    # are directly comparable across modes at equal max_num_frames.
    #
    # Per-task sampling overrides (e.g. StreamingBench clip_to_annotation).
    sampling_cfg = getattr(task, "sampling_override", None) or {}
    local_max_num_frames = sampling_cfg.get("max_num_frames", max_num_frames)
    local_sample_fps = sampling_cfg.get("sample_fps", sample_fps)
    local_max_pixels = sampling_cfg.get("max_pixels", max_pixels)
    # Rebuild mm_processor_kwargs with the (possibly overridden) max_pixels.
    mm_processor_kwargs = {"max_pixels": local_max_pixels} if local_max_pixels else None

    if sampling_cfg.get("clip_to_annotation"):
        # Trim to annotation [start, end] time range (e.g. StreamingBench).
        clip_start = float(doc.get("start", 0.0))
        clip_end = float(doc.get("end", duration_s))
        start_frame = max(0, int(clip_start * fps))
        end_frame = min(int(clip_end * fps), total_frames)
        effective_total_frames = max(1, end_frame - start_frame)
        effective_duration = clip_end - clip_start
        # Dynamic fps based on clip duration (mirrors official qwen2_5_vl_sf.py)
        if effective_duration <= 300:
            effective_fps = local_sample_fps
        elif effective_duration <= 600:
            effective_fps = 0.5
        else:
            effective_fps = 0.2
        # Choose frame sampling method based on config
        _frame_sampler = (
            _sample_global_frame_indices_strict_fps
            if frame_sampling_mode == "strict_fps"
            else _sample_global_frame_indices
        )
        local_indices = _frame_sampler(
            effective_total_frames, fps, effective_duration, effective_fps, local_max_num_frames,
        )
        # Offset back to full-video coordinate system
        global_frame_indices = [idx + start_frame for idx in local_indices]
    else:
        # Choose frame sampling method based on config
        _frame_sampler = (
            _sample_global_frame_indices_strict_fps
            if frame_sampling_mode == "strict_fps"
            else _sample_global_frame_indices
        )
        global_frame_indices = _frame_sampler(
            total_frames, fps, duration_s, local_sample_fps, local_max_num_frames,
        )

    picked_stream_think_times = None
    if sampling_cfg.get("skip_intermediate_thinking"):
        # Single chunk: skip all intermediate thinking, go straight to final answer.
        chunk_start_s = global_frame_indices[0] / fps if global_frame_indices else 0.0
        chunk_end_s = global_frame_indices[-1] / fps if global_frame_indices else 0.0
        chunks = [{
            "chunk_id": 0,
            "sampled_start_idx": 0,
            "sampled_end_idx": len(global_frame_indices),
            "num_input_frames": len(global_frame_indices),
            "frame_indices": global_frame_indices,
            "start_frame": global_frame_indices[0] if global_frame_indices else 0,
            "end_frame": (global_frame_indices[-1] + 1) if global_frame_indices else 0,
            "start_s": chunk_start_s,
            "end_s": chunk_end_s,
            "estimated_visual_tokens": len(global_frame_indices) * tokens_per_frame,
        }]
    elif chunking_mode == "vst_original":
        # Per-task stream_think_times override (e.g. VideoMME=1-1-3-4, LongVideoBench=1)
        local_stream_think_times = stream_think_times
        if sampling_cfg.get("stream_think_times"):
            local_stream_think_times = _resolve_stream_think_times(sampling_cfg["stream_think_times"])
        picked_stream_think_times = _pick_stream_think_times(duration_s, local_stream_think_times)
        # Dynamic resolution: replicate official _spatial_resize_video logic.
        # Compute per-frame max_pixels based on total pixel budget / nframes trade-off.
        dynamic_nframes, dynamic_max_pixels = _compute_dynamic_resolution(
            nframes=len(global_frame_indices),
            stream_think_times_count=picked_stream_think_times,
            max_stream_vid_tokens=max_stream_vid_tokens,
            min_pixels=200704,  # 256 * 28 * 28 (official VIDEO_MIN_PIXELS)
        )
        # If _compute_dynamic_resolution reduced nframes, uniformly thin our indices.
        if dynamic_nframes < len(global_frame_indices):
            thin_idx = np.linspace(0, len(global_frame_indices) - 1, dynamic_nframes, dtype=int).tolist()
            global_frame_indices = [global_frame_indices[i] for i in thin_idx]
        # Override max_pixels with the dynamic value for this sample.
        local_max_pixels = dynamic_max_pixels
        mm_processor_kwargs = {"max_pixels": local_max_pixels} if local_max_pixels else None

        chunks = _build_frame_chunks_vst_original(global_frame_indices, fps, duration_s, local_stream_think_times)
    else:
        chunks = _build_frame_chunks(global_frame_indices, fps, tokens_per_frame, stream_token_budget, max_chunks)


    video_items: list[tuple[Any, dict[str, Any]]] | None = None
    decoded_frame_shape: list[int] | None = None
    if use_predecoded_video:
        _report("decoding_video", num_chunks=len(chunks))
        if decode_timeout_s and decode_timeout_s > 0:
            decoded_frames = await asyncio.wait_for(
                asyncio.to_thread(
                    _decode_global_video_once,
                    video_path,
                    global_frame_indices,
                    decode_num_threads,
                    max_pixels=local_max_pixels,
                    use_torchvision_resize=use_torchvision_resize,
                ),
                timeout=decode_timeout_s,
            )
        else:
            decoded_frames = await asyncio.to_thread(
                _decode_global_video_once,
                video_path,
                global_frame_indices,
                decode_num_threads,
                max_pixels=local_max_pixels,
                use_torchvision_resize=use_torchvision_resize,
            )
        decoded_frame_shape = list(decoded_frames.shape)
        video_items = _slice_predecoded_chunks(decoded_frames, chunks)

    # Initial memory: official starts with "[System]\nYou are a Streaming Video Analyst.\n"
    _INITIAL_MEMORY = "[System]\nYou are a Streaming Video Analyst.\n"
    memory_parts: list[str] = [_INITIAL_MEMORY]
    detailed_chunks = []
    sample_t0 = time.perf_counter()

    # Intermediate "buffer -> think" steps for all but the last chunk.
    # Official message structure per intermediate think request:
    #   message[0] = {"role": "system", "content": "You are a helpful assistant."}
    #   message[1] = {"role": "previous text", "content": get_textual_memory(memory_parts)}
    #   message[2] = {"role": "user", "content": [{"text": "Time=X-Ys "}, {"video": chunk}]}
    # Memory accumulation: "Time=X-Ys " + model_output + "\n"
    for chunk in chunks[:-1]:
        _report("stream_think", chunk_id=chunk["chunk_id"], num_chunks=len(chunks))
        seg_text = f"Time={chunk['start_s']:.1f}-{chunk['end_s']:.1f}s "

        # Build memory string matching official get_textual_memory()
        if chunking_mode == "vst_original" and max_keep_memory > 0 and len(memory_parts) > (max_keep_memory + 1):
            memory_str = memory_parts[0] + "".join(memory_parts[-max_keep_memory:])
        else:
            memory_str = "".join(memory_parts)
        memory_before = memory_str

        if video_items is not None:
            result = await client.chat_with_predecoded_video_ttft(
                text_prompt=seg_text,
                video_item=video_items[chunk["chunk_id"]],
                sampling_params=sampling_params_think,
                system_prompt=_SYSTEM_PROMPT,
                memory_text=memory_str,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        else:
            messages = _build_segment_messages(video_path, seg_text, memory_str)
            processor_kwargs = {"max_num_frames": max_num_frames}
            if max_pixels:
                processor_kwargs["max_pixels"] = max_pixels
            result = await client.chat_with_ttft(messages, sampling_params_think, processor_kwargs)

        # Official memory format: "Time=X-Ys " + model_output + "\n"
        memory_update = seg_text + result['text'] + "\n"
        memory_parts.append(memory_update)
        truncated = False
        if chunking_mode == "vst_original":
            if max_keep_memory > 0 and len(memory_parts) > (max_keep_memory + 1):
                memory_parts = [memory_parts[0]] + memory_parts[-max_keep_memory:]
                truncated = True
        else:
            memory_joined = "".join(memory_parts)
            if max_memory_chars > 0 and len(memory_joined) > max_memory_chars:
                first = memory_parts[0]
                suffix = memory_joined[-max_memory_chars:]
                memory_parts = [first, suffix]
                truncated = True

        detailed_chunks.append({
            **chunk,
            "phase": "intermediate_think",
            "prompt_text": seg_text,
            "memory_before_chars": len(memory_before),
            "memory_update": result["text"],
            "memory_truncated_after_this_step": truncated,
            "latency_s": result["total_s"],
            "ttft_s": result["ttft_s"],  # informational only; not the reported headline metric
            "num_output_tokens": result["num_output_tokens"],
        })

    # Final chunk: this is the moment the "video is fully input" for the sample.
    # TTFT is measured on this request specifically.
    final_chunk = chunks[-1]
    _report("final_answer", chunk_id=final_chunk["chunk_id"], num_chunks=len(chunks))

    # Official final answer message structure:
    #   message[0] = {"role": "system", "content": "You are a helpful assistant."}
    #   message[1] = {"role": "previous text", "content": get_textual_memory(memory_parts)}
    #   message[-2] = {"role": "user", "content": [{"text": "Time=X-Ys "}, {"video": last_chunk}]}
    #   message[-1] = {"role": "user", "content": "Time=Ts " + question}
    # For clip_to_annotation tasks (e.g. StreamingBench), use annotation time range.
    if sampling_cfg.get("clip_to_annotation"):
        clip_start = float(doc.get("start", 0.0))
        clip_end = float(doc.get("end", duration_s))
        final_video_text = f"Time={clip_start:.1f}-{clip_end:.1f}s "
        final_question_text = f"Time={clip_end:.1f}s {task.build_prompt(doc)}"
    else:
        final_video_text = f"Time={final_chunk['start_s']:.1f}-{final_chunk['end_s']:.1f}s "
        total_seconds = duration_s
        final_question_text = f"Time={total_seconds:.1f}s {task.build_prompt(doc)}"

    # Build memory string for final request (same as intermediate)
    if chunking_mode == "vst_original" and max_keep_memory > 0 and len(memory_parts) > (max_keep_memory + 1):
        memory_before_final = memory_parts[0] + "".join(memory_parts[-max_keep_memory:])
    else:
        memory_before_final = "".join(memory_parts)

    # Per-task system prompt override (e.g. StreamingBench uses "You are a helpful assistant.")
    final_system_prompt = sampling_cfg.get("system_prompt", _SYSTEM_PROMPT)

    video_input_complete_time = time.perf_counter()
    if video_items is not None:
        final_result = await client.chat_with_predecoded_video_ttft(
            text_prompt=final_video_text,
            video_item=video_items[final_chunk["chunk_id"]],
            sampling_params=sampling_params_final,
            system_prompt=final_system_prompt,
            memory_text=memory_before_final,
            mm_processor_kwargs=mm_processor_kwargs,
            final_text_message=final_question_text,
        )
    else:
        final_messages = _build_segment_messages(video_path, final_video_text, memory_before_final)
        # Override system prompt in messages if needed
        if final_system_prompt != _SYSTEM_PROMPT and final_messages:
            final_messages[0] = {"role": "system", "content": final_system_prompt}
        # Append the final question as separate user message
        final_messages.append({"role": "user", "content": final_question_text})
        processor_kwargs = {"max_num_frames": local_max_num_frames}
        if local_max_pixels:
            processor_kwargs["max_pixels"] = local_max_pixels
        final_result = await client.chat_with_ttft(final_messages, sampling_params_final, processor_kwargs)
    sample_complete_time = time.perf_counter()


    score_row = task.score_one(doc, final_result["text"])

    detailed_chunks.append({
        **final_chunk,
        "phase": "final_answer",
        "prompt_text": final_video_text + " | " + final_question_text,
        "memory_before_chars": len(memory_before_final),
        "final_output": final_result["text"],
        "latency_s": final_result["total_s"],
        "ttft_s": final_result["ttft_s"],
        "num_output_tokens": final_result["num_output_tokens"],
    })

    return {
        "sample_id": doc.get("question_id") or doc.get("id") or f"{doc.get('video')}::{doc.get('question', '')[:40]}",
        "video": doc.get("video"),
        "resolved_video_path": video_path,
        "question": doc.get("question"),
        "gt_answer": score_row.get("gt_answer"),
        "raw_gt_answer": doc.get("answer"),
        "prediction": score_row.get("pred_answer"),
        "raw_prediction": final_result["text"],
        "score": score_row.get("score"),
        "task": score_row.get("task"),
        "subtask": score_row.get("subtask"),
        "subsubtask": score_row.get("subsubtask"),
        "video_stats": {"total_frames": total_frames, "fps": fps, "duration_s": duration_s},
        "stream_config": {
            "chunking_mode": chunking_mode,
            "frame_sampling_mode": "global_sample_fps_max_frames",
            "sample_fps": sample_fps,
            "max_num_frames": max_num_frames,
            "max_pixels": max_pixels,
            "global_num_frames": len(global_frame_indices),
            "global_frame_shape": decoded_frame_shape,
            "global_estimated_visual_tokens": len(global_frame_indices) * tokens_per_frame,
            "global_frame_indices": global_frame_indices,
            "max_chunks": max_chunks,
            "num_chunks": len(chunks),
            "use_predecoded_video": use_predecoded_video,
            "decode_num_threads": decode_num_threads,
            "decode_timeout_s": decode_timeout_s,
            "stream_token_budget": stream_token_budget,
            "tokens_per_frame": tokens_per_frame,
            "max_memory_chars": max_memory_chars,
            # vst_original-only fields; None/unused under token_budget mode.
            "stream_think_times": stream_think_times if chunking_mode == "vst_original" else None,
            "picked_stream_think_times": picked_stream_think_times,
            "max_stream_vid_tokens": max_stream_vid_tokens if chunking_mode == "vst_original" else None,
            "max_keep_memory": max_keep_memory if chunking_mode == "vst_original" else None,
        },
        "memory_final": "".join(memory_parts),
        "memory_final_chars": len("".join(memory_parts)),
        "chunks": detailed_chunks,
        "timing": {
            "sample_total_s": sample_complete_time - sample_t0,
            "video_input_to_final_complete_s": sample_complete_time - video_input_complete_time,
            # Headline metric: TTFT of the final-answer request, i.e. the delay
            # between "video fully input" and "model starts responding".
            "ttft_s": final_result["ttft_s"],
            "ttft_source": "async_llm_stream_first_token_wall_clock",
        },
    }


async def _eval_one_sample_from_args(
    *,
    args,
    client,
    task,
    doc: dict,
    local_idx: int,
    total_docs: int,
    sampling_params_think,
    sampling_params_final,
    stream_think_times: list[int],
) -> dict:
    return await _eval_one_sample(
        client=client,
        task=task,
        doc=doc,
        sampling_params_think=sampling_params_think,
        sampling_params_final=sampling_params_final,
        max_num_frames=args.max_num_frames,
        max_pixels=args.max_pixels,
        sample_fps=args.sample_fps,
        chunking_mode=args.chunking_mode,
        stream_token_budget=args.stream_token_budget,
        tokens_per_frame=args.tokens_per_frame,
        max_chunks=args.max_chunks,
        max_memory_chars=args.max_memory_chars,
        stream_think_times=stream_think_times,
        max_stream_vid_tokens=args.max_stream_vid_tokens,
        max_keep_memory=args.max_keep_memory,
        use_predecoded_video=bool(args.use_predecoded_video),
        decode_num_threads=args.decode_num_threads,
        decode_timeout_s=args.decode_timeout_s,
        use_torchvision_resize=getattr(args, 'use_torchvision_resize', False),
        frame_sampling_mode=getattr(args, 'frame_sampling_mode', 'linspace'),
        progress_file=None,
        local_idx=local_idx,
        total_docs=total_docs,
    )


async def _run_concurrent_samples(
    *,
    args,
    client,
    task,
    docs: list[dict],
    sampling_params_think,
    sampling_params_final,
    stream_think_times: list[int],
    log_f,
) -> list[dict]:
    total_docs = len(docs)
    sample_concurrency = max(1, int(args.sample_concurrency))
    detail_by_idx: dict[int, dict] = {}
    next_idx = 0
    done_count = 0
    correct_count = 0
    scored_count = 0
    timeout_count = 0
    error_count = 0
    inflight: set[asyncio.Task] = set()

    async def _launch(local_idx: int, doc: dict) -> tuple[int, dict]:
        try:
            detail = await _eval_one_sample_from_args(
                args=args,
                client=client,
                task=task,
                doc=doc,
                local_idx=local_idx,
                total_docs=total_docs,
                sampling_params_think=sampling_params_think,
                sampling_params_final=sampling_params_final,
                stream_think_times=stream_think_times,
            )
        except asyncio.TimeoutError as e:
            detail = {
                "sample_id": doc.get("question_id") or doc.get("id") or f"{doc.get('video')}::{doc.get('question', '')[:40]}",
                "video": doc.get("video"),
                "resolved_video_path": "",
                "question": task.build_prompt(doc),
                "prediction": "",
                "raw_prediction": "",
                "gt_answer": "",
                "raw_gt_answer": "",
                "score": 0,
                "error": repr(e),
                "error_type": "timeout",
                "timing": {"ttft_s": None, "ttft_source": "error", "sample_total_s": None},
                "chunks": [],
                "stream_config": {"chunking_mode": args.chunking_mode, "sample_concurrency": sample_concurrency},
            }
        except Exception as e:
            detail = {
                "sample_id": doc.get("question_id") or doc.get("id") or f"{doc.get('video')}::{doc.get('question', '')[:40]}",
                "video": doc.get("video"),
                "resolved_video_path": "",
                "question": task.build_prompt(doc),
                "prediction": "",
                "raw_prediction": "",
                "gt_answer": "",
                "raw_gt_answer": "",
                "score": 0,
                "error": repr(e),
                "error_type": "exception",
                "timing": {"ttft_s": None, "ttft_source": "error", "sample_total_s": None},
                "chunks": [],
                "stream_config": {"chunking_mode": args.chunking_mode, "sample_concurrency": sample_concurrency},
            }
        return local_idx, detail

    def _start_more() -> None:
        nonlocal next_idx
        while next_idx < total_docs and len(inflight) < sample_concurrency:
            task_obj = asyncio.create_task(_launch(next_idx, docs[next_idx]))
            inflight.add(task_obj)
            next_idx += 1

    _start_more()
    write_progress(args.progress_file, {
        "phase": "concurrent_eval",
        "done": done_count,
        "total": total_docs,
        "inflight": len(inflight),
        "sample_concurrency": sample_concurrency,
        "correct": correct_count,
        "scored": scored_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
    })

    while inflight:
        done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
        for task_obj in done:
            local_idx, detail = await task_obj
            detail_by_idx[local_idx] = detail
            log_f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            log_f.flush()
            done_count += 1
            # Track errors/timeouts
            if detail.get("error_type") == "timeout":
                timeout_count += 1
            elif detail.get("error"):
                error_count += 1
            score = detail.get("score")
            if score is not None:
                scored_count += 1
                if score:
                    correct_count += 1
            print(
                f"[gpu{args.gpu_id}] stream sample {done_count}/{total_docs} "
                f"score={detail.get('score')} ttft_s={detail.get('timing', {}).get('ttft_s')}"
                f"{' [TIMEOUT]' if detail.get('error_type') == 'timeout' else ''}"
            )
        _start_more()
        write_progress(args.progress_file, {
            "phase": "concurrent_eval",
            "done": done_count,
            "total": total_docs,
            "inflight": len(inflight),
            "sample_concurrency": sample_concurrency,
            "correct": correct_count,
            "scored": scored_count,
            "timeout_count": timeout_count,
            "error_count": error_count,
        })

    return [detail_by_idx[i] for i in sorted(detail_by_idx)]


async def _retry_failed_samples(
    *,
    args,
    client,
    task,
    docs: list[dict],
    details: list[dict],
    sampling_params_think,
    sampling_params_final,
    stream_think_times: list[int],
    log_f,
    max_rounds: int = 3,
) -> tuple[list[dict], dict]:
    """对 details 中有 error 的样本进行串行重试，最多 max_rounds 轮。

    Returns:
        (updated_details, retry_stats) where retry_stats contains:
        - retry_rounds_used: int
        - retry_recovered_count: int
        - retry_still_failed_count: int
        - retry_still_failed_samples: list[str]
    """
    total_recovered = 0
    rounds_used = 0

    # Create a retry-specific args copy with serial execution and doubled timeout
    retry_args = copy.copy(args)
    retry_args.sample_concurrency = 1
    if retry_args.decode_timeout_s and retry_args.decode_timeout_s > 0:
        retry_args.decode_timeout_s = retry_args.decode_timeout_s * 2

    for round_num in range(1, max_rounds + 1):
        failed_indices = [i for i, d in enumerate(details) if d.get("error")]
        if not failed_indices:
            break
        rounds_used = round_num

        print(
            f"[gpu{args.gpu_id}] 🔄 Retry round {round_num}/{max_rounds}: "
            f"{len(failed_indices)} failed samples to retry (serial mode, "
            f"decode_timeout={retry_args.decode_timeout_s:.0f}s)",
            flush=True,
        )

        recovered = 0
        for local_idx in failed_indices:
            doc = docs[local_idx]
            try:
                detail = await _eval_one_sample_from_args(
                    args=retry_args,
                    client=client,
                    task=task,
                    doc=doc,
                    local_idx=local_idx,
                    total_docs=len(docs),
                    sampling_params_think=sampling_params_think,
                    sampling_params_final=sampling_params_final,
                    stream_think_times=stream_think_times,
                )
                # Success - replace the failed detail
                detail["retry_round"] = round_num
                detail["retry_recovered"] = True
                details[local_idx] = detail
                log_f.write(json.dumps({"_retry_round": round_num, **detail}, ensure_ascii=False) + "\n")
                log_f.flush()
                recovered += 1
                print(
                    f"[gpu{args.gpu_id}]   ✓ recovered sample {local_idx} "
                    f"(video={doc.get('video', 'unknown')})",
                    flush=True,
                )
            except asyncio.TimeoutError as e:
                details[local_idx]["error"] = repr(e)
                details[local_idx]["error_type"] = "timeout"
                details[local_idx]["retry_round"] = round_num
            except Exception as e:
                details[local_idx]["error"] = repr(e)
                details[local_idx]["error_type"] = "exception"
                details[local_idx]["retry_round"] = round_num

        total_recovered += recovered
        still_failed = len(failed_indices) - recovered
        print(
            f"[gpu{args.gpu_id}] 🔄 Retry round {round_num} done: "
            f"{recovered} recovered, {still_failed} still failed",
            flush=True,
        )
        if still_failed == 0:
            break

    # Compute final stats
    final_failed = [i for i, d in enumerate(details) if d.get("error")]
    still_failed_samples = [
        details[i].get("sample_id", f"idx={i}") for i in final_failed
    ]

    retry_stats = {
        "retry_rounds_used": rounds_used,
        "retry_recovered_count": total_recovered,
        "retry_still_failed_count": len(final_failed),
        "retry_still_failed_samples": still_failed_samples,
    }

    if rounds_used > 0:
        if final_failed:
            print(
                f"[gpu{args.gpu_id}] ⚠️  After {rounds_used} retry round(s): "
                f"{total_recovered} recovered, {len(final_failed)} still failed",
                flush=True,
            )
        else:
            print(
                f"[gpu{args.gpu_id}] ✅ All decode failures recovered after "
                f"{rounds_used} retry round(s) ({total_recovered} samples)",
                flush=True,
            )

    return details, retry_stats


async def _run_worker_async(args, task, docs) -> dict:
    from vllm_eval.async_chat_client import StreamChatClient
    from vllm import SamplingParams

    total_docs = len(docs)
    write_progress(args.progress_file, {"phase": "loading_model", "done": 0, "total": total_docs})

    t_load0 = time.perf_counter()
    client = StreamChatClient(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1},
        enforce_eager=True,
        dtype="bfloat16",
        allowed_local_media_path=task.video_root,
    )
    load_s = time.perf_counter() - t_load0
    write_progress(args.progress_file, {"phase": "ready", "done": 0, "total": total_docs, "load_s": load_s})

    sampling_params_think = SamplingParams(max_tokens=args.think_max_tokens, temperature=0.0, top_p=1.0)
    sampling_params_final = SamplingParams(
        max_tokens=task.generation_kwargs.get("max_new_tokens", 16),
        temperature=task.generation_kwargs.get("temperature", 0.0),
        top_p=1.0,
    )

    details = []
    seq_correct_count = 0
    seq_scored_count = 0
    seq_timeout_count = 0
    seq_error_count = 0
    stream_think_times = _resolve_stream_think_times(args.stream_think_times)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl_log).parent.mkdir(parents=True, exist_ok=True)
    t_eval0 = time.perf_counter()

    with open(args.jsonl_log, "w", encoding="utf-8") as log_f:
        if int(args.sample_concurrency) <= 1:
            for local_idx, doc in enumerate(docs):
                try:
                    detail = await _eval_one_sample(
                        client=client,
                        task=task,
                        doc=doc,
                        sampling_params_think=sampling_params_think,
                        sampling_params_final=sampling_params_final,
                        max_num_frames=args.max_num_frames,
                        max_pixels=args.max_pixels,
                        sample_fps=args.sample_fps,
                        chunking_mode=args.chunking_mode,
                        stream_token_budget=args.stream_token_budget,
                        tokens_per_frame=args.tokens_per_frame,
                        max_chunks=args.max_chunks,
                        max_memory_chars=args.max_memory_chars,
                        stream_think_times=stream_think_times,
                        max_stream_vid_tokens=args.max_stream_vid_tokens,
                        max_keep_memory=args.max_keep_memory,
                        use_predecoded_video=bool(args.use_predecoded_video),
                        decode_num_threads=args.decode_num_threads,
                        decode_timeout_s=args.decode_timeout_s,
                        use_torchvision_resize=getattr(args, 'use_torchvision_resize', False),
                        frame_sampling_mode=getattr(args, 'frame_sampling_mode', 'linspace'),
                        progress_file=args.progress_file,
                        local_idx=local_idx,
                        total_docs=total_docs,
                    )
                except asyncio.TimeoutError as e:
                    detail = {
                        "sample_id": doc.get("question_id") or doc.get("id") or f"{doc.get('video')}::{doc.get('question', '')[:40]}",
                        "video": doc.get("video"),
                        "resolved_video_path": "",
                        "question": task.build_prompt(doc),
                        "prediction": "",
                        "raw_prediction": "",
                        "gt_answer": "",
                        "raw_gt_answer": "",
                        "score": 0,
                        "error": repr(e),
                        "error_type": "timeout",
                        "timing": {"ttft_s": None, "ttft_source": "error", "sample_total_s": None},
                    }
                    seq_timeout_count += 1
                except Exception as e:
                    detail = {
                        "sample_id": doc.get("question_id") or doc.get("id") or f"{doc.get('video')}::{doc.get('question', '')[:40]}",
                        "video": doc.get("video"),
                        "resolved_video_path": "",
                        "question": task.build_prompt(doc),
                        "prediction": "",
                        "raw_prediction": "",
                        "gt_answer": "",
                        "raw_gt_answer": "",
                        "score": 0,
                        "error": repr(e),
                        "error_type": "exception",
                        "timing": {"ttft_s": None, "ttft_source": "error", "sample_total_s": None},
                    }
                    seq_error_count += 1
                details.append(detail)
                log_f.write(json.dumps(detail, ensure_ascii=False) + "\n")
                log_f.flush()
                score = detail.get("score")
                if score is not None:
                    seq_scored_count += 1
                    if score:
                        seq_correct_count += 1
                write_progress(args.progress_file, {
                    "phase": "done",
                    "done": local_idx + 1,
                    "total": total_docs,
                    "correct": seq_correct_count,
                    "scored": seq_scored_count,
                    "timeout_count": seq_timeout_count,
                    "error_count": seq_error_count,
                })
                print(
                    f"[gpu{args.gpu_id}] stream sample {local_idx+1}/{total_docs} "
                    f"score={detail.get('score')} ttft_s={detail.get('timing', {}).get('ttft_s')}"
                    f"{' [TIMEOUT]' if detail.get('error_type') == 'timeout' else ''}",
                    flush=True,
                )
        else:
            details = await _run_concurrent_samples(
                args=args,
                client=client,
                task=task,
                docs=docs,
                sampling_params_think=sampling_params_think,
                sampling_params_final=sampling_params_final,
                stream_think_times=stream_think_times,
                log_f=log_f,
            )

    eval_s = time.perf_counter() - t_eval0

    # --- Retry failed samples (decode errors / timeouts) ---
    max_retry_rounds = int(getattr(args, "max_retry_rounds", 3))
    first_pass_timeout = sum(1 for d in details if d.get("error_type") == "timeout")
    first_pass_error = sum(1 for d in details if d.get("error") and d.get("error_type") != "timeout")

    if (first_pass_timeout or first_pass_error) and max_retry_rounds > 0:
        print(
            f"[gpu{args.gpu_id}] ⚠️  First pass: {first_pass_timeout} timeout(s), "
            f"{first_pass_error} other error(s) out of {total_docs} samples "
            f"({100.0 * (first_pass_timeout + first_pass_error) / total_docs:.2f}% failure rate). "
            f"Starting retry (max {max_retry_rounds} rounds)...",
            flush=True,
        )
        with open(args.jsonl_log, "a", encoding="utf-8") as retry_log_f:
            details, retry_stats = await _retry_failed_samples(
                args=args,
                client=client,
                task=task,
                docs=docs,
                details=details,
                sampling_params_think=sampling_params_think,
                sampling_params_final=sampling_params_final,
                stream_think_times=stream_think_times,
                log_f=retry_log_f,
                max_rounds=max_retry_rounds,
            )
    else:
        retry_stats = {
            "retry_rounds_used": 0,
            "retry_recovered_count": 0,
            "retry_still_failed_count": 0,
            "retry_still_failed_samples": [],
        }

    # Count timeouts and errors across all details (post-retry)
    total_timeout_count = sum(1 for d in details if d.get("error_type") == "timeout")
    total_error_count = sum(1 for d in details if d.get("error") and d.get("error_type") != "timeout")

    rows = [
        {
            "pred_answer": detail.get("prediction", ""),
            "gt_answer": detail.get("gt_answer", ""),
            "score": detail.get("score", 0),
            "task": detail.get("task", task.name),
            "subtask": detail.get("subtask"),
            "subsubtask": detail.get("subsubtask"),
            "raw_pred": detail.get("raw_prediction", ""),
            "ttft_s": detail.get("timing", {}).get("ttft_s"),
            "sample_total_s": detail.get("timing", {}).get("sample_total_s"),
        }
        for detail in details
    ]
    client.shutdown()

    write_progress(args.progress_file, {
        "phase": "done",
        "done": total_docs,
        "total": total_docs,
        "load_s": load_s,
        "eval_s": eval_s,
        "timeout_count": total_timeout_count,
        "error_count": total_error_count,
    })

    if total_timeout_count or total_error_count:
        print(
            f"[gpu{args.gpu_id}] ⚠️  Finished with {total_timeout_count} timeout(s), "
            f"{total_error_count} other error(s) out of {total_docs} samples "
            f"({100.0 * (total_timeout_count + total_error_count) / total_docs:.2f}% failure rate)",
            flush=True,
        )

    return {
        "task": args.task,
        "gpu_id": args.gpu_id,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_docs": len(docs),
        "load_s": load_s,
        "eval_s": eval_s,
        "runtime_config": {
            "sample_concurrency": int(args.sample_concurrency),
            "decode_timeout_s": float(args.decode_timeout_s),
        },
        "timeout_count": total_timeout_count,
        "error_count": total_error_count,
        "failure_rate": (total_timeout_count + total_error_count) / total_docs if total_docs else 0.0,
        **retry_stats,
        "rows": rows,
    }


def main():
    _s = DEFAULT_STREAM
    _sampling = DEFAULT_SAMPLING
    _v = DEFAULT_VST_ORIGINAL
    _e_max_model_len = 20480
    _e_gpu_mem_util = 0.75
    try:
        from eval_config import DEFAULT_ENGINE
        _e_max_model_len = DEFAULT_ENGINE.max_model_len
        _e_gpu_mem_util = DEFAULT_ENGINE.gpu_memory_utilization
    except ImportError:
        pass
    _default_max_num_frames = _sampling.max_num_frames if _sampling else 32
    _default_max_pixels = _sampling.max_pixels if _sampling else None
    _default_sample_fps = _sampling.sample_fps if _sampling else 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--anno-path", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-num-frames", type=int, default=_default_max_num_frames, help="Global max sampled frames for the whole video (shared across all chunks)")
    parser.add_argument("--max-pixels", type=int, default=_default_max_pixels, help="Per-frame pixel cap passed to the Qwen-VL mm_processor_kwargs; caps visual tokens per frame regardless of native video resolution")
    parser.add_argument("--sample-fps", type=float, default=_default_sample_fps, help="Candidate sampling fps used to pick the whole video's frame set before chunking")
    parser.add_argument("--chunking-mode", default=getattr(_s, "chunking_mode", "token_budget") if _s else "token_budget", choices=["token_budget", "vst_original"], help="token_budget: accumulate estimated visual tokens to stream-token-budget. vst_original: VST-style duration-bucket stream-think-times + even split")
    parser.add_argument("--stream-token-budget", type=int, default=_s.stream_token_budget if _s else 5000, help="[token_budget mode] Accumulate estimated visual tokens up to this budget before flushing a chunk into an intermediate-thinking request")
    parser.add_argument("--tokens-per-frame", type=int, default=_s.tokens_per_frame if _s else 256, help="[token_budget mode] Estimated visual tokens per sampled frame; combined with --stream-token-budget to size each chunk")
    parser.add_argument("--max-chunks", type=int, default=_s.max_chunks if _s else 8, help="[token_budget mode] Safety ceiling on chunk count; excess token-budget boundaries are folded into the last chunk")
    parser.add_argument("--max-memory-chars", type=int, default=_s.max_memory_chars if _s else 12000, help="[token_budget mode] Memory truncation by character count")
    parser.add_argument("--stream-think-times", default=_v.stream_think_times if _v else "2-3-5-5", help="[vst_original mode] '-'-separated 4-value duration-bucket segment counts (or a single int), e.g. '2-3-5-5'")
    parser.add_argument("--max-stream-vid-tokens", type=int, default=_v.max_stream_vid_tokens if _v else 8192, help="[vst_original mode] Per-round visual token budget used by VST to size video_total_pixels; recorded for logging/parity, not used to gate vLLM chunking")
    parser.add_argument("--max-keep-memory", type=int, default=_v.max_keep_memory if _v else 0, help="[vst_original mode] Memory truncation by round count (<=0 = unlimited), mirroring get_textual_memory()")
    parser.add_argument("--think-max-tokens", type=int, default=_s.think_max_tokens if _s else 512)
    parser.add_argument("--use-predecoded-video", type=int, default=1 if (_s is None or getattr(_s, "use_predecoded_video", True)) else 0)
    parser.add_argument("--decode-num-threads", type=int, default=getattr(_s, "decode_num_threads", 4) if _s else 4)
    parser.add_argument("--sample-concurrency", type=int, default=1, help="Number of samples to keep in flight inside one GPU worker; 1 preserves the serial baseline")
    parser.add_argument("--decode-timeout-s", type=float, default=0.0, help="Optional per-sample probe/decode timeout in seconds; <=0 disables timeout")
    parser.add_argument("--max-model-len", type=int, default=_e_max_model_len)
    parser.add_argument("--gpu-memory-utilization", type=float, default=_e_gpu_mem_util)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jsonl-log", required=True)
    parser.add_argument("--progress-file", default=None, help="Optional JSON file this worker periodically overwrites with {phase,done,total} for the engine's progress display")
    # Precision-alignment options (read from eval_config.StreamConfig)
    parser.add_argument("--use-torchvision-resize", type=int,
                        default=1 if (_s is not None and getattr(_s, "use_torchvision_resize", False)) else 0,
                        help="1: use torchvision resize with antialias (matches official HF path). 0: PIL resize (faster)")
    parser.add_argument("--frame-sampling-mode",
                        default=getattr(_s, "frame_sampling_mode", "linspace") if _s else "linspace",
                        choices=["linspace", "strict_fps"],
                        help="strict_fps: official timestamp-based sampling. linspace: uniform frame-index sampling")
    parser.add_argument("--max-retry-rounds", type=int, default=3,
                        help="Max retry rounds for failed (decode error/timeout) samples. "
                             "Retries run serially to avoid IO contention. 0 disables retry.")
    args = parser.parse_args()

    # Convert int flags to bool for downstream use
    args.use_torchvision_resize = bool(args.use_torchvision_resize)


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    task = build_task(args.task, args.anno_path, args.video_root)
    docs = task.load_docs()
    if args.limit > 0:
        docs = docs[: args.limit]
    shard_docs = _load_shard(docs, args.shard_id, args.num_shards)

    report = asyncio.run(_run_worker_async(args, task, shard_docs))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    print(
        f"[gpu{args.gpu_id}] stream done: {report['num_docs']} docs, "
        f"load={report['load_s']:.1f}s eval={report['eval_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
