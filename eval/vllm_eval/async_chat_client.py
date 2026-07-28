"""
Thin async wrapper around vLLM's AsyncLLM that exposes real per-request TTFT
(time from request submission to first generated token) by streaming
RequestOutput objects and timestamping the first one that carries a token.

We need this instead of the synchronous LLM.chat()/LLM.generate() because,
as verified empirically (see worker_stream.py header comment / analysis in
VLLM_MIGRATION_ANALYSIS.md), RequestOutput.metrics is not populated on the
sync engine path in vLLM 0.11.0 even with disable_log_stats=False -- it is
only meaningfully driven through the AsyncLLM streaming generator, where we
can measure TTFT ourselves from wall-clock timestamps around the first
yielded token, which is equivalent to (and just as accurate as) the engine's
internal first_token_time bookkeeping for the purposes of this evaluation.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.chat_utils import (
    apply_hf_chat_template,
    parse_chat_messages,
    resolve_chat_template_content_format,
)
from vllm.v1.engine.async_llm import AsyncLLM


class StreamChatClient:
    """Wraps a single AsyncLLM instance and exposes a sync-friendly
    `chat_with_ttft()` coroutine that returns (text, ttft_s, total_s).

    One instance owns one GPU (set CUDA_VISIBLE_DEVICES before constructing).
    """

    def __init__(self, **engine_kwargs: Any) -> None:
        engine_kwargs.setdefault("disable_log_stats", False)
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self._tokenizer = None
        self._model_config = None

    async def _ensure_meta(self):
        if self._tokenizer is None:
            self._tokenizer = await self.engine.get_tokenizer()
        if self._model_config is None:
            self._model_config = self.engine.model_config

    async def chat_with_ttft(
        self,
        messages: list[dict],
        sampling_params: SamplingParams,
        mm_processor_kwargs: Optional[dict] = None,
    ) -> dict:
        """Submit one chat conversation, stream tokens, and time TTFT.

        Returns a dict: {"text": str, "ttft_s": float, "total_s": float,
        "num_output_tokens": int, "submit_wall_time": float,
        "first_token_wall_time": float, "finish_wall_time": float}.
        """
        await self._ensure_meta()

        content_format = resolve_chat_template_content_format(
            None, None, "auto", self._tokenizer, model_config=self._model_config,
        )
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages, self._model_config, self._tokenizer, content_format,
        )
        prompt_str = apply_hf_chat_template(
            self._tokenizer,
            conversation=conversation,
            chat_template=None,
            tools=None,
            model_config=self._model_config,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt: dict[str, Any] = {"prompt": prompt_str}
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs

        request_id = uuid.uuid4().hex
        submit_wall_time = time.perf_counter()
        first_token_wall_time = None
        final_text = ""
        num_output_tokens = 0

        async for out in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if out.outputs:
                token_ids = out.outputs[0].token_ids
                if first_token_wall_time is None and len(token_ids) > 0:
                    first_token_wall_time = time.perf_counter()
                final_text = out.outputs[0].text
                num_output_tokens = len(token_ids)
            if out.finished:
                break

        finish_wall_time = time.perf_counter()
        ttft_s = (first_token_wall_time - submit_wall_time) if first_token_wall_time is not None else None

        return {
            "text": final_text.strip(),
            "ttft_s": ttft_s,
            "total_s": finish_wall_time - submit_wall_time,
            "num_output_tokens": num_output_tokens,
            "submit_wall_time": submit_wall_time,
            "first_token_wall_time": first_token_wall_time,
            "finish_wall_time": finish_wall_time,
        }

    async def chat_with_predecoded_video_ttft(
        self,
        *,
        text_prompt: str,
        video_item: tuple[Any, dict[str, Any]],
        sampling_params: SamplingParams,
        system_prompt: Optional[str] = None,
        memory_text: Optional[str] = None,
        mm_processor_kwargs: Optional[dict] = None,
        final_text_message: Optional[str] = None,
    ) -> dict:
        """Submit a chat request using already-decoded video frames.

        This bypasses vLLM's video_url media loader, which re-decodes the full
        video file for every request. We still ask vLLM's chat parser to insert
        the model-specific video placeholder by using an empty video_url part,
        then replace the parsed video item with our predecoded `(frames,
        metadata)` tuple.

        Message structure mirrors the official VST stream_think eval:
          message[0] = {"role": "system", "content": system_prompt}
          message[1] = {"role": "previous text", "content": memory_text}
          message[2] = {"role": "user", "content": [{"text": text_prompt}, {"video": ...}]}
          message[3] = {"role": "user", "content": final_text_message}  (final answer only)
        """
        await self._ensure_meta()

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if memory_text:
            messages.append({"role": "previous text", "content": memory_text})
        # User message with video chunk + text prefix
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": text_prompt},
            {"type": "video_url", "video_url": {"url": ""}},
        ]
        messages.append({"role": "user", "content": user_content})
        # Optional final text-only user message (for final answer requests)
        if final_text_message:
            messages.append({"role": "user", "content": final_text_message})

        content_format = resolve_chat_template_content_format(
            None, None, "auto", self._tokenizer, model_config=self._model_config,
        )
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages, self._model_config, self._tokenizer, content_format,
        )
        prompt_str = apply_hf_chat_template(
            self._tokenizer,
            conversation=conversation,
            chat_template=None,
            tools=None,
            model_config=self._model_config,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt: dict[str, Any] = {
            "prompt": prompt_str,
            "multi_modal_data": {"video": [video_item]},
        }
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs

        request_id = uuid.uuid4().hex
        submit_wall_time = time.perf_counter()
        first_token_wall_time = None
        final_text = ""
        num_output_tokens = 0

        async for out in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if out.outputs:
                token_ids = out.outputs[0].token_ids
                if first_token_wall_time is None and len(token_ids) > 0:
                    first_token_wall_time = time.perf_counter()
                final_text = out.outputs[0].text
                num_output_tokens = len(token_ids)
            if out.finished:
                break

        finish_wall_time = time.perf_counter()
        ttft_s = (first_token_wall_time - submit_wall_time) if first_token_wall_time is not None else None
        return {
            "text": final_text,
            "ttft_s": ttft_s,
            "total_s": finish_wall_time - submit_wall_time,
            "num_output_tokens": num_output_tokens,
            "submit_wall_time": submit_wall_time,
            "first_token_wall_time": first_token_wall_time,
            "finish_wall_time": finish_wall_time,
        }

    def shutdown(self):
        self.engine.shutdown()
