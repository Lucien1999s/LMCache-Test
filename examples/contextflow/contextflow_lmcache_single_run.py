# SPDX-License-Identifier: Apache-2.0
"""
ContextFlow LMCache CacheBlend single-condition diagnostic runner.

Runs one condition per process:
- method: full_recompute or lmcache_blend
- chunk_count: e.g., 2, 4, 8, 16
- chunk_size_tokens: e.g., 512
- model: Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-7B-Instruct

This is a first-stage diagnostic:
- generation_sec is full offline generate time, not true streaming TTFT.
- gpu_peak_mb includes model loading and vLLM KV-cache preallocation.
- Run one condition per Python process for cleaner memory/cache state.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynvml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.contextflow_profiler import (
    configure as configure_contextflow_profiler,
    record_event as cf_record_event,
    span as cf_span,
)


LMCACHE_ENV_KEYS = [
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_ENABLE_BLENDING",
    "LMCACHE_BLEND_SPECIAL_STR",
    "LMCACHE_USE_LAYERWISE",
    "LMCACHE_BLEND_CHECK_LAYERS",
    "LMCACHE_BLEND_RECOMPUTE_RATIOS",
    "LMCACHE_LOCAL_CPU",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
    "LMCACHE_LOCAL_DISK",
    "LMCACHE_MAX_LOCAL_DISK_SIZE",
    "LMCACHE_EXTRA_CONFIG",
]


def clear_lmcache_env() -> None:
    for key in LMCACHE_ENV_KEYS:
        os.environ.pop(key, None)


def setup_common_env() -> None:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf-cache"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/hf-cache/hub"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Force short tmp paths for LMCache ZMQ IPC sockets.
    # Long TMPDIR paths can exceed sockaddr_un.sun_path and break lookup_client init.
    os.environ["TMPDIR"] = "/tmp/cf"
    os.environ["TMP"] = "/tmp/cf"
    os.environ["TEMP"] = "/tmp/cf"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    Path("/tmp/cf").mkdir(parents=True, exist_ok=True)


def setup_lmcache_blend_env(
    blend_special_str: str,
    recompute_ratio: float,
    chunk_size: int = 256,
    use_disk: bool = False,
    enable_sparse: bool = False,
) -> None:
    clear_lmcache_env()

    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(recompute_ratio)

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_plain_llm(model: str, max_model_len: int, gpu_memory_utilization: float):
    clear_lmcache_env()

    llm_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(llm_args))
    yield llm


@contextlib.contextmanager
def build_lmcache_llm(model: str, max_model_len: int, gpu_memory_utilization: float):
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


class GPUMemorySampler:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.02):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = False
        self.peak_mb = 0.0
        self.start_mb = 0.0
        self.end_mb = 0.0

    def _sample_once_mb(self) -> float:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.used) / (1024 * 1024)

    def start(self) -> None:
        pynvml.nvmlInit()
        self.started = True
        self.start_mb = self._sample_once_mb()
        self.peak_mb = self.start_mb

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.peak_mb = max(self.peak_mb, self._sample_once_mb())
                except Exception:
                    pass
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.end_mb = self._sample_once_mb()
        pynvml.nvmlShutdown()


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def make_synthetic_chunk(tokenizer, idx: int, target_tokens: int) -> list[int]:
    base = (
        f"Document chunk {idx}. "
        f"This chunk contains synthetic evidence about topic {idx}. "
        f"The important fact in chunk {idx} is value_{idx}. "
        f"Use this information only when the question asks about topic {idx}. "
    )
    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(encode_no_special(tokenizer, base))
    return ids[:target_tokens]


def build_prompts(
    tokenizer,
    chunk_count: int,
    chunk_size_tokens: int,
    blend_special_str: str,
) -> dict[str, list[int]]:
    sys_prompt = encode_no_special(
        tokenizer,
        "You are a helpful assistant. Answer the question using the provided chunks. ",
    )

    first_query = encode_no_special(
        tokenizer,
        "Question: What is the name of the assistant? Answer:",
    )
    second_query = encode_no_special(
        tokenizer,
        "Question: Which chunks contain useful evidence? Answer:",
    )
    third_query = encode_no_special(
        tokenizer,
        "Question: Summarize the useful facts from the provided chunks. Answer:",
    )

    warmup = encode_no_special(tokenizer, "Nice to meet you. " * 200)

    chunks = [
        make_synthetic_chunk(tokenizer, i, chunk_size_tokens)
        for i in range(chunk_count)
    ]

    sep = encode_no_special(tokenizer, blend_special_str)

    order_first = list(range(chunk_count))
    if chunk_count >= 2:
        order_blend = [1, 0] + list(range(2, chunk_count))
    else:
        order_blend = order_first

    def concat(order: list[int], query: list[int]) -> list[int]:
        prompt = list(sys_prompt)
        prompt += sep
        for idx in order:
            prompt += chunks[idx]
            prompt += sep
        prompt += query
        return prompt

    return {
        "warmup": warmup,
        "first": concat(order_first, first_query),
        "second": concat(order_blend, second_query),
        "third": concat(order_blend, third_query),
    }


def _request_metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    keys = [
        "arrival_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
        "first_token_latency",
        "num_generation_tokens",
    ]
    values = {key: getattr(metrics, key, None) for key in keys}
    scheduled_ts = values.get("scheduled_ts") or 0.0
    first_token_ts = values.get("first_token_ts") or 0.0
    last_token_ts = values.get("last_token_ts") or 0.0
    if scheduled_ts and first_token_ts:
        values["prefill_time"] = first_token_ts - scheduled_ts
    if first_token_ts and last_token_ts:
        values["decode_time"] = last_token_ts - first_token_ts
    if scheduled_ts and last_token_ts:
        values["inference_time"] = last_token_ts - scheduled_ts
    return values


def generate_once(
    llm: LLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    request_label: str,
) -> tuple[str, float, int, dict[str, Any]]:
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    with cf_span(
        "offline_llm_generate_call",
        category="runner",
        device="CPU/GPU",
        metadata={
            "request_label": request_label,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
        },
    ):
        start = time.time()
        outputs = llm.generate(
            prompts={"prompt_token_ids": prompt_ids},
            sampling_params=params,
        )
        elapsed = time.time() - start
    text = outputs[0].outputs[0].text
    output_tokens = len(outputs[0].outputs[0].token_ids)
    metrics = _request_metrics_to_dict(getattr(outputs[0], "metrics", None))
    cf_record_event(
        "vllm_request_metrics",
        category="runner",
        device="CPU/GPU",
        metadata={
            "request_label": request_label,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "output_tokens": output_tokens,
            "num_cached_tokens": getattr(outputs[0], "num_cached_tokens", None),
            "metrics": metrics,
        },
    )
    return text, elapsed, output_tokens, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    setup_common_env()

    if args.stage_profile_output:
        condition = {
            "method": args.method,
            "model": args.model,
            "chunk_count": args.chunk_count,
            "chunk_size_tokens": args.chunk_size_tokens,
            "max_new_tokens": args.max_new_tokens,
            "recompute_ratio": args.recompute_ratio,
            "lmcache_chunk_size": args.lmcache_chunk_size,
            "enable_sparse": args.enable_sparse,
        }
        if args.stage_profile_reset_output:
            Path(args.stage_profile_output).unlink(missing_ok=True)
        configure_contextflow_profiler(
            args.stage_profile_output,
            enabled=True,
            cuda_sync=args.stage_profile_cuda_sync,
            include_memory=not args.stage_profile_no_memory,
            run_id=args.stage_profile_run_id or str(uuid.uuid4()),
            condition=condition,
        )

    with cf_span(
        "prompt_tokenizer_load",
        category="runner",
        device="CPU",
        metadata={"model": args.model},
    ):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    with cf_span(
        "prompt_build_and_tokenize",
        category="runner",
        device="CPU",
        metadata={
            "chunk_count": args.chunk_count,
            "chunk_size_tokens": args.chunk_size_tokens,
        },
    ):
        prompts = build_prompts(
            tokenizer=tokenizer,
            chunk_count=args.chunk_count,
            chunk_size_tokens=args.chunk_size_tokens,
            blend_special_str=args.blend_special_str,
        )

    result: dict[str, Any] = {
        "method": args.method,
        "model": args.model,
        "chunk_count": args.chunk_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "prompt_tokens": len(prompts["third"]),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "lmcache_recompute_ratio": args.recompute_ratio if args.method == "lmcache_blend" else None,
        "lmcache_chunk_size": args.lmcache_chunk_size if args.method == "lmcache_blend" else None,
        "success": False,
        "error": None,
    }

    sampler = GPUMemorySampler(gpu_index=args.gpu_index)

    try:
        sampler.start()

        if args.method == "lmcache_blend":
            setup_lmcache_blend_env(
                blend_special_str=args.blend_special_str,
                recompute_ratio=args.recompute_ratio,
                chunk_size=args.lmcache_chunk_size,
                use_disk=args.use_disk,
                enable_sparse=args.enable_sparse,
            )
            context = build_lmcache_llm(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        elif args.method == "full_recompute":
            clear_lmcache_env()
            context = build_plain_llm(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

        with context as llm:
            _, warmup_sec, _, warmup_metrics = generate_once(
                llm, prompts["warmup"], 1, "warmup"
            )

            if args.method == "lmcache_blend":
                _, first_sec, _, first_metrics = generate_once(
                    llm, prompts["first"], 1, "first_store"
                )
                time.sleep(args.sleep_between_requests)
                _, second_sec, _, second_metrics = generate_once(
                    llm, prompts["second"], 1, "second_blend_warmup"
                )
                time.sleep(args.sleep_between_requests)
                measured_prompt = prompts["third"]
            else:
                first_sec = None
                second_sec = None
                first_metrics = None
                second_metrics = None
                measured_prompt = prompts["third"]

            output_text, generation_sec, output_tokens, generation_metrics = generate_once(
                llm,
                measured_prompt,
                args.max_new_tokens,
                "measured",
            )

        result.update(
            {
                "success": True,
                "warmup_sec": warmup_sec,
                "first_sec": first_sec,
                "second_sec": second_sec,
                "generation_sec": generation_sec,
                "warmup_metrics": warmup_metrics,
                "first_metrics": first_metrics,
                "second_metrics": second_metrics,
                "generation_metrics": generation_metrics,
                "output_tokens": output_tokens,
                "output_text": output_text,
                "stage_profile_output": args.stage_profile_output,
            }
        )

    except Exception as e:
        result["error"] = repr(e)

    finally:
        sampler.stop()
        result.update(
            {
                "gpu_start_mb": sampler.start_mb,
                "gpu_peak_mb": sampler.peak_mb,
                "gpu_end_mb": sampler.end_mb,
                "gpu_peak_delta_mb": sampler.peak_mb - sampler.start_mb,
            }
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", choices=["full_recompute", "lmcache_blend"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=32648)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gpu-index", type=int, default=0)

    parser.add_argument("--blend-special-str", default="# #")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--use-disk", action="store_true")
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument("--sleep-between-requests", type=float, default=1.0)

    parser.add_argument("--stage-profile-output", default=None)
    parser.add_argument("--stage-profile-run-id", default=None)
    parser.add_argument("--stage-profile-cuda-sync", action="store_true")
    parser.add_argument("--stage-profile-no-memory", action="store_true")
    parser.add_argument("--stage-profile-reset-output", action="store_true")

    parser.add_argument("--output", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
