# SPDX-License-Identifier: Apache-2.0
"""Single-process clean latency runner for ContextFlow LMCache baselines.

This runner executes one condition and one repeat per Python process. It is
intended for clean E2E latency comparisons, not stage breakdown profiling.
"""

from __future__ import annotations

# Standard
import argparse
import contextlib
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

_FILE_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent for parent in _FILE_PATH.parents if (parent / "lmcache").is_dir()
)
EXPERIMENT_ROOT = next(
    parent
    for parent in _FILE_PATH.parents
    if (parent / "scripts").is_dir() and (parent / "src" / "common.py").is_file()
)
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Third Party
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

# Local
from common import (  # noqa: E402
    GPUMemorySampler,
    build_prompts,
    clear_lmcache_env,
    setup_common_env,
)


LMCACHE_METHODS = {
    "lmcache_prefix_chunked_reuse",
    "lmcache_naive_segment_reuse",
    "lmcache_blend",
}


def _set_extra_config(extra_config: dict[str, Any]) -> None:
    if extra_config:
        os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(
            extra_config,
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        os.environ.pop("LMCACHE_EXTRA_CONFIG", None)


def setup_lmcache_clean_env(
    *,
    method: str,
    blend_special_str: str,
    recompute_ratio: float,
    chunk_size: int,
    max_local_cpu_size_gb: float,
    enable_sparse: bool,
) -> None:
    """Configure LMCache for one clean latency method."""
    clear_lmcache_env()

    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(max_local_cpu_size_gb)

    extra_config: dict[str, Any] = {}
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        extra_config["enable_sparse"] = True

    if method == "lmcache_prefix_chunked_reuse":
        os.environ["LMCACHE_ENABLE_BLENDING"] = "False"
    elif method in {"lmcache_naive_segment_reuse", "lmcache_blend"}:
        os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
        os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
        os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
        os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(recompute_ratio)
        if method == "lmcache_naive_segment_reuse":
            extra_config["contextflow_naive_segment_reuse"] = True
    else:
        raise ValueError(f"Unsupported LMCache method: {method}")

    _set_extra_config(extra_config)


@contextlib.contextmanager
def build_plain_llm(
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    """Create a plain vLLM instance without LMCache."""
    clear_lmcache_env()

    llm_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        shutdown_vllm_engine(llm)
        cleanup_cuda_state()


@contextlib.contextmanager
def build_lmcache_llm(
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    """Create a vLLM instance with LMCacheConnectorV1 attached."""
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
        shutdown_vllm_engine(llm)
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        cleanup_cuda_state()


def shutdown_vllm_engine(llm: LLM) -> None:
    """Best-effort vLLM shutdown before destroying LMCache state."""
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if shutdown is None:
        return
    try:
        shutdown(timeout=5)
    except TypeError:
        try:
            shutdown()
        except Exception:
            pass
    except Exception:
        pass


def cleanup_cuda_state() -> None:
    """Best-effort cleanup for fresh-process latency isolation."""
    try:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


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


def _latency_fields(
    *,
    elapsed_sec: float,
    output_tokens: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    decode_sec = metrics.get("decode_time")
    num_generation_tokens = metrics.get("num_generation_tokens") or output_tokens

    engine_tpot_sec = None
    if decode_sec is not None and num_generation_tokens and num_generation_tokens > 1:
        engine_tpot_sec = decode_sec / (num_generation_tokens - 1)

    wall_tpot_sec = None
    if output_tokens > 0:
        wall_tpot_sec = elapsed_sec / output_tokens

    return {
        "engine_ttft_sec": metrics.get("first_token_latency"),
        "engine_prefill_sec": metrics.get("prefill_time"),
        "engine_decode_sec": decode_sec,
        "engine_inference_sec": metrics.get("inference_time"),
        "engine_tpot_sec": engine_tpot_sec,
        "wall_tpot_sec": wall_tpot_sec,
    }


def generate_timed(
    llm: LLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    request_label: str,
) -> dict[str, Any]:
    """Run one offline ``LLM.generate`` call and return timing metadata."""
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt_ids},
        sampling_params=params,
    )
    elapsed = time.time() - start

    request_output = outputs[0]
    completion = request_output.outputs[0]
    output_tokens = len(completion.token_ids)
    metrics = _request_metrics_to_dict(getattr(request_output, "metrics", None))

    return {
        "request_label": request_label,
        "elapsed_sec": elapsed,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "output_tokens": output_tokens,
        "output_text": completion.text,
        "num_cached_tokens": getattr(request_output, "num_cached_tokens", None),
        "metrics": metrics,
        **_latency_fields(
            elapsed_sec=elapsed,
            output_tokens=output_tokens,
            metrics=metrics,
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    setup_common_env()
    os.environ.setdefault("PYTHONHASHSEED", "0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(
        tokenizer=tokenizer,
        chunk_count=args.chunk_count,
        chunk_size_tokens=args.chunk_size_tokens,
        blend_special_str=args.blend_special_str,
    )
    warmup_prompt = prompts["warmup"]
    store_prompt = prompts["first"]
    measured_prompt = prompts["second"]

    result: dict[str, Any] = {
        "method": args.method,
        "repeat_id": args.repeat_id,
        "model": args.model,
        "chunk_count": args.chunk_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "store_prompt_tokens": len(store_prompt),
        "measured_prompt_tokens": len(measured_prompt),
        "prompt_tokens": len(measured_prompt),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "lmcache_recompute_ratio": (
            args.recompute_ratio if args.method == "lmcache_blend" else None
        ),
        "lmcache_chunk_size": (
            args.lmcache_chunk_size if args.method in LMCACHE_METHODS else None
        ),
        "success": False,
        "error": None,
    }

    sampler = GPUMemorySampler(
        gpu_index=args.gpu_index,
        interval_sec=args.memory_sample_interval_sec,
    )

    try:
        cleanup_cuda_state()
        sampler.start()

        if args.method == "full_recompute":
            context = build_plain_llm(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        elif args.method in LMCACHE_METHODS:
            setup_lmcache_clean_env(
                method=args.method,
                blend_special_str=args.blend_special_str,
                recompute_ratio=args.recompute_ratio,
                chunk_size=args.lmcache_chunk_size,
                max_local_cpu_size_gb=args.max_local_cpu_size_gb,
                enable_sparse=args.enable_sparse,
            )
            context = build_lmcache_llm(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

        with context as llm:
            warmup = generate_timed(llm, warmup_prompt, 1, "engine_warmup")

            store = None
            if args.method in LMCACHE_METHODS:
                store = generate_timed(llm, store_prompt, 1, "store_cache")
                time.sleep(args.sleep_between_requests)

            measured = generate_timed(
                llm,
                measured_prompt,
                args.max_new_tokens,
                "measured_reuse" if args.method in LMCACHE_METHODS else "measured",
            )

        result.update(
            {
                "success": True,
                "warmup": warmup,
                "store": store,
                "measured": measured,
                "generation_sec": measured["elapsed_sec"],
                "engine_ttft_sec": measured["engine_ttft_sec"],
                "engine_prefill_sec": measured["engine_prefill_sec"],
                "engine_decode_sec": measured["engine_decode_sec"],
                "engine_inference_sec": measured["engine_inference_sec"],
                "engine_tpot_sec": measured["engine_tpot_sec"],
                "wall_tpot_sec": measured["wall_tpot_sec"],
                "output_tokens": measured["output_tokens"],
                "output_text": measured["output_text"],
                "num_cached_tokens": measured["num_cached_tokens"],
            }
        )

    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()

    finally:
        sampler.stop()
        result.update(
            {
                "gpu_memory_available": sampler.available,
                "gpu_memory_error": sampler.error,
                "gpu_start_mb": sampler.start_mb,
                "gpu_peak_mb": sampler.peak_mb,
                "gpu_end_mb": sampler.end_mb,
                "gpu_peak_delta_mb": sampler.peak_mb - sampler.start_mb,
            }
        )
        cleanup_cuda_state()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=[
            "full_recompute",
            "lmcache_prefix_chunked_reuse",
            "lmcache_naive_segment_reuse",
            "lmcache_blend",
        ],
        required=True,
    )
    parser.add_argument("--repeat-id", type=int, default=0)
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
    parser.add_argument("--max-local-cpu-size-gb", type=float, default=5.0)
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument("--sleep-between-requests", type=float, default=1.0)
    parser.add_argument("--memory-sample-interval-sec", type=float, default=0.05)

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
