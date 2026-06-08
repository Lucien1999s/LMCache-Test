# SPDX-License-Identifier: Apache-2.0
"""Single-condition layer-wise KV drift diagnostic runner."""

from __future__ import annotations

# Standard
import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

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
from transformers import AutoTokenizer

# First Party
from lmcache.v1.contextflow_profiler import configure as configure_contextflow_profiler

# Local
from common import (  # noqa: E402
    GPUMemorySampler,
    build_lmcache_llm,
    build_prompts,
    generate_once,
    setup_common_env,
    setup_lmcache_blend_env,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one layer-wise drift diagnostic condition."""
    setup_common_env()
    os.environ["LMCACHE_CONTEXTFLOW_DRIFT_ANALYSIS"] = "1"

    run_id = args.profile_run_id or str(uuid.uuid4())
    condition = {
        "experiment": "drift_analysis",
        "method": "lmcache_blend",
        "model": args.model,
        "chunk_count": args.chunk_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "max_new_tokens": args.max_new_tokens,
        "recompute_ratio": args.recompute_ratio,
        "lmcache_chunk_size": args.lmcache_chunk_size,
        "blend_check_layers": args.blend_check_layers,
        "enable_sparse": args.enable_sparse,
    }
    if args.profile_reset_output:
        Path(args.profile_output).unlink(missing_ok=True)
    configure_contextflow_profiler(
        args.profile_output,
        enabled=True,
        cuda_sync=args.cuda_sync,
        include_memory=not args.no_profile_memory,
        run_id=run_id,
        condition=condition,
    )

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
        "run_id": run_id,
        "method": "lmcache_blend",
        "model": args.model,
        "chunk_count": args.chunk_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "store_prompt_tokens": len(store_prompt),
        "measured_prompt_tokens": len(measured_prompt),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "lmcache_recompute_ratio": args.recompute_ratio,
        "lmcache_chunk_size": args.lmcache_chunk_size,
        "blend_check_layers": args.blend_check_layers,
        "drift_profile_output": args.profile_output,
        "success": False,
        "error": None,
    }

    sampler = GPUMemorySampler(
        gpu_index=args.gpu_index,
        interval_sec=args.memory_sample_interval_sec,
    )

    try:
        setup_lmcache_blend_env(
            blend_special_str=args.blend_special_str,
            recompute_ratio=args.recompute_ratio,
            chunk_size=args.lmcache_chunk_size,
            use_disk=False,
            enable_sparse=args.enable_sparse,
        )
        os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = args.blend_check_layers

        context = build_lmcache_llm(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        with context as llm:
            sampler.start()
            warmup_result = None
            if not args.no_warmup:
                warmup_result = _run_request(llm, warmup_prompt, 1, "engine_warmup")

            store_result = _run_request(llm, store_prompt, 1, "store_cache")
            time.sleep(args.sleep_between_requests)
            measured_result = _run_request(
                llm,
                measured_prompt,
                args.max_new_tokens,
                "measured_reuse",
            )

        result.update(
            {
                "success": True,
                "warmup": warmup_result,
                "store": store_result,
                "measured": measured_result,
                "generation_sec": measured_result["elapsed_sec"],
                "output_tokens": measured_result["output_tokens"],
                "output_text": measured_result["output_text"],
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

    return result


def _run_request(
    llm: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    label: str,
) -> dict:
    text, elapsed_sec, output_tokens, metrics = generate_once(
        llm,
        prompt_ids,
        max_new_tokens,
        label,
    )
    return {
        "request_label": label,
        "elapsed_sec": elapsed_sec,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "output_tokens": output_tokens,
        "output_text": text,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32648)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gpu-index", type=int, default=0)

    parser.add_argument("--blend-special-str", default="# #")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--blend-check-layers", default="1")
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument("--sleep-between-requests", type=float, default=1.0)
    parser.add_argument("--memory-sample-interval-sec", type=float, default=0.05)

    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--profile-run-id", default=None)
    parser.add_argument("--profile-reset-output", action="store_true")
    parser.add_argument("--cuda-sync", action="store_true")
    parser.add_argument("--no-profile-memory", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """Run the requested condition and append one JSONL record."""
    args = parse_args()
    result = run(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
