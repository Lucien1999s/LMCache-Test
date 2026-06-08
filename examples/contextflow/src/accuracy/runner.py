# SPDX-License-Identifier: Apache-2.0
"""Run one ContextFlow accuracy condition.

The runner consumes a precomputed chunk-selection JSONL file and evaluates one
method/dataset/chunk-count/selection-mode condition in a fresh Python process.
"""

from __future__ import annotations

# Standard
import argparse
import json
import os
import re
import string
import sys
import time
import traceback
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
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Local
from accuracy.selection import chunk_context, to_answer_list
from clean_latency import (  # noqa: E402
    LMCACHE_METHODS,
    build_lmcache_llm,
    build_plain_llm,
    cleanup_cuda_state,
    setup_lmcache_clean_env,
)
from common import (  # noqa: E402
    clear_lmcache_env,
    encode_no_special,
    setup_common_env,
)


def normalize_text(text: str) -> str:
    """Normalize answer text for EM/F1 computation."""
    text = str(text).lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 against one ground-truth answer."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in pred_tokens:
        counts[token] = counts.get(token, 0) + 1

    overlap = 0
    for token in gold_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute normalized exact match against one answer."""
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def contains_match(prediction: str, ground_truth: str) -> float:
    """Compute normalized containment against one answer."""
    prediction_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)
    if not prediction_norm or not gold_norm:
        return 0.0
    return float(gold_norm in prediction_norm or prediction_norm in gold_norm)


def max_metric(prediction: str, answers: list[str], metric_fn: Any) -> float:
    """Return the best score over all acceptable answers."""
    if not answers:
        return 0.0
    return max(metric_fn(prediction, str(answer)) for answer in answers)


def postprocess_prediction(text: str) -> str:
    """Extract a compact answer span from model output."""
    text = str(text).strip()
    if not text:
        return ""
    first_line = re.split(r"[\r\n]", text, maxsplit=1)[0].strip()
    first_line = re.sub(
        r"^(final\s+answer|answer)\s*:\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip(" \t\"'")


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
    generation_tokens = metrics.get("num_generation_tokens") or output_tokens

    engine_tpot_sec = None
    if decode_sec is not None and generation_tokens and generation_tokens > 1:
        engine_tpot_sec = decode_sec / (generation_tokens - 1)

    wall_tpot_sec = elapsed_sec / output_tokens if output_tokens > 0 else None

    return {
        "engine_ttft_sec": metrics.get("first_token_latency"),
        "engine_prefill_sec": metrics.get("prefill_time"),
        "engine_decode_sec": decode_sec,
        "engine_inference_sec": metrics.get("inference_time"),
        "engine_tpot_sec": engine_tpot_sec,
        "wall_tpot_sec": wall_tpot_sec,
    }


def generate_qa_timed(
    llm: LLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    request_label: str,
) -> dict[str, Any]:
    """Generate once and return output text plus timing metadata."""
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["\n", "#"],
    )
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
        "prediction_text": postprocess_prediction(completion.text),
        "num_cached_tokens": getattr(request_output, "num_cached_tokens", None),
        "metrics": metrics,
        **_latency_fields(
            elapsed_sec=elapsed,
            output_tokens=output_tokens,
            metrics=metrics,
        ),
    }


def build_qa_prompt_ids(
    *,
    tokenizer: Any,
    selected_chunks: list[dict[str, Any]],
    question: str,
    blend_special_str: str,
    reorder_for_measured: bool,
) -> list[int]:
    """Build a stable segment prompt for cache store or measured QA."""
    system_text = (
        "You are a careful extractive QA assistant. Use only the provided "
        "context chunks. Return the shortest exact answer span from the "
        "context whenever possible. Do not explain. Do not add locations, "
        "dates, alternatives, or extra words unless they are part of the "
        "answer. For yes/no questions, answer only yes or no.\n"
    )
    sep_ids = encode_no_special(tokenizer, blend_special_str)

    ordered_chunks = sorted(selected_chunks, key=lambda chunk: int(chunk["index"]))
    if reorder_for_measured and len(ordered_chunks) >= 2:
        ordered_chunks = [ordered_chunks[1], ordered_chunks[0]] + ordered_chunks[2:]

    prompt = encode_no_special(tokenizer, system_text)
    prompt += sep_ids
    for chunk in ordered_chunks:
        stable_header = f"[Context original_chunk={int(chunk['index'])}]\n"
        prompt += encode_no_special(tokenizer, stable_header)
        prompt += list(chunk["ids"])
        prompt += sep_ids

    query_text = (
        f"\nQuestion: {question}\n"
        "Shortest exact answer span, within 5 words:\n"
        "Answer:"
    )
    prompt += encode_no_special(tokenizer, query_text)
    return prompt


def load_selection_rows(
    *,
    selection_file: Path,
    dataset_name: str,
    chunk_count: int,
    selection_mode: str,
    limit: int | None,
    start_index: int | None,
) -> list[dict[str, Any]]:
    """Load and filter selection rows for one run condition."""
    rows = []
    for line in selection_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != dataset_name:
            continue
        if int(row.get("chunk_count", -1)) != chunk_count:
            continue
        if row.get("selection_mode") != selection_mode:
            continue
        example_index = int(row["example_index"])
        if start_index is not None and example_index < start_index:
            continue
        if limit is not None and start_index is not None:
            if example_index >= start_index + limit:
                continue
        rows.append(row)

    rows.sort(key=lambda row: int(row["example_index"]))
    if limit is not None and start_index is None:
        rows = rows[:limit]
    return rows


def build_context(args: argparse.Namespace) -> Any:
    """Create the vLLM/LMCache context manager for one method."""
    if args.method == "full_recompute":
        clear_lmcache_env()
        return build_plain_llm(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if args.method in LMCACHE_METHODS:
        setup_lmcache_clean_env(
            method=args.method,
            blend_special_str=args.blend_special_str,
            recompute_ratio=args.recompute_ratio,
            chunk_size=args.lmcache_chunk_size,
            max_local_cpu_size_gb=args.max_local_cpu_size_gb,
            enable_sparse=args.enable_sparse,
        )
        return build_lmcache_llm(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    raise ValueError(f"Unknown method: {args.method}")


def run(args: argparse.Namespace) -> None:
    """Run one accuracy condition and append per-example JSONL results."""
    setup_common_env()
    os.environ.setdefault("PYTHONHASHSEED", "0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = pd.read_parquet(args.dataset_path)
    selection_rows = load_selection_rows(
        selection_file=Path(args.selection_file),
        dataset_name=args.dataset_name,
        chunk_count=args.chunk_count,
        selection_mode=args.selection_mode,
        limit=args.limit,
        start_index=args.start_index,
    )
    if not selection_rows:
        raise RuntimeError("No matching selection rows found")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleanup_cuda_state()
    with build_context(args) as llm, output_path.open("a") as f:
        if args.warmup:
            warmup_prompt = encode_no_special(tokenizer, "Warm up the engine. " * 64)
            generate_qa_timed(llm, warmup_prompt, 1, "engine_warmup")

        for selection in selection_rows:
            example_index = int(selection["example_index"])
            example = dataset.iloc[example_index].to_dict()
            question = str(example.get("input", ""))
            context_text = str(example.get("context", ""))
            answers = to_answer_list(example.get("answers"))

            result: dict[str, Any] = {
                "dataset": args.dataset_name,
                "method": args.method,
                "model": args.model,
                "selection_mode": args.selection_mode,
                "example_index": example_index,
                "example_id": str(example.get("_id", example_index)),
                "question": question,
                "answers": answers,
                "chunk_size_tokens": args.chunk_size_tokens,
                "chunk_count": args.chunk_count,
                "selected_chunk_indices": selection.get("selected_chunk_indices", []),
                "selected_contains_answer": selection.get(
                    "selected_contains_answer"
                ),
                "full_context_contains_answer": selection.get(
                    "full_context_contains_answer"
                ),
                "num_context_chunks": selection.get("num_context_chunks"),
                "num_answer_chunks": selection.get("num_answer_chunks"),
                "max_new_tokens": args.max_new_tokens,
                "success": False,
                "error": None,
            }

            try:
                chunks = chunk_context(tokenizer, context_text, args.chunk_size_tokens)
                selected_chunks = [
                    chunks[int(index)]
                    for index in selection["selected_chunk_indices"]
                    if int(index) < len(chunks)
                ]
                if len(selected_chunks) != len(selection["selected_chunk_indices"]):
                    raise RuntimeError("Selection index out of range for context")

                store_prompt = build_qa_prompt_ids(
                    tokenizer=tokenizer,
                    selected_chunks=selected_chunks,
                    question="Read the context and answer OK.",
                    blend_special_str=args.blend_special_str,
                    reorder_for_measured=False,
                )
                measured_prompt = build_qa_prompt_ids(
                    tokenizer=tokenizer,
                    selected_chunks=selected_chunks,
                    question=question,
                    blend_special_str=args.blend_special_str,
                    reorder_for_measured=True,
                )

                store_result = None
                if args.method in LMCACHE_METHODS:
                    store_result = generate_qa_timed(
                        llm,
                        store_prompt,
                        1,
                        "store_cache",
                    )
                    time.sleep(args.sleep_between_requests)

                measured = generate_qa_timed(
                    llm,
                    measured_prompt,
                    args.max_new_tokens,
                    "measured_reuse"
                    if args.method in LMCACHE_METHODS
                    else "measured",
                )
                prediction = measured["prediction_text"]

                result.update(
                    {
                        "success": True,
                        "store": store_result,
                        "measured": measured,
                        "store_prompt_tokens": len(store_prompt),
                        "prompt_tokens": len(measured_prompt),
                        "output_text": measured["output_text"],
                        "prediction_text": prediction,
                        "output_tokens": measured["output_tokens"],
                        "num_cached_tokens": measured["num_cached_tokens"],
                        "generation_sec": measured["elapsed_sec"],
                        "engine_ttft_sec": measured["engine_ttft_sec"],
                        "engine_prefill_sec": measured["engine_prefill_sec"],
                        "engine_decode_sec": measured["engine_decode_sec"],
                        "engine_inference_sec": measured["engine_inference_sec"],
                        "engine_tpot_sec": measured["engine_tpot_sec"],
                        "wall_tpot_sec": measured["wall_tpot_sec"],
                        "exact_match": max_metric(
                            prediction,
                            answers,
                            exact_match,
                        ),
                        "f1": max_metric(prediction, answers, f1_score),
                        "contains_match": max_metric(
                            prediction,
                            answers,
                            contains_match,
                        ),
                    }
                )
            except Exception as exc:
                result["error"] = repr(exc)
                result["traceback"] = traceback.format_exc()

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(result, ensure_ascii=False))

    cleanup_cuda_state()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--selection-mode", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-count", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=32648)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--output", required=True)

    parser.add_argument("--blend-special-str", default="# #")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--max-local-cpu-size-gb", type=float, default=5.0)
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument("--sleep-between-requests", type=float, default=0.2)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
