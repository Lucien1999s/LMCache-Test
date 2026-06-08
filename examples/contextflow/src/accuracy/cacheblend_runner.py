# SPDX-License-Identifier: Apache-2.0
"""Run CacheBlend-aligned QA accuracy generation from retrieval rows."""

from __future__ import annotations

# Standard
import argparse
import contextlib
import json
import os
import re
import string
import sys
import time
import traceback
from pathlib import Path
from statistics import mean
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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Local
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
    """Compute normalized answer containment against one answer."""
    prediction_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)
    if not prediction_norm or not gold_norm:
        return 0.0
    return float(gold_norm in prediction_norm or prediction_norm in gold_norm)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 from normalized token LCS."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    previous = [0] * (len(gold_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, gold_token in enumerate(gold_tokens, start=1):
            if pred_token == gold_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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
    text = text.split("# #", maxsplit=1)[0].strip()
    first_line = re.split(r"[\r\n]", text, maxsplit=1)[0].strip()
    first_line = re.sub(
        r"^(final\s+answer|answer)\s*:\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip(" \t\"'")


def build_judge_prompt(
    *,
    question: str,
    answers: list[str],
    prediction: str,
) -> str:
    """Build a fixed short-answer QA judge prompt."""
    references = json.dumps(answers, ensure_ascii=False)
    return (
        "You are judging a short-answer QA prediction.\n"
        "Decide whether the predicted answer is semantically equivalent to "
        "any reference answer.\n\n"
        "Rules:\n"
        "- Ignore harmless casing, punctuation, and leading phrases.\n"
        "- Mark correct if the prediction clearly contains the reference answer "
        "without contradiction.\n"
        "- Mark wrong if it gives a different entity, date, place, or yes/no "
        "answer.\n"
        "- Mark partial only when the prediction is incomplete but captures a "
        "material part of the reference.\n\n"
        f"Question: {question}\n"
        f"Reference answers: {references}\n"
        f"Predicted answer: {prediction}\n\n"
        "Return JSON only with this schema:\n"
        '{"label": "correct" | "partial" | "wrong", '
        '"score": 1.0 | 0.5 | 0.0, '
        '"reason": "brief reason"}'
    )


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


def generate_qa_timed(
    llm: LLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    request_label: str,
) -> dict[str, Any]:
    """Generate once and return output plus timing metadata."""
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["\n", "# #"],
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
        "engine_ttft_sec": metrics.get("first_token_latency"),
        "engine_prefill_sec": metrics.get("prefill_time"),
        "engine_decode_sec": metrics.get("decode_time"),
        "engine_inference_sec": metrics.get("inference_time"),
    }


def build_prompt_ids(
    *,
    tokenizer: Any,
    selected_chunks: list[dict[str, Any]],
    question: str,
    blend_special_str: str,
) -> list[int]:
    """Build CacheBlend-shaped QA prompt token IDs."""
    system_prompt = (
        "You are a careful extractive QA assistant. Use only the provided "
        "context chunks. Return only the final short answer span. Do not "
        "explain."
    )
    sep_ids = encode_no_special(tokenizer, blend_special_str)
    newline_ids = encode_no_special(tokenizer, "\n")
    prompt = encode_no_special(tokenizer, system_prompt)
    prompt += newline_ids + sep_ids + newline_ids
    for chunk in selected_chunks:
        header = (
            f"[Context rank={chunk['retrieval_rank']} "
            f"source={chunk['source_label']}]\n"
        )
        prompt += encode_no_special(tokenizer, header)
        prompt += encode_no_special(tokenizer, str(chunk["text"]))
        prompt += newline_ids + sep_ids + newline_ids
    query = f"Question: {question}\nAnswer within 5 words.\nAnswer:"
    prompt += encode_no_special(tokenizer, query)
    return prompt


def load_retrieval_rows(
    *,
    retrieval_file: Path,
    dataset_name: str,
    chunk_count: int,
    limit: int | None,
    start_index: int | None,
) -> list[dict[str, Any]]:
    """Load retrieval rows for one dataset and k."""
    rows = []
    for line in retrieval_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != dataset_name:
            continue
        if int(row.get("chunk_count", -1)) != chunk_count:
            continue
        example_index = int(row["example_index"])
        if start_index is not None and example_index < start_index:
            continue
        if limit is not None and len(rows) >= limit:
            continue
        rows.append(row)
    rows.sort(key=lambda row: int(row["example_index"]))
    return rows


@contextlib.contextmanager
def build_context(args: argparse.Namespace):
    """Create the vLLM/LMCache context manager for one method."""
    if args.method == "full_recompute":
        clear_lmcache_env()
        with build_plain_llm(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as llm:
            yield llm
        return

    if args.method in LMCACHE_METHODS:
        setup_lmcache_clean_env(
            method=args.method,
            blend_special_str=args.blend_special_str,
            recompute_ratio=args.recompute_ratio,
            chunk_size=args.lmcache_chunk_size,
            max_local_cpu_size_gb=args.max_local_cpu_size_gb,
            enable_sparse=args.enable_sparse,
        )
        with build_lmcache_llm(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as llm:
            yield llm
        return

    raise ValueError(f"Unknown method: {args.method}")


def run(args: argparse.Namespace) -> None:
    """Run one method/dataset/k condition."""
    setup_common_env()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_retrieval_rows(
        retrieval_file=Path(args.retrieval_file),
        dataset_name=args.dataset_name,
        chunk_count=args.chunk_count,
        limit=args.limit,
        start_index=args.start_index,
    )
    if not rows:
        raise RuntimeError("No matching retrieval rows found")

    prediction_path = Path(args.predictions_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path = (
        Path(args.llm_judge_inputs_output)
        if args.llm_judge_inputs_output
        else None
    )
    if judge_path is not None:
        judge_path.parent.mkdir(parents=True, exist_ok=True)

    condition_results: list[dict[str, Any]] = []
    cleanup_cuda_state()
    with build_context(args) as llm, prediction_path.open("a") as pred_out:
        judge_out = judge_path.open("a") if judge_path is not None else None
        try:
            if args.warmup:
                warmup = encode_no_special(tokenizer, "Warm up the engine. " * 64)
                generate_qa_timed(llm, warmup, 1, "engine_warmup")

            for idx, row in enumerate(rows):
                question = str(row["question"])
                answers = [str(answer) for answer in row.get("answers", [])]
                selected_chunks = row["selected_chunks"]

                result: dict[str, Any] = {
                    "dataset": args.dataset_name,
                    "method": args.method,
                    "model": args.model,
                    "sample_id": row["example_id"],
                    "example_index": row["example_index"],
                    "example_id": row["example_id"],
                    "question": question,
                    "answers": answers,
                    "chunk_size_tokens": row["chunk_size_tokens"],
                    "chunk_count": args.chunk_count,
                    "max_new_tokens": args.max_new_tokens,
                    "retrieved_answer_hit": row["retrieved_answer_hit"],
                    "answer_hit_in_retrieved_chunks": row["retrieved_answer_hit"],
                    "retrieved_answer_hit_count": row["retrieved_answer_hit_count"],
                    "support_evidence_available": row.get(
                        "support_evidence_available", False
                    ),
                    "retrieved_support_hit": row.get("retrieved_support_hit"),
                    "selected_chunk_ids": row["selected_chunk_ids"],
                    "retrieved_chunk_ids": row["selected_chunk_ids"],
                    "selected_source_ids": row["selected_source_ids"],
                    "retrieved_chunk_previews": row.get("retrieved_chunk_previews"),
                    "embedding_model": row.get("embedding_model"),
                    "embedding_backend": row.get("embedding_backend"),
                    "retrieval_metric": row.get("retrieval_metric"),
                    "retrieval_order": row.get("retrieval_order"),
                    "prompt_preview": (
                        row.get("prompt_preview")
                        if idx < args.preview_count
                        else None
                    ),
                    "judge_score": None,
                    "judge_label": None,
                    "judge_reason": None,
                    "success": False,
                    "error": None,
                }

                try:
                    measured_prompt = build_prompt_ids(
                        tokenizer=tokenizer,
                        selected_chunks=selected_chunks,
                        question=question,
                        blend_special_str=args.blend_special_str,
                    )
                    store_result = None
                    if args.method in LMCACHE_METHODS:
                        store_prompt = build_prompt_ids(
                            tokenizer=tokenizer,
                            selected_chunks=selected_chunks,
                            question="Read the context and answer OK.",
                            blend_special_str=args.blend_special_str,
                        )
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
                    f1 = max_metric(prediction, answers, f1_score)
                    em = max_metric(prediction, answers, exact_match)
                    contains = max_metric(prediction, answers, contains_match)
                    rouge_l = max_metric(prediction, answers, rouge_l_score)
                    result.update(
                        {
                            "success": True,
                            "store": store_result,
                            "measured": measured,
                            "prompt_tokens": len(measured_prompt),
                            "store_prompt_tokens": (
                                store_result["prompt_tokens"]
                                if store_result is not None
                                else None
                            ),
                            "output_text": measured["output_text"],
                            "prediction_text": prediction,
                            "output_tokens": measured["output_tokens"],
                            "num_cached_tokens": measured["num_cached_tokens"],
                            "generation_sec": measured["elapsed_sec"],
                            "latency_ms": measured["elapsed_sec"] * 1000,
                            "engine_ttft_sec": measured["engine_ttft_sec"],
                            "engine_prefill_sec": measured["engine_prefill_sec"],
                            "engine_decode_sec": measured["engine_decode_sec"],
                            "engine_inference_sec": measured["engine_inference_sec"],
                            "f1": f1,
                            "rouge_l": rouge_l,
                            "exact_match": em,
                            "em": em,
                            "contains_match": contains,
                        }
                    )
                except Exception as exc:
                    result["error"] = repr(exc)
                    result["traceback"] = traceback.format_exc()

                condition_results.append(result)
                pred_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                pred_out.flush()
                print(json.dumps(result, ensure_ascii=False))

                if judge_out is not None:
                    judge_out.write(
                        json.dumps(
                            {
                                "dataset": result["dataset"],
                                "method": result["method"],
                                "example_id": result["example_id"],
                                "question": result["question"],
                                "gold_answers": result["answers"],
                                "prediction": result.get("prediction_text"),
                                "f1": result.get("f1"),
                                "rouge_l": result.get("rouge_l"),
                                "exact_match": result.get("exact_match"),
                                "contains_match": result.get("contains_match"),
                                "judge_prompt": build_judge_prompt(
                                    question=result["question"],
                                    answers=result["answers"],
                                    prediction=str(
                                        result.get("prediction_text") or ""
                                    ),
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    judge_out.flush()
        finally:
            if judge_out is not None:
                judge_out.close()

    cleanup_cuda_state()

    successes = [row for row in condition_results if row.get("success")]
    run_record = {
        "dataset": args.dataset_name,
        "method": args.method,
        "model": args.model,
        "chunk_count": args.chunk_count,
        "chunk_size_tokens": args.chunk_size_tokens,
        "max_new_tokens": args.max_new_tokens,
        "count": len(condition_results),
        "success_count": len(successes),
        "failure_count": len(condition_results) - len(successes),
        "mean_f1": mean([row["f1"] for row in successes]) if successes else 0.0,
        "mean_exact_match": (
            mean([row["exact_match"] for row in successes]) if successes else 0.0
        ),
        "mean_contains_match": (
            mean([row["contains_match"] for row in successes]) if successes else 0.0
        ),
        "mean_retrieved_answer_hit": (
            mean([float(row["retrieved_answer_hit"]) for row in successes])
            if successes
            else 0.0
        ),
    }
    run_path = Path(args.runs_output)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("a") as f:
        f.write(json.dumps(run_record, ensure_ascii=False) + "\n")


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
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-count", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32648)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--blend-special-str", default="# #")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--lmcache-chunk-size", type=int, default=256)
    parser.add_argument("--max-local-cpu-size-gb", type=float, default=5.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.2)
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--runs-output", required=True)
    parser.add_argument("--llm-judge-inputs-output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
