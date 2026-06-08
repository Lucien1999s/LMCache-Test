# SPDX-License-Identifier: Apache-2.0
"""Summarize CacheBlend-aligned accuracy predictions."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def percentile(values: list[float], q: float) -> float:
    """Return a nearest-rank percentile for a non-empty value list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for line in Path(args.predictions_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            row.get("dataset"),
            row.get("method"),
            row.get("model"),
            row.get("chunk_size_tokens"),
            row.get("chunk_count"),
            row.get("embedding_model"),
            row.get("retrieval_metric"),
        )
        groups[(*key, "all")].append(row)
        if row.get("retrieved_answer_hit"):
            groups[(*key, "retrieved_answer_hit_only")].append(row)

    fieldnames = [
        "dataset",
        "method",
        "model",
        "chunk_size_tokens",
        "chunk_count",
        "embedding_model",
        "retrieval_metric",
        "analysis_subset",
        "count",
        "success_count",
        "failure_count",
        "retrieved_answer_hit_rate",
        "mean_rouge_l",
        "median_rouge_l",
        "mean_f1",
        "median_f1",
        "p95_f1",
        "max_f1",
        "min_f1",
        "mean_exact_match",
        "mean_em",
        "mean_contains_match",
        "mean_judge_score",
        "mean_generation_sec",
        "mean_latency_ms",
        "median_latency_ms",
        "median_generation_sec",
        "mean_engine_ttft_sec",
        "median_engine_ttft_sec",
        "mean_prompt_tokens",
        "mean_num_cached_tokens",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups, key=lambda item: tuple(str(v) for v in item)):
            rows = groups[key]
            successes = [row for row in rows if row.get("success")]
            f1s = [float(row.get("f1") or 0.0) for row in successes]
            rouge_l = [float(row.get("rouge_l") or 0.0) for row in successes]
            ems = [float(row.get("exact_match") or 0.0) for row in successes]
            contains = [float(row.get("contains_match") or 0.0) for row in successes]
            judge_scores = [
                float(row.get("judge_score"))
                for row in successes
                if row.get("judge_score") is not None
            ]
            generation = [
                float(row.get("generation_sec") or 0.0) for row in successes
            ]
            latency_ms = [
                float(row.get("latency_ms") or 0.0) for row in successes
            ]
            ttft = [
                float(row.get("engine_ttft_sec") or 0.0) for row in successes
            ]
            prompt_tokens = [
                float(row.get("prompt_tokens") or 0.0) for row in successes
            ]
            cached_tokens = [
                float(row.get("num_cached_tokens") or 0.0)
                for row in successes
                if row.get("num_cached_tokens") is not None
            ]
            retrieved_hits = [
                float(bool(row.get("retrieved_answer_hit"))) for row in successes
            ]
            writer.writerow(
                {
                    "dataset": key[0],
                    "method": key[1],
                    "model": key[2],
                    "chunk_size_tokens": key[3],
                    "chunk_count": key[4],
                    "embedding_model": key[5],
                    "retrieval_metric": key[6],
                    "analysis_subset": key[7],
                    "count": len(rows),
                    "success_count": len(successes),
                    "failure_count": len(rows) - len(successes),
                    "retrieved_answer_hit_rate": (
                        mean(retrieved_hits) if retrieved_hits else 0.0
                    ),
                    "mean_rouge_l": mean(rouge_l) if rouge_l else 0.0,
                    "median_rouge_l": median(rouge_l) if rouge_l else 0.0,
                    "mean_f1": mean(f1s) if f1s else 0.0,
                    "median_f1": median(f1s) if f1s else 0.0,
                    "p95_f1": percentile(f1s, 0.95),
                    "max_f1": max(f1s) if f1s else 0.0,
                    "min_f1": min(f1s) if f1s else 0.0,
                    "mean_exact_match": mean(ems) if ems else 0.0,
                    "mean_em": mean(ems) if ems else 0.0,
                    "mean_contains_match": mean(contains) if contains else 0.0,
                    "mean_judge_score": (
                        mean(judge_scores) if judge_scores else ""
                    ),
                    "mean_generation_sec": mean(generation) if generation else 0.0,
                    "mean_latency_ms": mean(latency_ms) if latency_ms else 0.0,
                    "median_latency_ms": (
                        median(latency_ms) if latency_ms else 0.0
                    ),
                    "median_generation_sec": (
                        median(generation) if generation else 0.0
                    ),
                    "mean_engine_ttft_sec": mean(ttft) if ttft else 0.0,
                    "median_engine_ttft_sec": median(ttft) if ttft else 0.0,
                    "mean_prompt_tokens": (
                        mean(prompt_tokens) if prompt_tokens else 0.0
                    ),
                    "mean_num_cached_tokens": (
                        mean(cached_tokens) if cached_tokens else 0.0
                    ),
                }
            )

    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
