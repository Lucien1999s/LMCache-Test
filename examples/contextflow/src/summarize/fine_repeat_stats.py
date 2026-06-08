# SPDX-License-Identifier: Apache-2.0
"""Summarize repeat-run exclusive breakdown stability and tail latency."""

from __future__ import annotations

# Standard
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def percentile(values: list[float], q: float) -> float:
    """Return a nearest-rank percentile for a non-empty value list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((len(ordered) - 1) * q)))
    return ordered[index]


def coefficient_of_variation(values: list[float]) -> float:
    """Return population std divided by mean."""
    if not values:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return pstdev(values) / avg


def condition_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable condition key for one exclusive summary row."""
    return (
        row["method"],
        row["model"],
        row["chunk_count"],
        row["chunk_size_tokens"],
        row["max_new_tokens"],
        row["recompute_ratio"],
        row["lmcache_chunk_size"],
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("exclusive_summary_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--slow-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.exclusive_summary_csv).open()))
    run_total_by_condition: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    bucket_by_condition: dict[tuple[tuple[str, ...], str], dict[str, float]] = (
        defaultdict(dict)
    )
    row_by_run_bucket: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        key = condition_key(row)
        run_id = row["run_id"]
        bucket = row["bucket"]
        row_by_run_bucket[(run_id, bucket)] = row
        if bucket == "cache_lookup_reuse_decision":
            run_total_by_condition[key][run_id] = float(row["measured_total_ms"])
        bucket_by_condition[(key, bucket)][run_id] = float(row["exclusive_ms"])

    slow_runs_by_condition: dict[tuple[str, ...], set[str]] = {}
    condition_tail: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, totals_by_run in run_total_by_condition.items():
        totals = list(totals_by_run.values())
        total_median = median(totals) if totals else 0.0
        total_p95 = percentile(totals, 0.95)
        slow_threshold = max(total_median * args.slow_multiplier, total_p95)
        slow_runs = {
            run_id
            for run_id, value in totals_by_run.items()
            if value > slow_threshold
        }
        slow_runs_by_condition[key] = slow_runs
        if totals_by_run:
            max_run_id, max_total = max(
                totals_by_run.items(),
                key=lambda item: item[1],
            )
            buckets = [
                row_by_run_bucket[(max_run_id, bucket)]
                for bucket in sorted(
                    {
                        bucket
                        for run, bucket in row_by_run_bucket
                        if run == max_run_id
                    }
                )
            ]
            top_bucket_row = max(
                buckets,
                key=lambda row: float(row["exclusive_ms"]),
            )
            condition_tail[key] = {
                "slow_threshold_ms": slow_threshold,
                "max_run_id": max_run_id,
                "max_total_ms": max_total,
                "max_run_top_bucket": top_bucket_row["bucket"],
                "max_run_top_bucket_ms": top_bucket_row["exclusive_ms"],
            }

    fieldnames = [
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "lmcache_chunk_size",
        "bucket",
        "repeat_count",
        "median_ms",
        "mean_ms",
        "std_ms",
        "p90_ms",
        "p95_ms",
        "max_ms",
        "min_ms",
        "coefficient_of_variation",
        "median_ratio",
        "mean_ratio",
        "slow_run_count",
        "slow_run_ids",
        "slow_threshold_ms",
        "condition_max_run_id",
        "condition_max_total_ms",
        "condition_max_run_top_bucket",
        "condition_max_run_top_bucket_ms",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (key, bucket), values_by_run in sorted(
            bucket_by_condition.items(),
            key=lambda item: tuple(str(part) for part in (*item[0][0], item[0][1])),
        ):
            values = list(values_by_run.values())
            totals_by_run = run_total_by_condition.get(key, {})
            ratios = [
                value / totals_by_run[run_id]
                for run_id, value in values_by_run.items()
                if totals_by_run.get(run_id, 0.0) > 0
            ]
            slow_runs = sorted(slow_runs_by_condition.get(key, set()))
            tail = condition_tail.get(key, {})
            writer.writerow(
                {
                    "method": key[0],
                    "model": key[1],
                    "chunk_count": key[2],
                    "chunk_size_tokens": key[3],
                    "max_new_tokens": key[4],
                    "recompute_ratio": key[5],
                    "lmcache_chunk_size": key[6],
                    "bucket": bucket,
                    "repeat_count": len(values),
                    "median_ms": median(values) if values else 0.0,
                    "mean_ms": mean(values) if values else 0.0,
                    "std_ms": pstdev(values) if len(values) > 1 else 0.0,
                    "p90_ms": percentile(values, 0.90),
                    "p95_ms": percentile(values, 0.95),
                    "max_ms": max(values) if values else 0.0,
                    "min_ms": min(values) if values else 0.0,
                    "coefficient_of_variation": coefficient_of_variation(values),
                    "median_ratio": median(ratios) if ratios else 0.0,
                    "mean_ratio": mean(ratios) if ratios else 0.0,
                    "slow_run_count": len(slow_runs),
                    "slow_run_ids": ";".join(slow_runs),
                    "slow_threshold_ms": tail.get("slow_threshold_ms"),
                    "condition_max_run_id": tail.get("max_run_id"),
                    "condition_max_total_ms": tail.get("max_total_ms"),
                    "condition_max_run_top_bucket": tail.get("max_run_top_bucket"),
                    "condition_max_run_top_bucket_ms": tail.get(
                        "max_run_top_bucket_ms"
                    ),
                }
            )

    print(f"Wrote repeat stats CSV: {output_path}")


if __name__ == "__main__":
    main()
