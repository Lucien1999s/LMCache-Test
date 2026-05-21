# SPDX-License-Identifier: Apache-2.0
"""Summarize ContextFlow stage-profile JSONL events into CSV."""

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


def condition_value(event: dict[str, Any], key: str) -> Any:
    """Return a condition value from one profiling event."""
    condition = event.get("condition") or {}
    if isinstance(condition, dict):
        return condition.get(key)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_path = (
        Path(args.output)
        if args.output is not None
        else jsonl_path.with_suffix(".summary.csv")
    )

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)

    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        duration_ms = event.get("duration_ms")
        if duration_ms is None:
            continue
        key = (
            event.get("run_id"),
            condition_value(event, "method"),
            condition_value(event, "model"),
            condition_value(event, "chunk_count"),
            condition_value(event, "chunk_size_tokens"),
            condition_value(event, "max_new_tokens"),
            condition_value(event, "recompute_ratio"),
            condition_value(event, "enable_sparse"),
            event.get("stage"),
            event.get("category"),
            event.get("device"),
            event.get("layer_id"),
        )
        groups[key].append(float(duration_ms))

    fieldnames = [
        "run_id",
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "enable_sparse",
        "stage",
        "category",
        "device",
        "layer_id",
        "count",
        "sum_ms",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "max_ms",
        "min_ms",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
            values = groups[key]
            writer.writerow(
                {
                    "run_id": key[0],
                    "method": key[1],
                    "model": key[2],
                    "chunk_count": key[3],
                    "chunk_size_tokens": key[4],
                    "max_new_tokens": key[5],
                    "recompute_ratio": key[6],
                    "enable_sparse": key[7],
                    "stage": key[8],
                    "category": key[9],
                    "device": key[10],
                    "layer_id": key[11],
                    "count": len(values),
                    "sum_ms": sum(values),
                    "mean_ms": mean(values),
                    "median_ms": median(values),
                    "p95_ms": percentile(values, 0.95),
                    "max_ms": max(values),
                    "min_ms": min(values),
                }
            )

    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
