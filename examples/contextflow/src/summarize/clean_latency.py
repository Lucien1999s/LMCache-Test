# SPDX-License-Identifier: Apache-2.0
"""Summarize ContextFlow clean latency JSONL results into CSV."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


GROUP_FIELDS = [
    "method",
    "model",
    "chunk_count",
    "chunk_size_tokens",
    "max_new_tokens",
]

METRIC_FIELDS = [
    "generation_sec",
    "engine_ttft_sec",
    "engine_prefill_sec",
    "engine_decode_sec",
    "engine_inference_sec",
    "engine_tpot_sec",
    "wall_tpot_sec",
    "gpu_peak_mb",
    "gpu_peak_delta_mb",
    "num_cached_tokens",
]


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _metric_summary(values: Iterable[Any]) -> dict[str, Optional[float]]:
    numeric = [v for value in values if (v := _as_float(value)) is not None]
    return {
        "median": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "max": max(numeric) if numeric else None,
        "min": min(numeric) if numeric else None,
    }


def summarize(input_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        grouped[key].append(row)

    output_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        out = {field: value for field, value in zip(GROUP_FIELDS, key, strict=False)}
        success_rows = [row for row in group_rows if row.get("success")]
        out["count"] = len(group_rows)
        out["success_count"] = len(success_rows)
        out["failure_count"] = len(group_rows) - len(success_rows)

        for field in METRIC_FIELDS:
            summary = _metric_summary(row.get(field) for row in success_rows)
            for stat_name, stat_value in summary.items():
                out[f"{stat_name}_{field}"] = stat_value

        output_rows.append(out)

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(args.input)

    fieldnames = list(GROUP_FIELDS) + ["count", "success_count", "failure_count"]
    for metric in METRIC_FIELDS:
        for stat_name in ["median", "p95", "max", "min"]:
            fieldnames.append(f"{stat_name}_{metric}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()
