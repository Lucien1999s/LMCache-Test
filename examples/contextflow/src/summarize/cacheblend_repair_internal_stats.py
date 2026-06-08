# SPDX-License-Identifier: Apache-2.0
"""Summarize CacheBlend repair-internal repeat-run stability."""

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
    """Return a stable condition key for one summary row."""
    return (
        row["method"],
        row["model"],
        row["chunk_count"],
        row["chunk_size_tokens"],
        row["max_new_tokens"],
        row["recompute_ratio"],
        row["lmcache_chunk_size"],
    )


def float_value(row: dict[str, Any], key: str) -> float:
    """Read a CSV float value."""
    raw = row.get(key)
    if raw in {None, ""}:
        return 0.0
    return float(raw)


def int_value(row: dict[str, Any], key: str) -> int:
    """Read a CSV int value."""
    raw = row.get(key)
    if raw in {None, ""}:
        return 0
    return int(float(raw))


def bool_value(row: dict[str, Any], key: str) -> bool:
    """Read a CSV bool value."""
    return str(row.get(key)).lower() == "true"


def build_layer_stats_rows(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]],
    dict[tuple[str, ...], set[str]],
]:
    """Group rows by condition, layer, and bucket."""
    grouped: dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    invalid_runs: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = condition_key(row)
        if not bool_value(row, "valid_sum_le_total"):
            invalid_runs[key].add(row["run_id"])
        grouped[(key, str(row["layer_id"]), row["bucket"])].append(row)
    return grouped, invalid_runs


def build_all_layer_stats_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]]:
    """Aggregate per-layer rows into per-run all-layer bucket rows."""
    bucket_ms: dict[tuple[tuple[str, ...], str, str], float] = defaultdict(float)
    totals_by_run: dict[tuple[tuple[str, ...], str], dict[str, Any]] = defaultdict(
        lambda: {
            "repair_internal_total_ms": 0.0,
            "repair_internal_residual_ms": 0.0,
            "selected_repair_tokens": 0,
            "cached_tokens": 0,
            "valid_sum_le_total": True,
            "example_row": None,
            "layers_seen": set(),
        }
    )
    for row in rows:
        key = condition_key(row)
        run_id = row["run_id"]
        bucket = row["bucket"]
        bucket_ms[(key, run_id, bucket)] += float_value(row, "exclusive_ms")
        run_total = totals_by_run[(key, run_id)]
        run_total["example_row"] = row
        run_total["valid_sum_le_total"] = (
            run_total["valid_sum_le_total"] and bool_value(row, "valid_sum_le_total")
        )
        if row["layer_id"] not in run_total["layers_seen"]:
            run_total["layers_seen"].add(row["layer_id"])
            run_total["selected_repair_tokens"] += int_value(
                row,
                "selected_repair_tokens",
            )
            run_total["cached_tokens"] += int_value(row, "cached_tokens")
            run_total["repair_internal_total_ms"] += float_value(
                row,
                "repair_internal_total_ms",
            )
            run_total["repair_internal_residual_ms"] += float_value(
                row,
                "repair_internal_residual_ms",
            )

    grouped: dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for (key, run_id, bucket), duration in bucket_ms.items():
        total = totals_by_run[(key, run_id)]
        example = total["example_row"]
        if example is None:
            continue
        grouped[(key, "all", bucket)].append(
            {
                **example,
                "layer_id": "all",
                "bucket": bucket,
                "exclusive_ms": duration,
                "exclusive_ratio": (
                    duration / total["repair_internal_total_ms"]
                    if total["repair_internal_total_ms"]
                    else 0.0
                ),
                "repair_internal_total_ms": total["repair_internal_total_ms"],
                "repair_internal_residual_ms": total[
                    "repair_internal_residual_ms"
                ],
                "repair_internal_residual_ratio": (
                    total["repair_internal_residual_ms"]
                    / total["repair_internal_total_ms"]
                    if total["repair_internal_total_ms"]
                    else 0.0
                ),
                "selected_repair_tokens": total["selected_repair_tokens"],
                "cached_tokens": total["cached_tokens"],
                "valid_sum_le_total": total["valid_sum_le_total"],
            }
        )
    return grouped


def write_stats(
    output_path: Path,
    grouped: dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]],
    invalid_runs: dict[tuple[str, ...], set[str]],
) -> None:
    """Write repair-internal repeat stats."""
    fieldnames = [
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "lmcache_chunk_size",
        "layer_id",
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
        "median_repair_internal_total_ms",
        "mean_repair_internal_total_ms",
        "p95_repair_internal_total_ms",
        "median_residual_ratio",
        "mean_residual_ratio",
        "median_selected_repair_tokens",
        "mean_selected_repair_tokens",
        "median_cached_tokens",
        "mean_cached_tokens",
        "invalid_accounting_count",
        "invalid_run_ids",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (key, layer_id, bucket), bucket_rows in sorted(
            grouped.items(),
            key=lambda item: (
                *[str(part) for part in item[0][0]],
                str(item[0][1]),
                item[0][2],
            ),
        ):
            values = [float_value(row, "exclusive_ms") for row in bucket_rows]
            ratios = [float_value(row, "exclusive_ratio") for row in bucket_rows]
            totals = [
                float_value(row, "repair_internal_total_ms") for row in bucket_rows
            ]
            residual_ratios = [
                float_value(row, "repair_internal_residual_ratio")
                for row in bucket_rows
            ]
            selected_tokens = [
                int_value(row, "selected_repair_tokens") for row in bucket_rows
            ]
            cached_tokens = [int_value(row, "cached_tokens") for row in bucket_rows]
            invalid = sorted(invalid_runs.get(key, set()))
            writer.writerow(
                {
                    "method": key[0],
                    "model": key[1],
                    "chunk_count": key[2],
                    "chunk_size_tokens": key[3],
                    "max_new_tokens": key[4],
                    "recompute_ratio": key[5],
                    "lmcache_chunk_size": key[6],
                    "layer_id": layer_id,
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
                    "median_repair_internal_total_ms": (
                        median(totals) if totals else 0.0
                    ),
                    "mean_repair_internal_total_ms": mean(totals) if totals else 0.0,
                    "p95_repair_internal_total_ms": percentile(totals, 0.95),
                    "median_residual_ratio": (
                        median(residual_ratios) if residual_ratios else 0.0
                    ),
                    "mean_residual_ratio": (
                        mean(residual_ratios) if residual_ratios else 0.0
                    ),
                    "median_selected_repair_tokens": (
                        median(selected_tokens) if selected_tokens else 0.0
                    ),
                    "mean_selected_repair_tokens": (
                        mean(selected_tokens) if selected_tokens else 0.0
                    ),
                    "median_cached_tokens": (
                        median(cached_tokens) if cached_tokens else 0.0
                    ),
                    "mean_cached_tokens": (
                        mean(cached_tokens) if cached_tokens else 0.0
                    ),
                    "invalid_accounting_count": len(invalid),
                    "invalid_run_ids": ";".join(invalid),
                }
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("exclusive_summary_csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.exclusive_summary_csv).open()))
    layer_grouped, invalid_runs = build_layer_stats_rows(rows)
    all_layer_grouped = build_all_layer_stats_rows(rows)
    grouped = {**layer_grouped, **all_layer_grouped}
    write_stats(Path(args.output), grouped, invalid_runs)
    print(f"Wrote repair-internal stats CSV: {args.output}")


if __name__ == "__main__":
    main()
