# SPDX-License-Identifier: Apache-2.0
"""Summarize ContextFlow layer-wise KV drift events into CSV."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


DRIFT_STAGE = "kv_drift_distribution_layer"

METADATA_FIELDS = [
    "drift_metric",
    "drift_scope",
    "cached_tokens",
    "evaluated_tokens",
    "evaluated_token_ratio",
    "drift_mean",
    "drift_median",
    "drift_p90",
    "drift_p95",
    "drift_p99",
    "drift_max",
    "drift_min",
    "drift_std",
    "drift_top1pct_mass_ratio",
    "drift_top5pct_mass_ratio",
    "drift_top10pct_mass_ratio",
    "selected_repair_tokens",
    "selected_ratio",
    "selected_drift_mean",
    "selected_drift_median",
    "selected_drift_p90",
    "selected_drift_p95",
    "selected_drift_p99",
    "selected_drift_max",
    "selected_drift_min",
    "selected_drift_std",
    "selected_drift_mass_ratio",
    "selected_position_min",
    "selected_position_max",
]


def main() -> None:
    """Read drift-analysis JSONL and write one CSV row per run/request/layer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_path = (
        Path(args.output)
        if args.output is not None
        else jsonl_path.with_suffix(".drift.summary.csv")
    )

    rows: list[dict[str, Any]] = []
    selected_sets: dict[int, set[int]] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("stage") != DRIFT_STAGE:
            continue
        row = _base_row(event)
        metadata = event.get("metadata") or {}
        for field in METADATA_FIELDS:
            row[field] = metadata.get(field)
        positions = metadata.get("selected_token_positions") or []
        ranges = metadata.get("selected_token_position_ranges") or []
        row["selected_position_range_count"] = len(ranges)
        row["selected_token_position_ranges_json"] = json.dumps(ranges)
        selected_sets[id(row)] = set(int(position) for position in positions)
        rows.append(row)

    _add_overlap_and_ranking(rows, selected_sets)

    fieldnames = [
        "run_id",
        "request_id",
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "blend_check_layers",
        "layer_id",
        *METADATA_FIELDS,
        "selected_position_range_count",
        "selected_token_position_ranges_json",
        "prev_layer_selected_jaccard",
        "prev_layer_selected_overlap_ratio",
        "next_layer_selected_jaccard",
        "next_layer_selected_overlap_ratio",
        "drift_mean_group_mean",
        "drift_mean_ratio_to_group_mean",
        "drift_mean_rank_desc",
        "drift_p95_rank_desc",
        "drift_is_gt_2x_group_mean",
        "drift_is_lt_half_group_mean",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item.get("run_id")),
                str(item.get("drift_scope")),
                int(item.get("layer_id")),
            ),
        ):
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(f"Wrote CSV: {output_path}")


def _base_row(event: dict[str, Any]) -> dict[str, Any]:
    condition = event.get("condition") or {}
    return {
        "run_id": event.get("run_id"),
        "request_id": event.get("request_id"),
        "method": condition.get("method"),
        "model": condition.get("model"),
        "chunk_count": condition.get("chunk_count"),
        "chunk_size_tokens": condition.get("chunk_size_tokens"),
        "max_new_tokens": condition.get("max_new_tokens"),
        "recompute_ratio": condition.get("recompute_ratio"),
        "blend_check_layers": condition.get("blend_check_layers"),
        "layer_id": event.get("layer_id"),
    }


def _add_overlap_and_ranking(
    rows: list[dict[str, Any]],
    selected_sets: dict[int, set[int]],
) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("run_id"),
            row.get("request_id"),
            row.get("drift_scope"),
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda item: int(item.get("layer_id")))
        for index, row in enumerate(group_rows):
            current = selected_sets[id(row)]
            previous = selected_sets[id(group_rows[index - 1])] if index > 0 else set()
            next_set = (
                selected_sets[id(group_rows[index + 1])]
                if index + 1 < len(group_rows)
                else set()
            )
            row["prev_layer_selected_jaccard"] = _jaccard(current, previous)
            row["prev_layer_selected_overlap_ratio"] = _overlap_ratio(
                current,
                previous,
            )
            row["next_layer_selected_jaccard"] = _jaccard(current, next_set)
            row["next_layer_selected_overlap_ratio"] = _overlap_ratio(
                current,
                next_set,
            )

        means = [
            float(row["drift_mean"])
            for row in group_rows
            if row.get("drift_mean") is not None
        ]
        group_mean = sum(means) / len(means) if means else None
        mean_rank = _rank_rows(group_rows, "drift_mean")
        p95_rank = _rank_rows(group_rows, "drift_p95")
        for row in group_rows:
            row["drift_mean_group_mean"] = group_mean
            row["drift_mean_ratio_to_group_mean"] = _safe_ratio(
                row.get("drift_mean"),
                group_mean,
            )
            row["drift_mean_rank_desc"] = mean_rank.get(id(row))
            row["drift_p95_rank_desc"] = p95_rank.get(id(row))
            ratio = row["drift_mean_ratio_to_group_mean"]
            row["drift_is_gt_2x_group_mean"] = ratio is not None and ratio > 2.0
            row["drift_is_lt_half_group_mean"] = ratio is not None and ratio < 0.5


def _rank_rows(rows: list[dict[str, Any]], field: str) -> dict[int, int]:
    ranked = sorted(
        [row for row in rows if row.get(field) is not None],
        key=lambda item: float(item[field]),
        reverse=True,
    )
    return {id(row): rank for rank, row in enumerate(ranked, start=1)}


def _jaccard(left: set[int], right: set[int]) -> Optional[float]:
    if not left and not right:
        return None
    union = len(left | right)
    if union == 0:
        return None
    return len(left & right) / union


def _overlap_ratio(left: set[int], right: set[int]) -> Optional[float]:
    if not left:
        return None
    return len(left & right) / len(left)


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
