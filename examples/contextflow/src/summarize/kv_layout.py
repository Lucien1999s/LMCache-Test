# SPDX-License-Identifier: Apache-2.0
"""Summarize ContextFlow KV layout JSONL events into CSV."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from pathlib import Path
from typing import Any


KV_LAYOUT_STAGES = {
    "kv_layout_cpu_to_gpu_load_layer",
    "kv_layout_writeback_layer",
    "kv_layout_selected_repair_layer",
}


def condition_value(event: dict[str, Any], key: str) -> Any:
    """Return a condition value from one profiling event."""
    condition = event.get("condition") or {}
    if isinstance(condition, dict):
        return condition.get(key)
    return None


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    """Return numerator / denominator when both values are usable."""
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def main() -> None:
    """Read KV layout JSONL and write one CSV row per run/request/layer."""
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

    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        stage = event.get("stage")
        if stage not in KV_LAYOUT_STAGES:
            continue

        key = (
            event.get("run_id"),
            event.get("request_id"),
            condition_value(event, "method"),
            condition_value(event, "model"),
            condition_value(event, "chunk_count"),
            condition_value(event, "chunk_size_tokens"),
            condition_value(event, "max_new_tokens"),
            condition_value(event, "recompute_ratio"),
            event.get("layer_id"),
        )
        row = rows.setdefault(
            key,
            {
                "run_id": key[0],
                "request_id": key[1],
                "method": key[2],
                "model": key[3],
                "chunk_count": key[4],
                "chunk_size_tokens": key[5],
                "max_new_tokens": key[6],
                "recompute_ratio": key[7],
                "layer_id": key[8],
            },
        )
        metadata = event.get("metadata") or {}
        if stage == "kv_layout_cpu_to_gpu_load_layer":
            _copy_fields(
                row,
                metadata,
                [
                    "retrieved_tokens",
                    "retrieved_segments",
                    "actual_loaded_bytes",
                    "actual_loaded_physical_bytes",
                    "number_of_copy_ops",
                    "loaded_valid_slot_count",
                    "loaded_num_contiguous_slot_ranges",
                    "loaded_avg_contiguous_slot_range_length",
                    "loaded_max_contiguous_slot_range_length",
                    "loaded_vllm_blocks_touched",
                ],
            )
            row["cpu_to_gpu_copy_ops"] = metadata.get("number_of_copy_ops")
        elif stage == "kv_layout_writeback_layer":
            _copy_fields(
                row,
                metadata,
                [
                    "writeback_span_tokens",
                    "span_gap_tokens",
                    "actual_writeback_bytes",
                    "writeback_kernel_ops",
                    "writeback_valid_slot_count",
                    "writeback_num_contiguous_slot_ranges",
                    "writeback_avg_contiguous_slot_range_length",
                    "writeback_max_contiguous_slot_range_length",
                    "writeback_vllm_blocks_touched",
                ],
            )
        elif stage == "kv_layout_selected_repair_layer":
            _copy_fields(
                row,
                metadata,
                [
                    "selected_repair_tokens",
                    "theoretical_selected_token_bytes",
                    "bytes_per_selected_token",
                    "selected_valid_slot_count",
                    "selected_num_contiguous_slot_ranges",
                    "selected_avg_contiguous_slot_range_length",
                    "selected_max_contiguous_slot_range_length",
                    "selected_vllm_blocks_touched",
                ],
            )

    for row in rows.values():
        selected_bytes = row.get("theoretical_selected_token_bytes")
        loaded_bytes = row.get("actual_loaded_bytes")
        writeback_bytes = row.get("actual_writeback_bytes")
        row["useful_bytes_to_loaded_ratio"] = safe_ratio(selected_bytes, loaded_bytes)
        row["over_transfer_ratio"] = safe_ratio(loaded_bytes, selected_bytes)
        row["writeback_to_loaded_ratio"] = safe_ratio(writeback_bytes, loaded_bytes)
        row["writeback_over_selected_ratio"] = safe_ratio(
            writeback_bytes,
            selected_bytes,
        )

    fieldnames = [
        "run_id",
        "request_id",
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "layer_id",
        "retrieved_tokens",
        "retrieved_segments",
        "actual_loaded_bytes",
        "actual_loaded_physical_bytes",
        "cpu_to_gpu_copy_ops",
        "loaded_valid_slot_count",
        "loaded_num_contiguous_slot_ranges",
        "loaded_avg_contiguous_slot_range_length",
        "loaded_max_contiguous_slot_range_length",
        "loaded_vllm_blocks_touched",
        "writeback_span_tokens",
        "span_gap_tokens",
        "actual_writeback_bytes",
        "writeback_kernel_ops",
        "writeback_valid_slot_count",
        "writeback_num_contiguous_slot_ranges",
        "writeback_avg_contiguous_slot_range_length",
        "writeback_max_contiguous_slot_range_length",
        "writeback_vllm_blocks_touched",
        "selected_repair_tokens",
        "theoretical_selected_token_bytes",
        "bytes_per_selected_token",
        "selected_valid_slot_count",
        "selected_num_contiguous_slot_ranges",
        "selected_avg_contiguous_slot_range_length",
        "selected_max_contiguous_slot_range_length",
        "selected_vllm_blocks_touched",
        "useful_bytes_to_loaded_ratio",
        "over_transfer_ratio",
        "writeback_to_loaded_ratio",
        "writeback_over_selected_ratio",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows, key=lambda item: tuple(str(value) for value in item)):
            writer.writerow({field: rows[key].get(field) for field in fieldnames})

    print(f"Wrote CSV: {output_path}")


def _copy_fields(
    row: dict[str, Any],
    metadata: dict[str, Any],
    fields: list[str],
) -> None:
    for field in fields:
        row[field] = metadata.get(field)


if __name__ == "__main__":
    main()
