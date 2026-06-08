# SPDX-License-Identifier: Apache-2.0
"""Extract raw physical KV slot layouts from ContextFlow KV layout events."""

from __future__ import annotations

# Standard
import argparse
import json
from pathlib import Path
from typing import Any


KV_LAYOUT_STAGES = {
    "kv_layout_cpu_to_gpu_load_layer": "loaded",
    "kv_layout_writeback_layer": "writeback",
    "kv_layout_selected_repair_layer": "selected",
}

RAW_FIELDS = [
    "physical_slot_ranges",
    "physical_slot_ranges_are_end_exclusive",
    "slot_order_physical_slot_runs",
    "slot_order_runs_are_end_exclusive",
    "touched_block_ids",
    "logical_token_positions",
    "physical_slots",
    "logical_to_physical_slots",
]

LOAD_ONLY_FIELDS = [
    "copy_ops",
    "lmcache_gpu_buffer_ranges",
]

WRITEBACK_ONLY_FIELDS = [
    "logical_range",
    "gap_logical_positions",
]


def main() -> None:
    """Read ContextFlow event JSONL and write merged physical-layout JSONL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    jsonl_path = Path(args.jsonl)
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        stage = event.get("stage")
        prefix = KV_LAYOUT_STAGES.get(stage)
        if prefix is None:
            continue

        key = _row_key(event)
        row = rows.setdefault(key, _base_row(event))
        metadata = event.get("metadata") or {}
        _copy_prefixed_fields(row, metadata, prefix, RAW_FIELDS)
        if prefix == "loaded":
            _copy_prefixed_fields(row, metadata, prefix, LOAD_ONLY_FIELDS)
        elif prefix == "writeback":
            _copy_prefixed_fields(row, metadata, prefix, WRITEBACK_ONLY_FIELDS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for key in sorted(rows, key=lambda item: tuple(str(value) for value in item)):
            f.write(json.dumps(rows[key], ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Wrote physical layout JSONL: {output_path}")


def _row_key(event: dict[str, Any]) -> tuple[Any, ...]:
    condition = event.get("condition") or {}
    return (
        event.get("run_id"),
        event.get("request_id"),
        condition.get("method"),
        condition.get("model"),
        condition.get("chunk_count"),
        condition.get("chunk_size_tokens"),
        condition.get("max_new_tokens"),
        condition.get("recompute_ratio"),
        event.get("layer_id"),
    )


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
        "layer_id": event.get("layer_id"),
    }


def _copy_prefixed_fields(
    row: dict[str, Any],
    metadata: dict[str, Any],
    prefix: str,
    fields: list[str],
) -> None:
    for field in fields:
        source = f"{prefix}_{field}"
        if source in metadata:
            row[source] = metadata[source]


if __name__ == "__main__":
    main()
