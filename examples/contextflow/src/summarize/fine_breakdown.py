# SPDX-License-Identifier: Apache-2.0
"""Summarize ContextFlow fine-breakdown profiling JSONL events."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional


OUTER_STAGES = {
    "offline_llm_generate_call",
    "cacheblend_load_recompute_repair_writeback",
    "naive_segment_reuse_load_writeback",
    "prefix_chunked_reuse_load_writeback",
    "layerwise_model_recompute_and_repair_layer",
    "lmcache_engine_lookup_total",
    "lookup_client_bypass_lookup_total",
    "storage_manager_batched_put_total",
}


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


def metadata_value(event: dict[str, Any], key: str) -> Any:
    """Return a metadata value from one profiling event."""
    metadata = event.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a path if it exists."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def infer_phase(
    event: dict[str, Any],
    intervals_by_run: dict[str, list[tuple[str, int, int]]],
) -> Optional[str]:
    """Infer request phase from runner generate intervals."""
    explicit = event.get("request_phase")
    if explicit:
        return str(explicit)
    run_id = event.get("run_id")
    if not run_id:
        return None
    start_ns = event.get("start_ns")
    end_ns = event.get("end_ns", start_ns)
    if start_ns is None:
        return None
    for label, interval_start, interval_end in intervals_by_run.get(run_id, []):
        if int(start_ns) <= interval_end and int(end_ns) >= interval_start:
            return label
    return None


def device_bucket(device: Any) -> str:
    """Classify a device string into a coarse bucket."""
    text = str(device or "")
    if text == "CPU":
        return "cpu_side"
    if "wait" in text.lower() or "sync" in text.lower():
        return "sync_wait"
    if "CPU" in text and "GPU" in text:
        return "cpu_control_gpu_work"
    if "GPU" in text:
        return "gpu_side"
    return "other"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("events_jsonl")
    parser.add_argument("--runs-jsonl", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    events_path = Path(args.events_jsonl)
    output_path = (
        Path(args.output)
        if args.output is not None
        else events_path.with_suffix(".summary.csv")
    )

    events = load_jsonl(events_path)
    intervals_by_run: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    measured_total_ms: dict[str, float] = {}
    run_condition: dict[str, dict[str, Any]] = {}

    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue
        if run_id not in run_condition:
            condition = event.get("condition") or {}
            run_condition[run_id] = condition if isinstance(condition, dict) else {}
        if event.get("stage") != "offline_llm_generate_call":
            continue
        label = metadata_value(event, "request_label")
        start_ns = event.get("start_ns")
        end_ns = event.get("end_ns")
        if label and start_ns is not None and end_ns is not None:
            intervals_by_run[run_id].append((str(label), int(start_ns), int(end_ns)))
            if label == "measured" and event.get("duration_ms") is not None:
                measured_total_ms[run_id] = float(event["duration_ms"])

    stage_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    run_leaf_ms: dict[str, float] = defaultdict(float)
    run_bucket_ms: dict[tuple[str, str], float] = defaultdict(float)

    for event in events:
        duration_ms = event.get("duration_ms")
        if duration_ms is None:
            continue
        duration = float(duration_ms)
        run_id = event.get("run_id")
        phase = infer_phase(event, intervals_by_run)
        key = (
            run_id,
            phase,
            condition_value(event, "method"),
            condition_value(event, "model"),
            condition_value(event, "chunk_count"),
            condition_value(event, "chunk_size_tokens"),
            condition_value(event, "max_new_tokens"),
            condition_value(event, "recompute_ratio"),
            condition_value(event, "lmcache_chunk_size"),
            event.get("stage"),
            event.get("category"),
            event.get("device"),
            event.get("layer_id"),
        )
        stage_groups[key].append(duration)
        if run_id and phase == "measured" and event.get("stage") not in OUTER_STAGES:
            run_leaf_ms[str(run_id)] += duration
            run_bucket_ms[(str(run_id), device_bucket(event.get("device")))] += duration
            stage_text = str(event.get("stage") or "").lower()
            if "wait" in stage_text or "sync" in stage_text:
                run_bucket_ms[(str(run_id), "sync_wait")] += duration

    fieldnames = [
        "row_type",
        "run_id",
        "request_phase",
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "lmcache_chunk_size",
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
        "measured_total_ms",
        "sum_known_profiled_ms",
        "unattributed_ms",
        "unattributed_ratio",
        "cpu_side_ms",
        "gpu_side_ms",
        "cpu_control_gpu_work_ms",
        "sync_wait_ms",
        "known_exceeds_measured_total",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_id in sorted(measured_total_ms):
            condition = run_condition.get(run_id, {})
            measured = measured_total_ms[run_id]
            known = run_leaf_ms.get(run_id, 0.0)
            unattributed = max(measured - known, 0.0)
            writer.writerow(
                {
                    "row_type": "run_total",
                    "run_id": run_id,
                    "request_phase": "measured",
                    "method": condition.get("method"),
                    "model": condition.get("model"),
                    "chunk_count": condition.get("chunk_count"),
                    "chunk_size_tokens": condition.get("chunk_size_tokens"),
                    "max_new_tokens": condition.get("max_new_tokens"),
                    "recompute_ratio": condition.get("recompute_ratio"),
                    "lmcache_chunk_size": condition.get("lmcache_chunk_size"),
                    "stage": "__measured_reuse_total__",
                    "category": "summary",
                    "device": "mixed",
                    "layer_id": None,
                    "count": 1,
                    "sum_ms": measured,
                    "mean_ms": measured,
                    "median_ms": measured,
                    "p95_ms": measured,
                    "max_ms": measured,
                    "min_ms": measured,
                    "measured_total_ms": measured,
                    "sum_known_profiled_ms": known,
                    "unattributed_ms": unattributed,
                    "unattributed_ratio": unattributed / measured if measured else 0.0,
                    "cpu_side_ms": run_bucket_ms.get((run_id, "cpu_side"), 0.0),
                    "gpu_side_ms": run_bucket_ms.get((run_id, "gpu_side"), 0.0),
                    "cpu_control_gpu_work_ms": run_bucket_ms.get(
                        (run_id, "cpu_control_gpu_work"), 0.0
                    ),
                    "sync_wait_ms": run_bucket_ms.get((run_id, "sync_wait"), 0.0),
                    "known_exceeds_measured_total": known > measured,
                }
            )

        for key in sorted(stage_groups, key=lambda item: tuple(str(v) for v in item)):
            values = stage_groups[key]
            run_id = str(key[0]) if key[0] is not None else ""
            measured = measured_total_ms.get(run_id)
            known = run_leaf_ms.get(run_id, 0.0)
            unattributed = (
                max(measured - known, 0.0) if measured is not None else None
            )
            writer.writerow(
                {
                    "row_type": "stage",
                    "run_id": key[0],
                    "request_phase": key[1],
                    "method": key[2],
                    "model": key[3],
                    "chunk_count": key[4],
                    "chunk_size_tokens": key[5],
                    "max_new_tokens": key[6],
                    "recompute_ratio": key[7],
                    "lmcache_chunk_size": key[8],
                    "stage": key[9],
                    "category": key[10],
                    "device": key[11],
                    "layer_id": key[12],
                    "count": len(values),
                    "sum_ms": sum(values),
                    "mean_ms": mean(values),
                    "median_ms": median(values),
                    "p95_ms": percentile(values, 0.95),
                    "max_ms": max(values),
                    "min_ms": min(values),
                    "measured_total_ms": measured,
                    "sum_known_profiled_ms": known,
                    "unattributed_ms": unattributed,
                    "unattributed_ratio": (
                        unattributed / measured
                        if measured is not None and measured
                        else None
                    ),
                    "cpu_side_ms": run_bucket_ms.get((run_id, "cpu_side"), 0.0),
                    "gpu_side_ms": run_bucket_ms.get((run_id, "gpu_side"), 0.0),
                    "cpu_control_gpu_work_ms": run_bucket_ms.get(
                        (run_id, "cpu_control_gpu_work"), 0.0
                    ),
                    "sync_wait_ms": run_bucket_ms.get((run_id, "sync_wait"), 0.0),
                    "known_exceeds_measured_total": (
                        known > measured if measured is not None else None
                    ),
                }
            )

    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
