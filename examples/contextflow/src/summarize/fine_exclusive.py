# SPDX-License-Identifier: Apache-2.0
"""Build wall-clock valid exclusive breakdowns from fine profiling events."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


BUCKET_ORDER = [
    "cache_lookup_reuse_decision",
    "scheduler_kv_allocation",
    "connector_metadata_slot_mapping",
    "lmcache_retrieve_load_orchestration",
    "cpu_to_gpu_kv_copy_wait",
    "cacheblend_repair_compute",
    "repaired_kv_writeback",
    "vllm_decode_sampling_output",
    "residual_unknown",
]

BUCKET_PRIORITY = {
    "vllm_decode_sampling_output": 0,
    "repaired_kv_writeback": 1,
    "cpu_to_gpu_kv_copy_wait": 2,
    "cacheblend_repair_compute": 3,
    "cache_lookup_reuse_decision": 4,
    "scheduler_kv_allocation": 5,
    "connector_metadata_slot_mapping": 6,
    "lmcache_retrieve_load_orchestration": 7,
}

OUTER_ONLY_STAGES = {
    "offline_llm_generate_call",
    "cacheblend_load_recompute_repair_writeback",
    "naive_segment_reuse_load_writeback",
    "prefix_chunked_reuse_load_writeback",
}

OUTER_SCOPE_PRIORITY = [
    "cacheblend_load_recompute_repair_writeback",
    "naive_segment_reuse_load_writeback",
    "prefix_chunked_reuse_load_writeback",
    "later_request_cache_lookup",
    "lmcache_engine_lookup_total",
    "cacheblend_retrieve_layer_step",
    "layerwise_model_recompute_and_repair_layer",
    "offline_llm_generate_call",
]


def condition_value(event: dict[str, Any], key: str) -> Any:
    """Return a value from an event condition object."""
    condition = event.get("condition") or {}
    if isinstance(condition, dict):
        return condition.get(key)
    return None


def infer_repeat_id(run_id: str) -> Optional[int]:
    """Infer repeat id from ContextFlow run ids containing ``-rN-``."""
    match = re.search(r"-r(\d+)-", run_id)
    if match is None:
        return None
    return int(match.group(1))


def metadata_value(event: dict[str, Any], key: str) -> Any:
    """Return a value from an event metadata object."""
    metadata = event.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records."""
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def interval_overlap(
    start: int,
    end: int,
    window_start: int,
    window_end: int,
) -> Optional[tuple[int, int]]:
    """Clip an interval to a window."""
    clipped_start = max(start, window_start)
    clipped_end = min(end, window_end)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def classify_stage(event: dict[str, Any]) -> Optional[str]:
    """Classify one event into a mutually exclusive top-level bucket."""
    stage = str(event.get("stage") or "")
    category = str(event.get("category") or "")
    device = str(event.get("device") or "")
    lower = stage.lower()

    if stage in OUTER_ONLY_STAGES:
        return None

    if lower == "vllm_decode_sampling_output":
        return "vllm_decode_sampling_output"

    if "writeback" in lower or lower in {
        "repaired_kv_writeback_layer",
        "kv_layout_writeback_layer",
    }:
        return "repaired_kv_writeback"

    if (
        "cpu_to_gpu" in lower
        or "to_gpu_buffer" in lower
        or "gpu_buffer_stream_sync" in lower
        or "rope_position_correction" in lower
        or "gap_zeroing" in lower
        or device == "CPU->GPU"
    ):
        return "cpu_to_gpu_kv_copy_wait"

    if (
        category in {"model", "attention"}
        or lower in {
            "layerwise_model_recompute_and_repair_layer",
            "rope_on_new_qk",
            "kv_drift_measurement",
            "important_token_topk",
            "important_token_slice",
            "repair_attention_metadata_update",
            "selective_kv_replacement",
            "selected_token_attention_repair",
            "layernorm_qkv_proj",
            "post_attention_mlp",
        }
    ):
        return "cacheblend_repair_compute"

    if category == "connector_metadata" or "slot_mapping_construction" in lower:
        return "connector_metadata_slot_mapping"

    if (
        lower.startswith("scheduler_update_state_after_alloc")
        or lower == "kv_slot_allocation_result"
    ):
        return "scheduler_kv_allocation"

    if (
        "lookup" in lower
        or "contains" in lower
        or "segment_token_database" in lower
        or category in {"lookup", "lookup_client"}
    ):
        return "cache_lookup_reuse_decision"

    if (
        "retrieve" in lower
        or "load_preparation" in lower
        or "load_finalize" in lower
        or "layerwise_batched_get" in lower
        or "batched_get_non_blocking" in lower
        or "send_to_gpu_connector" in lower
        or "token_mask_construction" in lower
        or "start_load_kv" in lower
    ):
        return "lmcache_retrieve_load_orchestration"

    return None


def build_measured_windows(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Find measured offline generate windows by run id."""
    windows: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("stage") != "offline_llm_generate_call":
            continue
        if metadata_value(event, "request_label") != "measured":
            continue
        run_id = event.get("run_id")
        start = event.get("start_ns")
        end = event.get("end_ns")
        if run_id is None or start is None or end is None:
            continue
        windows[str(run_id)] = {
            "run_id": str(run_id),
            "start_ns": int(start),
            "end_ns": int(end),
            "measured_total_ms": (int(end) - int(start)) / 1_000_000,
            "condition": event.get("condition") or {},
        }
    return windows


def add_decode_candidates(
    events: list[dict[str, Any]],
    candidates_by_run: dict[str, list[dict[str, Any]]],
    windows: dict[str, dict[str, Any]],
) -> None:
    """Add vLLM decode/sampling intervals from request metrics."""
    for event in events:
        if event.get("stage") != "vllm_request_metrics":
            continue
        if metadata_value(event, "request_label") != "measured":
            continue
        run_id = event.get("run_id")
        if run_id is None or str(run_id) not in windows:
            continue
        metrics = metadata_value(event, "metrics") or {}
        if not isinstance(metrics, dict):
            continue
        first_token_ts = metrics.get("first_token_ts")
        last_token_ts = metrics.get("last_token_ts")
        if not first_token_ts or not last_token_ts:
            continue
        start = int(float(first_token_ts) * 1_000_000_000)
        end = int(float(last_token_ts) * 1_000_000_000)
        clipped = interval_overlap(
            start,
            end,
            windows[str(run_id)]["start_ns"],
            windows[str(run_id)]["end_ns"],
        )
        if clipped is None:
            continue
        candidates_by_run[str(run_id)].append(
            {
                "bucket": "vllm_decode_sampling_output",
                "start_ns": clipped[0],
                "end_ns": clipped[1],
                "stage": "vllm_decode_sampling_output",
                "category": "vllm_metrics",
                "device": "CPU/GPU",
            }
        )


def build_candidates(
    events: list[dict[str, Any]],
    windows: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Build bucket candidates and outer-scope intervals per run."""
    candidates_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outer_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        run_id = event.get("run_id")
        if run_id is None or str(run_id) not in windows:
            continue
        if event.get("duration_ms") is None:
            continue
        start = event.get("start_ns")
        end = event.get("end_ns")
        if start is None or end is None:
            continue
        clipped = interval_overlap(
            int(start),
            int(end),
            windows[str(run_id)]["start_ns"],
            windows[str(run_id)]["end_ns"],
        )
        if clipped is None:
            continue

        stage = str(event.get("stage") or "")
        if stage in OUTER_SCOPE_PRIORITY:
            outer_by_run[str(run_id)].append(
                {
                    "stage": stage,
                    "start_ns": clipped[0],
                    "end_ns": clipped[1],
                }
            )

        bucket = classify_stage(event)
        if bucket is None:
            continue
        candidates_by_run[str(run_id)].append(
            {
                "bucket": bucket,
                "start_ns": clipped[0],
                "end_ns": clipped[1],
                "stage": stage,
                "category": event.get("category"),
                "device": event.get("device"),
            }
        )

    add_decode_candidates(events, candidates_by_run, windows)
    return candidates_by_run, outer_by_run


def most_specific_outer_scope(
    start_ns: int,
    end_ns: int,
    outer_intervals: list[dict[str, Any]],
) -> str:
    """Return the most specific known outer scope covering a segment."""
    covering = []
    for outer in outer_intervals:
        if outer["start_ns"] <= start_ns and outer["end_ns"] >= end_ns:
            covering.append(str(outer["stage"]))
    for stage in OUTER_SCOPE_PRIORITY:
        if stage in covering:
            return stage
    return "measured_generate_outer"


def segment_run(
    run: dict[str, Any],
    candidates: list[dict[str, Any]],
    outer_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split a measured run into exclusive atomic segments."""
    start = run["start_ns"]
    end = run["end_ns"]
    points = {start, end}
    for candidate in candidates:
        points.add(candidate["start_ns"])
        points.add(candidate["end_ns"])
    ordered_points = sorted(points)

    segments: list[dict[str, Any]] = []
    for left, right in zip(ordered_points, ordered_points[1:], strict=False):
        if right <= left:
            continue
        active = [
            candidate
            for candidate in candidates
            if candidate["start_ns"] <= left and candidate["end_ns"] >= right
        ]
        if active:
            active.sort(
                key=lambda item: (
                    BUCKET_PRIORITY.get(item["bucket"], 999),
                    -(item["end_ns"] - item["start_ns"]),
                )
            )
            chosen = active[0]
            bucket = chosen["bucket"]
            contributing_stages = sorted({item["stage"] for item in active})
            contributing_devices = sorted({str(item["device"]) for item in active})
            outer_scope = None
        else:
            bucket = "residual_unknown"
            contributing_stages = []
            contributing_devices = []
            outer_scope = most_specific_outer_scope(left, right, outer_intervals)

        segments.append(
            {
                "run_id": run["run_id"],
                "bucket": bucket,
                "start_ns": left,
                "end_ns": right,
                "duration_ms": (right - left) / 1_000_000,
                "contributing_stages": contributing_stages,
                "contributing_devices": contributing_devices,
                "residual_outer_scope": outer_scope,
                "condition": run["condition"],
                "measured_total_ms": run["measured_total_ms"],
            }
        )
    return merge_adjacent_segments(segments)


def merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent segments with the same bucket and residual scope."""
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if (
            merged
            and merged[-1]["bucket"] == segment["bucket"]
            and merged[-1].get("residual_outer_scope")
            == segment.get("residual_outer_scope")
            and merged[-1]["end_ns"] == segment["start_ns"]
        ):
            previous = merged[-1]
            previous["end_ns"] = segment["end_ns"]
            previous["duration_ms"] += segment["duration_ms"]
            previous["contributing_stages"] = sorted(
                set(previous["contributing_stages"])
                | set(segment["contributing_stages"])
            )
            previous["contributing_devices"] = sorted(
                set(previous["contributing_devices"])
                | set(segment["contributing_devices"])
            )
            continue
        merged.append(segment)
    return merged


def write_events(path: Path, segments_by_run: dict[str, list[dict[str, Any]]]) -> None:
    """Write exclusive segment JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for run_id in sorted(segments_by_run):
            for segment in segments_by_run[run_id]:
                f.write(json.dumps(segment, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(
    path: Path,
    windows: dict[str, dict[str, Any]],
    segments_by_run: dict[str, list[dict[str, Any]]],
) -> None:
    """Write exclusive summary CSV."""
    fieldnames = [
        "run_id",
        "repeat_id",
        "method",
        "model",
        "chunk_count",
        "chunk_size_tokens",
        "max_new_tokens",
        "recompute_ratio",
        "lmcache_chunk_size",
        "bucket",
        "exclusive_ms",
        "exclusive_ratio",
        "measured_total_ms",
        "sum_exclusive_non_residual_ms",
        "residual_ms",
        "residual_ratio",
        "valid_sum_le_total",
        "segment_count",
        "residual_outer_scopes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in sorted(windows):
            run = windows[run_id]
            measured = float(run["measured_total_ms"])
            bucket_ms: dict[str, float] = defaultdict(float)
            bucket_segments: dict[str, int] = defaultdict(int)
            residual_scopes = set()
            for segment in segments_by_run.get(run_id, []):
                bucket = segment["bucket"]
                bucket_ms[bucket] += float(segment["duration_ms"])
                bucket_segments[bucket] += 1
                if bucket == "residual_unknown" and segment.get(
                    "residual_outer_scope"
                ):
                    residual_scopes.add(str(segment["residual_outer_scope"]))
            non_residual = sum(
                duration
                for bucket, duration in bucket_ms.items()
                if bucket != "residual_unknown"
            )
            residual = bucket_ms.get("residual_unknown", 0.0)
            valid = non_residual + residual <= measured + 1e-6
            condition = run["condition"]
            for bucket in BUCKET_ORDER:
                duration = bucket_ms.get(bucket, 0.0)
                writer.writerow(
                    {
                        "run_id": run_id,
                        "repeat_id": infer_repeat_id(run_id),
                        "method": condition.get("method"),
                        "model": condition.get("model"),
                        "chunk_count": condition.get("chunk_count"),
                        "chunk_size_tokens": condition.get("chunk_size_tokens"),
                        "max_new_tokens": condition.get("max_new_tokens"),
                        "recompute_ratio": condition.get("recompute_ratio"),
                        "lmcache_chunk_size": condition.get("lmcache_chunk_size"),
                        "bucket": bucket,
                        "exclusive_ms": duration,
                        "exclusive_ratio": duration / measured if measured else 0.0,
                        "measured_total_ms": measured,
                        "sum_exclusive_non_residual_ms": non_residual,
                        "residual_ms": residual,
                        "residual_ratio": residual / measured if measured else 0.0,
                        "valid_sum_le_total": valid,
                        "segment_count": bucket_segments.get(bucket, 0),
                        "residual_outer_scopes": (
                            ";".join(sorted(residual_scopes))
                            if bucket == "residual_unknown"
                            else ""
                        ),
                    }
                )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("events_jsonl")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--events-output", required=True)
    args = parser.parse_args()

    events = load_jsonl(Path(args.events_jsonl))
    windows = build_measured_windows(events)
    candidates_by_run, outer_by_run = build_candidates(events, windows)
    segments_by_run = {
        run_id: segment_run(
            run,
            candidates_by_run.get(run_id, []),
            outer_by_run.get(run_id, []),
        )
        for run_id, run in windows.items()
    }
    write_events(Path(args.events_output), segments_by_run)
    write_summary(Path(args.summary_output), windows, segments_by_run)
    print(f"Wrote exclusive events JSONL: {args.events_output}")
    print(f"Wrote exclusive summary CSV: {args.summary_output}")


if __name__ == "__main__":
    main()
