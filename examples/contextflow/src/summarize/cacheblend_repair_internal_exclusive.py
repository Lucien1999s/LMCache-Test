# SPDX-License-Identifier: Apache-2.0
"""Build exclusive CacheBlend repair-internal breakdowns from fine events."""

from __future__ import annotations

# Standard
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


PARENT_STAGES = {
    "layerwise_model_recompute_and_repair_layer",
    "cached_kv_gpu_buffer_stream_sync_layer",
    "cached_k_rope_position_correction_layer",
    "cached_kv_gap_zeroing_layer",
    "repaired_kv_writeback_layer",
}

BUCKET_ORDER = [
    "qkv_recompute",
    "rope_on_new_qk",
    "cached_kv_sync_wait",
    "cached_k_position_correction_rope",
    "cached_k_gap_zeroing",
    "kv_drift_measurement",
    "high_drift_token_topk",
    "important_token_slice",
    "selective_kv_replacement",
    "attention_metadata_update",
    "selected_token_attention_repair",
    "post_attention_mlp",
    "repaired_kv_writeback",
    "repair_internal_residual",
]

BUCKET_PRIORITY = {
    "cached_kv_sync_wait": 0,
    "cached_k_position_correction_rope": 1,
    "cached_k_gap_zeroing": 2,
    "rope_on_new_qk": 3,
    "kv_drift_measurement": 4,
    "high_drift_token_topk": 5,
    "important_token_slice": 6,
    "selective_kv_replacement": 7,
    "attention_metadata_update": 8,
    "selected_token_attention_repair": 9,
    "post_attention_mlp": 10,
    "repaired_kv_writeback": 11,
    # This span wraps process_qkv in the current implementation, so it must
    # lose to the finer repair children and only keep its exclusive remainder.
    "qkv_recompute": 12,
}

STAGE_TO_BUCKET = {
    "layernorm_qkv_proj": "qkv_recompute",
    "rope_on_new_qk": "rope_on_new_qk",
    "cached_kv_gpu_buffer_stream_sync_layer": "cached_kv_sync_wait",
    "cached_k_rope_position_correction_layer": "cached_k_position_correction_rope",
    "cached_kv_gap_zeroing_layer": "cached_k_gap_zeroing",
    "kv_drift_measurement": "kv_drift_measurement",
    "important_token_topk": "high_drift_token_topk",
    "important_token_slice": "important_token_slice",
    "selective_kv_replacement": "selective_kv_replacement",
    "repair_attention_metadata_update": "attention_metadata_update",
    "selected_token_attention_repair": "selected_token_attention_repair",
    "post_attention_mlp": "post_attention_mlp",
    "repaired_kv_writeback_layer": "repaired_kv_writeback",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a profiling event file."""
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def infer_repeat_id(run_id: str) -> Optional[int]:
    """Infer repeat id from ContextFlow run ids containing ``-rN-``."""
    match = re.search(r"-r(\d+)-", run_id)
    if match is None:
        return None
    return int(match.group(1))


def metadata(event: dict[str, Any]) -> dict[str, Any]:
    """Return event metadata as a dict."""
    raw = event.get("metadata")
    if isinstance(raw, dict):
        return raw
    return {}


def int_or_none(value: Any) -> Optional[int]:
    """Convert a value to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clip_interval(
    start_ns: int,
    end_ns: int,
    window_start_ns: int,
    window_end_ns: int,
) -> Optional[tuple[int, int]]:
    """Clip an interval to a parent window."""
    clipped_start = max(start_ns, window_start_ns)
    clipped_end = min(end_ns, window_end_ns)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def classify_child(event: dict[str, Any]) -> Optional[str]:
    """Classify a fine event into a repair-internal child bucket."""
    stage = str(event.get("stage") or "")
    return STAGE_TO_BUCKET.get(stage)


def condition_value(event: dict[str, Any], key: str) -> Any:
    """Return a condition value from a profiling event."""
    condition = event.get("condition")
    if isinstance(condition, dict):
        return condition.get(key)
    return None


def span_event(event: dict[str, Any]) -> bool:
    """Return whether an event has a non-empty wall-clock span."""
    return (
        event.get("event_type") == "span"
        and event.get("start_ns") is not None
        and event.get("end_ns") is not None
        and int(event["end_ns"]) > int(event["start_ns"])
    )


def layer_key(event: dict[str, Any]) -> Optional[tuple[str, int]]:
    """Return the run/layer key for one event."""
    run_id = event.get("run_id")
    layer_id = event.get("layer_id")
    if run_id is None or layer_id is None:
        return None
    return str(run_id), int(layer_id)


def collect_token_counts(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, int]]:
    """Collect cached and selected token counts by run/layer."""
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"cached_tokens": 0, "selected_repair_tokens": 0}
    )
    for event in events:
        key = layer_key(event)
        if key is None:
            continue
        meta = metadata(event)
        cached_candidates = [
            int_or_none(meta.get("tokens")),
            int_or_none(meta.get("kv_tokens")),
        ]
        selected_candidates = [
            int_or_none(meta.get("selected_tokens")),
            int_or_none(meta.get("topk")),
            int_or_none(meta.get("query_tokens"))
            if event.get("stage") == "selected_token_attention_repair"
            else None,
        ]
        for value in cached_candidates:
            if value is not None:
                counts[key]["cached_tokens"] = max(
                    counts[key]["cached_tokens"],
                    value,
                )
        for value in selected_candidates:
            if value is not None:
                counts[key]["selected_repair_tokens"] = max(
                    counts[key]["selected_repair_tokens"],
                    value,
                )
    return counts


def build_condition_by_run(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build run-level condition metadata."""
    condition_by_run: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id")
        condition = event.get("condition")
        if run_id is not None and isinstance(condition, dict):
            condition_by_run.setdefault(str(run_id), condition)
    return condition_by_run


def build_parent_intervals(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Build repair-internal parent intervals by run/layer.

    The layerwise model span is the broad parent for q/k/v recompute, token
    selection, attention repair, and MLP. Position correction, stream sync,
    gap zeroing, and writeback are outside that model span in the layerwise
    GPU connector pipeline, so they become additional parent intervals for
    the same layer.
    """
    intervals_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not span_event(event):
            continue
        if event.get("stage") not in PARENT_STAGES:
            continue
        key = layer_key(event)
        if key is None:
            continue
        intervals_by_key[key].append(
            {
                "start_ns": int(event["start_ns"]),
                "end_ns": int(event["end_ns"]),
                "stage": event.get("stage"),
            }
        )
    return {
        key: merge_parent_intervals(intervals)
        for key, intervals in intervals_by_key.items()
    }


def merge_parent_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping parent intervals while preserving stage labels."""
    ordered = sorted(intervals, key=lambda item: (item["start_ns"], item["end_ns"]))
    merged: list[dict[str, Any]] = []
    for interval in ordered:
        if merged and interval["start_ns"] <= merged[-1]["end_ns"]:
            previous = merged[-1]
            previous["end_ns"] = max(previous["end_ns"], interval["end_ns"])
            previous["parent_stages"] = sorted(
                set(previous["parent_stages"]) | {str(interval["stage"])}
            )
            continue
        merged.append(
            {
                "start_ns": interval["start_ns"],
                "end_ns": interval["end_ns"],
                "parent_stages": [str(interval["stage"])],
            }
        )
    return merged


def build_child_candidates(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Build child bucket candidates by run/layer."""
    candidates_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not span_event(event):
            continue
        bucket = classify_child(event)
        if bucket is None:
            continue
        key = layer_key(event)
        if key is None:
            continue
        candidates_by_key[key].append(
            {
                "bucket": bucket,
                "stage": str(event.get("stage")),
                "device": str(event.get("device")),
                "category": str(event.get("category")),
                "start_ns": int(event["start_ns"]),
                "end_ns": int(event["end_ns"]),
                "metadata": metadata(event),
            }
        )
    return candidates_by_key


def segment_parent_interval(
    key: tuple[str, int],
    parent: dict[str, Any],
    candidates: list[dict[str, Any]],
    counts: dict[str, int],
    condition: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split one parent interval into exclusive atomic repair segments."""
    run_id, layer_id = key
    parent_start = int(parent["start_ns"])
    parent_end = int(parent["end_ns"])
    clipped_candidates = []
    points = {parent_start, parent_end}
    for candidate in candidates:
        clipped = clip_interval(
            int(candidate["start_ns"]),
            int(candidate["end_ns"]),
            parent_start,
            parent_end,
        )
        if clipped is None:
            continue
        copied = dict(candidate)
        copied["start_ns"] = clipped[0]
        copied["end_ns"] = clipped[1]
        clipped_candidates.append(copied)
        points.update(clipped)

    segments: list[dict[str, Any]] = []
    ordered_points = sorted(points)
    for left, right in zip(ordered_points, ordered_points[1:], strict=False):
        if right <= left:
            continue
        active = [
            candidate
            for candidate in clipped_candidates
            if candidate["start_ns"] <= left and candidate["end_ns"] >= right
        ]
        if active:
            active.sort(
                key=lambda item: (
                    BUCKET_PRIORITY.get(str(item["bucket"]), 999),
                    -(int(item["end_ns"]) - int(item["start_ns"])),
                )
            )
            chosen = active[0]
            bucket = str(chosen["bucket"])
            event_name = str(chosen["stage"])
            device_type = str(chosen["device"])
            contributing_stages = sorted({str(item["stage"]) for item in active})
        else:
            bucket = "repair_internal_residual"
            event_name = "repair_internal_residual"
            device_type = "unknown"
            contributing_stages = []

        segments.append(
            {
                "run_id": run_id,
                "repeat_id": infer_repeat_id(run_id),
                "chunk_count": condition.get("chunk_count"),
                "max_new_tokens": condition.get("max_new_tokens"),
                "layer_id": layer_id,
                "event_name": event_name,
                "bucket": bucket,
                "start_ns": left,
                "end_ns": right,
                "duration_ms": (right - left) / 1_000_000,
                "device_type": device_type,
                "token_count": counts.get("cached_tokens", 0),
                "selected_repair_tokens": counts.get("selected_repair_tokens", 0),
                "cached_tokens": counts.get("cached_tokens", 0),
                "parent_event": "cacheblend_repair_internal_total",
                "parent_stages": parent["parent_stages"],
                "contributing_stages": contributing_stages,
                "condition": condition,
            }
        )
    return merge_adjacent_segments(segments)


def merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent repair segments with identical attribution."""
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if (
            merged
            and merged[-1]["bucket"] == segment["bucket"]
            and merged[-1]["event_name"] == segment["event_name"]
            and merged[-1]["device_type"] == segment["device_type"]
            and merged[-1]["end_ns"] == segment["start_ns"]
        ):
            previous = merged[-1]
            previous["end_ns"] = segment["end_ns"]
            previous["duration_ms"] += segment["duration_ms"]
            previous["parent_stages"] = sorted(
                set(previous["parent_stages"]) | set(segment["parent_stages"])
            )
            previous["contributing_stages"] = sorted(
                set(previous["contributing_stages"])
                | set(segment["contributing_stages"])
            )
            continue
        merged.append(segment)
    return merged


def build_segments(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Build all repair-internal exclusive segments."""
    parent_intervals = build_parent_intervals(events)
    child_candidates = build_child_candidates(events)
    token_counts = collect_token_counts(events)
    condition_by_run = build_condition_by_run(events)

    segments_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key in sorted(parent_intervals):
        run_id, _ = key
        condition = condition_by_run.get(run_id, {})
        counts = token_counts.get(
            key,
            {"cached_tokens": 0, "selected_repair_tokens": 0},
        )
        segments: list[dict[str, Any]] = []
        for parent in parent_intervals[key]:
            segments.extend(
                segment_parent_interval(
                    key,
                    parent,
                    child_candidates.get(key, []),
                    counts,
                    condition,
                )
            )
        segments_by_key[key] = merge_adjacent_segments(
            sorted(segments, key=lambda item: (item["start_ns"], item["end_ns"]))
        )
    return segments_by_key


def write_events(
    path: Path,
    segments_by_key: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    """Write exclusive repair-internal events as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for key in sorted(segments_by_key):
            for segment in segments_by_key[key]:
                f.write(json.dumps(segment, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(
    path: Path,
    segments_by_key: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    """Write per-run, per-layer, per-bucket repair summary."""
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
        "layer_id",
        "bucket",
        "exclusive_ms",
        "exclusive_ratio",
        "repair_internal_total_ms",
        "sum_exclusive_non_residual_ms",
        "repair_internal_residual_ms",
        "repair_internal_residual_ratio",
        "selected_repair_tokens",
        "cached_tokens",
        "valid_sum_le_total",
        "segment_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(segments_by_key):
            run_id, layer_id = key
            segments = segments_by_key[key]
            if not segments:
                continue
            condition = segments[0].get("condition") or {}
            total_ms = sum(float(segment["duration_ms"]) for segment in segments)
            bucket_ms: dict[str, float] = defaultdict(float)
            bucket_count: dict[str, int] = defaultdict(int)
            selected_tokens = 0
            cached_tokens = 0
            for segment in segments:
                bucket = str(segment["bucket"])
                bucket_ms[bucket] += float(segment["duration_ms"])
                bucket_count[bucket] += 1
                selected_tokens = max(
                    selected_tokens,
                    int(segment.get("selected_repair_tokens") or 0),
                )
                cached_tokens = max(
                    cached_tokens,
                    int(segment.get("cached_tokens") or 0),
                )

            non_residual_ms = sum(
                value
                for bucket, value in bucket_ms.items()
                if bucket != "repair_internal_residual"
            )
            residual_ms = bucket_ms.get("repair_internal_residual", 0.0)
            valid = non_residual_ms + residual_ms <= total_ms + 1e-6
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
                        "layer_id": layer_id,
                        "bucket": bucket,
                        "exclusive_ms": duration,
                        "exclusive_ratio": duration / total_ms if total_ms else 0.0,
                        "repair_internal_total_ms": total_ms,
                        "sum_exclusive_non_residual_ms": non_residual_ms,
                        "repair_internal_residual_ms": residual_ms,
                        "repair_internal_residual_ratio": (
                            residual_ms / total_ms if total_ms else 0.0
                        ),
                        "selected_repair_tokens": selected_tokens,
                        "cached_tokens": cached_tokens,
                        "valid_sum_le_total": valid,
                        "segment_count": bucket_count.get(bucket, 0),
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
    segments_by_key = build_segments(events)
    write_events(Path(args.events_output), segments_by_key)
    write_summary(Path(args.summary_output), segments_by_key)
    print(f"Wrote repair-internal exclusive events JSONL: {args.events_output}")
    print(f"Wrote repair-internal exclusive summary CSV: {args.summary_output}")


if __name__ == "__main__":
    main()
