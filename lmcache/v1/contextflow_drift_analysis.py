# SPDX-License-Identifier: Apache-2.0
"""ContextFlow layer-wise KV drift analysis helpers."""

from __future__ import annotations

# Standard
import os
from typing import Any, Optional, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.contextflow_profiler import (
    is_enabled as cf_is_enabled,
    record_event as cf_record_event,
)


_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return whether ContextFlow drift analysis profiling is enabled."""
    return (
        cf_is_enabled()
        and os.environ.get("LMCACHE_CONTEXTFLOW_DRIFT_ANALYSIS", "").lower()
        in _TRUE_VALUES
    )


def record_layer_drift(
    *,
    request_id: Optional[str],
    layer_id: int,
    drift_scores: torch.Tensor,
    drift_scope: str,
    cached_tokens: int,
    selected_indices: Optional[torch.Tensor],
    recompute_ratio: Optional[float],
    scores_are_selected_subset: bool = False,
) -> None:
    """Record one layer's KV drift distribution and repair-token behavior."""
    if not is_enabled():
        return

    scores = drift_scores.detach().to(device="cpu", dtype=torch.float32).flatten()
    selected_positions = _selected_positions(selected_indices)
    selected_scores = _selected_scores(
        scores=scores,
        selected_indices=selected_indices,
        scores_are_selected_subset=scores_are_selected_subset,
    )

    metadata: dict[str, Any] = {
        "drift_metric": "sum((rope(new_k)-cached_old_k)^2) over hidden dim",
        "drift_scope": drift_scope,
        "cached_tokens": int(cached_tokens),
        "evaluated_tokens": int(scores.numel()),
        "evaluated_token_ratio": _safe_ratio(scores.numel(), cached_tokens),
        "recompute_ratio": recompute_ratio,
    }
    metadata.update(_stats(scores, prefix="drift"))
    metadata.update(_mass_metrics(scores, prefix="drift"))

    selected_count = len(selected_positions)
    metadata.update(
        {
            "selected_repair_tokens": selected_count,
            "selected_ratio": _safe_ratio(selected_count, cached_tokens),
            "selected_token_positions": selected_positions,
            "selected_token_position_ranges": _coalesced_ranges(selected_positions),
            "selected_position_min": min(selected_positions)
            if selected_positions
            else None,
            "selected_position_max": max(selected_positions)
            if selected_positions
            else None,
        }
    )
    metadata.update(_stats(selected_scores, prefix="selected_drift"))
    metadata["selected_drift_mass_ratio"] = _safe_ratio(
        float(selected_scores.sum().item()) if selected_scores.numel() else 0.0,
        float(scores.sum().item()) if scores.numel() else 0.0,
    )

    cf_record_event(
        "kv_drift_distribution_layer",
        category="drift_analysis",
        device="GPU",
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def _selected_positions(selected_indices: Optional[torch.Tensor]) -> list[int]:
    if selected_indices is None:
        return []
    selected_values = (
        selected_indices.detach().to(device="cpu", dtype=torch.long).tolist()
    )
    return [int(value) for value in selected_values]


def _selected_scores(
    *,
    scores: torch.Tensor,
    selected_indices: Optional[torch.Tensor],
    scores_are_selected_subset: bool,
) -> torch.Tensor:
    if selected_indices is None:
        return torch.empty(0, dtype=scores.dtype)
    if scores_are_selected_subset:
        return scores
    indices = selected_indices.detach().to(device="cpu", dtype=torch.long)
    if indices.numel() == 0:
        return torch.empty(0, dtype=scores.dtype)
    return scores[indices]


def _stats(values: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    if values.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
            f"{prefix}_min": None,
            f"{prefix}_std": None,
        }

    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_median": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(values, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
        f"{prefix}_p99": float(torch.quantile(values, 0.99).item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
    }


def _mass_metrics(values: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    if values.numel() == 0:
        return {
            f"{prefix}_top1pct_mass_ratio": None,
            f"{prefix}_top5pct_mass_ratio": None,
            f"{prefix}_top10pct_mass_ratio": None,
        }
    return {
        f"{prefix}_top1pct_mass_ratio": _top_mass_ratio(values, 0.01),
        f"{prefix}_top5pct_mass_ratio": _top_mass_ratio(values, 0.05),
        f"{prefix}_top10pct_mass_ratio": _top_mass_ratio(values, 0.10),
    }


def _top_mass_ratio(values: torch.Tensor, fraction: float) -> Optional[float]:
    total = float(values.sum().item())
    if total == 0.0:
        return None
    k = max(1, int(values.numel() * fraction))
    top_values = torch.topk(values, k=k).values
    return float(top_values.sum().item()) / total


def _safe_ratio(numerator: float | int, denominator: float | int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _coalesced_ranges(values: Sequence[int]) -> list[list[int]]:
    if not values:
        return []
    unique_values = sorted(set(int(value) for value in values))
    ranges: list[list[int]] = []
    start = unique_values[0]
    previous = unique_values[0]
    for value in unique_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous + 1])
        start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges
