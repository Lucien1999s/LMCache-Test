# SPDX-License-Identifier: Apache-2.0
"""ContextFlow KV layout accounting helpers.

This module is intentionally passive unless both the ContextFlow profiler and
``LMCACHE_CONTEXTFLOW_KV_LAYOUT`` are enabled. It records byte/layout metadata
for CacheBlend diagnostics without changing KV load, repair, or writeback
behavior.
"""

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
    """Return whether ContextFlow KV layout diagnostics are enabled."""
    return (
        cf_is_enabled()
        and os.environ.get("LMCACHE_CONTEXTFLOW_KV_LAYOUT", "").lower()
        in _TRUE_VALUES
    )


def record_cpu_to_gpu_load_layer(
    *,
    request_id: Optional[str],
    layer_id: int,
    starts: Sequence[int],
    ends: Sequence[int],
    memory_objs: Sequence[Any],
    slot_mapping: torch.Tensor,
    block_size: Optional[int],
) -> None:
    """Record per-layer CPU-to-GPU cached KV load accounting."""
    if not is_enabled():
        return

    retrieved_tokens = _range_token_count(starts, ends)
    metadata = {
        "retrieved_tokens": retrieved_tokens,
        "retrieved_segments": len(starts),
        "retrieved_logical_ranges": _ranges(starts, ends),
        "actual_loaded_bytes": sum(
            _memory_obj_logical_bytes(obj) for obj in memory_objs
        ),
        "actual_loaded_physical_bytes": sum(
            _memory_obj_physical_bytes(obj) for obj in memory_objs
        ),
        "number_of_copy_ops": len(memory_objs) * 2,
    }
    metadata.update(
        _slot_metrics(
            _gather_range_slots(slot_mapping, starts, ends),
            block_size=block_size,
            prefix="loaded",
        )
    )
    if _include_scatter():
        loaded_positions, loaded_slots = _range_positions_and_slots(
            slot_mapping,
            starts,
            ends,
        )
        metadata.update(
            _raw_layout_metadata(
                logical_positions=loaded_positions,
                slots=loaded_slots,
                block_size=block_size,
                prefix="loaded",
            )
        )
        metadata["loaded_copy_ops"] = _load_copy_ops(starts, ends)
        metadata["loaded_lmcache_gpu_buffer_ranges"] = _buffer_ranges(starts, ends)
    cf_record_event(
        "kv_layout_cpu_to_gpu_load_layer",
        category="kv_layout",
        device="CPU->GPU",
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def record_writeback_layer(
    *,
    request_id: Optional[str],
    layer_id: int,
    slot_mapping_full: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    hidden_dim_size: int,
    element_size: int,
    block_size: Optional[int],
) -> None:
    """Record per-layer LMCache-buffer-to-vLLM-paged-cache writeback accounting."""
    if not is_enabled():
        return

    writeback_tokens = int(slot_mapping_full.numel())
    retrieved_tokens = _range_token_count(starts, ends)
    metadata = {
        "writeback_span_tokens": writeback_tokens,
        "retrieved_tokens": retrieved_tokens,
        "span_gap_tokens": max(0, writeback_tokens - retrieved_tokens),
        "actual_writeback_bytes": writeback_tokens
        * 2
        * int(hidden_dim_size)
        * int(element_size),
        "number_of_copy_ops": 1,
        "writeback_kernel_ops": 1,
    }
    metadata.update(
        _slot_metrics(slot_mapping_full, block_size=block_size, prefix="writeback")
    )
    if _include_scatter():
        logical_start = int(starts[0]) if starts else 0
        writeback_positions = list(
            range(logical_start, logical_start + writeback_tokens)
        )
        metadata.update(
            _raw_layout_metadata(
                logical_positions=writeback_positions,
                slots=slot_mapping_full,
                block_size=block_size,
                prefix="writeback",
            )
        )
        metadata["writeback_logical_range"] = [
            logical_start,
            logical_start + writeback_tokens,
        ]
        metadata["writeback_gap_logical_positions"] = _gap_positions(starts, ends)
    cf_record_event(
        "kv_layout_writeback_layer",
        category="kv_layout",
        device="GPU->GPU",
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def record_selected_repair_layer(
    *,
    request_id: Optional[str],
    layer_id: int,
    selected_indices: torch.Tensor,
    slot_mapping: Optional[torch.Tensor],
    block_size: Optional[int],
    key_tensor: torch.Tensor,
    value_tensor: torch.Tensor,
) -> None:
    """Record per-layer selected-token repair accounting."""
    if not is_enabled():
        return

    selected_tokens = int(selected_indices.numel())
    bytes_per_selected_token = _bytes_per_token(key_tensor) + _bytes_per_token(
        value_tensor
    )
    metadata: dict[str, Any] = {
        "selected_repair_tokens": selected_tokens,
        "theoretical_selected_token_bytes": selected_tokens
        * bytes_per_selected_token,
        "bytes_per_selected_token": bytes_per_selected_token,
    }

    selected_slots = _selected_slots(slot_mapping, selected_indices)
    if selected_slots is not None:
        metadata.update(
            _slot_metrics(selected_slots, block_size=block_size, prefix="selected")
        )
        if _include_scatter():
            selected_positions = _tensor_to_int_list(selected_indices)
            metadata.update(
                _raw_layout_metadata(
                    logical_positions=selected_positions,
                    slots=selected_slots,
                    block_size=block_size,
                    prefix="selected",
                )
            )
            metadata["selected_index_space"] = "request_logical_token_position"

    cf_record_event(
        "kv_layout_selected_repair_layer",
        category="kv_layout",
        device="GPU",
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def _include_scatter() -> bool:
    return (
        os.environ.get("LMCACHE_CONTEXTFLOW_KV_LAYOUT_SCATTER", "").lower()
        in _TRUE_VALUES
    )


def _bytes_per_token(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return int(tensor.element_size())
    return int(tensor[0].numel() * tensor.element_size())


def _memory_obj_logical_bytes(memory_obj: Any) -> int:
    tensor = getattr(memory_obj, "tensor", None)
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _memory_obj_physical_bytes(memory_obj: Any) -> int:
    try:
        return int(memory_obj.get_physical_size())
    except Exception:
        return 0


def _range_token_count(starts: Sequence[int], ends: Sequence[int]) -> int:
    return int(sum(int(end) - int(start) for start, end in zip(starts, ends)))


def _ranges(starts: Sequence[int], ends: Sequence[int]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def _gather_range_slots(
    slot_mapping: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
) -> torch.Tensor:
    if not starts:
        return torch.empty(0, dtype=torch.long, device=slot_mapping.device)
    chunks = [slot_mapping[int(start) : int(end)] for start, end in zip(starts, ends)]
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def _range_positions_and_slots(
    slot_mapping: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
) -> tuple[list[int], torch.Tensor]:
    positions: list[int] = []
    chunks: list[torch.Tensor] = []
    for start, end in zip(starts, ends):
        int_start = int(start)
        int_end = int(end)
        positions.extend(range(int_start, int_end))
        chunks.append(slot_mapping[int_start:int_end])
    if not chunks:
        return positions, torch.empty(0, dtype=torch.long, device=slot_mapping.device)
    return positions, torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def _selected_slots(
    slot_mapping: Optional[torch.Tensor],
    selected_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if slot_mapping is None:
        return None
    try:
        return slot_mapping[selected_indices.to(device=slot_mapping.device)]
    except Exception:
        return None


def _slot_metrics(
    slots: torch.Tensor,
    *,
    block_size: Optional[int],
    prefix: str,
) -> dict[str, Any]:
    values = [
        int(slot)
        for slot in slots.detach().to(device="cpu", dtype=torch.long).tolist()
        if int(slot) >= 0
    ]
    metrics: dict[str, Any] = {
        f"{prefix}_valid_slot_count": len(values),
        f"{prefix}_invalid_slot_count": int(slots.numel()) - len(values),
        f"{prefix}_min_physical_slot": min(values) if values else None,
        f"{prefix}_max_physical_slot": max(values) if values else None,
    }
    metrics.update(_contiguous_range_metrics(values, prefix=prefix))

    if block_size is not None and block_size > 0 and values:
        blocks = [slot // int(block_size) for slot in values]
        unique_blocks = sorted(set(blocks))
        metrics.update(
            {
                f"{prefix}_block_size": int(block_size),
                f"{prefix}_vllm_blocks_touched": len(unique_blocks),
                f"{prefix}_min_vllm_block": unique_blocks[0],
                f"{prefix}_max_vllm_block": unique_blocks[-1],
            }
        )
    else:
        metrics.update(
            {
                f"{prefix}_block_size": block_size,
                f"{prefix}_vllm_blocks_touched": None,
                f"{prefix}_min_vllm_block": None,
                f"{prefix}_max_vllm_block": None,
            }
        )
    return metrics


def _raw_layout_metadata(
    *,
    logical_positions: Sequence[int],
    slots: torch.Tensor,
    block_size: Optional[int],
    prefix: str,
) -> dict[str, Any]:
    logical_values = [int(position) for position in logical_positions]
    pairs = _valid_logical_slot_pairs(logical_values, slots)
    slot_values = [slot for _, slot in pairs]
    metadata: dict[str, Any] = {
        f"{prefix}_logical_token_positions": [position for position, _ in pairs],
        f"{prefix}_physical_slots": slot_values,
        f"{prefix}_logical_to_physical_slots": pairs,
        f"{prefix}_physical_slot_ranges": _coalesced_ranges(slot_values),
        f"{prefix}_physical_slot_ranges_are_end_exclusive": True,
        f"{prefix}_slot_order_physical_slot_runs": _ordered_contiguous_runs(
            slot_values
        ),
        f"{prefix}_slot_order_runs_are_end_exclusive": True,
    }
    if block_size is not None and block_size > 0:
        block_ids = sorted({slot // int(block_size) for slot in slot_values})
        metadata[f"{prefix}_touched_block_ids"] = block_ids
    else:
        metadata[f"{prefix}_touched_block_ids"] = []
    return metadata


def _valid_logical_slot_pairs(
    logical_positions: Sequence[int],
    slots: torch.Tensor,
) -> list[list[int]]:
    raw_slots = slots.detach().to(device="cpu", dtype=torch.long).tolist()
    pairs: list[list[int]] = []
    for position, slot in zip(logical_positions, raw_slots):
        int_slot = int(slot)
        if int_slot >= 0:
            pairs.append([int(position), int_slot])
    return pairs


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


def _ordered_contiguous_runs(values: Sequence[int]) -> list[list[int]]:
    if not values:
        return []
    ranges: list[list[int]] = []
    start = int(values[0])
    previous = int(values[0])
    for value in values[1:]:
        int_value = int(value)
        if int_value == previous + 1:
            previous = int_value
            continue
        ranges.append([start, previous + 1])
        start = int_value
        previous = int_value
    ranges.append([start, previous + 1])
    return ranges


def _load_copy_ops(starts: Sequence[int], ends: Sequence[int]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        int_start = int(start)
        int_end = int(end)
        ops.append(
            {
                "segment_index": index,
                "logical_range": [int_start, int_end],
                "lmcache_gpu_buffer_range": [
                    int_start - int(starts[0]),
                    int_end - int(starts[0]),
                ],
                "tokens": int_end - int_start,
                "kv_tensor_copy_ops": 2,
            }
        )
    return ops


def _buffer_ranges(starts: Sequence[int], ends: Sequence[int]) -> list[list[int]]:
    if not starts:
        return []
    offset = int(starts[0])
    return [
        [int(start) - offset, int(end) - offset]
        for start, end in zip(starts, ends)
    ]


def _gap_positions(starts: Sequence[int], ends: Sequence[int]) -> list[int]:
    if not starts:
        return []
    covered = set()
    for start, end in zip(starts, ends):
        covered.update(range(int(start), int(end)))
    return [
        position
        for position in range(int(starts[0]), int(ends[-1]))
        if position not in covered
    ]


def _contiguous_range_metrics(values: Sequence[int], *, prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_num_contiguous_slot_ranges": 0,
            f"{prefix}_avg_contiguous_slot_range_length": 0.0,
            f"{prefix}_max_contiguous_slot_range_length": 0,
        }

    lengths: list[int] = []
    current_len = 1
    previous = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == previous + 1:
            current_len += 1
        else:
            lengths.append(current_len)
            current_len = 1
        previous = value
    lengths.append(current_len)
    return {
        f"{prefix}_num_contiguous_slot_ranges": len(lengths),
        f"{prefix}_avg_contiguous_slot_range_length": sum(lengths) / len(lengths),
        f"{prefix}_max_contiguous_slot_range_length": max(lengths),
    }


def _tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().to(device="cpu").tolist()]
