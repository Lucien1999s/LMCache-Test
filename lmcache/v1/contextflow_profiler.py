# SPDX-License-Identifier: Apache-2.0
"""Lightweight stage profiler for ContextFlow CacheBlend diagnostics.

This module is intentionally disabled by default. It records JSONL events only
when ``LMCACHE_CONTEXTFLOW_PROFILE`` is enabled, so production LMCache paths do
not pay file I/O costs unless a ContextFlow experiment asks for them.
"""

from __future__ import annotations

# Standard
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional


_WRITE_LOCK = threading.Lock()
_TRUE_VALUES = {"1", "true", "yes", "on"}


def configure(
    output_path: str,
    *,
    enabled: bool = True,
    cuda_sync: bool = False,
    include_memory: bool = True,
    run_id: Optional[str] = None,
    condition: Optional[dict[str, Any]] = None,
) -> None:
    """Configure ContextFlow stage profiling through process environment.

    Args:
        output_path: JSONL file path where profiling events should be appended.
        enabled: Whether profiling should be enabled.
        cuda_sync: Whether to synchronize CUDA before stage boundaries.
        include_memory: Whether to attach best-effort GPU memory snapshots.
        run_id: Optional run identifier added to every event.
        condition: Optional experiment metadata added to every event.
    """
    os.environ["LMCACHE_CONTEXTFLOW_PROFILE"] = "1" if enabled else "0"
    os.environ["LMCACHE_CONTEXTFLOW_PROFILE_PATH"] = output_path
    os.environ["LMCACHE_CONTEXTFLOW_PROFILE_CUDA_SYNC"] = "1" if cuda_sync else "0"
    os.environ["LMCACHE_CONTEXTFLOW_PROFILE_MEMORY"] = "1" if include_memory else "0"
    if run_id is not None:
        os.environ["LMCACHE_CONTEXTFLOW_PROFILE_RUN_ID"] = run_id
    if condition is not None:
        os.environ["LMCACHE_CONTEXTFLOW_PROFILE_CONDITION"] = json.dumps(
            _json_safe(condition),
            ensure_ascii=False,
            sort_keys=True,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def is_enabled() -> bool:
    """Return whether ContextFlow stage profiling is currently enabled."""
    return os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE", "").lower() in _TRUE_VALUES


class StageTimer:
    """Manual timer for profiling spans with multiple return paths."""

    def __init__(
        self,
        stage: str,
        *,
        category: str,
        device: str,
        request_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.stage = stage
        self.category = category
        self.device = device
        self.request_id = request_id
        self.layer_id = layer_id
        self.metadata = dict(metadata or {})
        self._enabled = is_enabled()
        self._finished = False
        self._start_ns = 0
        self._start_gpu_memory_mb: Optional[float] = None
        if self._enabled:
            _cuda_sync_if_requested()
            self._start_gpu_memory_mb = _gpu_memory_mb()
            self._start_ns = time.perf_counter_ns()

    def add_metadata(self, **metadata: Any) -> None:
        """Add metadata that will be emitted when the timer finishes."""
        self.metadata.update(metadata)

    def finish(self, **metadata: Any) -> None:
        """Finish the span and append one JSONL event."""
        if self._finished:
            return
        self._finished = True
        if metadata:
            self.metadata.update(metadata)
        if not self._enabled:
            return
        _cuda_sync_if_requested()
        end_ns = time.perf_counter_ns()
        end_gpu_memory_mb = _gpu_memory_mb()
        record_event(
            self.stage,
            event_type="span",
            category=self.category,
            device=self.device,
            request_id=self.request_id,
            layer_id=self.layer_id,
            start_ns=self._start_ns,
            end_ns=end_ns,
            duration_ms=(end_ns - self._start_ns) / 1_000_000,
            gpu_memory_mb=end_gpu_memory_mb,
            metadata={
                **self.metadata,
                "gpu_memory_start_mb": self._start_gpu_memory_mb,
                "gpu_memory_end_mb": end_gpu_memory_mb,
            },
        )


class _SpanContext:
    def __init__(
        self,
        stage: str,
        *,
        category: str,
        device: str,
        request_id: Optional[str] = None,
        layer_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._timer = start_span(
            stage,
            category=category,
            device=device,
            request_id=request_id,
            layer_id=layer_id,
            metadata=metadata,
        )

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        metadata: dict[str, Any] = {}
        if exc_value is not None:
            metadata["exception"] = repr(exc_value)
        self._timer.finish(**metadata)
        return False


def start_span(
    stage: str,
    *,
    category: str,
    device: str,
    request_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> StageTimer:
    """Start a manually finished ContextFlow profiling span.

    Args:
        stage: Stable stage name.
        category: Stage group such as ``runner``, ``scheduler``, or ``worker``.
        device: Human-readable execution device, for example ``CPU`` or ``GPU``.
        request_id: Optional request identifier.
        layer_id: Optional model layer index.
        metadata: Optional JSON-serializable stage metadata.

    Returns:
        A timer object. Calling ``finish`` is safe even when profiling is disabled.
    """
    return StageTimer(
        stage,
        category=category,
        device=device,
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def span(
    stage: str,
    *,
    category: str,
    device: str,
    request_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> _SpanContext:
    """Profile a block as one ContextFlow span."""
    return _SpanContext(
        stage,
        category=category,
        device=device,
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )


def record_event(
    stage: str,
    *,
    event_type: str = "instant",
    category: str,
    device: str,
    request_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    duration_ms: Optional[float] = None,
    gpu_memory_mb: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Append a ContextFlow profiling event to the configured JSONL file."""
    if not is_enabled():
        return
    path = os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE_PATH")
    if not path:
        return

    now_ns = time.perf_counter_ns()
    event = {
        "schema_version": 1,
        "event_type": event_type,
        "stage": stage,
        "category": category,
        "device": device,
        "request_id": request_id,
        "layer_id": layer_id,
        "start_ns": start_ns if start_ns is not None else now_ns,
        "end_ns": end_ns if end_ns is not None else now_ns,
        "duration_ms": duration_ms,
        "wall_time_unix_ns": time.time_ns(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "run_id": os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE_RUN_ID"),
        "condition": _condition(),
        "gpu_memory_mb": gpu_memory_mb if gpu_memory_mb is not None else _gpu_memory_mb(),
        "metadata": _json_safe(metadata or {}),
    }

    line = json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True)
    with _WRITE_LOCK:
        with Path(path).open("a") as f:
            f.write(line + "\n")


def _condition() -> dict[str, Any]:
    raw = os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE_CONDITION")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _cuda_sync_if_requested() -> None:
    if os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE_CUDA_SYNC", "").lower() not in (
        _TRUE_VALUES
    ):
        return
    try:
        # Third Party
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _gpu_memory_mb() -> Optional[float]:
    if os.environ.get("LMCACHE_CONTEXTFLOW_PROFILE_MEMORY", "").lower() not in (
        _TRUE_VALUES
    ):
        return None
    try:
        # Third Party
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / (1024 * 1024)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", None)),
        }
    return str(value)
