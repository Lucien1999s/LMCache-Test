# SPDX-License-Identifier: Apache-2.0
"""Fine-grained ContextFlow profiling helpers.

This module gates additional fine-breakdown hooks behind an explicit
experiment flag. The regular ContextFlow stage profiler may be enabled for
coarse experiments; these helpers only emit events when
``CONTEXTFLOW_FINE_BREAKDOWN`` or ``LMCACHE_CONTEXTFLOW_FINE_BREAKDOWN`` is
also enabled.
"""

from __future__ import annotations

# Standard
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional
import os

# First Party
from lmcache.v1.contextflow_profiler import (
    record_event as cf_record_event,
    span as cf_span,
    start_span as cf_start_span,
)
from lmcache.v1.contextflow_profiler import is_enabled as cf_is_enabled

_TRUE_VALUES = {"1", "true", "yes", "on"}


class _NoOpTimer:
    """No-op manual timer matching the ContextFlow StageTimer surface."""

    def add_metadata(self, **metadata: Any) -> None:
        """Accept metadata without recording it."""
        return

    def finish(self, **metadata: Any) -> None:
        """Finish without recording any event."""
        return


def is_enabled() -> bool:
    """Return whether fine ContextFlow breakdown profiling is enabled."""
    return cf_is_enabled() and (
        os.environ.get("CONTEXTFLOW_FINE_BREAKDOWN", "").lower() in _TRUE_VALUES
        or os.environ.get("LMCACHE_CONTEXTFLOW_FINE_BREAKDOWN", "").lower()
        in _TRUE_VALUES
    )


def start_span(
    stage: str,
    *,
    category: str,
    device: str,
    request_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Start a manually finished fine-breakdown span.

    Args:
        stage: Stable fine-breakdown stage name.
        category: Stage group such as ``lookup`` or ``storage``.
        device: Human-readable execution device.
        request_id: Optional request identifier.
        layer_id: Optional model layer index.
        metadata: Optional JSON-serializable stage metadata.

    Returns:
        A ContextFlow timer when fine breakdown is enabled, otherwise a no-op
        timer with the same ``finish`` method.
    """
    if not is_enabled():
        return _NoOpTimer()
    return cf_start_span(
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
) -> AbstractContextManager[Any]:
    """Return a profiling span when fine breakdown is enabled.

    Args:
        stage: Stable fine-breakdown stage name.
        category: Stage group such as ``lookup`` or ``storage``.
        device: Human-readable execution device.
        request_id: Optional request identifier.
        layer_id: Optional model layer index.
        metadata: Optional JSON-serializable stage metadata.

    Returns:
        A context manager that records a span, or a no-op context manager.
    """
    if not is_enabled():
        return nullcontext()
    return cf_span(
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
    category: str,
    device: str,
    request_id: Optional[str] = None,
    layer_id: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Record a fine-breakdown instant event when enabled."""
    if not is_enabled():
        return
    cf_record_event(
        stage,
        category=category,
        device=device,
        request_id=request_id,
        layer_id=layer_id,
        metadata=metadata,
    )
