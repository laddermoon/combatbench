"""``sac_metrics_v1`` JSONL event writer.

Canonical storage is ``<run_dir>/metrics/events.jsonl``.  The writer is the
single producer for structured SAC metrics; ad-hoc ``__RAW_STATS__`` output
may mirror it later, but is not the source of truth.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .clocks import CLOCK_FIELDS, SACClockState
from .metric_catalog import MetricCatalog


SAC_METRICS_SCHEMA = "sac_metrics_v1"
EVENTS_RELATIVE_PATH = Path("metrics") / "events.jsonl"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 10_000:
            raise ValueError(
                f"metric event context array is too large: shape={value.shape}"
            )
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metric event contains non-finite float {value}")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _validate_metric_value(name: str, value: Any) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"metric {name!r} must be a finite numeric scalar, "
            f"got {type(value).__name__}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"metric {name!r} is non-finite: {value}")
    return out


@dataclass(frozen=True)
class MetricEvent:
    event_type: str
    clocks: Mapping[str, int]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SAC_METRICS_SCHEMA
    timestamp_unix: float = field(default_factory=time.time)

    def validate(self, catalog: Optional[MetricCatalog] = None) -> "MetricEvent":
        catalog = catalog or MetricCatalog()
        if self.schema_version != SAC_METRICS_SCHEMA:
            raise ValueError(
                f"unsupported metrics schema {self.schema_version!r}; expected "
                f"{SAC_METRICS_SCHEMA!r}"
            )
        catalog.validate_metrics(self.event_type, self.metrics)
        clock = dict(self.clocks)
        missing = [name for name in CLOCK_FIELDS if name not in clock]
        if missing:
            raise ValueError(f"metric event clocks missing {missing}")
        for name in CLOCK_FIELDS:
            value = clock[name]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(
                    f"clock {name!r} must be a non-negative integer, got {value!r}"
                )
        normalized_metrics = {
            str(name): _validate_metric_value(str(name), value)
            for name, value in self.metrics.items()
        }
        _to_jsonable(self.context)
        return MetricEvent(
            event_type=self.event_type,
            clocks={name: int(clock[name]) for name in CLOCK_FIELDS},
            metrics=normalized_metrics,
            context=_to_jsonable(self.context),
            schema_version=self.schema_version,
            timestamp_unix=float(self.timestamp_unix),
        )

    def to_dict(self) -> Dict[str, Any]:
        normalized = self.validate()
        return {
            "schema_version": normalized.schema_version,
            "event_type": normalized.event_type,
            "timestamp_unix": normalized.timestamp_unix,
            "clocks": dict(normalized.clocks),
            "metrics": dict(normalized.metrics),
            "context": _to_jsonable(normalized.context),
        }


class SACMetricsWriter:
    """Append-only JSONL writer for canonical SAC metrics events."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        fsync: bool = False,
        catalog: Optional[MetricCatalog] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.path = self.metrics_dir / "events.jsonl"
        self.fsync = bool(fsync)
        self.catalog = catalog or MetricCatalog()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def emit(
        self,
        event_type: str,
        *,
        clocks: SACClockState | Mapping[str, int],
        metrics: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> MetricEvent:
        if self._closed:
            raise RuntimeError("SACMetricsWriter is closed")
        clock_map = (
            clocks.snapshot()
            if isinstance(clocks, SACClockState)
            else SACClockState.from_mapping(clocks).snapshot()
        )
        event = MetricEvent(
            event_type=str(event_type),
            clocks=clock_map,
            metrics=dict(metrics or {}),
            context=dict(context or {}),
        ).validate(self.catalog)
        line = json.dumps(
            event.to_dict(), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            if self.fsync:
                os.fsync(self._file.fileno())
        return event

    def emit_round(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        return self.emit("round", clocks=clocks, metrics=metrics, context=context)

    def emit_tick(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        return self.emit("tick", clocks=clocks, metrics=metrics, context=context)

    def emit_eval(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        return self.emit("eval", clocks=clocks, metrics=metrics, context=context)

    def emit_export(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        return self.emit("export", clocks=clocks, metrics=metrics, context=context)

    def emit_checkpoint(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        return self.emit("checkpoint", clocks=clocks, metrics=metrics, context=context)

    def emit_debug(
        self,
        clocks: SACClockState | Mapping[str, int],
        metrics: Optional[Mapping[str, Any]] = None,
        **context: Any,
    ) -> MetricEvent:
        return self.emit("debug", clocks=clocks, metrics=metrics, context=context)

    def emit_config(
        self,
        clocks: SACClockState | Mapping[str, int],
        config: Mapping[str, Any],
        **context: Any,
    ) -> MetricEvent:
        merged = {"config": dict(config)}
        merged.update(context)
        return self.emit("config", clocks=clocks, metrics={}, context=merged)

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "SACMetricsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_events(path: str | Path) -> list[MetricEvent]:
    events: list[MetricEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                event = MetricEvent(
                    schema_version=str(data["schema_version"]),
                    event_type=str(data["event_type"]),
                    timestamp_unix=float(data["timestamp_unix"]),
                    clocks=data["clocks"],
                    metrics=data.get("metrics") or {},
                    context=data.get("context") or {},
                ).validate()
            except Exception as exc:
                raise ValueError(
                    f"invalid SAC metrics event at {path}:{line_no}: {exc}"
                ) from exc
            events.append(event)
    return events


__all__ = [
    "EVENTS_RELATIVE_PATH",
    "MetricEvent",
    "SACMetricsWriter",
    "SAC_METRICS_SCHEMA",
    "load_events",
]
