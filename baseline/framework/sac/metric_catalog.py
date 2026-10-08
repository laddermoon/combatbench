"""Metric catalog for ``sac_metrics_v1`` events.

The catalog intentionally uses SAC namespaces only.  PPO-specific names such
as ``advantage.*``, ``ratio.*``, ``clip.*`` or ``gae.*`` are rejected rather
than being silently reinterpreted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple


EVENT_TYPES = (
    "round",
    "tick",
    "eval",
    "export",
    "checkpoint",
    "debug",
    "config",
)

_ALLOWED_PREFIXES = {
    "round": (
        "collection.",
        "episode.",
        "reward.",
        "replay.",
        "timing.",
        "task.",
    ),
    "tick": (
        "batch.",
        "critic.",
        "actor.",
        "temperature.",
        "target.",
        "replay.",
        "timing.",
        "debug.",
    ),
    "eval": ("eval.", "collection.", "timing.", "task."),
    "export": ("export.", "timing."),
    "checkpoint": ("checkpoint.", "timing.", "replay."),
    "debug": ("debug.", "timing.", "replay.", "collection.", "batch."),
    # Config events carry their payload in context; metrics are optional but
    # are kept in a distinct namespace if present.
    "config": ("config.",),
}

_FORBIDDEN_PREFIXES = (
    "advantage.",
    "ratio.",
    "clip.",
    "gae.",
    "ppo.",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    event_type: str
    description: str = ""


class MetricCatalog:
    def allowed_prefixes(self, event_type: str) -> Tuple[str, ...]:
        try:
            return _ALLOWED_PREFIXES[event_type]
        except KeyError as exc:
            allowed = ", ".join(EVENT_TYPES)
            raise ValueError(
                f"unknown SAC metric event type {event_type!r}; allowed={allowed}"
            ) from exc

    def validate_metric_name(self, event_type: str, name: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("metric name must be a non-empty string")
        if name.startswith(_FORBIDDEN_PREFIXES):
            raise ValueError(
                f"metric {name!r} uses a PPO-only namespace and is not "
                "part of sac_metrics_v1"
            )
        prefixes = self.allowed_prefixes(event_type)
        if not any(name.startswith(prefix) for prefix in prefixes):
            raise ValueError(
                f"metric {name!r} is not allowed for event {event_type!r}; "
                f"expected prefixes {prefixes}"
            )

    def validate_metrics(
        self,
        event_type: str,
        metrics: Mapping[str, object],
    ) -> None:
        for name in metrics:
            self.validate_metric_name(event_type, str(name))


DEFAULT_METRIC_SPECS = tuple(
    MetricSpec(name=name, event_type=event)
    for event, names in {
        "round": (
            "collection.episodes",
            "collection.env_steps",
            "collection.agent_transitions",
            "collection.wall_time_s",
            "replay.transitions_added",
            "replay.size",
            "timing.collection_s",
            "timing.slice_s",
            "task.online_success",
            "task.final_potential_mean",
        ),
        "tick": (
            "batch.size",
            "critic.loss",
            "critic.q_mean",
            "critic.td_mean",
            "actor.loss",
            "actor.log_prob_mean",
            "temperature.alpha",
            "temperature.loss",
            "target.tau",
            "replay.size",
            "timing.update_s",
        ),
        "eval": (
            "eval.episodes",
            "eval.success_rate",
            "eval.survival_rate",
            "eval.ep_len_mean",
            "eval.max_pot",
            "eval.max_stage",
            "eval.max_h",
            "eval.final_pot",
            "timing.eval_s",
        ),
        "export": ("export.bytes", "timing.export_s"),
        "checkpoint": ("checkpoint.bytes", "timing.checkpoint_s"),
        "debug": ("debug.status", "debug.capture_s", "debug.bytes"),
        "config": (),
    }.items()
    for name in names
)


def metric_specs() -> Iterable[MetricSpec]:
    return DEFAULT_METRIC_SPECS


__all__ = [
    "DEFAULT_METRIC_SPECS",
    "EVENT_TYPES",
    "MetricCatalog",
    "MetricSpec",
    "metric_specs",
]
