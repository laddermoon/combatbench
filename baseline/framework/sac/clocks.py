"""Canonical SAC timing/counter state.

All metrics and dump identities share this clock vocabulary.  Counts are
kept distinct: environment action steps and admitted agent transitions are
not interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


CLOCK_FIELDS = (
    "collection_round",
    "env_step",
    "agent_transition",
    "critic_tick",
    "actor_tick",
    "temperature_tick",
    "target_tick",
    "eval_tick",
    "export_tick",
    "checkpoint_tick",
)


@dataclass
class SACClockState:
    collection_round: int = 0
    env_step: int = 0
    agent_transition: int = 0
    critic_tick: int = 0
    actor_tick: int = 0
    temperature_tick: int = 0
    target_tick: int = 0
    eval_tick: int = 0
    export_tick: int = 0
    checkpoint_tick: int = 0

    def __post_init__(self) -> None:
        for name in CLOCK_FIELDS:
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"clock {name} must be >= 0, got {value}")
            setattr(self, name, value)

    def snapshot(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in CLOCK_FIELDS}

    def advance_collection(self, *, env_steps: int, agent_transitions: int) -> int:
        self.collection_round += 1
        self.env_step += int(env_steps)
        self.agent_transition += int(agent_transitions)
        self.__post_init__()
        return self.collection_round

    def tick_critic(self) -> int:
        self.critic_tick += 1
        return self.critic_tick

    def tick_actor(self) -> int:
        self.actor_tick += 1
        return self.actor_tick

    def tick_temperature(self) -> int:
        self.temperature_tick += 1
        return self.temperature_tick

    def tick_target(self) -> int:
        self.target_tick += 1
        return self.target_tick

    def tick_eval(self) -> int:
        self.eval_tick += 1
        return self.eval_tick

    def tick_export(self) -> int:
        self.export_tick += 1
        return self.export_tick

    def tick_checkpoint(self) -> int:
        self.checkpoint_tick += 1
        return self.checkpoint_tick

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SACClockState":
        return cls(**{name: int(data.get(name, 0)) for name in CLOCK_FIELDS})


__all__ = ["CLOCK_FIELDS", "SACClockState"]
