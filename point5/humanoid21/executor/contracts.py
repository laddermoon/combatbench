"""Executor contracts — the stateless GenAction interface.

    Action  = Executor.GenAction(Context)
    Context = Observation (the simulator's own observation) +
              P5Clip (remaining instruction, frame 0 anchored)

The Executor holds no execution state; the caller owns progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


CTRL_DT = 0.05             # simulator control period (20 Hz)

# Humanoid21 total body mass (kg) — from
# envs.humanoid21.meta.Humanoid21Meta.build_runtime_tables
# ['robot_a']['total_mass'].
MASS_KG = 40.844


class ExecutorError(RuntimeError):
    pass


class AnchorError(ExecutorError):
    """Clip frame 0 does not match the observation's P5."""


class PlanError(ExecutorError):
    """Centroidal/reference planner found no acceptable plan."""


# The policy's observation — the flat 96-dim vector that a policy
# receives (get_derived_state()[robot_id]['observation']).  The
# Executor is a policy backend: this is ALL it may see.  Everything
# it needs (P5, centroidal state estimate, support cues) must be
# reconstructed from this vector — no core state, no sim handles.
Observation = np.ndarray


# simulator input: the (21,) normalized PD target for one control step
Action = np.ndarray


@dataclass
class Context:
    """One GenAction call's input."""

    observation: Observation
    clip: Any                            # P5Clip — remaining instruction
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FootSupport:
    """One foot's support state inside a LandMark.

    ``force_ratio`` is normal load divided by total body weight
    (matches obs[52:54]); 0 means no contact.
    """

    force_ratio: float = 0.0
    pos: np.ndarray = None   # (2,) local


@dataclass
class SupportState:
    """
    不规定坐标系，可以用于任意坐标系
    """

    com: np.ndarray                    # (3,) target COM, local frame
    feet: Dict[str, FootSupport]       # 'right'/'left'
    torso_yaw: Optional[float] = None  # rad; None = not handled (v1)
    for_frame: int = 0                 # for fullfil source clip frame (metadata)

@dataclass
class SupportPlan:
    """算法一的输出 — 支撑状态序列。

    所有 SupportState 坐标统一在**首帧 P5 局部帧**（坐标系约定挂
    在 plan 层，state 本身帧无关）。landmarks[i] 存绝对值——
    帧间转移是需要时算的差，不是存储模型。
    """
    landmarks: List[SupportState] = field(default_factory=list)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, i):
        return self.landmarks[i]