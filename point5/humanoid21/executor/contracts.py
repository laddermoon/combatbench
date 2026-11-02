"""Executor contracts: config, reference frames, results, errors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

DT = 0.05                      # control period (s)
PHY_DT = 0.002
PHY_STEPS = 25                 # DT / PHY_DT

EXEC_STATUS = ("RUNNING", "SETTLING", "HOLDING", "FAILED")


class ExecutorError(RuntimeError):
    pass


class AnchorError(ExecutorError):
    """Submitted clip's first frame does not match the current P5."""


class PlanError(ExecutorError):
    """Reference/timing planner could not produce a feasible plan."""


@dataclass(frozen=True)
class ExecutorConfig:
    anchor_atol: float = 1e-6        # first-frame anchoring tolerance (m)
    dt: float = DT
    horizon: int = 6                 # MPC window (control steps)
    n_samples: int = 32
    n_iters: int = 2
    n_elite: int = 8
    sigma0: float = 0.20
    seed: int = 0
    max_replans: int = 2
    replan_wait: float = 0.30        # grace past a missed keyframe time
    task_budget_s: float = 6.0
    settle_seconds: float = 2.0
    # capability envelope (big-logic, conservative)
    v_hand: float = 3.0              # m/s
    a_hand: float = 30.0             # m/s^2
    v_foot: float = 2.5
    a_foot: float = 25.0
    v_head: float = 1.5
    a_head: float = 15.0
    v_com: float = 0.6
    a_com: float = 6.0
    w_yaw: float = 3.0               # rad/s
    min_seg_dur: float = 0.10
    land_settle: float = 0.20        # extra time after a foot lands
    mu: float = 0.7                  # friction coefficient (conservative)
    total_mass: float = 40.84        # kg
    max_duration_scale: float = 1.25
    max_retries: int = 6
    # scoring weights (normalized per-channel residuals)
    w_point: float = 2.0
    w_point_vel: float = 0.3
    w_com: float = 1.5
    w_com_vel: float = 0.4
    w_yaw: float = 0.8
    w_support_slip: float = 3.0
    w_extra_contact: float = 4.0
    w_action: float = 0.02
    w_action_delta: float = 0.05
    w_pose_pref: float = 0.15
    disc: float = 0.9
    # tolerances used when keyframes carry no own specs
    tol_default: float = 0.10
    tol_com: float = 0.08
    tol_yaw: float = 0.35            # rad
    slip_v: float = 0.10             # m/s allowed for planted foot
    # completion / safety
    hold_com_vel: float = 0.10       # m/s RMS for HOLDING
    tilt_fail: float = 1.05          # rad (~60 deg) from upright
    head_min_z: float = 0.35         # head below this = fallen
    foot_contact_z: float = 0.16     # ankle z below this counts planted
    action_clip: float = 1.0


@dataclass
class RefFrame:
    """One control-step reference (aligned to action-end time)."""

    t: float                            # seconds since plan start
    p5_world: np.ndarray                # (5,3)
    p5_vel_world: np.ndarray            # (5,3)
    com_pos: np.ndarray                 # (3,)
    com_vel: np.ndarray                 # (3,)
    yaw: float                          # rad (world frame yaw schedule)
    yaw_rate: float                     # rad/s
    support: np.ndarray                 # (2,) bool [right, left]
    tol_point: np.ndarray               # (5,) position tolerances
    seg: int = -1
    s: float = 0.0
    event: str = ""                     # keyframe label at boundary
    kf_idx: int = -1


@dataclass
class StepResult:
    action: np.ndarray                  # (21,) to apply once
    status: str
    diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClipSubmission:
    """Resolved instruction kept by the executor."""

    clip: Any                           # P5Clip
    world_frames: list                  # FramePose per keyframe
    world_pts: np.ndarray               # (K,5,3)
    hand: str
    target: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
