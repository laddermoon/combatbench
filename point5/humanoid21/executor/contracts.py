"""Executor contracts — single source of truth for phase-1 boundaries.

Everything downstream (reference planner, scorer, MPC, demo) must read
tolerances/budgets/capabilities from ``ExecutorConfig`` — no magic
numbers scattered in code.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Command ops (instruction cache)
# ---------------------------------------------------------------------------

class Op(enum.Enum):
    KEEP = "keep"          # no submission: continue current task
    REPLACE = "replace"    # validated clip replaces the whole cache
    CLEAR = "clear"        # explicit empty: settle -> hold balance


class ExecStatus(enum.Enum):
    RUNNING = "running"        # tracking a submitted clip
    SETTLING = "settling"      # clip done or cleared: decelerating
    HOLDING = "holding"        # static balance hold
    FAILED = "failed"          # unrecoverable (fall, infeasible, timeout)


class ExecutorError(RuntimeError):
    pass


class AnchorError(ExecutorError):
    """Submitted clip frame 0 does not match current P5."""


class PlanError(ExecutorError):
    """Reference/timing planner could not produce a feasible plan."""


class StateError(ExecutorError):
    """Current-state extraction or snapshot restore inconsistency."""


# ---------------------------------------------------------------------------
# Reference frame (per control step, aligned to action-end time)
# ---------------------------------------------------------------------------

@dataclass
class RefFrame:
    t: float                       # seconds since plan start
    p5_world: np.ndarray           # (5,3) world positions
    p5_vel_world: np.ndarray       # (5,3) world velocities
    com_pos: np.ndarray            # (3,)
    com_vel: np.ndarray            # (3,)
    yaw: float                     # rad, torso facing schedule
    yaw_rate: float                # rad/s
    support: np.ndarray            # (2,) bool [right, left] planted
    tol_point: np.ndarray          # (5,) position tolerances (m)
    seg: int = -1                  # source clip segment
    event: str = ""                # keyframe label when boundary hit
    kf_idx: int = -1               # keyframe index confirmed at boundary


@dataclass
class StepResult:
    action: np.ndarray             # (21,) normalized PD target to apply
    status: ExecStatus
    diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Submission:
    """A clip adopted by the cache after anchor validation."""

    clip: Any                      # P5Clip
    f1: Tuple[np.ndarray, float]   # world FramePose of frame 0
    world_frames: list             # FramePose per keyframe
    world_pts: np.ndarray          # (K,5,3) world keyframe positions
    com_proxy: np.ndarray          # (K,3) world COM proxy per keyframe
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixed acceptance cases (E0 §5): local-frame targets, metres
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    hand: str
    target: Tuple[float, float, float]
    seed: int
    required: bool


CASES: Tuple[Case, ...] = tuple(
    [Case(f"std_{h}_s{s}", h, (1.05, 0.0, 1.15), s, True)
     for h in ("left", "right") for s in (0, 1)] +
    [Case(f"ext_{tgt[0]:.2f}_{tgt[1]:+.2f}_{tgt[2]:.2f}_{h}", h, tgt, 0,
          False)
     for tgt in ((1.20, 0.0, 1.15), (1.10, 0.15, 1.10),
                 (1.10, -0.15, 1.10))
     for h in ("left", "right")]
)


# ---------------------------------------------------------------------------
# Config — every tolerance / budget / capability in one place
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutorConfig:
    # -- acceptance / anchoring
    anchor_atol: float = 1e-6        # m, first-frame identity check
    # -- control timing (validated against simulator, see sim_iface)
    dt: float = 0.05                 # control period (s)
    phy_dt: float = 0.002
    # -- capability envelope (big-logic, conservative)
    v_hand: float = 3.0              # m/s
    a_hand: float = 30.0
    v_head: float = 1.5
    a_head: float = 15.0
    v_foot: float = 2.5
    a_foot: float = 25.0
    v_com: float = 0.6
    a_com: float = 6.0
    w_yaw: float = 2.5               # rad/s
    min_seg_dur: float = 0.10
    land_settle: float = 0.20
    mu: float = 0.7                  # friction estimate
    total_mass: float = 40.84        # kg
    # -- planner retries
    max_retries: int = 6
    duration_growth: float = 1.25
    task_budget_s: float = 6.0
    max_replans: int = 2
    replan_wait_s: float = 0.30
    # -- MPC budget
    horizon: int = 6
    n_samples: int = 32
    n_iters: int = 2
    n_elite: int = 8
    sigma0: float = 0.20
    seed: int = 0
    # -- scoring weights
    w_point: float = 2.0
    w_point_vel: float = 0.3
    w_com: float = 1.5
    w_com_vel: float = 0.4
    w_yaw: float = 0.8
    w_slip: float = 3.0
    w_com_support: float = 1.2       # COM over anchor during swing
    support_tol: float = 0.25        # allowed COM-anchor offset (m)
    w_extra_contact: float = 4.0
    w_action: float = 0.02
    w_action_delta: float = 0.05
    w_pose_pref: float = 0.15
    disc: float = 0.9
    fall_penalty: float = 50.0
    # -- tolerances / safety
    tol_default: float = 0.10        # point tol when no keyframe specs
    tol_com: float = 0.08
    tol_yaw: float = 0.35            # rad
    slip_v: float = 0.10             # m/s allowed for a planted foot
    kf_reach_tol: float = 0.12       # keyframe arrival confirm (m)
    hold_com_vel: float = 0.10       # m/s -> HOLDING
    tilt_fail: float = 1.05          # rad from upright
    head_min_z: float = 0.35         # head z below -> fallen
    ankle_plant_z: float = 0.16      # ankle z below -> planted
    settle_watch_s: float = 2.0      # post-clip observation window
    action_clip: float = 1.0

    @property
    def phy_steps(self) -> int:
        n = round(self.dt / self.phy_dt)
        assert abs(n * self.phy_dt - self.dt) < 1e-9
        return n
