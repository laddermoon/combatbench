"""ExecutorPolicy — framework Policy ABC wrapper around the Executor.

Caller-side contract (per executor design):
  - the caller owns intent progress.  Each ``act`` re-anchors the
    remaining clip's first keyframe to the P5 decoded from the current
    observation and advances through keyframes on their
    ``Transfer.dur`` budget (time-based, v1 — progress tracking is the
    caller's job, not the Executor's).
  - when the clip is exhausted the policy holds the last computed
    action (PD targets are absolute joint goals -> holds final pose).

Known semantic gap (v1): naive re-anchor means each remaining frame's
transfer is re-expressed against the CURRENT pose, so displacement
targets drift with the robot (e.g. "step +0.35" re-issued after a
partial step overshoots).  Acceptable for the e2e smoke; a faithful
implementation needs a persistent virtual anchor across calls.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.framework.policy import Policy
from point5.contracts import Keyframe, P5Clip

from .centroidal import CTRL_DT
from .centroidal_planner import ObsDecoder
from .contracts import Context
from .executor import Executor


def _default_clip() -> P5Clip:
    from point5.humanoid21.stand_punch import plan_stand_punch
    plan = plan_stand_punch(np.array([1.05, 0.0, 1.15]), "left")
    if plan.task is None:
        raise RuntimeError(f"default punch clip unreachable: {plan.diag}")
    return plan.task


class ExecutorPolicy(Policy):
    """RoundRunner-compatible wrapper: obs -> Executor -> (21,) action."""

    def __init__(self, robot_id: str = "robot_a",
                 clip: Optional[P5Clip] = None,
                 **_ignored: Any):
        self.robot_id = robot_id
        self.ex = Executor(robot_id=robot_id)
        self.dec = ObsDecoder(robot_id)
        self.clip = clip if clip is not None else _default_clip()
        self._idx = 0                    # keyframes[:idx+1] already issued
        self._elapsed = 0.0              # time spent in current segment
        self._last_action = np.zeros(21, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        self._idx = 0
        self._elapsed = 0.0

    def act(self, observation: Any,
            want_extra: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        obs = np.asarray(observation, dtype=np.float64)
        kfs = self.clip.keyframes
        if self._idx + 1 < len(kfs):
            local_p5, _ = self.dec.decode_p5_local(obs)
            remaining = [Keyframe(pos=local_p5)] + list(kfs[self._idx + 1:])
            clip = P5Clip(name=self.clip.name, keyframes=tuple(remaining))
            self._last_action = np.asarray(
                self.ex.gen_action(Context(observation=obs, clip=clip)),
                dtype=np.float64)
            self._advance(kfs)
        return self._last_action.astype(np.float32), None

    def _advance(self, kfs) -> None:
        """Consume Transfer.dur budgets: move _idx past elapsed frames."""
        self._elapsed += CTRL_DT
        while (self._idx + 1 < len(kfs)
               and kfs[self._idx + 1].transfer is not None
               and self._elapsed >= kfs[self._idx + 1].transfer.dur):
            self._elapsed -= kfs[self._idx + 1].transfer.dur
            self._idx += 1

    def close(self) -> None:
        wsim = getattr(self.ex, "wsim", None)
        if wsim is not None and hasattr(wsim, "close"):
            wsim.close()
