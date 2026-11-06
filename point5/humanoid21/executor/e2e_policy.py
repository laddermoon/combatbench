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
from point5.contracts import Keyframe, P5Clip, Transfer

from .contracts import CTRL_DT
from .support_planner import ObsDecoder
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
        self._hold_p5 = None             # posture frozen at clip end

    def reset(self, seed: Optional[int] = None) -> None:
        self._idx = 0
        self._elapsed = 0.0

    def act(self, observation: Any,
            want_extra: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        obs = np.asarray(observation, dtype=np.float64)
        kfs = self.clip.keyframes
        local_p5, _ = self.dec.decode_p5_local(obs)
        if self._idx + 1 < len(kfs):
            remaining = [Keyframe(pos=local_p5)] + list(kfs[self._idx + 1:])
            clip = P5Clip(name=self.clip.name, keyframes=tuple(remaining))
        else:
            # intent exhausted -> boxing semantics: retract to stance.
            # Target = canonical standing P5 (NOT the pose at exhaustion,
            # which is already mid-fall; freezing it anchors the lean).
            if self._hold_p5 is None:
                from point5.humanoid21.stand_punch import stand_p5
                self._hold_p5 = stand_p5()
            stay = Keyframe(pos=self._hold_p5,
                            transfer=Transfer(0.0, 0.0, 0.0, 5.0))
            clip = P5Clip(name="hold", keyframes=(
                Keyframe(pos=local_p5), stay))
        self._last_action = np.asarray(
            self.ex.gen_action(Context(
                observation=obs, clip=clip,
                meta={"last_action": self._last_action})),
            dtype=np.float64)
        if self._idx + 1 < len(kfs):
            self._advance(kfs, local_p5)
        return self._last_action.astype(np.float32), None

    _REACH_EPS = 0.30          # full-P5 err counting a frame as reached
    _MAX_DWELL = 1.2           # s — give up on a frame after this

    def _advance(self, kfs, local_p5) -> None:
        """Progress-gated advance: consume the next keyframe when its
        pose requirement is actually met (or the dwell cap expires).
        Pure dur-budget burning lets the intent expire while the body
        is still mid-motion — the 90% rule says carry the intent, not
        the clock.  (v4 semantics — a longer stepping dwell doubled the
        tilt exposure window and every video with it fell.)"""
        self._elapsed += CTRL_DT
        while self._idx + 1 < len(kfs):
            tr = kfs[self._idx + 1].transfer
            dur = tr.dur if tr is not None else 1.0
            if self._elapsed >= max(3.0 * dur, self._MAX_DWELL):
                self._elapsed -= dur
                self._idx += 1
                continue
            if (self._elapsed >= 0.5 * dur
                    and self._frame_err(kfs, local_p5) < self._REACH_EPS):
                self._elapsed = 0.0
                self._idx += 1
                continue
            break

    def _frame_err(self, kfs, local_p5) -> float:
        """||target P5 of kfs[idx+1] (expressed in frame0) - local_p5||"""
        from point5.contracts import compose_frame, to_world
        nxt = self._idx + 1
        pose = (np.zeros(2), 0.0)
        for i in range(1, nxt + 1):
            pose = compose_frame(*pose, kfs[i].transfer)
        target = to_world(np.asarray(kfs[nxt].pos, np.float64), *pose)
        return float(np.linalg.norm(target - local_p5))

    def close(self) -> None:
        wsim = getattr(self.ex, "wsim", None)
        if wsim is not None and hasattr(wsim, "close"):
            wsim.close()
