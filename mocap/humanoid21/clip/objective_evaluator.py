"""Objective evaluators for path-based reclip.

The evaluator turns a simulated rollout into progress, event and score
metrics.  It deliberately separates path geometry (TargetPath) from the
objective's time/progress semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .core_clip import CONTROL_DT
from .target_path import TargetPath, pose_error, velocity_error


@dataclass
class ReclipObjectiveConfig:
    """Weights and limits used by ObjectiveEvaluator.

    None of the weights is a tolerance.  Completion is driven by shared
    path progress; pose/velocity deviations are reported as metrics.
    """

    kind: str                  # track | pass | reach-stop
    time_budget_frames: int
    ds_max: float = 0.2
    progress_weight: float = 1.0
    path_weight: float = 1.0
    reg_weight: float = 1e-3
    terminal_pose_weight: float = 0.0
    terminal_vel_weight: float = 0.0
    fall_weight: float = 0.0
    track_s: Optional[np.ndarray] = None


@dataclass
class RolloutEvaluation:
    score: float
    status: str                # completed | budget_exhausted
    event_frame: Optional[int]
    progress: np.ndarray
    pose_errors: List[Dict[str, Any]]
    terminal_errors: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)


class ObjectiveEvaluator:
    """Evaluate simulated states against a pose path."""

    _KINDS = {"track", "pass", "reach-stop"}

    def __init__(self, path: TargetPath,
                 config: ReclipObjectiveConfig) -> None:
        if config.kind == "reach-state":
            raise NotImplementedError(
                "reach-state is reserved semantically but not implemented")
        if config.kind not in self._KINDS:
            raise ValueError(f"unknown objective {config.kind!r}")
        if config.kind == "track" and path.root_pos_mode == "free":
            raise ValueError(
                "track requires absolute root position; "
                "root_pos_mode='free' is only supported by "
                "pass and reach-stop")
        if config.time_budget_frames < 0:
            raise ValueError("time_budget_frames must be >= 0")
        if config.ds_max <= 0:
            raise ValueError("ds_max must be positive")
        self.path = path
        self.config = config

    def _target_s(self, n_states: int, frame_offset: int) -> np.ndarray:
        if self.config.track_s is not None:
            s = np.asarray(self.config.track_s, dtype=np.float64)
        else:
            s = self.path.knot_s
        out = s[frame_offset:frame_offset + n_states]
        if len(out) != n_states:
            raise ValueError(
                f"track needs {n_states} targets at frame_offset "
                f"{frame_offset}, got {len(out)}")
        return out

    def _project_progress(self, states: List[Dict[str, np.ndarray]],
                          start_s: float) -> np.ndarray:
        s = float(np.clip(start_s, 0.0, self.path.s_max))
        progress = [s]
        for st in states[1:]:
            proj = self.path.project(st, s_hint=s,
                                     ds_max=self.config.ds_max)
            s = proj.s
            progress.append(s)
        return np.asarray(progress, dtype=np.float64)

    def _pose_errors(self, states: List[Dict[str, np.ndarray]],
                     progress: np.ndarray) -> List[Dict[str, Any]]:
        errors = []
        for st, s in zip(states, progress):
            target = self.path.sample(float(s)).as_state()
            errors.append(pose_error(st, target, self.path.pose_weights))
        return errors

    @staticmethod
    def _fall_frames(states: List[Dict[str, np.ndarray]],
                     upto: int) -> List[int]:
        return [i for i, st in enumerate(states[:upto])
                if float(st["root_pos"][2]) < 0.7]

    @staticmethod
    def _action_reg(actions: np.ndarray, upto: int) -> float:
        if upto <= 0:
            return 0.0
        return float(np.mean(np.square(
            np.asarray(actions[:upto], dtype=np.float64))))

    def evaluate_rollout(
            self, states: List[Dict[str, np.ndarray]],
            actions: np.ndarray, start_s: float = 0.0,
            frame_offset: int = 0) -> RolloutEvaluation:
        if not states:
            raise ValueError("states must include at least the initial state")
        actions = np.asarray(actions, dtype=np.float64)
        if len(actions) != len(states) - 1:
            raise ValueError("actions must have len(states)-1 entries")

        if self.config.kind == "track":
            progress = self._target_s(len(states), frame_offset)
        else:
            progress = self._project_progress(states, start_s)
        pose_errors = self._pose_errors(states, progress)

        if self.config.kind == "track":
            event_frame = len(states) - 1
            status = "completed"
        else:
            hits = np.nonzero(progress >= self.path.s_end)[0]
            event_frame = int(hits[0]) if len(hits) else None
            status = ("completed" if event_frame is not None
                      else "budget_exhausted")

        # Anything after a pass/reach event is outside the objective and
        # does not affect completion or the scalar score.
        score_upto = (len(states) if event_frame is None
                      else event_frame + 1)
        action_upto = 0 if event_frame is None else event_frame
        if self.config.kind == "track":
            action_upto = len(actions)

        path_totals = np.asarray(
            [e["total"] for e in pose_errors[:score_upto]],
            dtype=np.float64)
        fall_frames = self._fall_frames(states, score_upto)
        fall_frac = (len(fall_frames) / max(1, score_upto)
                     if score_upto else 0.0)
        action_reg = self._action_reg(actions, action_upto)

        terminal_idx = (len(states) - 1 if event_frame is None
                        else event_frame)
        terminal_pose = pose_error(
            states[terminal_idx], self.path.sample(1.0).as_state(),
            self.path.pose_weights)
        zero_vel = dict(states[terminal_idx])
        zero_vel["root_vel_local"] = np.zeros(3, np.float32)
        zero_vel["root_angular_vel_local"] = np.zeros(3, np.float32)
        zero_vel["joint_vel_norm"] = np.zeros(21, np.float32)
        terminal_vel = velocity_error(states[terminal_idx], zero_vel)

        if self.config.kind == "track":
            event_term = 0.0
        elif event_frame is not None:
            event_term = float(event_frame)
        else:
            # Unfinished rollouts are always worse than any completed
            # event; within them, larger final progress is better.
            event_term = (float(self.config.time_budget_frames)
                          + self.config.progress_weight
                          * float(1.0 - progress[-1]))

        score = (
            event_term
            + self.config.path_weight * float(path_totals.mean())
            + self.config.reg_weight * action_reg
            + self.config.fall_weight * fall_frac)
        if self.config.kind == "reach-stop":
            score += (self.config.terminal_pose_weight
                      * terminal_pose["total"]
                      + self.config.terminal_vel_weight
                      * terminal_vel["total"])

        metrics: Dict[str, Any] = {
            "s_final": float(progress[-1]),
            "s_max_seen": float(progress.max()),
            "event_frame": event_frame,
            "event_time": (None if event_frame is None
                           else float(event_frame * CONTROL_DT)),
            "path_error_mean": float(path_totals.mean()),
            "path_error_max": float(path_totals.max()),
            "action_reg_mean": action_reg,
            "fall_frame": (fall_frames[0] if fall_frames else None),
            "fall_count": len(fall_frames),
            "terminal_pose_error": float(terminal_pose["total"]),
            "terminal_vel_error": float(terminal_vel["total"]),
        }
        return RolloutEvaluation(
            score=float(score), status=status, event_frame=event_frame,
            progress=progress, pose_errors=pose_errors,
            terminal_errors={"pose": terminal_pose,
                             "velocity": terminal_vel},
            metrics=metrics)
