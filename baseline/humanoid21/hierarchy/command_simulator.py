"""Command-conditioned simulator — 66-dim obs = 57 proprio + 9 cmd.

Strips the 39-dim opponent block (obs[57:96]) from the canonical
Humanoid21Simulator observation and appends a command vector for the
low-level executor.  See ``COMMAND_DESIGN.md`` for the full contract.

Two command layouts are supported via ``cmd_version``:

v2 (default, 9-dim, 66-dim obs) — finalized contract:

    cmd[0]  vx        ego-frame forward target speed   (m/s → /V_MAX)
    cmd[1]  vy        ego-frame lateral target speed   (m/s → /V_MAX)
    cmd[2]  w         target yaw rate, +left           (rad/s → /W_MAX)
    cmd[3]  h         torso height offset from nominal (m → /H_SCALE)
    cmd[4]  eff_type  0 none, +1 hand, -1 foot
    cmd[5]  eff_side  +1 left, -1 right (ignored if eff_type=0)
    cmd[6]  tx        effector target, ego forward     (m → /TGT_SCALE)
    cmd[7]  ty        effector target, ego lateral     (m → /TGT_SCALE)
    cmd[8]  tz        effector target, ego vertical    (m → /TGT_SCALE)

v1 (legacy 8-dim, 65-dim obs) — kept only so in-flight 65-dim runs
(e.g. hier_s2_balance_s42b) remain resumable under current code:

    cmd[0] eff_type {-1 foot, 0 torso, +1 hand}
    cmd[1] eff_side {-1 right, 0 bilateral, +1 left}
    cmd[2] act_mode {-1 locate, 0 idle, +1 strike}
    cmd[3] phase ∈ [0,1]
    cmd[4:7] target xyz
    cmd[7] target_aux

Instantaneous semantics: every field holds "what is wanted right now"
until overwritten — there are no action macros, modes, phases or
urgency fields.  Retracting a hand is just a new target point (or
``clear_effector``); stopping motion is a zero velocity command.
All-zero command = nominal-height in-place standard stepping.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator

BODY_DIM = 57  # obs[0:57] — proprioception, no opponent

# ---------------------------------------------------------------------------
# v2 command field indices within the 9-dim vector.
# ---------------------------------------------------------------------------
VX, VY, WZ, HT = 0, 1, 2, 3
EFF_TYPE, EFF_SIDE = 4, 5
TGT_X, TGT_Y, TGT_Z = 6, 7, 8

# Normalisation scales — fixed for the whole training lineage so a
# given command value always means the same physical quantity.
V_MAX = 1.5       # m/s
W_MAX = 2.0 * np.pi  # rad/s
H_SCALE = 0.4     # m  (h field = height offset / H_SCALE)
TGT_SCALE = 1.5   # m  (workspace radius for effector targets)

CMD_V1_DIM = 8
CMD_V2_DIM = 9


class CommandSimulator(Humanoid21Simulator):
    """Humanoid21Simulator → 57-dim proprio + N-dim command observation.

    ``cmd_version`` selects the command layout (``v2`` default = the
    finalized 9-dim contract; ``v1`` = legacy 8-dim, kept for resuming
    65-dim runs).  Commands persist until changed via ``set_motion`` /
    ``set_effector`` / ``clear_effector`` / ``reset_command``; fresh
    episodes start at the all-zero command.
    """

    OBS_BASE_DIM = 96

    def __init__(self, cmd_version: str = "v2", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if cmd_version not in ("v1", "v2"):
            raise ValueError(f"cmd_version must be 'v1' or 'v2', got {cmd_version!r}")
        self._cmd_version = cmd_version
        self.cmd_dim = CMD_V1_DIM if cmd_version == "v1" else CMD_V2_DIM
        self.OBS_OUT_DIM = BODY_DIM + self.cmd_dim
        self._action_step = 0
        self._cmd: Dict[str, np.ndarray] = {}
        self._schedules: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Step accounting — mirrors GaitClockSimulator convention.
    # ------------------------------------------------------------------

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._action_step += 1

    def reset(self, seed=None, options=None) -> None:
        super().reset(seed=seed, options=options)
        self._action_step = 0
        # Fresh episode → every agent back to the all-zero command.
        self._cmd = {}
        # Optional per-episode command schedule from episode_options:
        #   options["move_schedule"] = {robot_id: [[n_steps, vx, vy, w], ...]}
        # speeds in m/s, w in rad/s (v2 semantics; h stays 0).
        self._schedules: Dict[str, np.ndarray] = {}
        sched_opts = (options or {}).get("move_schedule") or {}
        for rid, segs in sched_opts.items():
            self._schedules[rid] = self._build_schedule(rid, segs)

    def _build_schedule(self, robot_id: str, segs) -> np.ndarray:
        """Compile [[n_steps, vx, vy, w], ...] into a (T, cmd_dim) table."""
        if self._cmd_version != "v2":
            raise RuntimeError("move_schedule requires cmd_version='v2'")
        rows = []
        for seg in segs:
            n_steps, vx, vy, w = int(seg[0]), seg[1], seg[2], seg[3]
            c = np.zeros(self.cmd_dim, dtype=np.float32)
            c[VX] = vx / V_MAX
            c[VY] = vy / V_MAX
            c[WZ] = w / W_MAX
            rows.append(np.broadcast_to(c, (max(1, n_steps), self.cmd_dim)))
        return np.concatenate(rows, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Command interface (v2 semantics; v1 accepts raw vectors only)
    # ------------------------------------------------------------------

    def _require_v2(self) -> None:
        if self._cmd_version != "v2":
            raise RuntimeError("structured command setters require cmd_version='v2'")

    def set_motion(self, robot_id: str, vx: float, vy: float, w: float,
                   h: float = 0.0) -> None:
        """Set the body channel; keeps the current effector channel.

        ``vx``/``vy`` in m/s, ``w`` in rad/s, ``h`` = torso height
        offset from nominal in m.
        """
        self._require_v2()
        cmd = self._current_cmd(robot_id)
        cmd[VX] = vx / V_MAX
        cmd[VY] = vy / V_MAX
        cmd[WZ] = w / W_MAX
        cmd[HT] = h / H_SCALE

    def set_effector(self, robot_id: str, eff_type: int, eff_side: int,
                     target_xyz: Tuple[float, float, float]) -> None:
        """Command one effector to reach ``target_xyz`` (ego frame, m).

        ``eff_type``: +1 hand, -1 foot.  ``eff_side``: +1 left, -1 right.
        The target persists until changed — speed of approach is what
        the reward measures, not a commanded mode.
        """
        self._require_v2()
        assert eff_type in (+1, -1) and eff_side in (+1, -1)
        cmd = self._current_cmd(robot_id)
        cmd[EFF_TYPE] = float(eff_type)
        cmd[EFF_SIDE] = float(eff_side)
        cmd[TGT_X:TGT_Z + 1] = np.asarray(target_xyz) / TGT_SCALE

    def clear_effector(self, robot_id: str) -> None:
        """Release the effector task (back to default posture); motion
        channel is untouched."""
        self._require_v2()
        cmd = self._current_cmd(robot_id)
        cmd[EFF_TYPE:TGT_Z + 1] = 0.0

    def set_command(self, robot_id: str, cmd: np.ndarray) -> None:
        """Overwrite the full command vector (already normalised)."""
        cmd = np.asarray(cmd, dtype=np.float32)
        if cmd.shape != (self.cmd_dim,):
            raise ValueError(f"expected ({self.cmd_dim},), got {cmd.shape}")
        self._cmd[robot_id] = cmd.copy()

    def reset_command(self, robot_id: str) -> None:
        """Back to the all-zero command (standard stepping)."""
        self._cmd[robot_id] = np.zeros(self.cmd_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Command storage — persistent until changed.
    # ------------------------------------------------------------------

    def _next_command(self, robot_id: str) -> np.ndarray:
        """Initial command for *robot_id* (all zeros = STAND).

        Extension point for command schedulers: override to inject a
        scripted or curriculum-driven command sequence per episode.
        """
        return np.zeros(self.cmd_dim, dtype=np.float32)

    def _current_cmd(self, robot_id: str) -> np.ndarray:
        sched = self._schedules.get(robot_id)
        if sched is not None:
            t = min(self._action_step, sched.shape[0] - 1)
            return sched[t]
        if robot_id not in self._cmd:
            self._cmd[robot_id] = self._next_command(robot_id)
        return self._cmd[robot_id]

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, Any]:
        if "_obs_cmd" in self._data_cache:
            return self._data_cache["_obs_cmd"]
        base = super().get_observation()
        out = {}
        for rid, o in base.items():
            proprio = np.asarray(o, dtype=np.float32)[:BODY_DIM]
            cmd = self._current_cmd(rid)
            out[rid] = np.concatenate([proprio, cmd])
        self._data_cache["_obs_cmd"] = out
        return out

    # ------------------------------------------------------------------
    # Expose command for debugging / reward computation
    # ------------------------------------------------------------------

    def get_command(self, robot_id: str) -> np.ndarray:
        """Current command for *robot_id* (read-only copy)."""
        return self._current_cmd(robot_id).copy()
