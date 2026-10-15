"""Command-conditioned simulator — 65-dim obs = 57 proprio + 8 cmd.

Strips the 39-dim opponent block (obs[57:96]) from the canonical
Humanoid21Simulator observation and appends an 8-dim physical-semantic
command vector, producing a 65-dim input for the low-level executor.

Stage 0 contract: the command schedule always emits STAND (all zeros).
Future stages will manage STAND → PUNCH/MOVE → STAND segment switching
inside this class.

Command layout (all values already in [-1.5, 1.5] O(1) range):

    cmd[0]  eff_type   ∈ {-1(foot), 0(torso), +1(hand)}
    cmd[1]  eff_side   ∈ {-1(right), 0(bilateral), +1(left)}
    cmd[2]  act_mode   ∈ {-1(locate), 0(idle), +1(strike)}
    cmd[3]  phase      ∈ [0.0, 1.0]   — normalized progress in current
                                        command window (0 for STAND)
    cmd[4]  target_x   ∈ [-1.5, 1.5]  — ego-frame forward coord (m)
    cmd[5]  target_y   ∈ [-1.5, 1.5]  — ego-frame lateral coord (m)
    cmd[6]  target_z   ∈ [-1.5, 1.5]  — ego-frame vertical coord (m)
    cmd[7]  target_aux ∈ [-1.0, 1.0]  — MOVE: Δψ/π;  PUNCH: force coef
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator

# ---------------------------------------------------------------------------
# Command field indices within the 8-dim vector.
# ---------------------------------------------------------------------------
EFF_TYPE = 0
EFF_SIDE = 1
ACT_MODE = 2
PHASE = 3
TGT_X = 4
TGT_Y = 5
TGT_Z = 6
TGT_AUX = 7

CMD_DIM = 8
BODY_DIM = 57  # obs[0:57] — proprioception, no opponent

# Canonical STAND command — all zeros.
STAND_CMD = np.zeros(CMD_DIM, dtype=np.float32)


class CommandSimulator(Humanoid21Simulator):
    """Humanoid21Simulator → 65-dim observation (57 proprio + 8 cmd).

    For Stage 0 the schedule emits STAND for every step.  The
    command-generation interface (``_next_command``) is the single
    extension point for later stages.
    """

    OBS_BASE_DIM = 96
    OBS_OUT_DIM = BODY_DIM + CMD_DIM  # 65

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._action_step = 0
        # Per-agent current command (Stage 0: always STAND).
        self._cmd: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Step accounting — mirrors GaitClockSimulator convention.
    # ------------------------------------------------------------------

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._action_step += 1

    def reset(self, seed=None, options=None) -> None:
        super().reset(seed=seed, options=options)
        self._action_step = 0
        # Fresh episode → every agent back to STAND.
        self._cmd = {}

    # ------------------------------------------------------------------
    # Command generation — extension point for Stage 1+.
    # ------------------------------------------------------------------

    def _next_command(self, robot_id: str) -> np.ndarray:
        """Return the 8-dim command for *robot_id* at this step.

        Stage 0: always STAND (all zeros).  Later stages override
        this to implement segment scheduling, launch gating, and
        mid-action abort.
        """
        return STAND_CMD

    def _current_cmd(self, robot_id: str) -> np.ndarray:
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
        """Current 8-dim command for *robot_id* (read-only copy)."""
        return self._current_cmd(robot_id).copy()
