"""Gait-clock simulator — Humanoid21Simulator + observable step command.

Appends 3 dims to the 96-dim observation (indices 96..98)::

    obs[96] = cmd_L   1.0 during the left foot's commanded swing window
    obs[97] = cmd_R   1.0 during the right foot's commanded swing window
    obs[98] = wprog   progress within the current window, [0, 1)

The command schedule is a deterministic function of the action-step
index — the same index that labels recorded trajectory frames — so the
reward side (``clock_foot_weights`` in stepping_state_machine.py) can
reproduce it exactly from frame indices.  The stepping "command" thus
becomes a learnable state feature for the policy instead of invisible
advantage shaping (the u01651 dump showed +W intents fire on ~30% of
standing frames but convert to real lifts only ~0.5% of the time —
the policy simply cannot see them).

The clock counts *action* steps (``set_action`` calls), unaffected by
the physics-level fallen-state settling during reset.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.humanoid21.simulator import Humanoid21Simulator


class GaitClockSimulator(Humanoid21Simulator):
    """Humanoid21Simulator whose observation carries a gait command clock."""

    OBS_BASE_DIM = 96
    GAIT_DIM = 3  # cmd_L, cmd_R, window progress

    def __init__(self, gait_period: int = 40, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gait_period = int(gait_period)
        self._action_step = 0

    # ------------------------------------------------------------------
    # Step accounting: obs for frame t is read before set_action(t), so
    # ``_action_step`` equals the recorded frame index at read time.
    # ------------------------------------------------------------------
    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._action_step += 1

    def reset(self, seed=None, options=None) -> None:
        super().reset(seed=seed, options=options)
        self._action_step = 0

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------
    def gait_clock(self) -> np.ndarray:
        """(3,) gait command for the current frame: cmd_L, cmd_R, wprog."""
        half = self.gait_period // 2
        pos = self._action_step % self.gait_period
        if pos < half:
            cmd_l, cmd_r, w = 1.0, 0.0, pos / half
        else:
            cmd_l, cmd_r, w = 0.0, 1.0, (pos - half) / half
        return np.array([cmd_l, cmd_r, w], dtype=np.float32)

    def get_observation(self) -> Dict[str, Any]:
        if '_obs_gait' in self._data_cache:
            return self._data_cache['_obs_gait']
        base = super().get_observation()
        ext = self.gait_clock()
        out = {
            rid: np.concatenate([np.asarray(o, dtype=np.float32), ext])
            for rid, o in base.items()
        }
        self._data_cache['_obs_gait'] = out
        return out
