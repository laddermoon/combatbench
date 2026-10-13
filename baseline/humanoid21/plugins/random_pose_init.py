"""Random standing/leaning pose initialization for a target robot.

Same internal-sim pattern as ``RandomFallenStatePlugin``, but the
acceptance criterion is inverted: instead of stepping until the robot
FALLS below a height threshold, we perturb it briefly and keep the
state only if it is still standing (or leaning) above ``min_height``.

Per ``on_pre_episode``:

1. Copy the real env's core state into the internal sim.
2. Apply a random action (uniform[-scale, scale]) to the target robot
   for ``perturb_phy_steps`` physics substeps — it twitches, leans,
   maybe takes a step.
3. Hold the last random action for ``settle_phy_steps`` more substeps
   so the dynamics settle into a physically real configuration.
4. Accept iff the target's root height >= ``min_height``; otherwise
   resample (up to ``max_attempts``).

The accepted state is written back via ``set_core_state``.  The move
plugin (``PoseMovePlugin``) then captures whatever pose this produces
and freezes it for the episode — no direct coupling needed.

If all attempts fail, the target's ORIGINAL (unperturbed) state is
restored and ``<rid>_pose_init_fallback`` is set to 1 in metrics —
visible in eval output rather than silently masked.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BasePlugin, SimContext


class RandomPoseInitPlugin(BasePlugin):
    """Perturb-then-accept-if-standing pose randomizer."""

    def __init__(
        self,
        target_robot: str = "robot_b",
        max_attempts: int = 20,
        perturb_phy_steps: int = 100,
        settle_phy_steps: int = 100,
        action_scale: float = 0.6,
        min_height: float = 0.9,
        reset_interval: int = 50,
        random_seed: Optional[int] = None,
    ) -> None:
        if target_robot not in ("robot_a", "robot_b"):
            raise ValueError(f"Invalid target_robot: {target_robot}")
        self.target_robot = target_robot
        self.max_attempts = int(max_attempts)
        self.perturb_phy_steps = int(perturb_phy_steps)
        self.settle_phy_steps = int(settle_phy_steps)
        self.action_scale = float(action_scale)
        self.min_height = float(min_height)
        self.reset_interval = int(reset_interval)
        self._rng = np.random.RandomState(random_seed)
        self._internal_sim: Optional[Any] = None

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    @property
    def name(self) -> str:
        return f"random_pose_init_{self.target_robot}"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "target_robot": self.target_robot,
            "max_attempts": self.max_attempts,
            "perturb_phy_steps": self.perturb_phy_steps,
            "settle_phy_steps": self.settle_phy_steps,
            "action_scale": self.action_scale,
            "min_height": self.min_height,
            "reset_interval": self.reset_interval,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RandomPoseInitPlugin":
        return cls(**config)

    def _ensure_internal_sim(self) -> Any:
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
        return self._internal_sim

    def on_detach(self) -> None:
        if self._internal_sim is not None:
            self._internal_sim.close()
            self._internal_sim = None

    @staticmethod
    def _tilt_cos(root_rot: np.ndarray) -> float:
        """cos of the body-z vs world-z angle from a (w,x,y,z) quat."""
        w, x, y, z = np.asarray(root_rot, dtype=np.float64)
        return float(1.0 - 2.0 * (x * x + y * y))

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal_sim()
        real_state = ctx.accessor.get_core_state()
        if self.target_robot not in real_state:
            raise ValueError(
                f"RandomPoseInitPlugin: {self.target_robot} not in core state"
            )

        non_target_state = {
            rid: {k: v.copy() for k, v in state.items()}
            for rid, state in real_state.items()
            if rid != self.target_robot
        }

        accepted: Optional[Dict[str, Any]] = None
        attempts = 0
        for attempt in range(self.max_attempts):
            attempts = attempt + 1
            sim.reset()
            sim.set_core_state(real_state)

            action = {}
            for rid in ("robot_a", "robot_b"):
                if rid == self.target_robot:
                    action[rid] = self._rng.uniform(
                        -self.action_scale, self.action_scale, size=(21,)
                    ).astype(np.float32)
                else:
                    action[rid] = real_state[rid].get(
                        "joint_pos_norm",
                        np.zeros(21, dtype=np.float32),
                    )
            sim.set_action(action)

            for step in range(self.perturb_phy_steps + self.settle_phy_steps):
                sim.physical_step()
                if non_target_state and (step + 1) % self.reset_interval == 0:
                    sim.set_core_state(non_target_state)

            core = sim.get_core_state()
            height = float(core[self.target_robot]["root_pos"][2])
            if height >= self.min_height:
                accepted = core[self.target_robot]
                break

        rid = self.target_robot
        if accepted is not None:
            ctx.mutator.set_core_state({rid: accepted})
            ctx.metrics[f"{rid}_pose_init_fallback"] = 0
            ctx.metrics[f"{rid}_pose_init_height"] = float(
                accepted["root_pos"][2])
            ctx.metrics[f"{rid}_pose_init_tilt_cos"] = self._tilt_cos(
                accepted["root_rot"])
        else:
            # All attempts produced sub-threshold states — keep the
            # unperturbed pose and mark it loudly in metrics.
            ctx.metrics[f"{rid}_pose_init_fallback"] = 1
            ctx.metrics[f"{rid}_pose_init_height"] = float(
                real_state[rid]["root_pos"][2])
            ctx.metrics[f"{rid}_pose_init_tilt_cos"] = self._tilt_cos(
                real_state[rid]["root_rot"])
        ctx.metrics[f"{rid}_pose_init_attempts"] = attempts
