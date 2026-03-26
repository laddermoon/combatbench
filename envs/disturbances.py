from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


class BaseDisturbance:
    name = "base"

    def reset(self, env: Any, rng: np.random.Generator) -> None:
        pass

    def before_substep(
        self,
        env: Any,
        rng: np.random.Generator,
        step_index: int,
        substep_index: int,
    ) -> Optional[Dict[str, Any]]:
        return None


class RandomPushDisturbance(BaseDisturbance):
    name = "random_push"

    def __init__(
        self,
        robot_ids: Sequence[str] = ("robot_a",),
        body_name: str = "torso",
        probability_per_substep: float = 0.0,
        horizontal_force_range: tuple[float, float] = (0.0, 0.0),
        vertical_force_range: tuple[float, float] = (0.0, 0.0),
        torque_range: tuple[float, float] = (0.0, 0.0),
    ):
        self.robot_ids = tuple(robot_ids)
        self.body_name = str(body_name)
        self.probability_per_substep = float(probability_per_substep)
        self.horizontal_force_range = tuple(float(value) for value in horizontal_force_range)
        self.vertical_force_range = tuple(float(value) for value in vertical_force_range)
        self.torque_range = tuple(float(value) for value in torque_range)

    def _sample_signed_magnitude(
        self,
        rng: np.random.Generator,
        magnitude_range: tuple[float, float],
    ) -> float:
        low, high = magnitude_range
        magnitude = float(low) if np.isclose(low, high) else float(rng.uniform(low, high))
        sign = -1.0 if float(rng.uniform()) < 0.5 else 1.0
        return sign * magnitude

    def before_substep(
        self,
        env: Any,
        rng: np.random.Generator,
        step_index: int,
        substep_index: int,
    ) -> Optional[Dict[str, Any]]:
        if self.probability_per_substep <= 0.0:
            return None
        if float(rng.uniform()) >= self.probability_per_substep:
            return None
        robot_id = str(rng.choice(np.asarray(self.robot_ids, dtype=object)))
        horizontal_force = np.array(
            [
                self._sample_signed_magnitude(rng, self.horizontal_force_range),
                self._sample_signed_magnitude(rng, self.horizontal_force_range),
                self._sample_signed_magnitude(rng, self.vertical_force_range),
            ],
            dtype=np.float64,
        )
        torque = np.array(
            [
                self._sample_signed_magnitude(rng, self.torque_range),
                self._sample_signed_magnitude(rng, self.torque_range),
                self._sample_signed_magnitude(rng, self.torque_range),
            ],
            dtype=np.float64,
        )
        return env.apply_body_wrench(
            robot_id,
            self.body_name,
            force=horizontal_force,
            torque=torque,
            source=self.name,
            step_index=step_index,
            substep_index=substep_index,
        )


class ScheduledPushDisturbance(BaseDisturbance):
    name = "scheduled_push"

    def __init__(
        self,
        schedule: Iterable[Dict[str, Any]],
        default_body_name: str = "torso",
    ):
        self.schedule = [dict(item) for item in schedule]
        self.default_body_name = str(default_body_name)

    def before_substep(
        self,
        env: Any,
        rng: np.random.Generator,
        step_index: int,
        substep_index: int,
    ) -> Optional[Dict[str, Any]]:
        for item in self.schedule:
            if int(item.get("step_index", -1)) != int(step_index):
                continue
            if int(item.get("substep_index", 0)) != int(substep_index):
                continue
            robot_id = str(item.get("robot_id", "robot_a"))
            body_name = str(item.get("body_name", self.default_body_name))
            force = np.asarray(item.get("force", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            torque = np.asarray(item.get("torque", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            return env.apply_body_wrench(
                robot_id,
                body_name,
                force=force,
                torque=torque,
                source=self.name,
                step_index=step_index,
                substep_index=substep_index,
            )
        return None
