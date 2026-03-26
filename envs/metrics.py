from typing import Any, Dict, Optional

import numpy as np


class BaseMetricCollector:
    name = "base"

    def reset(self, env: Any) -> None:
        pass

    def collect(
        self,
        env: Any,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Optional[Dict[str, Any]]:
        return None


class CoreMetricCollector(BaseMetricCollector):
    name = "core"

    def collect(
        self,
        env: Any,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        robot_states = info.get("robot_states", {})
        relative_metrics = info.get("relative_metrics", {})
        scores = info.get("scores", {})
        initial_health = float(getattr(env.score_calculator, "INITIAL_HEALTH", 100.0))
        current_health = {
            robot_id: float(scores.get(robot_id, 0.0))
            for robot_id in ("robot_a", "robot_b")
        }
        damage_taken = {
            robot_id: max(0.0, float(initial_health - current_health[robot_id]))
            for robot_id in ("robot_a", "robot_b")
        }
        damage_dealt = {
            robot_id: damage_taken["robot_b" if robot_id == "robot_a" else "robot_a"]
            for robot_id in ("robot_a", "robot_b")
        }
        robot_metrics: Dict[str, Any] = {}
        for robot_id in ("robot_a", "robot_b"):
            state = robot_states.get(robot_id, {})
            linear_velocity = np.asarray(state.get("linear_velocity", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            angular_velocity = np.asarray(state.get("angular_velocity", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            robot_metrics[robot_id] = {
                "uprightness": float(state.get("uprightness", 0.0)),
                "height": float(np.asarray(state.get("torso_position", np.zeros(3, dtype=np.float32)), dtype=np.float32)[2]),
                "linear_speed": float(np.linalg.norm(linear_velocity)),
                "horizontal_speed": float(np.linalg.norm(linear_velocity[:2])),
                "angular_speed": float(np.linalg.norm(angular_velocity)),
                "damage_dealt": float(damage_dealt[robot_id]),
                "damage_taken": float(damage_taken[robot_id]),
                "hit_count_received": int(len(info.get("hit_records", {}).get(robot_id, []))),
                "hit_count_dealt": int(len(info.get("hit_records", {}).get("robot_b" if robot_id == "robot_a" else "robot_a", []))),
            }
        return {
            "robot": robot_metrics,
            "relative": relative_metrics,
            "episode": {
                "current_step": int(info.get("current_step", 0)),
                "physics_step_count": int(info.get("physics_step_count", 0)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        }


class ConstraintMetricCollector(BaseMetricCollector):
    name = "constraints"

    def collect(
        self,
        env: Any,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        return {
            "constraints": info.get("constraints", {}),
        }


class DisturbanceMetricCollector(BaseMetricCollector):
    name = "disturbances"

    def collect(
        self,
        env: Any,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        disturbances = info.get("disturbances", [])
        return {
            "count": int(len(disturbances)),
            "entries": disturbances,
        }
