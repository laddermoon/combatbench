"""
================================================================================
OBSOLETE - 此文件已过时，请使用 preset_envs.py 中的环境
================================================================================

此文件保留用于参考，新代码请使用：
- things.combatbench.envs.preset_envs.Humanoid21NonFallEnv
- things.combatbench.envs.preset_envs.Humanoid21FallEnv
"""

import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, List, Optional, Sequence

from ..core.physics import PhysicsEngine
from ..humanoid21.robot import HumanoidRobot
from ..humanoid21.collision import CollisionDetector
from ..humanoid21.scoring import ScoreCalculator
from .control_modes import BaseControlMode, build_default_control_modes
from .disturbances import BaseDisturbance
from .metrics import BaseMetricCollector, ConstraintMetricCollector, CoreMetricCollector, DisturbanceMetricCollector
from .resetters import BaseResetter, SymmetricStandResetter

class CombatGymEnv(gym.Env):
    """
    Dual Robot Combat Gym Environment (V1.0 - 21DOF)
    Single round Episode (30s)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    DEFAULT_ROOT_HEIGHT = 1.282

    def __init__(
        self,
        render_mode=None,
        arena_xml=None,
        dt=0.002,
        initial_distance=2.0,  # Rule: Initial distance 2 meters facing each other
        control_frequency=20,
        video_sample_frequency=30,
        match_duration=30.0,   # Single roundduration 30 seconds
        damage_scale=100.0,
        resetter: Optional[BaseResetter] = None,
        constraints: Optional[Sequence[Any]] = None,
        disturbances: Optional[Sequence[BaseDisturbance]] = None,
        metric_collectors: Optional[Sequence[BaseMetricCollector]] = None,
        control_modes: Optional[Dict[str, BaseControlMode]] = None,
        add_default_metric_collectors: bool = True,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = dt
        self.initial_distance = initial_distance
        self.match_duration = match_duration
        self.damage_scale = float(damage_scale)
        self.resetter = resetter or SymmetricStandResetter(initial_distance=initial_distance, root_height=self.DEFAULT_ROOT_HEIGHT)
        self.constraints: List[Any] = []
        self.disturbances: List[BaseDisturbance] = []
        self.metric_collectors: List[BaseMetricCollector] = []
        self.control_modes: Dict[str, BaseControlMode] = build_default_control_modes()
        self._rng = np.random.default_rng()

        self.sim_frequency = 1.0 / dt
        self.control_frequency = control_frequency
        self.video_sample_frequency = video_sample_frequency

        self.action_steps = max(1, int(round(self.sim_frequency / control_frequency)))
        self.video_sample_steps = max(1, int(round(self.sim_frequency / video_sample_frequency)))

        if arena_xml is None:
            arena_xml = os.path.join(
                os.path.dirname(__file__),
                '../assets/battle_v1.xml'
            )

        self.physics = PhysicsEngine(
            gui=(render_mode == "human"),
            dt=dt,
            arena_xml=arena_xml
        )

        action_dim = HumanoidRobot.ACTION_DIM
        self._joint_names = tuple(HumanoidRobot.CONTROLLED_JOINTS)
        self._joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(self._joint_names)}

        self.action_space = spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        })

        # Observation space：42 + 13 + 8 + 64 = 127dims
        obs_dim = 127
        self.observation_space = spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        })
        self.observation_slices = HumanoidRobot.OBSERVATION_SLICES

        self.collision_detector = CollisionDetector()
        self.score_calculator = ScoreCalculator(damage_scale=self.damage_scale)

        self.robot_a = None
        self.robot_b = None
        self._robots_created = False
        self._initial_qpos = None
        self._initial_qvel = None
        self._initial_ctrl = None
        self.actions = {"robot_a": None, "robot_b": None}
        self._default_controller_kp = np.full(action_dim, 4.0, dtype=np.float32)
        self._default_controller_kd = np.full(action_dim, 0.4, dtype=np.float32)
        self._controller_kp = self._default_controller_kp.copy()
        self._controller_kd = self._default_controller_kd.copy()
        self._default_controller_action_scale = {
            "robot_a": np.ones(action_dim, dtype=np.float32),
            "robot_b": np.ones(action_dim, dtype=np.float32),
        }
        self._controller_reference_positions = {
            "robot_a": np.zeros(action_dim, dtype=np.float32),
            "robot_b": np.zeros(action_dim, dtype=np.float32),
        }
        self._controller_action_scale = {
            "robot_a": np.ones(action_dim, dtype=np.float32),
            "robot_b": np.ones(action_dim, dtype=np.float32),
        }
        self._controller_target_positions = {
            "robot_a": np.zeros(action_dim, dtype=np.float32),
            "robot_b": np.zeros(action_dim, dtype=np.float32),
        }
        self._controller_joint_limits = None
        self._controller_ctrl_limits = None
        self.video_buffer = []
        self.hit_records = {'robot_a': [], 'robot_b': []}
        
        self.current_step = 0
        self.physics_step_count = 0
        self.max_steps = int(match_duration * control_frequency)

        # Cache camera state for smooth tracking
        self._prev_cam_pos = None
        self._prev_lookat = None

        self._root_joint_cache = None
        self._last_constraint_results: Dict[str, Any] = {}
        self._last_disturbance_events: List[Dict[str, Any]] = []
        self._last_metric_payloads: Dict[str, Any] = {}
        self._last_reset_payload: Dict[str, Any] = {}

        self.set_constraints(constraints)
        self.set_disturbances(disturbances)
        self.set_metric_collectors(metric_collectors, add_defaults=add_default_metric_collectors)
        self.set_control_modes(control_modes)

    def set_resetter(self, resetter: Optional[BaseResetter]) -> None:
        if resetter is not None:
            self.resetter = resetter

    def set_constraints(self, constraints: Optional[Sequence[Any]]) -> None:
        self.constraints = [] if constraints is None else list(constraints)

    def set_disturbances(self, disturbances: Optional[Sequence[BaseDisturbance]]) -> None:
        self.disturbances = [] if disturbances is None else list(disturbances)

    def set_metric_collectors(
        self,
        metric_collectors: Optional[Sequence[BaseMetricCollector]],
        *,
        add_defaults: bool = True,
    ) -> None:
        collector_list: List[BaseMetricCollector] = []
        if add_defaults:
            collector_list.extend([
                CoreMetricCollector(),
                ConstraintMetricCollector(),
                DisturbanceMetricCollector(),
            ])
        if metric_collectors is not None:
            collector_list.extend(metric_collectors)
        self.metric_collectors = collector_list

    def set_control_modes(self, control_modes: Optional[Dict[str, BaseControlMode]]) -> None:
        self.control_modes = build_default_control_modes()
        if control_modes is not None:
            self.control_modes.update(control_modes)

    def _yaw_deg_to_wxyz(self, yaw_deg: float) -> np.ndarray:
        quat_xyzw = R.from_euler('z', float(yaw_deg), degrees=True).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    def build_symmetric_root_poses(
        self,
        *,
        initial_distance: Optional[float] = None,
        root_height: Optional[float] = None,
        lateral_offset: float = 0.0,
        yaw_jitter_deg: float = 0.0,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        distance = float(self.initial_distance if initial_distance is None else initial_distance)
        height = float(self.DEFAULT_ROOT_HEIGHT if root_height is None else root_height)
        lateral_offset = float(lateral_offset)
        yaw_jitter_deg = float(yaw_jitter_deg)
        return {
            'robot_a': {
                'joint_name': 'root_red',
                'position': np.array([-distance / 2.0, lateral_offset, height], dtype=np.float64),
                'orientation': self._yaw_deg_to_wxyz(yaw_jitter_deg),
            },
            'robot_b': {
                'joint_name': 'root_blue',
                'position': np.array([distance / 2.0, -lateral_offset, height], dtype=np.float64),
                'orientation': self._yaw_deg_to_wxyz(180.0 + yaw_jitter_deg),
            },
        }

    def apply_root_poses(self, root_poses: Dict[str, Dict[str, np.ndarray]]) -> None:
        for root_pose in root_poses.values():
            joint_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_JOINT, root_pose['joint_name'])
            if joint_id < 0:
                continue
            qpos_adr = self.physics.model.jnt_qposadr[joint_id]
            qvel_adr = self.physics.model.jnt_dofadr[joint_id]
            self.physics.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(root_pose['position'], dtype=np.float64)
            self.physics.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(root_pose['orientation'], dtype=np.float64)
            self.physics.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def get_root_joint_cache(self):
        if self._root_joint_cache is not None:
            return self._root_joint_cache
        self._root_joint_cache = {}
        for robot_id, root_pose in self.build_symmetric_root_poses().items():
            joint_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_JOINT, root_pose['joint_name'])
            if joint_id < 0:
                continue
            self._root_joint_cache[robot_id] = {
                'joint_id': joint_id,
                'qpos_adr': int(self.physics.model.jnt_qposadr[joint_id]),
                'qvel_adr': int(self.physics.model.jnt_dofadr[joint_id]),
            }
        return self._root_joint_cache

    def _resolve_robot_body_name(self, robot_id: str, body_name: str) -> str:
        robot = self._get_robot_lookup().get(robot_id)
        suffix = '' if robot is None else robot.suffix
        if body_name.endswith(suffix):
            return body_name
        return f"{body_name}{suffix}"

    def apply_body_wrench(
        self,
        robot_id: str,
        body_name: str,
        *,
        force: np.ndarray,
        torque: np.ndarray,
        source: str,
        step_index: int,
        substep_index: int,
    ) -> Optional[Dict[str, Any]]:
        full_body_name = self._resolve_robot_body_name(robot_id, body_name)
        body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
        if body_id < 0:
            return None
        point = np.asarray(self.physics.data.xpos[body_id], dtype=np.float64).copy()
        mujoco.mj_applyFT(
            self.physics.model,
            self.physics.data,
            np.asarray(force, dtype=np.float64).reshape(3),
            np.asarray(torque, dtype=np.float64).reshape(3),
            point,
            body_id,
            self.physics.data.qfrc_applied,
        )
        event = {
            'source': source,
            'robot_id': robot_id,
            'body_name': full_body_name,
            'force': np.asarray(force, dtype=np.float64).reshape(3).astype(np.float32).tolist(),
            'torque': np.asarray(torque, dtype=np.float64).reshape(3).astype(np.float32).tolist(),
            'step_index': int(step_index),
            'substep_index': int(substep_index),
        }
        self._last_disturbance_events.append(event)
        return event

    def apply_constraints(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for constraint in self.constraints:
            if hasattr(constraint, 'apply'):
                result = constraint.apply(self)
                if result is not None:
                    results[getattr(constraint, 'name', constraint.__class__.__name__)] = result
        self._last_constraint_results = results
        return results

    def apply_disturbances(self, step_index: int, substep_index: int) -> List[Dict[str, Any]]:
        self.physics.data.qfrc_applied[:] = 0.0
        disturbance_events: List[Dict[str, Any]] = []
        for disturbance in self.disturbances:
            event = disturbance.before_substep(self, self._rng, step_index, substep_index)
            if event is not None:
                disturbance_events.append(event)
        self._last_disturbance_events.extend(disturbance_events)
        return disturbance_events

    def collect_metric_payloads(
        self,
        observation: Dict[str, np.ndarray],
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for collector in self.metric_collectors:
            payload = collector.collect(
                self,
                observation,
                info,
                terminated=terminated,
                truncated=truncated,
            )
            if payload is not None:
                metrics[getattr(collector, 'name', collector.__class__.__name__)] = payload
        self._last_metric_payloads = metrics
        return metrics

    def _build_relative_metrics(self, robot_states):
        relative_metrics = {}
        for robot_id, opponent_id in [('robot_a', 'robot_b'), ('robot_b', 'robot_a')]:
            self_state = robot_states[robot_id]
            opponent_state = robot_states[opponent_id]
            relative_position = opponent_state['torso_position'] - self_state['torso_position']
            distance = float(np.linalg.norm(relative_position))
            horizontal_distance = float(np.linalg.norm(relative_position[:2]))
            if distance > 1e-8:
                direction_to_opponent = relative_position / distance
            else:
                direction_to_opponent = np.zeros(3, dtype=np.float32)

            relative_metrics[robot_id] = {
                'distance': distance,
                'horizontal_distance': horizontal_distance,
                'relative_position': relative_position.astype(np.float32),
                'direction_to_opponent': direction_to_opponent.astype(np.float32),
                'facing_opponent': float(np.dot(self_state['forward_vector'], direction_to_opponent)),
            }
        return relative_metrics

    def _reset_runtime_modules(self) -> None:
        for constraint in self.constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset(self)
        for disturbance in self.disturbances:
            if hasattr(disturbance, 'reset'):
                disturbance.reset(self, self._rng)
        for collector in self.metric_collectors:
            if hasattr(collector, 'reset'):
                collector.reset(self)
        for robot_id, control_mode in self.control_modes.items():
            if hasattr(control_mode, 'reset'):
                control_mode.reset(self, robot_id)

    def _build_info(self, collisions=None, winner=None, end_reason=None, terminated=False, truncated=False):
        if collisions is None:
            collisions = []

        robot_states = {
            'robot_a': self.robot_a.get_state_summary(),
            'robot_b': self.robot_b.get_state_summary(),
        }

        info = {
            'scores': self.score_calculator.get_health(),
            'collisions': collisions,
            'positions': {
                'robot_a': self.robot_a.get_position(),
                'robot_b': self.robot_b.get_position(),
            },
            'torso_positions': {
                'robot_a': robot_states['robot_a']['torso_position'],
                'robot_b': robot_states['robot_b']['torso_position'],
            },
            'robot_states': robot_states,
            'relative_metrics': self._build_relative_metrics(robot_states),
            'hit_records': self.hit_records.copy(),
            'winner': winner if (terminated or truncated) else None,
            'end_reason': end_reason,
            'current_step': self.current_step,
            'physics_step_count': self.physics_step_count,
            'controller_state': {
                'robot_a': {
                    'reference_positions': self._controller_reference_positions['robot_a'].copy(),
                    'target_positions': self._controller_target_positions['robot_a'].copy(),
                    'action_scale': self._controller_action_scale['robot_a'].copy(),
                },
                'robot_b': {
                    'reference_positions': self._controller_reference_positions['robot_b'].copy(),
                    'target_positions': self._controller_target_positions['robot_b'].copy(),
                    'action_scale': self._controller_action_scale['robot_b'].copy(),
                },
            },
            'control_modes': {
                robot_id: getattr(control_mode, 'name', control_mode.__class__.__name__)
                for robot_id, control_mode in self.control_modes.items()
            },
            'reset': dict(self._last_reset_payload),
            'constraints': dict(self._last_constraint_results),
            'disturbances': list(self._last_disturbance_events),
            'metrics': dict(self._last_metric_payloads),
            'raw_state': {
                'qpos': self.physics.data.qpos.copy(),
                'qvel': self.physics.data.qvel.copy(),
                'ctrl': self.physics.data.ctrl.copy(),
            },
            'observation_slices': self.observation_slices,
        }
        return info

    def _get_robot_lookup(self):
        return {
            'robot_a': self.robot_a,
            'robot_b': self.robot_b,
        }

    def _default_action_scale_from_joint_limits(self, lower_limits, upper_limits):
        default_scale = np.full(HumanoidRobot.ACTION_DIM, 0.25, dtype=np.float32)
        finite_mask = np.isfinite(lower_limits) & np.isfinite(upper_limits)
        default_scale[finite_mask] = 0.25 * (upper_limits[finite_mask] - lower_limits[finite_mask])
        return np.maximum(default_scale, 1e-3).astype(np.float32)

    def _initialize_controller_state(self):
        if self.robot_a is None or self.robot_b is None:
            return

        self._controller_joint_limits = {}
        self._controller_ctrl_limits = {}
        for robot_id, robot in self._get_robot_lookup().items():
            joint_limits = robot.get_joint_position_limits()
            ctrl_limits = robot.get_actuator_ctrl_limits()
            default_scale = self._default_action_scale_from_joint_limits(
                joint_limits['lower'],
                joint_limits['upper'],
            )
            self._controller_joint_limits[robot_id] = joint_limits
            self._controller_ctrl_limits[robot_id] = ctrl_limits
            self._default_controller_action_scale[robot_id] = default_scale.copy()

    def _coerce_joint_vector(self, robot_id, joint_values, base_vector):
        vector = base_vector.copy()
        if joint_values is None:
            return vector
        if isinstance(joint_values, dict):
            for joint_name, joint_value in joint_values.items():
                joint_index = self._joint_name_to_index.get(joint_name)
                if joint_index is None:
                    continue
                vector[joint_index] = float(joint_value)
            return vector.astype(np.float32)
        return np.asarray(joint_values, dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM)

    def reset_controller_config(self):
        self._initialize_controller_state()
        self._controller_kp = self._default_controller_kp.copy()
        self._controller_kd = self._default_controller_kd.copy()
        for robot_id in ('robot_a', 'robot_b'):
            self._controller_reference_positions[robot_id] = np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)
            self._controller_action_scale[robot_id] = self._default_controller_action_scale[robot_id].copy()
            self._controller_target_positions[robot_id] = self._controller_reference_positions[robot_id].copy()

    def set_controller_reference_positions(self, joint_positions):
        self._initialize_controller_state()
        for robot_id, joint_values in joint_positions.items():
            if joint_values is None:
                continue
            reference_positions = self._coerce_joint_vector(
                robot_id,
                joint_values,
                self._controller_reference_positions[robot_id],
            )
            joint_limits = self._controller_joint_limits[robot_id]
            reference_positions = np.clip(
                reference_positions,
                joint_limits['lower'],
                joint_limits['upper'],
            ).astype(np.float32)
            self._controller_reference_positions[robot_id] = reference_positions
            self._controller_target_positions[robot_id] = reference_positions.copy()

    def set_controller_action_scale(self, action_scales):
        self._initialize_controller_state()
        for robot_id, scale_values in action_scales.items():
            if scale_values is None:
                continue
            action_scale = self._coerce_joint_vector(
                robot_id,
                scale_values,
                self._controller_action_scale[robot_id],
            )
            self._controller_action_scale[robot_id] = np.maximum(action_scale, 0.0).astype(np.float32)

    def set_controller_gains(self, kp=None, kd=None):
        self._initialize_controller_state()
        if kp is not None:
            if np.isscalar(kp):
                self._controller_kp = np.full(HumanoidRobot.ACTION_DIM, float(kp), dtype=np.float32)
            else:
                self._controller_kp = np.asarray(kp, dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM)
        if kd is not None:
            if np.isscalar(kd):
                self._controller_kd = np.full(HumanoidRobot.ACTION_DIM, float(kd), dtype=np.float32)
            else:
                self._controller_kd = np.asarray(kd, dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM)

    def _compute_target_positions(self, robot_id, residual_action):
        joint_limits = self._controller_joint_limits[robot_id]
        target_positions = self._controller_reference_positions[robot_id] + self._controller_action_scale[robot_id] * residual_action
        target_positions = np.clip(
            target_positions,
            joint_limits['lower'],
            joint_limits['upper'],
        ).astype(np.float32)
        self._controller_target_positions[robot_id] = target_positions
        return target_positions

    def _compute_torque_action(self, robot_id, target_positions):
        robot = self._get_robot_lookup()[robot_id]
        joint_states = robot.get_joint_states()
        current_positions = joint_states['positions']
        current_velocities = joint_states['velocities']
        torque_action = self._controller_kp * (target_positions - current_positions) - self._controller_kd * current_velocities
        ctrl_limits = self._controller_ctrl_limits[robot_id]
        return np.clip(
            torque_action,
            ctrl_limits['lower'],
            ctrl_limits['upper'],
        ).astype(np.float32)

    def _update_cached_actions(self, action_dict):
        if action_dict is None:
            return

        if 'robot_a' in action_dict and action_dict['robot_a'] is not None:
            self.actions['robot_a'] = np.clip(
                np.asarray(action_dict['robot_a'], dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM),
                -1.0,
                1.0,
            )
        if 'robot_b' in action_dict and action_dict['robot_b'] is not None:
            self.actions['robot_b'] = np.clip(
                np.asarray(action_dict['robot_b'], dtype=np.float32).reshape(HumanoidRobot.ACTION_DIM),
                -1.0,
                1.0,
            )

    def _apply_cached_actions(self):
        self._initialize_controller_state()
        resolved_action_a = self.control_modes['robot_a'].resolve_action(self, 'robot_a', self.actions['robot_a'])
        if resolved_action_a is not None:
            target_positions = self._compute_target_positions('robot_a', resolved_action_a)
            torque_action = self._compute_torque_action('robot_a', target_positions)
            self.robot_a.apply_action(torque_action)
        resolved_action_b = self.control_modes['robot_b'].resolve_action(self, 'robot_b', self.actions['robot_b'])
        if resolved_action_b is not None:
            target_positions = self._compute_target_positions('robot_b', resolved_action_b)
            torque_action = self._compute_torque_action('robot_b', target_positions)
            self.robot_b.apply_action(torque_action)

    def set_robot_joint_positions(self, joint_positions, update_controller_reference=True):
        robot_lookup = {
            'robot_a': self.robot_a,
            'robot_b': self.robot_b,
        }

        for robot_id, joint_targets in joint_positions.items():
            robot = robot_lookup.get(robot_id)
            if robot is None or joint_targets is None:
                continue

            joint_limits = robot.get_joint_position_limits()

            for joint_name, joint_value in joint_targets.items():
                joint_id = robot._joint_indices.get(joint_name)
                if joint_id is None:
                    continue
                joint_index = self._joint_name_to_index.get(joint_name)
                if joint_index is not None:
                    joint_value = float(
                        np.clip(
                            joint_value,
                            joint_limits['lower'][joint_index],
                            joint_limits['upper'][joint_index],
                        )
                    )
                qpos_idx = self.physics.model.jnt_qposadr[joint_id]
                qvel_idx = self.physics.model.jnt_dofadr[joint_id]
                self.physics.data.qpos[qpos_idx] = float(joint_value)
                self.physics.data.qvel[qvel_idx] = 0.0

        mujoco.mj_forward(self.physics.model, self.physics.data)
        if update_controller_reference:
            self.set_controller_reference_positions(joint_positions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and 'initial_distance' in options:
            self.initial_distance = float(options['initial_distance'])

        if not self._robots_created:
            # Robot A: facing +X 
            pos_a = [-self.initial_distance / 2, 0, 1.4]
            orn_a = [1, 0, 0, 0] 

            self.robot_a = HumanoidRobot(
                self.physics, pos_a, orn_a, robot_id="robot_a", color=(0.8, 0.2, 0.2)
            )

            # Robot B: facing -X 
            pos_b = [self.initial_distance / 2, 0, 1.4]
            orn_b = [0, 0, 0, 1]  #Rotate 180 degrees around Z axis

            self.robot_b = HumanoidRobot(
                self.physics, pos_b, orn_b, robot_id="robot_b", color=(0.2, 0.2, 0.8)
            )
            self._robots_created = True

        if self._initial_qpos is not None:
            self.physics.data.qpos[:] = self._initial_qpos
            self.physics.data.qvel[:] = self._initial_qvel
            self.physics.data.ctrl[:] = self._initial_ctrl

        self._last_constraint_results = {}
        self._last_disturbance_events = []
        self._last_metric_payloads = {}
        self._last_reset_payload = self.resetter.reset(self, self._rng, options)
        self.physics.data.qvel[:] = 0.0
        self.physics.data.ctrl[:] = 0.0
        self.physics.data.qfrc_applied[:] = 0.0
        self._initialize_controller_state()
        self.reset_controller_config()

        self.score_calculator.reset()
        self.actions = {"robot_a": None, "robot_b": None}
        self.current_step = 0
        self.physics_step_count = 0
        self.video_buffer = []
        self.hit_records = {'robot_a': [], 'robot_b': []}
        self._prev_cam_pos = None
        self._prev_lookat = None
        self._prev_azi = None
        self._prev_ele = None
        self._prev_dist = None
        self._reset_runtime_modules()

        # Trigger one forward to apply position and velocity
        mujoco.mj_forward(self.physics.model, self.physics.data)

        if self._initial_qpos is None:
            self._initial_qpos = self.physics.data.qpos.copy()
            self._initial_qvel = self.physics.data.qvel.copy()
            self._initial_ctrl = self.physics.data.ctrl.copy()

        observation = self._get_obs()

        info = self._build_info()
        info['metrics'] = self.collect_metric_payloads(
            observation,
            info,
            terminated=False,
            truncated=False,
        )

        return observation, info

    def step(self, action_dict=None, action_callback=None):
        self.hit_records = {'robot_a': [], 'robot_b': []}
        self._last_constraint_results = {}
        self._last_disturbance_events = []
        self._last_metric_payloads = {}
        for constraint in self.constraints:
            if hasattr(constraint, 'begin_step'):
                constraint.begin_step(self)
        self._update_cached_actions(action_dict)

        all_collisions = []
        for i in range(self.action_steps):
            if action_callback is not None:
                callback_actions = action_callback(self, i)
                self._update_cached_actions(callback_actions)

            self._apply_cached_actions()
            self.apply_disturbances(self.current_step, i)

            self.physics.step()
            self.apply_constraints()
            self.physics_step_count += 1

            collisions = self.collision_detector.check_collisions(
                self.robot_a, self.robot_b, self.physics
            )
            all_collisions.extend(collisions)

            for collision in collisions:
                defender = collision['defender']
                hit_part = collision['hit_part']
                damage_part = self.collision_detector.get_damage_part(hit_part)
                damage = self.score_calculator.take_damage(
                    defender,
                    damage_part,
                    collision.get('impulse', 0.0),
                )

                if damage < 0:
                    self.hit_records[defender].append({
                        'hit_part': hit_part,
                        'damage_part': damage_part,
                        'damage': damage,
                        'velocity': collision.get('velocity', 0),
                        'force': collision.get('force', 0.0),
                        'impulse': collision.get('impulse', 0.0),
                        'contact_count': collision.get('contact_count', 0),
                    })

            if self.render_mode is not None and self.physics_step_count % self.video_sample_steps == 0:
                frame = self.get_broadcast_view()
                self.video_buffer.append(frame)

        terminated = False
        truncated = False
        end_reason = None
        winner = None

        is_over, temp_winner, reason = self.score_calculator.check_match_over()
        if is_over:
            terminated = True
            winner = temp_winner
            end_reason = reason

        self.current_step += 1
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            winner = self.score_calculator.get_winner_by_health()
            if winner == 'draw':
                end_reason = f"Time limit reached ({self.match_duration}s), draw"
            else:
                end_reason = f"Time limit reached ({self.match_duration}s), {winner} wins by health"

        reward = {'robot_a': 0.0, 'robot_b': 0.0}
        observation = self._get_obs()

        info = self._build_info(
            collisions=all_collisions,
            winner=winner,
            end_reason=end_reason,
            terminated=terminated,
            truncated=truncated,
        )
        info['metrics'] = self.collect_metric_payloads(
            observation,
            info,
            terminated=terminated,
            truncated=truncated,
        )

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs_a = self.robot_a.get_observation(opponent_robot=self.robot_b)
        obs_b = self.robot_b.get_observation(opponent_robot=self.robot_a)

        if obs_a.shape[0] != HumanoidRobot.OBSERVATION_DIM or obs_b.shape[0] != HumanoidRobot.OBSERVATION_DIM:
            raise ValueError(
                f"Observation shape mismatch: robot_a={obs_a.shape}, robot_b={obs_b.shape}, expected={(HumanoidRobot.OBSERVATION_DIM,)}"
            )

        return {
            'robot_a_obs': obs_a.astype(np.float32),
            'robot_b_obs': obs_b.astype(np.float32),
        }

    def get_broadcast_view(self):
        import mujoco
        try:
            pos_a = self.robot_a.get_position()
            pos_b = self.robot_b.get_position()
            center = (pos_a + pos_b) / 2.0
            
            # Base viewpoint: center of two robots, height slightly lowered (waist level)
            target_lookat = center.copy()
            target_lookat[2] = 1.0  
            
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0])

            # Expect to look from the side of the two (Azimuth angle corresponding to the normal vector of direction)
            # arctan2(y, x) Get the angle of the vector on the XY plane
            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            
            # Camera is on the side, so azimuth + 90 degrees
            target_azi = dir_angle + 90.0
            target_ele = -20.0  # look down 20 degrees
            
            # Camera distance: base distance is spacing * 1.5, limited between 2.5 and 4.0
            target_dist = max(2.5, min(4.0, dist_ab * 1.5))
            
            # --- Boundary limit (prevent camera from moving outside walls) ---
            # Room boundary is approx x,y \in [-3.05, 3.05]
            # We reserve 0.5 safe distance -> wall limit at 2.55
            limit = 2.55
            
            # In MuJoCo, given azimuth, elevation and distance, camera's horizontal offset in world coords is approx:
            # dx = -dist * cos(azi) * cos(ele)
            # dy = -dist * sin(azi) * cos(ele)
            azi_rad = np.radians(target_azi)
            ele_rad = np.radians(target_ele)
            
            dx = -target_dist * np.cos(azi_rad) * np.cos(ele_rad)
            dy = -target_dist * np.sin(azi_rad) * np.cos(ele_rad)
            
            cam_x = target_lookat[0] + dx
            cam_y = target_lookat[1] + dy
            
            # If expected X exceeds room, shorten distance to approach wall
            if abs(cam_x) > limit:
                max_dx = limit - target_lookat[0] if cam_x > 0 else -limit - target_lookat[0]
                factor = -np.cos(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dx / factor))
                    
            # If expected Y exceeds room
            if abs(cam_y) > limit:
                max_dy = limit - target_lookat[1] if cam_y > 0 else -limit - target_lookat[1]
                factor = -np.sin(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dy / factor))

            # --- Smooth filtering (EMA) ---
            alpha_pos = 0.05  # Smoothing coefficient for polar coords and distance
            alpha_look = 0.1  # Smoothing coefficient for observation focus
            
            if getattr(self, '_prev_azi', None) is None:
                azi = target_azi
                ele = target_ele
                dist = target_dist
                lookat = target_lookat.copy()
            else:
                # 角degrees平滑需要处理 360 degrees循环跳变
                diff = (target_azi - self._prev_azi + 180) % 360 - 180
                azi = self._prev_azi + diff * alpha_pos
                ele = self._prev_ele * (1.0 - alpha_pos) + target_ele * alpha_pos
                dist = self._prev_dist * (1.0 - alpha_pos) + target_dist * alpha_pos
                lookat = self._prev_lookat * (1.0 - alpha_look) + target_lookat * alpha_look
                
            # Update cache
            self._prev_azi = azi
            self._prev_ele = ele
            self._prev_dist = dist
            self._prev_lookat = lookat.copy()

            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = lookat
            cam.distance = dist
            cam.elevation = ele
            cam.azimuth = azi

            renderer = mujoco.Renderer(self.physics.model, height=720, width=1280)
            renderer.update_scene(self.physics.data, camera=cam)
            image = renderer.render()
            del renderer
            return image
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to render broadcast view: {e}")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.get_broadcast_view()
        return None

    def get_video_buffer(self):
        return self.video_buffer

    def clear_video_buffer(self):
        self.video_buffer = []

    def save_video(self, filepath, fps=10):
        try:
            import cv2
            if len(self.video_buffer) == 0:
                print("Warning: No video frames to save")
                return

            height, width = self.video_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            for frame in self.video_buffer:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"Video saved to {filepath} ({len(self.video_buffer)} frames)")
        except ImportError:
            print("Warning: opencv-python not installed")
        except Exception as e:
            print(f"Error saving video: {e}")

    def close(self):
        try:
            self.physics.close()
        except:
            pass
