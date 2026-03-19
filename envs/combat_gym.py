import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces
import os
from pathlib import Path
import sys

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.physics import PhysicsEngine
from core.humanoid_robot import HumanoidRobot
from core.collision import CollisionDetector
from core.scoring import ScoreCalculator

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
        video_sample_frequency=10,
        match_duration=30.0,   # Single roundduration 30 seconds
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = dt
        self.initial_distance = initial_distance
        self.match_duration = match_duration

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

        self.collision_detector = CollisionDetector(velocity_threshold=1.0)
        self.score_calculator = ScoreCalculator()

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

    def _get_root_pose_targets(self):
        return {
            'robot_a': {
                'joint_name': 'root_red',
                'position': np.array([-self.initial_distance / 2.0, 0.0, self.DEFAULT_ROOT_HEIGHT], dtype=np.float64),
                'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
            'robot_b': {
                'joint_name': 'root_blue',
                'position': np.array([self.initial_distance / 2.0, 0.0, self.DEFAULT_ROOT_HEIGHT], dtype=np.float64),
                'orientation': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            },
        }

    def _apply_initial_root_poses(self):
        for root_pose in self._get_root_pose_targets().values():
            joint_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_JOINT, root_pose['joint_name'])
            if joint_id < 0:
                continue

            qpos_adr = self.physics.model.jnt_qposadr[joint_id]
            qvel_adr = self.physics.model.jnt_dofadr[joint_id]
            self.physics.data.qpos[qpos_adr:qpos_adr + 3] = root_pose['position']
            self.physics.data.qpos[qpos_adr + 3:qpos_adr + 7] = root_pose['orientation']
            self.physics.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

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

    def _build_info(self, collisions=None, winner=None, end_reason=None, terminated=False, truncated=False):
        if collisions is None:
            collisions = []

        robot_states = {
            'robot_a': self.robot_a.get_state_summary(),
            'robot_b': self.robot_b.get_state_summary(),
        }

        return {
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
            'observation_slices': self.observation_slices,
        }

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
        if self.actions['robot_a'] is not None:
            target_positions = self._compute_target_positions('robot_a', self.actions['robot_a'])
            torque_action = self._compute_torque_action('robot_a', target_positions)
            self.robot_a.apply_action(torque_action)
        if self.actions['robot_b'] is not None:
            target_positions = self._compute_target_positions('robot_b', self.actions['robot_b'])
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

        self._apply_initial_root_poses()
        self.physics.data.qvel[:] = 0.0
        self.physics.data.ctrl[:] = 0.0
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

        # Trigger one forward to apply position and velocity
        mujoco.mj_forward(self.physics.model, self.physics.data)

        if self._initial_qpos is None:
            self._initial_qpos = self.physics.data.qpos.copy()
            self._initial_qvel = self.physics.data.qvel.copy()
            self._initial_ctrl = self.physics.data.ctrl.copy()

        observation = self._get_obs()

        info = self._build_info()

        return observation, info

    def step(self, action_dict=None, action_callback=None):
        self.hit_records = {'robot_a': [], 'robot_b': []}
        self._update_cached_actions(action_dict)

        all_collisions = []
        for i in range(self.action_steps):
            if action_callback is not None:
                callback_actions = action_callback(self, i)
                self._update_cached_actions(callback_actions)

            self._apply_cached_actions()

            self.physics.step()
            self.physics_step_count += 1

            collisions = self.collision_detector.check_collisions(
                self.robot_a, self.robot_b, self.physics
            )
            all_collisions.extend(collisions)

            for collision in collisions:
                defender = collision['defender']
                hit_part = collision['hit_part']
                damage_part = self.collision_detector.get_damage_part(hit_part)
                damage = self.score_calculator.take_damage(defender, damage_part)

                if damage < 0:
                    self.hit_records[defender].append({
                        'hit_part': hit_part,
                        'damage_part': damage_part,
                        'damage': damage,
                        'velocity': collision.get('velocity', 0),
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
