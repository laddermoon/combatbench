import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from pathlib import Path
import sys

# 添加父目录到路径以导入本地模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.physics import PhysicsEngine
from core.humanoid_robot import HumanoidRobot
from core.collision import CollisionDetector
from core.scoring import ScoreCalculator

class CombatGymEnv(gym.Env):
    """
    双机器人对抗 Gym 环境 (V1.0 - 21DOF)
    单回合 Episode (30s)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        arena_xml=None,
        dt=0.002,
        initial_distance=2.0,  # 规则：初始距离两米相向而立
        control_frequency=20,
        video_sample_frequency=10,
        match_duration=30.0,   # 单回合时长 30 秒
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = dt
        self.initial_distance = initial_distance
        self.match_duration = match_duration

        self.sim_frequency = 1.0 / dt
        self.control_frequency = control_frequency
        self.video_sample_frequency = video_sample_frequency

        self.action_steps = int(self.sim_frequency / control_frequency)
        self.video_sample_steps = int(self.sim_frequency / video_sample_frequency)

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

        self.action_space = spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        })

        # 观测空间：42 + 13 + 8 + 64 = 127维
        obs_dim = 127
        self.observation_space = spaces.Dict({
            "robot_a_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "robot_b_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        })

        self.collision_detector = CollisionDetector(velocity_threshold=1.0)
        self.score_calculator = ScoreCalculator()

        self.robot_a = None
        self.robot_b = None
        self.actions = {"robot_a": None, "robot_b": None}
        self.video_buffer = []
        self.hit_records = {'robot_a': [], 'robot_b': []}
        
        self.current_step = 0
        self.physics_step_count = 0
        self.max_steps = int(match_duration * control_frequency)

        # 缓存相机状态用于平滑跟随
        self._prev_cam_pos = None
        self._prev_lookat = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not hasattr(self, '_robots_created'):
            # Robot A: 面向 +X 
            pos_a = [-self.initial_distance / 2, 0, 1.4]
            orn_a = [1, 0, 0, 0] 

            self.robot_a = HumanoidRobot(
                self.physics, pos_a, orn_a, robot_id="robot_a", color=(0.8, 0.2, 0.2)
            )

            # Robot B: 面向 -X 
            pos_b = [self.initial_distance / 2, 0, 1.4]
            orn_b = [0, 0, 0, 1]  #绕Z轴旋转180度

            self.robot_b = HumanoidRobot(
                self.physics, pos_b, orn_b, robot_id="robot_b", color=(0.2, 0.2, 0.8)
            )

            self._initial_qpos = self.physics.data.qpos.copy()
            self._robots_created = True
        else:
            self.physics.data.qpos[:] = self._initial_qpos
            self.physics.data.qvel[:] = 0

        self.score_calculator.reset()
        self.actions = {"robot_a": None, "robot_b": None}
        self.current_step = 0
        self.physics_step_count = 0
        self.video_buffer = []
        self.hit_records = {'robot_a': [], 'robot_b': []}
        self._prev_cam_pos = None
        self._prev_lookat = None

        # 触发一次 forward 让位置和速度生效
        import mujoco
        mujoco.mj_forward(self.physics.model, self.physics.data)

        observation = self._get_obs()

        info = {
            'scores': self.score_calculator.get_health(),
            'positions': {
                'robot_a': self.robot_a.get_position(),
                'robot_b': self.robot_b.get_position(),
            },
            'hit_records': self.hit_records.copy(),
        }

        return observation, info

    def step(self, action_dict):
        self.hit_records = {'robot_a': [], 'robot_b': []}

        if 'robot_a' in action_dict and action_dict['robot_a'] is not None:
            self.actions['robot_a'] = action_dict['robot_a']
        if 'robot_b' in action_dict and action_dict['robot_b'] is not None:
            self.actions['robot_b'] = action_dict['robot_b']

        if self.actions['robot_a'] is not None:
            self.robot_a.set_joint_targets(self.actions['robot_a'])
        if self.actions['robot_b'] is not None:
            self.robot_b.set_joint_targets(self.actions['robot_b'])

        all_collisions = []
        for i in range(self.action_steps):
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

        info = {
            'scores': self.score_calculator.get_health(),
            'collisions': all_collisions,
            'positions': {
                'robot_a': self.robot_a.get_position(),
                'robot_b': self.robot_b.get_position(),
            },
            'hit_records': self.hit_records.copy(),
            'winner': winner if (terminated or truncated) else None,
            'end_reason': end_reason,
            'physics_step_count': self.physics_step_count,
        }

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        health_a = self.score_calculator.get_health('robot_a')
        health_b = self.score_calculator.get_health('robot_b')

        obs_a = self.robot_a.get_observation(opponent_robot=self.robot_b)
        obs_b = self.robot_b.get_observation(opponent_robot=self.robot_a)

        return {
            'robot_a_obs': obs_a.astype(np.float32),
            'robot_b_obs': obs_b.astype(np.float32),
        }

    def get_broadcast_view(self):
        import mujoco
        import os
        os.environ['MUJOCO_GL'] = 'egl'
        try:
            pos_a = self.robot_a.get_position()
            pos_b = self.robot_b.get_position()
            center = (pos_a + pos_b) / 2.0
            
            # 基础视点：两个机器人的中心，高度稍微调低一点(看腰部)
            target_lookat = center.copy()
            target_lookat[2] = 1.0  
            
            direction = pos_b - pos_a
            dist_ab = np.linalg.norm(direction)
            if dist_ab > 1e-6:
                direction = direction / dist_ab
            else:
                direction = np.array([1.0, 0.0, 0.0])

            # 期望从两人的侧面看过去 (方向的法向量对应的方位角)
            # arctan2(y, x) 得到向量在 XY 平面的角度
            dir_angle = np.degrees(np.arctan2(direction[1], direction[0]))
            
            # 摄像机位于侧面，所以方位角 + 90 度
            target_azi = dir_angle + 90.0
            target_ele = -20.0  # 俯视 20 度
            
            # 摄像机距离：基础距离为两人的间距乘以1.5，限制在 2.5 ~ 4.0 之间
            target_dist = max(2.5, min(4.0, dist_ab * 1.5))
            
            # --- 边界限制 (防止摄像机退到墙外) ---
            # 房间边界大约是 x,y \in [-3.05, 3.05]
            # 我们预留 0.5 的安全距离 -> 墙壁限制在 2.55
            limit = 2.55
            
            # 在 MuJoCo 中，给定方位角、仰角和距离，相机在世界坐标系的水平偏移大概是：
            # dx = -dist * cos(azi) * cos(ele)
            # dy = -dist * sin(azi) * cos(ele)
            azi_rad = np.radians(target_azi)
            ele_rad = np.radians(target_ele)
            
            dx = -target_dist * np.cos(azi_rad) * np.cos(ele_rad)
            dy = -target_dist * np.sin(azi_rad) * np.cos(ele_rad)
            
            cam_x = target_lookat[0] + dx
            cam_y = target_lookat[1] + dy
            
            # 如果预期的 X 坐标超出房间，通过缩短 distance 来逼近墙壁
            if abs(cam_x) > limit:
                max_dx = limit - target_lookat[0] if cam_x > 0 else -limit - target_lookat[0]
                factor = -np.cos(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dx / factor))
                    
            # 如果预期的 Y 坐标超出房间
            if abs(cam_y) > limit:
                max_dy = limit - target_lookat[1] if cam_y > 0 else -limit - target_lookat[1]
                factor = -np.sin(azi_rad) * np.cos(ele_rad)
                if abs(factor) > 1e-6:
                    target_dist = min(target_dist, abs(max_dy / factor))

            # --- 平滑滤波 (EMA) ---
            alpha_pos = 0.05  # 平滑极坐标和距离的系数
            alpha_look = 0.1  # 平滑观测焦点的系数
            
            if getattr(self, '_prev_azi', None) is None:
                azi = target_azi
                ele = target_ele
                dist = target_dist
                lookat = target_lookat.copy()
            else:
                # 角度平滑需要处理 360 度循环跳变
                diff = (target_azi - self._prev_azi + 180) % 360 - 180
                azi = self._prev_azi + diff * alpha_pos
                ele = self._prev_ele * (1.0 - alpha_pos) + target_ele * alpha_pos
                dist = self._prev_dist * (1.0 - alpha_pos) + target_dist * alpha_pos
                lookat = self._prev_lookat * (1.0 - alpha_look) + target_lookat * alpha_look
                
            # 更新缓存
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
