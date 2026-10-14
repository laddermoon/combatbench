"""MotionTrackingSimulator — humanoid21 参考运动跟踪仿真器（M2/L2）。

在 Humanoid21Simulator 之上：

- reset 时做 **RSI**（Reference State Initialization）：按 seed 随机选
  控制帧起点，把被跟踪机器人的 qpos/qvel 一次性写到参考帧状态
  （根 pos/quat/线速度/机体系角速度 + 21 关节 pos/vel）。这是
  episode 初始化，不是逐帧覆写。
- ``set_action`` 每调用一次（20Hz 控制步）相位 +1；参考帧相位就是
  episode 时间轴。
- 观测扩展（仅被跟踪机器人，对手保持 96 维）：
    [phase(1), ref_q_norm(21), ref_qv_norm(21)*0.2, ref_root_linvel_ref(3),
     root_disp_ego(3)*0.5, dyaw_sin_cos(2), ref_root_angvel_ref(3)]
  → obs_dim = 96 + 54 = 150。**参考帧取 frame+1（下一控制步目标），
  与步末奖励所对照的参考帧一致。**
- ``get_derived_state`` 增加 ``motion_track`` 字段供奖励/终止插件使用：
  ref_q_norm / ref_qv_norm / ref_root_pos / ref_keypoints(15,3) /
  self_keypoints / frame / phase / ended。

robot_b 留在 standing 姿态由外部 standing policy 驱动，仅作场景存在。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.tracking.ref_data import RefClip, REF_STRIDE

import mujoco


class MotionTrackingSimulator(Humanoid21Simulator):

    OBS_EXT_DIM = 54  # 1 phase + 21 q + 21 qv + 3 root linvel
                      # + 3 root pos + 2 yaw(sin,cos) + 3 root angvel

    def __init__(self, clip_id: str, track_robot: str = 'robot_a',
                 rsi: str = 'random', clips_dir: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if track_robot not in ('robot_a', 'robot_b'):
            raise ValueError(f'track_robot must be robot_a/robot_b, got {track_robot}')
        self._track_robot = track_robot
        self._rsi = rsi
        self.ref = RefClip(clip_id, clips_dir or
                           '/data1/dev/combatbench_mocap/clips')
        self._frame = 0

        # 被跟踪机器人的 keypoint body ids（与 npz keypoint_names 同名 + 后缀）
        cache = self._robots[track_robot]
        suffix = cache['suffix']
        self._kp_ids = {}
        for name in self.ref.keypoint_names:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f'{name}{suffix}')
            if bid >= 0:
                self._kp_ids[name] = bid

    # ------------------------------------------------------------------
    @property
    def frame_idx(self) -> int:
        return self._frame

    @property
    def track_ended(self) -> bool:
        return self._frame >= self.ref.n_ctrl - 1

    def ref_at(self, k: Optional[int] = None) -> Dict[str, np.ndarray]:
        return self.ref.at(self._frame if k is None else k)

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)

        opts = options or {}
        if 'start_frame' in opts:
            k0 = int(opts['start_frame'])
        elif self._rsi == 'first':
            k0 = self.ref.min_start_ctrl
        else:
            rng = np.random.default_rng(seed)
            k0 = int(rng.integers(self.ref.min_start_ctrl, self.ref.n_ctrl))
        k0 = int(np.clip(k0, 0, self.ref.n_ctrl - 1))
        self._frame = k0

        f0 = min(k0 * REF_STRIDE, self.ref.n_ctrl - 1)  # ref 已按 20Hz 存
        rf = self.ref.at(k0)

        cache = self._robots[self._track_robot]
        root_qpos_adr = cache['root_qpos_adr']
        root_qvel_adr = cache['root_qvel_adr']
        qpos_idx = cache['qpos_indices']
        qvel_idx = cache['qvel_indices']
        norm = self._norm_params[self._track_robot]

        # 根 pos/quat（世界系）
        self.data.qpos[root_qpos_adr:root_qpos_adr + 3] = rf['root_pos']
        self.data.qpos[root_qpos_adr + 3:root_qpos_adr + 7] = rf['root_quat']

        # 关节角（rad）与关节速度（rad/s）
        self.data.qpos[qpos_idx] = rf['q_rad']
        self.data.qvel[qvel_idx] = rf['qv_rad']

        # freejoint qvel: [0:3] 世界系线速度, [3:6] 机体系角速度
        Rr = R.from_quat([rf['root_quat'][1], rf['root_quat'][2],
                          rf['root_quat'][3], rf['root_quat'][0]])
        self.data.qvel[root_qvel_adr:root_qvel_adr + 3] = rf['root_linvel']
        self.data.qvel[root_qvel_adr + 3:root_qvel_adr + 6] = \
            Rr.inv().as_matrix() @ rf['root_angvel']

        # PD 目标对齐到当前参考帧，避免首步冲击
        self._target_pos_norm[self._track_robot] = rf['q_norm'].astype(np.float32)

        self._data_cache.clear()
        mujoco.mj_forward(self.model, self.data)

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._frame += 1

    # ------------------------------------------------------------------
    def _self_keypoints(self) -> Dict[str, np.ndarray]:
        return {n: self.data.xpos[bid].copy()
                for n, bid in self._kp_ids.items()}

    def _ref_root_vel_local(self, rf: Dict[str, np.ndarray]) -> np.ndarray:
        # 参考线速度投到**参考朝向**系：表达"目标朝向下的期望速度"，
        # 与 dyaw 观测配合时语义一致（投到自身体系会随自身朝向漂移而失真）。
        Rr = R.from_quat([rf['root_quat'][1], rf['root_quat'][2],
                          rf['root_quat'][3], rf['root_quat'][0]])
        return Rr.inv().as_matrix() @ rf['root_linvel']

    def _ref_root_angvel_local(self, rf: Dict[str, np.ndarray]) -> np.ndarray:
        Rr = R.from_quat([rf['root_quat'][1], rf['root_quat'][2],
                          rf['root_quat'][3], rf['root_quat'][0]])
        return Rr.inv().as_matrix() @ rf['root_angvel']

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        if '_ds_track' in self._data_cache:
            cached = self._data_cache['_ds_track']
        else:
            rf = self.ref_at()
            self_kp = self._self_keypoints()
            kp_names = self.ref.keypoint_names
            ref_kp = rf['keypoints']
            cached = {
                'frame': self._frame,
                'phase': rf['phase'],
                'ended': self.track_ended,
                'track_robot': self._track_robot,
                'ref_q_norm': rf['q_norm'].astype(np.float32),
                'ref_qv_norm': rf['qv_norm'].astype(np.float32),
                'ref_root_pos': rf['root_pos'].astype(np.float32),
                'ref_root_quat': rf['root_quat'].astype(np.float32),
                'ref_keypoints': ref_kp.astype(np.float32),
                'ref_keypoint_names': kp_names,
                'self_keypoints': {k: v.astype(np.float32)
                                   for k, v in self_kp.items()},
            }
            self._data_cache['_ds_track'] = cached
        ds = super().get_derived_state(fields)
        ds['motion_track'] = cached
        return ds

    def get_observation(self) -> Dict[str, Any]:
        if '_obs_track' in self._data_cache:
            return self._data_cache['_obs_track']
        base = super().get_observation()
        # 观测给“下一帧”目标：本控制步内 robot 从 frame k 运动到 k+1，
        # 奖励在步末对照 ref[k+1] 评分；给 ref[k] 会引入一步目标滞后
        # （快速出拳段 ~10°/控制步 的系统性相位差）。
        rf = self.ref_at(self._frame + 1)
        # 根级目标必须可观测**且可比较**：96 维基础观测不含绝对 xy/yaw
        # （OBSERVATION_zh.md §4 刻意设计），因此给绝对 ref_root_pos/ref_yaw
        # 策略无从对照自身 —— v4 拳击 dxy 仍漂移 1.0m/47° 的直接原因。
        # 这里改发 ego 系跟踪误差：disp = R_self⁻¹(ref_pos − self_pos)，
        # dyaw = wrap(yaw_ref − yaw_self)。
        cache = self._robots[self._track_robot]
        self_pos = self.data.qpos[cache['root_qpos_adr']:
                                  cache['root_qpos_adr'] + 3]
        self_quat = self.data.qpos[cache['root_qpos_adr'] + 3:
                                   cache['root_qpos_adr'] + 7]
        R_self = R.from_quat([self_quat[1], self_quat[2],
                              self_quat[3], self_quat[0]])
        yaw_self = float(R_self.as_euler('xyz')[2])
        yaw_ref = float(R.from_quat(
            [rf['root_quat'][1], rf['root_quat'][2],
             rf['root_quat'][3], rf['root_quat'][0]]).as_euler('xyz')[2])
        dyaw = (yaw_ref - yaw_self + np.pi) % (2 * np.pi) - np.pi
        disp_ego = R_self.inv().as_matrix() @ (rf['root_pos'] - self_pos)
        ext = np.concatenate([
            [rf['phase']],
            rf['q_norm'],
            rf['qv_norm'] * 0.2,
            self._ref_root_vel_local(rf),
            disp_ego * 0.5,
            [np.sin(dyaw), np.cos(dyaw)],
            self._ref_root_angvel_local(rf),
        ]).astype(np.float32)
        out = {}
        for rid, o in base.items():
            o = np.asarray(o, dtype=np.float32)
            out[rid] = (np.concatenate([o, ext]) if rid == self._track_robot
                        else o)
        self._data_cache['_obs_track'] = out
        return out
