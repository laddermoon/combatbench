"""DeepMimicTrackSimulator — humanoid21 单 clip 运动跟踪仿真器。

在 Humanoid21Simulator 之上：

- reset 时做 **RSI**：从 ``ref.rsi_frames``（= 全帧 − meta.rsi_bad_frames）
  采控制帧起点，``set_core_state({track_robot: clip.state(k0)})`` 一次性
  落位——CoreState 六字段同名同语义，无任何手动索引。
- ``set_action`` 每调用一次（20Hz 控制步）相位 +1。
- 观测扩展（仅被跟踪机器人，对手保持 96 维）：

    ``ref_obs='full'``（默认，54 维）：
      [phase(1), ref_q_norm(21), ref_qv_norm(21)*0.2,
       ref_root_linvel_refframe(3), root_disp_ego(3)*0.5,
       dyaw_sin_cos(2), ref_root_angvel_refframe(3)]
    → obs_dim = 96 + 54 = 150。**参考帧取 frame+1（下一控制步目标），
    与步末奖励所对照的参考帧一致**（M2 v3 教训：给 ref[k] 引入一步
    目标滞后，快拳段 ~10°/控制步的系统性相位差）。

    ``ref_obs='phase_only'``（论文设定消融，1 维）：[phase]
    → obs_dim = 97。根级/姿态目标皆不可观测，只留标量相位。

- ``get_derived_state`` 增加 ``motion_track`` 字段供奖励/终止插件：
  ref_q_norm / ref_qv_norm / ref_root_pos / ref_root_quat /
  ref_keypoints / self_keypoints / nonfoot_body_ids /
  track_robot_aff / frame / phase / ended。

robot_b 留在 standing 姿态由外部 policy 驱动，仅作场景（不参与跟踪评分，
也不出现在 motion_track 中）。

坐标系说明：CoreClip 的 ``root_vel_local`` / ``root_angular_vel_local``
本就以参考根朝向系存储——正是"目标朝向下的期望速度"语义（M2 v5 需要
手动把世界系速度投到 ref 系；CoreClip 免费给出）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator

from .ref import CoreRefView

_FOOT_BODIES = ('foot_left', 'foot_right')


def _quat_mat_inv(quat_wxyz: np.ndarray) -> np.ndarray:
    """R^T for wxyz quaternion (scipy xyzw 转换一行版）。"""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return R.from_quat([q[1], q[2], q[3], q[0]]).inv().as_matrix()


def _yaw(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')[2])


class DeepMimicTrackSimulator(Humanoid21Simulator):

    OBS_EXT_FULL = 54   # 1 phase + 21 q + 21 qv + 3 linvel + 3 disp + 2 dyaw + 3 angvel
    OBS_EXT_PHASE_ONLY = 1

    def __init__(self, clip_id: str, track_robot: str = 'robot_a',
                 rsi: str = 'random', clips_dir: Optional[str] = None,
                 ref_obs: str = 'full', **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if track_robot not in ('robot_a', 'robot_b'):
            raise ValueError(f'track_robot must be robot_a/robot_b, got {track_robot}')
        if ref_obs not in ('full', 'phase_only'):
            raise ValueError(f"ref_obs must be 'full'/'phase_only', got {ref_obs}")
        self._track_robot = track_robot
        self._rsi = rsi
        self._ref_obs = ref_obs
        self.ref = CoreRefView(clip_id, clips_dir)
        self._frame = 0

        cache = self._robots[track_robot]
        suffix = cache['suffix']
        # KEYPOINT_BODIES + 后缀 → 自身关键点 body ids（世界系 xpos 直读）
        self._kp_ids = {}
        for name in self.ref.keypoint_names:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f'{name}{suffix}')
            if bid >= 0:
                self._kp_ids[name] = bid
        # 非足 body id 集合（contact 模式摔倒判定用）
        foot_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      f'{n}{suffix}') for n in _FOOT_BODIES}
        self._nonfoot_body_ids = frozenset(
            bid for bid in range(self.model.nbody)
            if self._body_id_to_aff.get(bid) == self._track_aff
            and bid not in foot_ids)

    @property
    def _track_aff(self) -> int:
        return 1 if self._track_robot == 'robot_a' else 2

    @property
    def obs_ext_dim(self) -> int:
        return (self.OBS_EXT_FULL if self._ref_obs == 'full'
                else self.OBS_EXT_PHASE_ONLY)

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
            k0 = int(self.ref.rsi_frames[0])
        else:
            rng = np.random.default_rng(seed)
            k0 = int(rng.choice(self.ref.rsi_frames))
        k0 = int(np.clip(k0, 0, self.ref.n_ctrl - 1))
        self._frame = k0

        st = self.ref.core_state(k0)
        self.set_core_state({self._track_robot: st})

        # PD 目标对齐当前参考帧，避免首步冲击。
        # _target_pos_norm 的语义是 action = (q − ref) / scale_eff；
        # q_rad = jpn*scale + mid ⇒ action = jpn * scale/scale_eff。
        np_ = self._norm_params[self._track_robot]
        self._target_pos_norm[self._track_robot] = (
            np.asarray(st['joint_pos_norm'], dtype=np.float32)
            * (np_['scale'] / np_['scale_eff'])).astype(np.float32)

        self._data_cache.clear()

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        super().set_action(action)
        self._frame += 1
        # 相位已推进：上一帧的 track 缓存作废（基类只在 physical_step 全清，
        # on_pre_phy_step 钩子里读 derived_state 会拿到旧帧数据）。
        self._data_cache.pop('_ds_track', None)
        self._data_cache.pop('_obs_track', None)

    # ------------------------------------------------------------------
    def _self_keypoints(self) -> Dict[str, np.ndarray]:
        return {n: self.data.xpos[bid].copy()
                for n, bid in self._kp_ids.items()}

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        if '_ds_track' in self._data_cache:
            cached = self._data_cache['_ds_track']
        else:
            rf = self.ref_at()
            cached = {
                'frame': self._frame,
                'phase': rf['phase'],
                'ended': self.track_ended,
                'track_robot': self._track_robot,
                'track_robot_aff': self._track_aff,
                'nonfoot_body_ids': self._nonfoot_body_ids,
                'ref_q_norm': rf['q_norm'].astype(np.float32),
                'ref_qv_norm': rf['qv_norm'].astype(np.float32),
                'ref_root_pos': rf['root_pos'].astype(np.float32),
                'ref_root_quat': rf['root_quat'].astype(np.float32),
                'ref_keypoints': rf['keypoints'].astype(np.float32),
                'ref_keypoint_names': self.ref.keypoint_names,
                'self_keypoints': {k: v.astype(np.float32)
                                   for k, v in self._self_keypoints().items()},
            }
            self._data_cache['_ds_track'] = cached
        ds = super().get_derived_state(fields)
        ds['motion_track'] = cached
        return ds

    def get_observation(self) -> Dict[str, Any]:
        if '_obs_track' in self._data_cache:
            return self._data_cache['_obs_track']
        base = super().get_observation()
        rf = self.ref_at(self._frame + 1)
        if self._ref_obs == 'phase_only':
            ext = np.array([rf['phase']], dtype=np.float32)
        else:
            cache = self._robots[self._track_robot]
            self_pos = self.data.qpos[cache['root_qpos_adr']:
                                      cache['root_qpos_adr'] + 3]
            self_quat = self.data.qpos[cache['root_qpos_adr'] + 3:
                                       cache['root_qpos_adr'] + 7]
            R_self_inv = _quat_mat_inv(self_quat)
            dyaw = (_yaw(rf['root_quat']) - _yaw(self_quat)
                    + np.pi) % (2 * np.pi) - np.pi
            disp_ego = R_self_inv @ (rf['root_pos'] - self_pos)
            ext = np.concatenate([
                [rf['phase']],
                rf['q_norm'],
                rf['qv_norm'] * 0.2,
                rf['root_vel_local'],
                disp_ego * 0.5,
                [np.sin(dyaw), np.cos(dyaw)],
                rf['root_angvel_local'],
            ]).astype(np.float32)
        out = {}
        for rid, o in base.items():
            o = np.asarray(o, dtype=np.float32)
            out[rid] = (np.concatenate([o, ext]) if rid == self._track_robot
                        else o)
        self._data_cache['_obs_track'] = out
        return out
