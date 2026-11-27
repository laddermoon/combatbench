"""DeepMimic28 跟踪仿真器：RSI + 相位推进 + MimicKit 精确 obs/reward 数据。

设计对齐 MimicKit deepmimic_env（单角色跟踪，robot_b 仅作场景陪衬）：
- reset: RSI——按均匀随机相位采样动作时间，将参考状态直接落入 robot_a
  （root pos/rot/vel/angvel + 28 hinge 角/角速度），WRAP clip 相位模回绕
- obs（robot_a, 464 维）= char_obs(140) + tar_obs(3×108)：
    char_obs = [root_h(1), root_rot tannorm(6), root_vel(3), root_angvel(3),
                joint_rot tannorm(14×6=84), dof_vel(28), key_pos 根相对(15)]
    tar_obs(每步) = [root_pos 偏移(3,z为绝对高), root_rot tannorm(6),
                joint_rot tannorm(84), key_pos 根相对(15)]
    tar 步 = 未来 [1,2,3] 个控制步的参考帧；全局系（global_obs=True）
- robot_b 保持 base 110 维战斗 obs（陪衬，冻结）
- reward/termination 所需的当前帧参考数据由 track_obs_data() 暴露给插件
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from envs.humanoid28.simulator import Humanoid28Simulator
from envs.humanoid28.meta import Humanoid28Meta

from .refmotion import RefMotion, load_ref_motion, KIN_JOINTS, hinge_layout
from .mathutil import (quat_mul, quat_conj, quat_rotate, quat_diff_angle,
                       quat_to_tan_norm, nlerp_quat, compose_group_quat)

# 关节组轴索引（hinge 关节）
HINGE_AXIS = {'right_elbow': 2, 'left_elbow': 2, 'right_knee': 1, 'left_knee': 1}
SPH_GROUPS = ['abdomen', 'neck', 'right_shoulder', 'left_shoulder',
              'right_hip', 'right_ankle', 'left_hip', 'left_ankle']
KEY_BODIES = ['head', 'right_hand', 'left_hand', 'right_foot', 'left_foot']

# MimicKit deepmimic_humanoid_env.yaml 的关节组误差权重（14 组，含 fixed 手=0）
JOINT_ERR_W = np.array([1.0, 0.6, 0.6, 0.4, 0.0, 0.6, 0.4, 0.0,
                        1.0, 0.6, 0.4, 1.0, 0.6, 0.4], dtype=np.float64)

CONTROL_DT = 1.0 / 20.0  # 20Hz 决策
TAR_OBS_STEPS = (1, 2, 3)

OBS_DIM = 464
CHAR_OBS_DIM = 140


class DeepMimic28Simulator(Humanoid28Simulator):
    """MimicKit 语义的单 clip 跟踪仿真器（robot_a 跟踪，robot_b 陪衬）。"""

    def __init__(self, motion: str = 'humanoid_spinkick',
                 motion_dir: Optional[str] = None,
                 rand_reset: bool = True,
                 episode_steps: int = 200,  # 10s @20Hz，对齐上游 episode_length
                 **kwargs):
        super().__init__(**kwargs)
        self.ref: RefMotion = load_ref_motion(motion, motion_dir)
        self.rand_reset = rand_reset
        self.episode_steps = int(episode_steps)

        # ---- 查表：CONTROLLED_JOINTS 顺序的 hinge → 组/轴 ----
        self._hl = hinge_layout()
        # 组 list 索引（KIN_JOINTS 顺序 14）
        self._group_of_hinge = [KIN_JOINTS.index(g) for _, g, _, _ in self._hl]
        # 每 hinge 的 (aidx, is_sph)
        self._hinge_info = [(a, s) for _, _, a, s in self._hl]

        # dof 误差权重：按组权重广播到 28 hinge（CONTROLLED_JOINTS 顺序）
        self._dof_err_w = np.zeros(28)
        for i, (hinge, group, aidx, sph) in enumerate(self._hl):
            self._dof_err_w[i] = JOINT_ERR_W[KIN_JOINTS.index(group)]

        # robot_a 全部 body（world 除外），pose 偏差终止用（对齐上游 body_pos 全体）
        self._all_body_ids = []
        for b in range(1, self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ''
            if nm.endswith('_a'):
                self._all_body_ids.append(b)

        # key body id（robot_a 后缀）
        self._key_body_ids = []
        for bn in KEY_BODIES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn + '_a')
            assert bid >= 0, f'missing body {bn}_a'
            self._key_body_ids.append(bid)

        # 参考 FK 用的 scratch 数据（独立 MjData，避免污染主仿真状态）
        self._fk_data = mujoco.MjData(self.model)

        # action 归一化参数（MimicKit box normalizer：center/half-range）
        # 策略在归一化空间输出（std=0.05 亦为归一化单位），此处还原为弧度
        self._a_center = 0.5 * (self._action_high + self._action_low)
        self._a_half = 0.5 * (self._action_high - self._action_low)

        # 帧对齐加速：fps=60 时 1 控制步 = 3 帧，RSI 帧对齐后全部查询精确落帧
        self._fps = self.ref.fps
        self._frames_per_step = CONTROL_DT * self._fps  # 60fps→3.0
        self._frame_aligned = abs(self._frames_per_step - round(self._frames_per_step)) < 1e-9
        self._fk_frames: Optional[list] = None   # 全帧 FK 状态缓存（lazy）
        self._t0_frame = 0

        # 相位状态
        self._motion_time0 = 0.0   # RSI 采样的起始动作时间
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._ref_cache: Dict[int, Dict[str, np.ndarray]] = {}  # 控制步索引 → ref_state

    # ---------------------------------------------------------
    # 相位
    # ---------------------------------------------------------
    def motion_time(self, steps_ahead: int = 0) -> float:
        t = self._motion_time0 + (self._step_count + steps_ahead) * CONTROL_DT
        if self.ref.loop_mode == 1:  # WRAP
            t = t % self.ref.duration
        return min(t, self.ref.duration)

    # ---------------------------------------------------------
    # 参考数据（带 FK key positions）
    # ---------------------------------------------------------
    def _build_fk_frames(self) -> None:
        """一次性预计算全部帧的 FK 增强状态（key/body 世界位置）。"""
        self._fk_frames = []
        for f in range(self.ref.n_frames):
            self._fk_frames.append(self.ref_state(f / self._fps))
        # WRAP：时间取模后到帧 n-1 等于帧 0 的位姿但属合法查询，保留原样即可

    def _frame_of_step(self, step_idx: int) -> int:
        f = self._t0_frame + int(round(self._frames_per_step)) * step_idx
        if self.ref.loop_mode == 1:
            f = f % (self.ref.n_frames - 1)
        else:
            f = min(f, self.ref.n_frames - 1)
        return f

    def ref_state_at_step(self, step_idx: int) -> Dict[str, np.ndarray]:
        """按控制步索引取参考状态（memoized——reward 与 tar_obs 在同一时刻复用）。"""
        if step_idx not in self._ref_cache:
            if self._frame_aligned:
                if self._fk_frames is None:
                    self._build_fk_frames()
                self._ref_cache[step_idx] = self._fk_frames[self._frame_of_step(step_idx)]
            else:
                self._ref_cache[step_idx] = self.ref_state(self.motion_time_at_step(step_idx))
        return self._ref_cache[step_idx]

    def motion_time_at_step(self, step_idx: int) -> float:
        t = self._motion_time0 + step_idx * CONTROL_DT
        if self.ref.loop_mode == 1:
            t = t % self.ref.duration
        return min(t, self.ref.duration)

    def ref_state(self, t: float) -> Dict[str, np.ndarray]:
        st = self.ref.state_at(t)
        # FK：把 hinge_pos+root 落入 scratch，读 key body 世界位置
        cache = self._robot('robot_a')
        qp = cache['qpos_indices']
        rq = cache['root_qpos_adr']
        self._fk_data.qpos[:] = 0.0
        self._fk_data.qvel[:] = 0.0
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self._fk_data.qpos[self.model.jnt_qposadr[j] + 3] = 1.0
        self._fk_data.qpos[qp] = st['hinge_pos']
        self._fk_data.qpos[rq:rq + 3] = st['root_pos']
        qw = st['root_quat']
        self._fk_data.qpos[rq + 3:rq + 7] = [qw[3], qw[0], qw[1], qw[2]]
        # robot_b 挪远避免干扰（FK 只是运动学，接触不影响 xpos）
        mujoco.mj_forward(self.model, self._fk_data)
        st['key_pos'] = np.array([self._fk_data.xpos[b].copy() for b in self._key_body_ids])
        st['body_pos'] = np.array([self._fk_data.xpos[b].copy() for b in self._all_body_ids])
        return st

    # ---------------------------------------------------------
    # 角色自身状态（MimicKit kin 语义）
    # ---------------------------------------------------------
    def char_state(self) -> Dict[str, np.ndarray]:
        cache = self._robot('robot_a')
        qp = cache['qpos_indices']
        qv = cache['qvel_indices']
        rq = cache['root_qpos_adr']
        rqv = cache['root_qvel_adr']

        root_pos = self.data.qpos[rq:rq + 3].copy()
        qw = self.data.qpos[rq + 3:rq + 7].copy()  # wxyz
        root_quat = np.array([qw[1], qw[2], qw[3], qw[0]])  # xyzw
        root_vel = self.data.qvel[rqv:rqv + 3].copy()       # 世界系
        # qvel[3:6] 是 body-local 系（已数值验证）→ 转世界系对齐 MimicKit
        ang_local = self.data.qvel[rqv + 3:rqv + 6].copy()
        root_ang_vel = quat_rotate(root_quat, ang_local)

        hinge_pos = self.data.qpos[qp].copy()
        dof_vel = self.data.qvel[qv].copy()

        # 组 quat（14 组含 fixed 手=恒等）
        joint_quat = np.zeros((14, 4))
        joint_quat[:, 3] = 1.0
        # spherical 组：逐组组合
        for g in SPH_GROUPS:
            jidx = KIN_JOINTS.index(g)
            angs = {}
            for i, (hinge, group, aidx, sph) in enumerate(self._hl):
                if group == g:
                    angs[aidx] = hinge_pos[i]
            joint_quat[jidx] = compose_group_quat(
                (angs.get(0, 0.0), angs.get(1, 0.0), angs.get(2, 0.0)), 'sph')
        # hinge 关节
        for g, axis in HINGE_AXIS.items():
            jidx = KIN_JOINTS.index(g)
            for i, (hinge, group, aidx, sph) in enumerate(self._hl):
                if group == g:
                    joint_quat[jidx] = compose_group_quat(hinge_pos[i], ('hinge', axis))

        key_pos = np.array([self.data.xpos[b].copy() for b in self._key_body_ids])
        body_pos = np.array([self.data.xpos[b].copy() for b in self._all_body_ids])
        return dict(root_pos=root_pos, root_quat=root_quat, root_vel=root_vel,
                    root_ang_vel=root_ang_vel, joint_quat=joint_quat,
                    dof_vel=dof_vel, key_pos=key_pos, body_pos=body_pos, hinge_pos=hinge_pos)

    # ---------------------------------------------------------
    # reset：RSI
    # ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        if self.rand_reset:
            if self._frame_aligned:
                # 帧量化 RSI（D13）：起始帧均匀采样 → 后续所有查询精确落帧，
                # ref 查询变纯查表。上游为连续时间采样——方法影响可忽略。
                nf = self.ref.n_frames - 1 if self.ref.loop_mode == 1 else self.ref.n_frames
                self._t0_frame = int(self._rng.integers(0, nf))
                t0 = self._t0_frame / self._fps
            else:
                t0 = float(self._rng.uniform(0.0, self.ref.duration))
        else:
            self._t0_frame = 0
            t0 = 0.0
        if 'start_time' in options:
            t0 = float(options['start_time'])
            self._t0_frame = int(round(t0 * self._fps))
        self._motion_time0 = t0
        self._step_count = 0
        self._ref_cache = {}

        # RSI：参考状态落入 robot_a
        st = self.ref_state(t0)
        cache = self._robot('robot_a')
        rq = cache['root_qpos_adr']
        rqv = cache['root_qvel_adr']
        qp = cache['qpos_indices']
        qv = cache['qvel_indices']

        self.data.qpos[rq:rq + 3] = st['root_pos']
        qw = st['root_quat']
        self.data.qpos[rq + 3:rq + 7] = [qw[3], qw[0], qw[1], qw[2]]
        self.data.qpos[qp] = st['hinge_pos']

        # 速度：root_vel 是世界系 qvel[0:3]；角速度 expmap(world) ≈ qvel[3:6]
        # （与 humanoid21 CoreClip 不同——MimicKit 直接用世界系差分）
        self.data.qvel[rqv:rqv + 3] = st['root_vel']
        # ref 角速度是世界系（expmap 差分）→ 转 body-local 写入 qvel[3:6]
        ang_local = quat_rotate(quat_conj(st['root_quat']), st['root_ang_vel'])
        self.data.qvel[rqv + 3:rqv + 6] = ang_local
        self.data.qvel[qv] = st['dof_vel']

        # PD 目标 = 当前关节角（保持姿态起步）
        self._target_pos['robot_a'] = st['hinge_pos'].astype(np.float32).copy()

        mujoco.mj_forward(self.model, self.data)
        self._data_cache.clear()

    def physical_step(self) -> None:
        # 每 action step 25 物理子步由 runtime 驱动；相位在 action 步推进。
        # humanoid28 runtime 每次 step 调一次物理子步循环——在
        # set_action 时推进相位（每个控制步一次）。
        super().physical_step()

    def set_action(self, action: Dict[str, Optional[Any]]) -> None:
        # 策略输出为归一化动作：a_rad = a_norm * half + center
        # （MimicKit Normalizer.unnormalize，之后由父类 clip 到扩展限位）
        conv: Dict[str, Optional[Any]] = {}
        for rid in ('robot_a', 'robot_b'):
            a = action.get(rid)
            if a is None:
                conv[rid] = None
                continue
            arr = np.asarray(a.detach().cpu().numpy() if hasattr(a, 'detach') else a,
                             dtype=np.float32)
            conv[rid] = arr * self._a_half + self._a_center
        super().set_action(conv)
        self._step_count += 1
        for k in ('_dm_obs', '_char_state', '_dm_track'):
            self._data_cache.pop(k, None)

    # ---------------------------------------------------------
    # 观测（MimicKit 精确配方）
    # ---------------------------------------------------------
    def _char(self) -> Dict[str, np.ndarray]:
        if '_char_state' not in self._data_cache:
            self._data_cache['_char_state'] = self.char_state()
        return self._data_cache['_char_state']

    @staticmethod
    def _key_rel(key_pos: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
        return key_pos - root_pos[None, :]

    def _ensure_obs_norm(self):
        """惰性加载 obs 归一化统计（MimicKit normalizer 保真项）。"""
        if getattr(self, '_obs_norm_loaded', False):
            return
        self._obs_norm_loaded = True
        import os
        npz = os.path.join(os.path.dirname(__file__), 'data', 'obs_norm.npz')
        if os.path.exists(npz):
            d = np.load(npz)
            self._obs_norm_mean = d['mean'].astype(np.float32)
            self._obs_norm_std = d['std'].astype(np.float32)
        else:
            self._obs_norm_mean = None
            self._obs_norm_std = None

    def compute_obs_a(self) -> np.ndarray:
        self._ensure_obs_norm()
        ch = self._char()
        # ---- char_obs (140) ----
        root_h = ch['root_pos'][2:3]                                    # 1
        root_rot_tn = quat_to_tan_norm(ch['root_quat'])                 # 6
        jrot_tn = quat_to_tan_norm(ch['joint_quat']).reshape(-1)        # 84
        key_rel = self._key_rel(ch['key_pos'], ch['root_pos']).reshape(-1)  # 15
        char_obs = np.concatenate([
            root_h, root_rot_tn, ch['root_vel'], ch['root_ang_vel'],
            jrot_tn, ch['dof_vel'], key_rel])
        assert char_obs.shape[0] == CHAR_OBS_DIM, char_obs.shape

        # ---- tar_obs 3×108 ----
        tar_all = []
        for k in TAR_OBS_STEPS:
            ref = self.ref_state_at_step(self._step_count + k)
            # root_pos 偏移（相对角色 root；z 为参考绝对高度）
            rp = ref['root_pos'] - ch['root_pos']
            rp[2] = ref['root_pos'][2]
            rrot_tn = quat_to_tan_norm(ref['root_quat'])                # 6
            rj_tn = quat_to_tan_norm(ref['joint_quat']).reshape(-1)     # 84
            rkey = self._key_rel(ref['key_pos'], ref['root_pos']).reshape(-1)  # 15
            tar_all.append(np.concatenate([rp, rrot_tn, rj_tn, rkey]))
        tar_obs = np.concatenate(tar_all)                               # 324
        obs = np.concatenate([char_obs, tar_obs]).astype(np.float32)
        assert obs.shape[0] == OBS_DIM, obs.shape
        # MimicKit 观测归一化（上游 normalizer + clip±10 保真项）：
        # fixed stats 见 data/obs_norm.npz（生成脚本见 DESIGN.md §9）
        if self._obs_norm_mean is not None:
            obs = (obs - self._obs_norm_mean) / self._obs_norm_std
            np.clip(obs, -10.0, 10.0, out=obs)
        return obs

    def get_observation(self) -> Dict[str, Any]:
        if '_dm_obs' in self._data_cache:
            return self._data_cache['_dm_obs']
        obs_a = self.compute_obs_a()
        # robot_b：base 110 维战斗 obs（陪衬）
        obs_b = self._get_robot_view('robot_b', 'robot_a')['observation']
        out = {'robot_a': obs_a, 'robot_b': obs_b}
        self._data_cache['_dm_obs'] = out
        return out

    # ---------------------------------------------------------
    # reward / termination 数据（当前帧）
    # ---------------------------------------------------------
    def track_data(self) -> Dict[str, Any]:
        """插件用：当前帧角色状态 + 当前时间参考状态。"""
        ch = self._char()
        ref = self.ref_state_at_step(self._step_count)
        return {'char': ch, 'ref': ref,
                'time': self._motion_time0 + self._step_count * CONTROL_DT,
                'step': self._step_count}

    def get_static_data(self) -> Dict[str, Any]:
        out = super().get_static_data()
        out['dm_dof_err_w'] = self._dof_err_w.copy()
        out['dm_episode_steps'] = self.episode_steps
        out['dm_foot_body_ids'] = list(self._key_body_ids[3:5])
        out['dm_loop_wrap'] = bool(self.ref.loop_mode == 1)
        out['dm_duration'] = float(self.ref.duration)
        return out

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        """扩展标准接口：'dm_track' 字段暴露跟踪数据（插件经 accessor 读取）。"""
        if fields is not None and 'dm_track' in fields:
            rest = [f for f in fields if f != 'dm_track']
            out = super().get_derived_state(fields=rest) if rest else {}
            if '_dm_track' not in self._data_cache:
                self._data_cache['_dm_track'] = self.track_data()
            out['dm_track'] = self._data_cache['_dm_track']
            return out
        return super().get_derived_state(fields=fields)
