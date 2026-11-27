"""DeepMimic28 奖励观测器 + 终止插件——MimicKit compute_reward/compute_done 精确移植。

奖励（单通道加权和，对齐 MimicKit deepmimic_env.compute_reward）：
  r = 0.5*r_pose + 0.1*r_vel + 0.15*r_root_pose + 0.1*r_root_vel + 0.15*r_key_pos
  r_pose      = exp(-0.25 * Σ_g w_g * angle(q_g tar⁻¹ ⊗ q_g char)²)   # 14 组四元数误差
  r_vel       = exp(-0.01 * Σ_d w_d * (dof_vel diff)²)                # 28 hinge
  r_root_pose = exp(-5.0 * (|Δpos|² + 0.1 * angle(rot diff)²))
  r_root_vel  = exp(-1.0 * (|Δvel|² + 0.1 * |Δangvel|²))
  r_key_pos   = exp(-10.0 * mean(|Δkey_pos root-rel|²))               # 5 key body

多通道变体（实验用）：五通道各自独立，由框架 multi-critic 分别估值。

终止（compute_done）：
  - 接触摔倒：非 whitelist 身体（脚之外）接触力 → FAIL
  - pose 偏差：任一键位身体世界位置偏差 > 1.0m（含 root，global_obs=True）→ FAIL
  - clip 结束（motion_end）→ SUCC；超时 → TIMEOUT
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework.observer_plugin import BaseObserverPlugin
from envs.framework.plugin import BasePlugin

from .mathutil import quat_diff_angle, quat_mul, quat_conj
from .sim import DeepMimic28Simulator, JOINT_ERR_W

# MimicKit deepmimic_humanoid_env.yaml 奖励配方
W_POSE, W_VEL, W_ROOT_POSE, W_ROOT_VEL, W_KEY = 0.5, 0.1, 0.15, 0.1, 0.15
S_POSE, S_VEL, S_ROOT_POSE, S_ROOT_VEL, S_KEY = 0.25, 0.01, 5.0, 1.0, 10.0

POSE_TERMINATION_DIST = 1.0
# 接触白名单：双脚（MimicKit contact_bodies = feet）
CONTACT_WHITELIST = ('right_foot_a', 'left_foot_a')

CHANNELS = ('pose', 'vel', 'root_pose', 'root_vel', 'key_pos')


def compute_reward_components(track: Dict[str, Any], dof_err_w: np.ndarray) -> Dict[str, float]:
    """MimicKit compute_reward 逐行移植。输入为 sim.track_data()。"""
    ch, ref = track['char'], track['ref']

    # --- pose: 14 组四元数误差（加权平方和）---
    diff_ang = quat_diff_angle(ref['joint_quat'], ch['joint_quat'])  # (14,)
    pose_err = float(np.sum(JOINT_ERR_W * diff_ang ** 2))
    r_pose = np.exp(-S_POSE * pose_err)

    # --- vel: 28 hinge 角速度误差 ---
    dvel = ref['dof_vel'] - ch['dof_vel']
    vel_err = float(np.sum(dof_err_w * dvel ** 2))
    r_vel = np.exp(-S_VEL * vel_err)

    # --- root pose: pos² + 0.1*rot² ---
    dpos = ref['root_pos'] - ch['root_pos']
    rrot_ang = float(quat_diff_angle(ref['root_quat'][None, :], ch['root_quat'][None, :])[0])
    root_pose_err = float(dpos @ dpos) + 0.1 * rrot_ang ** 2
    r_root_pose = np.exp(-S_ROOT_POSE * root_pose_err)

    # --- root vel ---
    dv = ref['root_vel'] - ch['root_vel']
    dav = ref['root_ang_vel'] - ch['root_ang_vel']
    root_vel_err = float(dv @ dv) + 0.1 * float(dav @ dav)
    r_root_vel = np.exp(-S_ROOT_VEL * root_vel_err)

    # --- key pos: root-relative ---
    key_rel = ch['key_pos'] - ch['root_pos'][None, :]
    ref_key_rel = ref['key_pos'] - ref['root_pos'][None, :]
    key_err = float(np.sum((ref_key_rel - key_rel) ** 2))  # 上游为全键位求和
    r_key = np.exp(-S_KEY * key_err)

    return dict(pose=float(r_pose), vel=float(r_vel), root_pose=float(r_root_pose),
                root_vel=float(r_root_vel), key_pos=float(r_key),
                _pose_err=pose_err, _vel_err=vel_err,
                _root_pose_err=root_pose_err, _root_vel_err=root_vel_err,
                _key_err=key_err)


class DeepMimic28RewardObserver(BaseObserverPlugin):
    """跟踪奖励观测器（dict 输出）。

    字段：
      reward      — MimicKit 加权和（单通道实验用）
      r_pose/r_vel/r_root_pose/r_root_vel/r_key_pos — 五通道分量
      （多通道实验各通道独立 critic；actor_weight 由实验侧按权重给）
      key_err/pose_err/root_dev — 诊断指标
    """

    def __init__(self, agent_id: str = 'robot_a', **_):
        self.agent_id = agent_id
        self._output: Dict[str, float] = {'reward': 0.0}

    @property
    def name(self) -> str:
        return 'deepmimic28_reward'

    def on_pre_episode(self, ctx) -> None:
        self._output = {'reward': 0.0}

    def on_post_action_step(self, ctx) -> None:
        static = ctx.accessor.get_static_data()
        track = ctx.accessor.get_derived_state(fields=['dm_track'])['dm_track']
        comp = compute_reward_components(track, static['dm_dof_err_w'])
        total = (W_POSE * comp['pose'] + W_VEL * comp['vel'] +
                 W_ROOT_POSE * comp['root_pose'] + W_ROOT_VEL * comp['root_vel'] +
                 W_KEY * comp['key_pos'])
        self._output = {
            'reward': float(total),
            'r_pose': comp['pose'], 'r_vel': comp['vel'],
            'r_root_pose': comp['root_pose'], 'r_root_vel': comp['root_vel'],
            'r_key_pos': comp['key_pos'],
            'pose_err': comp['_pose_err'], 'vel_err': comp['_vel_err'],
            'key_err': comp['_key_err'],
            'root_dev': float(np.linalg.norm(
                track['char']['root_pos'] - track['ref']['root_pos'])),
            'phase': float(track['step'] / max(static['dm_episode_steps'], 1)),
        }

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)


class DeepMimic28TerminationPlugin(BasePlugin):
    """MimicKit compute_done：接触摔倒 + pose 偏差 + clip 结束。

    - contact_fall: 非白名单身体（双脚之外）受力 → track_fall
    - pose_dev: 5 key body + root 的世界系位置偏差 > 1m → track_deviation
      （global_obs=True ⇒ track_root=True，root 参与判定）
    - clip_end: 动作播完（WRAP clip 不终止）→ track_clip_end
    """

    def __init__(self, contact_fall: bool = True, pose_dev: bool = True):
        self._contact_fall = contact_fall
        self._pose_dev = pose_dev

    @property
    def name(self) -> str:
        return 'deepmimic28_termination'

    @property
    def priority(self) -> int:
        return 10

    def on_post_action_step(self, ctx) -> None:
        if ctx.episode_step <= 1:
            return  # RSI 落位帧不判（接触力未稳定）

        # --- pose 偏差（键位身体世界系，含 root）---
        static = ctx.accessor.get_static_data()
        track = ctx.accessor.get_derived_state(fields=['dm_track'])['dm_track']
        if self._pose_dev:
            ch, ref = track['char'], track['ref']
            # 上游：全部身体世界系位置误差取 max（含 root，global+track_root）
            key_dev = float(np.linalg.norm(ch['body_pos'] - ref['body_pos'], axis=-1).max())
            root_dev = float(np.linalg.norm(ch['root_pos'] - ref['root_pos']))
            ctx.metrics['dm_key_dev'] = float(key_dev)
            ctx.metrics['dm_root_dev'] = root_dev
            if key_dev > POSE_TERMINATION_DIST or root_dev > POSE_TERMINATION_DIST:
                ctx.metrics['dm_done_reason'] = 'deviation'
                ctx.request_termination('track_deviation')
                return

        # --- 接触摔倒：robot_a 非白名单身体与外部（地面/robot_b）接触 ---
        if self._contact_fall:
            contacts = ctx.accessor.get_derived_state(fields=['contacts'])['contacts']
            hit = False
            ncon = int(contacts.get('ncon', 0))
            if ncon > 0:
                body1 = np.asarray(contacts['body1'])
                body2 = np.asarray(contacts['body2'])
                aff1 = np.asarray(contacts['aff1'])
                aff2 = np.asarray(contacts['aff2'])
                fmag = np.asarray(contacts['force_mag'])
                # robot_a 侧 body id；whitelist 仅双脚
                wl = set(static['dm_foot_body_ids'])  # right_foot_a, left_foot_a
                for i in range(ncon):
                    if fmag[i] < 0.1:  # 上游阈值 0.1N
                        continue
                    a1, a2 = int(aff1[i]), int(aff2[i])
                    if a1 == 1 and a2 == 1:
                        continue  # robot_a 自碰撞（Isaac 语义不计）
                    if a1 == 1 and int(body1[i]) not in wl:
                        hit = True
                        break
                    if a2 == 1 and int(body2[i]) not in wl:
                        hit = True
                        break
            if hit:
                ctx.metrics['dm_done_reason'] = 'contact'
                ctx.request_termination('track_fall')
                return

        # --- clip 结束 ---
        if not static.get('dm_loop_wrap', True):  # 非 WRAP
            if track['time'] >= static['dm_duration']:
                ctx.metrics['dm_done_reason'] = 'clip_end'
                ctx.request_termination('track_clip_end')
