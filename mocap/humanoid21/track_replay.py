"""M2: 参考运动的物理跟踪回放（state-non-overwriting）。

把一个 retargeted clip（cmu_XX_YY.npz, 120Hz, joint space = humanoid21）
作为 open-loop PD 目标在真实 battle_circular_v2 模型里回放：

  - t=0 用 set_core_state 一次性初始化（根位置/姿态/速度 + 关节位置/速度），
    之后绝不再覆写物理状态 —— 这是与 compare_render.py 的本质区别。
  - 每个控制步（20Hz = 25 个物理步 @ 2ms）把参考帧的关节角映射为
    归一化 action = (q_ref - mid)/半量程，喂给固定 KP/KD PD。
  - robot_b 保持 standing 停在场地远端，不参与跟踪。

记录（按控制帧）：
  actual/ref joint pos, root pos/quat, keypoint body 位置(foot/torso/head),
  ctrl 饱和度, 足接触高度 —— 输出 npz + metrics json, 可选 mp4。

摔倒判据（用于 tracked_until，不会因此提前覆写状态）：
  torso z < FALL_TORSO_Z 或 head z < FALL_HEAD_Z 或躯干朝上向量为负。
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from envs.humanoid21.simulator import Humanoid21Simulator

CLIPS_DIR = Path('/data1/dev/combatbench_mocap/clips')
OUT_DIR = Path('/data1/dev/combatbench_mocap/track')

PHY_DT = 0.002                  # simulator DT
PHY_STEPS_PER_ACTION = 25       # 20Hz control
REF_FPS = 120.0
REF_STRIDE = 6                  # 120Hz / 20Hz

FALL_TORSO_Z = 0.55             # 站立 torso z ≈ 1.28
FALL_HEAD_Z = 0.45              # 站立 head z ≈ 1.7


def load_clip(clip_id: str):
    path = CLIPS_DIR / f'{clip_id}.npz'
    if not path.exists():
        path = CLIPS_DIR / f'cmu_{clip_id}.npz'
    if not path.exists():
        raise FileNotFoundError(f'clip npz not found: {clip_id}')
    return np.load(path, allow_pickle=True), path


def body_ids(sim, suffix='_a'):
    m = sim.model
    ids = {}
    for name in ['torso', 'head', 'foot_left', 'foot_right', 'hand_left', 'hand_right']:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{name}{suffix}')
        if bid < 0:
            raise ValueError(f'body {name}{suffix} not found')
        ids[name] = bid
    return ids


def quat_wxyz_to_R(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])


def run_track(clip_id: str, robot_id: str = 'robot_a', center_xy: bool = True,
              keep_going_after_fall: bool = False, video: bool = False,
              max_seconds: float = None):
    z, path = load_clip(clip_id)
    joint_names = [str(n) for n in z['joint_names']]
    q_ref = z['joint_angles'].astype(np.float64)          # (T,21) rad
    root_pos = z['root_pos'].astype(np.float64).copy()    # (T,3)
    root_quat = z['root_quat'].astype(np.float64)         # (T,4) wxyz
    root_linvel = z['root_linvel'].astype(np.float64)
    root_angvel = z['root_angvel'].astype(np.float64)
    qvel_ref = z['joint_velocities'].astype(np.float64)

    T = q_ref.shape[0]
    if max_seconds is not None:
        T = min(T, int(max_seconds * REF_FPS))
        q_ref = q_ref[:T]; root_pos = root_pos[:T]; root_quat = root_quat[:T]
        root_linvel = root_linvel[:T]; root_angvel = root_angvel[:T]
        qvel_ref = qvel_ref[:T]

    # 水平平移：把整段轨迹的 xy bbox 中心挪到场地原点，记录偏移量（溯源用）。
    xy_offset = np.zeros(2)
    if center_xy:
        lo = root_pos[:, :2].min(axis=0)
        hi = root_pos[:, :2].max(axis=0)
        xy_offset = -0.5 * (lo + hi)
        root_pos[:, :2] += xy_offset

    suffix = {'robot_a': '_a', 'robot_b': '_b'}[robot_id]
    other = 'robot_b' if robot_id == 'robot_a' else 'robot_a'

    sim = Humanoid21Simulator(initial_distance=6.0)
    sim.reset()

    cache = sim._robot(robot_id)
    jnt_ranges = np.asarray(cache['jnt_ranges'], dtype=np.float64)
    jnt_lo, jnt_hi = jnt_ranges[:, 0], jnt_ranges[:, 1]
    mid = 0.5 * (jnt_lo + jnt_hi)
    half = 0.5 * (jnt_hi - jnt_lo)

    # CONTROLLED_JOINTS 顺序应与 npz joint_names 一致；做一次显式校验。
    ctrl_names = [n.replace(suffix, '') if n.endswith(suffix) else n
                  for n in cache['controlled_joint_names']]
    if ctrl_names != joint_names:
        # 不同序时按名字重排参考角。
        idx = [joint_names.index(n) for n in ctrl_names]
        q_ref = q_ref[:, idx]
        qvel_ref = qvel_ref[:, idx]

    # ---- t=0 一次性状态初始化（此后不再覆写）----
    R0 = quat_wxyz_to_R(root_quat[0])
    state = {
        robot_id: {
            'root_pos': root_pos[0],
            'root_rot': root_quat[0],
            'root_vel_local': R0.inv().as_matrix() @ root_linvel[0],
            'root_angular_vel_local': R0.inv().as_matrix() @ root_angvel[0],
            'joint_pos_norm': (q_ref[0] - mid) / half,
            'joint_vel_norm': qvel_ref[0] / half,
        }
    }
    sim.set_core_state(state)

    bids = body_ids(sim, suffix)
    actuator_ids = np.asarray(cache['actuator_ids'], dtype=np.int64)
    ctrlrange = np.asarray(sim.model.actuator_ctrlrange[actuator_ids], dtype=np.float64)

    n_ctrl = int(np.ceil(T / REF_STRIDE))
    rec = {k: np.zeros((n_ctrl, d), dtype=np.float64) for k, d in {
        'ref_q': 21, 'act_q': 21, 'ref_root': 3, 'act_root': 3,
        'ref_quat': 4, 'act_quat': 4,
    }.items()}
    rec_kp_ref = np.zeros((n_ctrl, 6, 3))
    rec_kp_act = np.zeros((n_ctrl, 6, 3))
    rec_ctrl_sat = np.zeros((n_ctrl, 21))
    rec_times = np.zeros(n_ctrl)

    kp_names = list(bids.keys())
    ref_kp_idx = None
    if 'keypoints' in z.files:
        kp_src_names = [str(n) for n in z['keypoint_names']]
        # npz keypoint names: pelvis torso head thigh_* shin_* foot_* upper_arm_* lower_arm_* hand_*
        kp_map = {'torso': 'torso', 'head': 'head', 'foot_left': 'foot_left',
                  'foot_right': 'foot_right', 'hand_left': 'hand_left', 'hand_right': 'hand_right'}
        ref_kp_idx = [kp_src_names.index(kp_map[n]) if kp_map[n] in kp_src_names else None
                      for n in kp_names]
        ref_kps = z['keypoints'].astype(np.float64).copy()
        ref_kps[:, :, 0] += xy_offset[0]
        ref_kps[:, :, 1] += xy_offset[1]
    else:
        ref_kps = None

    frames = []
    tracked_until = n_ctrl - 1
    fell_at = None

    for k in range(n_ctrl):
        f = min(k * REF_STRIDE, T - 1)
        action_norm = np.clip((q_ref[f] - mid) / half, -1.0, 1.0)
        sim.set_action({robot_id: action_norm.astype(np.float32), other: None})
        for _ in range(PHY_STEPS_PER_ACTION):
            sim.physical_step()

        d = sim.data
        rec['ref_q'][k] = q_ref[f]
        rec['act_q'][k] = d.qpos[cache['qpos_indices']]
        rec['ref_root'][k] = root_pos[f]
        rec['act_root'][k] = d.qpos[cache['root_qpos_adr']:cache['root_qpos_adr'] + 3]
        rec['ref_quat'][k] = root_quat[f]
        rec['act_quat'][k] = d.qpos[cache['root_qpos_adr'] + 3:cache['root_qpos_adr'] + 7]
        for j, name in enumerate(kp_names):
            rec_kp_act[k, j] = d.xpos[bids[name]]
            if ref_kps is not None and ref_kp_idx[j] is not None:
                rec_kp_ref[k, j] = ref_kps[f, ref_kp_idx[j]]
        ctrl = np.asarray(d.ctrl[actuator_ids], dtype=np.float64)
        lo, hi = ctrlrange[:, 0], ctrlrange[:, 1]
        rng = np.where(hi > lo, hi - lo, 1.0)
        rec_ctrl_sat[k] = np.clip((ctrl - lo) / rng, 0, 1)  # 0=下限饱和 1=上限饱和
        rec_times[k] = k * PHY_STEPS_PER_ACTION * PHY_DT

        if video:
            img = sim.get_broadcastview_image()
            if img is not None:
                frames.append(img)

        torso_z = d.xpos[bids['torso']][2]
        head_z = d.xpos[bids['head']][2]
        if fell_at is None and (torso_z < FALL_TORSO_Z or head_z < FALL_HEAD_Z):
            fell_at = rec_times[k]
            tracked_until = k
            if not keep_going_after_fall:
                break

    n_done = k + 1
    for key in rec:
        rec[key] = rec[key][:n_done]
    rec_kp_ref = rec_kp_ref[:n_done]; rec_kp_act = rec_kp_act[:n_done]
    rec_ctrl_sat = rec_ctrl_sat[:n_done]; rec_times = rec_times[:n_done]

    # ---- 指标汇总 ----
    n_tr = min(tracked_until + 1, n_done)  # 摔倒前（含摔倒帧）的跟踪窗口
    q_err = np.degrees(np.abs(rec['act_q'][:n_tr] - rec['ref_q'][:n_tr]))
    root_err = np.linalg.norm(rec['act_root'][:n_tr] - rec['ref_root'][:n_tr], axis=1)
    kp_err = np.linalg.norm(rec_kp_act[:n_tr] - rec_kp_ref[:n_tr], axis=2)  # (n,6)
    # 根姿态误差角
    rot_err = np.array([
        (quat_wxyz_to_R(rec['act_quat'][i]).inv() * quat_wxyz_to_R(rec['ref_quat'][i])).magnitude()
        for i in range(n_tr)])
    rot_err = np.degrees(rot_err)

    sat = (rec_ctrl_sat[:n_tr] < 0.02) | (rec_ctrl_sat[:n_tr] > 0.98)

    metrics = {
        'clip_id': clip_id, 'robot_id': robot_id,
        'npz': str(path), 'xy_offset': xy_offset.tolist(),
        'n_ctrl_frames_total': n_ctrl, 'n_ctrl_frames_run': int(n_done),
        'duration_s': float(rec_times[-1]),
        'fell_at_s': fell_at, 'fell': fell_at is not None,
        'tracked_until_s': float(rec_times[n_tr - 1]),
        'joint_err_deg': {
            'rmse': float(np.sqrt((q_err ** 2).mean())),
            'p95': float(np.percentile(q_err, 95)),
            'max': float(q_err.max()),
        },
        'root_pos_err_m': {'mean': float(root_err.mean()), 'max': float(root_err.max())},
        'root_rot_err_deg': {'mean': float(rot_err.mean()), 'max': float(rot_err.max())},
        'keypoint_err_m': {
            n: {'mean': float(kp_err[:, j].mean()), 'max': float(kp_err[:, j].max())}
            for j, n in enumerate(kp_names)},
        'ctrl_sat_frac': float(sat.mean()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_out = OUT_DIR / f'{clip_id}_track.npz'
    np.savez_compressed(
        npz_out,
        times=rec_times, ref_q=rec['ref_q'], act_q=rec['act_q'],
        ref_root=rec['ref_root'], act_root=rec['act_root'],
        ref_quat=rec['ref_quat'], act_quat=rec['act_quat'],
        kp_names=np.array(kp_names), kp_ref=rec_kp_ref, kp_act=rec_kp_act,
        ctrl_sat=rec_ctrl_sat,
        joint_names=np.array(ctrl_names))
    json_out = OUT_DIR / f'{clip_id}_track.json'
    json_out.write_text(json.dumps(metrics, indent=1))

    if video and frames:
        import imageio.v2 as imageio
        vid_out = OUT_DIR / f'{clip_id}_track.mp4'
        imageio.mimsave(str(vid_out), frames, fps=int(1 / (PHY_STEPS_PER_ACTION * PHY_DT)))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip_id')
    ap.add_argument('--robot', default='robot_a')
    ap.add_argument('--no-center', action='store_true')
    ap.add_argument('--keep-going', action='store_true')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=None)
    a = ap.parse_args()
    m = run_track(a.clip_id, robot_id=a.robot, center_xy=not a.no_center,
                  keep_going_after_fall=a.keep_going, video=a.video,
                  max_seconds=a.max_seconds)
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
