"""AMP 风格判别器的运动特征提取（M3）。

逐帧特征（67 维，按控制帧采样）：
  [0:21]   关节角（限位归一化到 [-1,1]，与 track 环境同约定）
  [21:42]  关节角速度 ×0.1 缩放（rad/s）
  [42]     根高度 z（m）
  [43:46]  根线速度 → 根 yaw 对齐系（m/s）
  [46:49]  根角速度 → 根 yaw 对齐系（rad/s）
  [49:67]  关键点相对骨盆位置 → 根 yaw 对齐系（6×3: torso/head/双手/双足）

状态转换 = concat(f_t, f_{t+1})，134 维，作为判别器输入。

不变性设计（与世界系 xy/yaw 无关，与 OBSERVATION_zh.md 同一哲学）：
- 水平位移/yaw 旋转均被消去——风格判别只关心"怎么动"，不关心"在哪动"；
- 离线路径（npz 参考库）与在线路径（simulator derived state）共用同一
  extract()，避免两套语义漂移。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

CLIPS_DIR = Path('/data1/dev/combatbench_mocap/clips')
REF_STRIDE = 6          # 120Hz → 20Hz 控制帧
VEL_SCALE = 0.1         # 关节角速度缩放（~10 rad/s 满量程）

AMP_KP_NAMES = ('torso', 'head', 'hand_left', 'hand_right',
                'foot_left', 'foot_right')

_MID = np.array([0.5 * (joint_limits(n)[0] + joint_limits(n)[1])
                 for n in CONTROLLED_JOINTS])
_HALF = np.array([0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
                  for n in CONTROLLED_JOINTS])

FEAT_DIM = 67
TRANS_DIM = 2 * FEAT_DIM


def _yaw_rot(quat_wxyz: np.ndarray) -> R:
    """世界系四元数 → 只保留 yaw 分量的旋转（用于消去朝向）。"""
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    yaw = r.as_euler('xyz')[2]
    return R.from_euler('z', yaw)


def extract(joint_angles: np.ndarray,
            joint_velocities: np.ndarray,
            root_pos: np.ndarray,
            root_quat_wxyz: np.ndarray,
            root_linvel: np.ndarray,
            root_angvel: np.ndarray,
            keypoints: np.ndarray,
            keypoint_names: List[str]) -> np.ndarray:
    """单帧特征提取。所有输入为世界系物理单位，输出 (67,) float32。

    keypoints: (K,3) 世界系；keypoint_names 含 'pelvis' 与 AMP_KP_NAMES。
    """
    qn = (np.asarray(joint_angles, dtype=np.float64) - _MID) / _HALF
    qv = np.asarray(joint_velocities, dtype=np.float64) * VEL_SCALE

    Ry = _yaw_rot(np.asarray(root_quat_wxyz))
    pelvis = np.asarray(keypoints)[list(keypoint_names).index('pelvis')]
    kp_rel = []
    for n in AMP_KP_NAMES:
        p = np.asarray(keypoints)[list(keypoint_names).index(n)] - pelvis
        kp_rel.append(Ry.inv().as_matrix() @ p)
    kp_rel = np.concatenate(kp_rel)

    return np.concatenate([
        np.clip(qn, -1.5, 1.5),
        np.clip(qv, -1.5, 1.5),
        [root_pos[2]],
        Ry.inv().as_matrix() @ np.asarray(root_linvel),
        Ry.inv().as_matrix() @ np.asarray(root_angvel),
        kp_rel,
    ]).astype(np.float32)


def clip_frame_features(npz_path: Path,
                        stride: int = REF_STRIDE) -> np.ndarray:
    """整段 npz → (T_ctrl, 67) 特征序列（lead_stable 之前跳过）。"""
    d = np.load(npz_path)
    t0 = int(d['lead_stable_frame']) if 'lead_stable_frame' in d else 0
    sl = slice(t0, None, stride)
    kn = [str(x) for x in d['keypoint_names']]
    mask = d['valid_mask'][sl] if 'valid_mask' in d else None
    feats = np.stack([
        extract(d['joint_angles'][sl][i], d['joint_velocities'][sl][i],
                d['root_pos'][sl][i], d['root_quat'][sl][i],
                d['root_linvel'][sl][i], d['root_angvel'][sl][i],
                d['keypoints'][sl][i], kn)
        for i in range(len(d['root_pos'][sl]))])
    if mask is not None:
        feats = feats[mask.astype(bool)]
    return feats


def transitions(feats: np.ndarray, gap: int = 3) -> np.ndarray:
    """(T,67) → (T-gap,134) 转换对，默认间隔 3 控制帧（150ms）。

    单步（50ms）真实位移增量极小（归一化后 ~0.01），与"冻结"负例
    （增量恰为 0）之间无分辨余量；150ms 窗内真实增量达 0.1-0.3
    归一化单位，冻结/倒放/变速才有可学信号。
    """
    return np.concatenate([feats[:-gap], feats[gap:]], axis=1)


TRANS_GAP = 3


def library_transitions(clip_ids: Optional[List[str]] = None,
                        clips_dir: Path = CLIPS_DIR) -> np.ndarray:
    """整个参考库的正例转换矩阵。"""
    out = []
    for p in sorted(clips_dir.glob('cmu_*.npz')):
        cid = p.stem
        if cid.endswith('_src'):            # 62-dof 源骨架数据，非 humanoid21
            continue
        if clip_ids is not None and cid not in clip_ids:
            continue
        f = clip_frame_features(p)
        if len(f) >= 2:
            out.append(transitions(f))
    return np.concatenate(out, axis=0) if out else np.zeros((0, TRANS_DIM))
