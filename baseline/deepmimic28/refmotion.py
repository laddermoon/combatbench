"""MimicKit 动作数据 → humanoid28 参考动作库。

数据链路（严格对齐 MimicKit motion_lib 语义）：

    pkl frame (34d) = root_pos(3) + root_expmap(3)
        + abdomen(3) + neck(3) + rshoulder(3) + relbow(1)
        + lshoulder(3) + lelbow(1) + rhip(3) + rknee(1) + rankle(3)
        + lhip(3) + lknee(1) + lankle(3)

    3 维关节组 = spherical 旋转的指数映射(expmap)；1 维 = hinge 角度。
    关节组顺序 = XML 深度优先（与 CONTROLLED_JOINTS 的组序一致）。

内部表示：
    - root_pos (T,3)、root_quat (T,4) xyzw（scipy 约定）
    - joint_quat (T,14,4) xyzw —— 每个关节组（含两个 fixed 手关节，
      恒等旋转）的净旋转四元数，与 MimicKit kin_char_model 语义一致
    - hinge_pos (T,28) —— 破解到 MuJoCo 串联 hinge 的角度，
      分解约定 q = qx⊗qy⊗qz（已对 MuJoCo FK 数值验证）
    - 速度：root_vel(world)、root_ang_vel(expmap 差分，与上游一致)、
      dof_vel（28 维 hinge 角速度，有限差分）

时间索引：帧率 fps（spinkick=60），支持非整数时间线性插值
（位置 lerp、旋转 slerp），与上游 calc_motion_frame 一致。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# 帧内关节组布局（MimicKit README 记录的确切顺序）
# (name, n_dims, is_spherical)
FRAME_JOINTS = [
    ('abdomen', 3, True),
    ('neck', 3, True),
    ('right_shoulder', 3, True),
    ('right_elbow', 1, False),
    ('left_shoulder', 3, True),
    ('left_elbow', 1, False),
    ('right_hip', 3, True),
    ('right_knee', 1, False),
    ('right_ankle', 3, True),
    ('left_hip', 3, True),
    ('left_knee', 1, False),
    ('left_ankle', 3, True),
]

# kin_char_model 的 14 个非根关节（含 fixed 手关节，权重 0），
# 顺序 = XML 深度优先。joint_err_w 按此顺序。
KIN_JOINTS = [
    'abdomen', 'neck',
    'right_shoulder', 'right_elbow', 'right_hand',
    'left_shoulder', 'left_elbow', 'left_hand',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
]

# humanoid28 CONTROLLED_JOINTS 的 28 个 hinge 按组归属：
# abdomen_x/y/z → abdomen 组（顺序 x,y,z），elbow 单 hinge 等。
GROUP_HINGES = {
    'abdomen': ['abdomen_x', 'abdomen_y', 'abdomen_z'],
    'neck': ['neck_x', 'neck_y', 'neck_z'],
    'right_shoulder': ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
    'right_elbow': ['right_elbow'],
    'left_shoulder': ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
    'left_elbow': ['left_elbow'],
    'right_hip': ['right_hip_x', 'right_hip_y', 'right_hip_z'],
    'right_knee': ['right_knee'],
    'right_ankle': ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
    'left_hip': ['left_hip_x', 'left_hip_y', 'left_hip_z'],
    'left_knee': ['left_knee'],
    'left_ankle': ['left_ankle_x', 'left_ankle_y', 'left_ankle_z'],
}

# CONTROLLED_JOINTS 顺序（与 meta.py 一致）的 hinge → (group, axis_idx)
def _hinge_index():
    from envs.humanoid28.meta import Humanoid28Meta
    out = []
    for group, n, sph in FRAME_JOINTS:
        for i, hn in enumerate(GROUP_HINGES[group]):
            out.append((hn, group, i, sph))
    order = {n: i for i, n in enumerate(Humanoid28Meta.CONTROLLED_JOINTS)}
    out.sort(key=lambda t: order[t[0]])
    return out  # 按 CONTROLLED_JOINTS 顺序的 (hinge, group, axis_idx, is_sph)

HINGE_LAYOUT = None  # lazy


def hinge_layout():
    global HINGE_LAYOUT
    if HINGE_LAYOUT is None:
        HINGE_LAYOUT = _hinge_index()
    return HINGE_LAYOUT


def expmap_to_quat_xyzw(e: np.ndarray) -> np.ndarray:
    """expmap (...,3) → quat xyzw。零向量 → 恒等。"""
    angle = np.linalg.norm(e, axis=-1, keepdims=True)  # (...,1)
    axis = np.divide(e, np.maximum(angle, 1e-9))
    half = 0.5 * angle[..., 0]
    w = np.cos(half)
    xyz = axis * np.sin(half)[..., None]
    return np.concatenate([xyz, w[..., None]], axis=-1)


def quat_xyzw_to_expmap(q: np.ndarray) -> np.ndarray:
    """quat xyzw (...,4) → expmap。先翻转到 w≥0 半球（quat_pos 语义）。"""
    q = q.copy()
    neg = q[..., 3] < 0
    q[neg] = -q[neg]
    xyz = q[..., :3]
    length = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, q[..., 3:4])  # (...,1)
    scale = np.where(length > 1e-5, angle / np.maximum(length, 1e-9), 0.0)
    return xyz * scale


def decompose_sph_to_hinges(quat_xyzw: np.ndarray) -> np.ndarray:
    """spherical 组净旋转 quat(xyzw) → 串联 hinge 角度 (x,y,z)。

    约定 q = qx⊗qy⊗qz（MuJoCo FK 数值验证）。scipy 中对应**外旋** 'XYZ'
    （数值验证 roundtrip 误差=0；内旋 'xyz' 是另一种组合，勿混淆）。
    """
    r = R.from_quat(quat_xyzw)
    return r.as_euler('XYZ')  # (3,) — (x角, y角, z角)


def compose_hinges_to_quat(ax: float, ay: float, az: float) -> np.ndarray:
    """串联 hinge 角度 → 组净旋转 quat xyzw。q = qx⊗qy⊗qz。"""
    r = R.from_rotvec([ax, 0, 0]) * R.from_rotvec([0, ay, 0]) * R.from_rotvec([0, 0, az])
    return r.as_quat()


class RefMotion:
    """单条 MimicKit 动作 clip 的参考库（numpy 版 motion_lib）。"""

    def __init__(self, pkl_path: str):
        d = pickle.load(open(pkl_path, 'rb'))
        self.fps = float(d['fps'])
        self.loop_mode = int(d['loop_mode'])  # 0=CLAMP 1=WRAP
        frames = np.asarray(d['frames'], dtype=np.float64)
        self.n_frames = frames.shape[0]
        assert frames.shape[1] == 34, f'frame dim {frames.shape} != 34'
        self.duration = (self.n_frames - 1) / self.fps

        # ---- 解析帧 ----
        self.root_pos = frames[:, 0:3].copy()                     # (T,3)
        root_exp = frames[:, 3:6]                                  # (T,3)
        self.root_quat = expmap_to_quat_xyzw(root_exp)             # (T,4) xyzw

        # 关节组 expmap/角度 → 组 quat；fixed 手关节 = 恒等
        T = self.n_frames
        self.joint_quat = np.zeros((T, 14, 4), dtype=np.float64)
        self.joint_quat[:, :, 3] = 1.0  # 默认恒等（fixed 关节保持）
        off = 6
        group_slice = {}
        for name, n, sph in FRAME_JOINTS:
            raw = frames[:, off:off + n]                           # (T,n)
            jidx = KIN_JOINTS.index(name)
            if sph:
                self.joint_quat[:, jidx] = expmap_to_quat_xyzw(raw)
            else:
                # 单 hinge：axis 来自上游 XML——elbow 为 z 轴、knee 为 y 轴
                axis_idx = {'right_elbow': 2, 'left_elbow': 2,
                            'right_knee': 1, 'left_knee': 1}[name]
                ang = raw[:, 0]
                q = np.zeros((T, 4))
                q[:, axis_idx] = np.sin(ang / 2.0)
                q[:, 3] = np.cos(ang / 2.0)
                self.joint_quat[:, jidx] = q
            group_slice[name] = (jidx, raw)
            off += n

        # ---- 破解到 28 hinge 角度（MuJoCo qpos 语义）----
        self.hinge_pos = np.zeros((T, 28), dtype=np.float64)
        for hinge, group, aidx, sph in hinge_layout():
            col = {n: i for i, n in enumerate(
                [h for hh in hinge_layout() for h in [hh[0]]])}[hinge]
            jidx = KIN_JOINTS.index(group)
            if sph:
                eul = np.array([decompose_sph_to_hinges(self.joint_quat[t, jidx])
                                for t in range(T)])                # (T,3) xyz序
                self.hinge_pos[:, col] = eul[:, aidx]
            else:
                self.hinge_pos[:, col] = group_slice[group][1][:, 0]

        # ---- 速度（有限差分，对齐上游 _process_data）----
        fps = self.fps
        self.root_vel = np.zeros_like(self.root_pos)
        self.root_vel[:-1] = fps * (self.root_pos[1:] - self.root_pos[:-1])
        self.root_vel[-1] = self.root_vel[-2]

        # root 角速度：quat_diff 的 expmap × fps（世界系近似，与上游一致）
        self.root_ang_vel = np.zeros_like(self.root_pos)
        q0 = R.from_quat(self.root_quat[:-1])
        q1 = R.from_quat(self.root_quat[1:])
        dq = (q0.inv() * q1).as_quat()  # 身体系差分
        # 上游：quat_diff(q0,q1) 定义见 torch_util —— 用世界系差分 q1*q0^-1
        dq_world = (q1 * q0.inv()).as_quat()
        self.root_ang_vel[:-1] = fps * quat_xyzw_to_expmap(dq_world)
        self.root_ang_vel[-1] = self.root_ang_vel[-2]

        # dof 速度：hinge 角速度有限差分
        self.dof_vel = np.zeros_like(self.hinge_pos)
        # 处理角度缠绕：差分钳到 [-pi, pi]
        diff = self.hinge_pos[1:] - self.hinge_pos[:-1]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self.dof_vel[:-1] = fps * diff
        self.dof_vel[-1] = self.dof_vel[-2]

        # wrap 位移（cyclic 用，上游 _motion_wrap_delta）
        self.wrap_delta = self.root_pos[-1] - self.root_pos[0]
        self.wrap_delta[2] = 0.0

    # ----------------------------------------------------------
    # 时间索引
    # ----------------------------------------------------------
    def clamp_time(self, t: float) -> float:
        return float(np.clip(t, 0.0, self.duration))

    def _blend(self, t: float):
        """返回 (i0, i1, alpha)。WRAP 模式由调用方处理相位回绕。"""
        t = self.clamp_time(t)
        x = t * self.fps
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, self.n_frames - 1)
        a = x - i0
        return i0, i1, a

    def state_at(self, t: float) -> Dict[str, np.ndarray]:
        """任意时间的插值参考状态（rotation 用 slerp 近似——
        上游 torch_util 用 nlerp+normalize，数值差异可忽略）。"""
        i0, i1, a = self._blend(t)
        out = {}
        out['root_pos'] = (1 - a) * self.root_pos[i0] + a * self.root_pos[i1]
        q0 = R.from_quat(self.root_quat[i0])
        q1 = R.from_quat(self.root_quat[i1])
        # scipy slerp
        from scipy.spatial.transform import Slerp
        if i0 == i1 or a == 0.0:
            out['root_quat'] = self.root_quat[i0]
        else:
            sl = Slerp([0, 1], R.from_quat([self.root_quat[i0], self.root_quat[i1]]))
            out['root_quat'] = sl([a])[0].as_quat()
        out['root_vel'] = (1 - a) * self.root_vel[i0] + a * self.root_vel[i1]
        out['root_ang_vel'] = (1 - a) * self.root_ang_vel[i0] + a * self.root_ang_vel[i1]

        jq = np.zeros((14, 4))
        for j in range(14):
            qa = self.joint_quat[i0, j]
            qb = self.joint_quat[i1, j]
            if i0 == i1 or a == 0.0:
                jq[j] = qa
            else:
                sl = Slerp([0, 1], R.from_quat([qa, qb]))
                jq[j] = sl([a])[0].as_quat()
        out['joint_quat'] = jq
        # hinge_pos 从插值后的 joint_quat 派生（单一一致路径：
        # quat 为主表示，hinge 查询时分解，避免两条插值路径不一致）
        hp = np.zeros(28)
        for hinge, group, aidx, sph in hinge_layout():
            col = [h[0] for h in hinge_layout()].index(hinge)
            jidx = KIN_JOINTS.index(group)
            if sph:
                eul = decompose_sph_to_hinges(jq[jidx])
                hp[col] = eul[aidx]
            else:
                axis_idx = {'right_elbow': 2, 'left_elbow': 2,
                            'right_knee': 1, 'left_knee': 1}[group]
                # 从 hinge quat 取绕轴角（小角近似 atan2 精确解）
                q = jq[jidx]
                s, c = q[axis_idx], q[3]
                hp[col] = 2.0 * np.arctan2(s, c)
        out['hinge_pos'] = hp
        out['dof_vel'] = (1 - a) * self.dof_vel[i0] + a * self.dof_vel[i1]
        return out


_DEFAULT_DIR = Path(__file__).resolve().parent / 'data' / 'motions'
_cache: Dict[str, RefMotion] = {}


def load_ref_motion(name_or_path: str, data_dir: Optional[str] = None) -> RefMotion:
    """按名字（如 'humanoid_spinkick'）或路径加载，带缓存。"""
    if name_or_path in _cache:
        return _cache[name_or_path]
    p = Path(name_or_path)
    if not p.exists():
        base = Path(data_dir) if data_dir else _DEFAULT_DIR
        p = base / (name_or_path if name_or_path.endswith('.pkl') else name_or_path + '.pkl')
    rm = RefMotion(str(p))
    _cache[name_or_path] = rm
    return rm
