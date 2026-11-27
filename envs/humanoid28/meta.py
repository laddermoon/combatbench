"""
Humanoid28 仿真环境元数据定义与管理

设计理念与 Humanoid21Meta 相同（决策 #9/#12：实现完全参照 humanoid21）：
  以 MuJoCo 原生概念模型为基准，在代码中显式定义环境结构。

与 humanoid21 的核心差异：
  * 机器人资产来自 MimicKit（Apache-2.0，见 README.md 归属说明），
    28 个受控 DOF、root body 为 pelvis、质量约 50.05 kg。
  * **不存在归一化层**：action 是弧度制 PD 位置目标（MimicKit 控制格式，
    决策 #4），关节观测也是原始弧度/弧度每秒（用户决策：原始值无压缩）。
    因此本文件没有 humanoid21 的 reference/scale 归一化参数。
  * PD 增益 kp/kd 与扭矩上限 gear 来自 MimicKit XML 的 joint
    stiffness/damping 与 motor gear —— 上游语义中它们就是 PD 控制增益
    （Newton 引擎会把被动值清零后转作控制增益，见 generate_arena_xml.py
    的转换说明）。数值存于 robot_actuation.json（脚本生成，勿手改）。

两层结构:
  Layer 1 — MuJoCo 概念 (与 battle_humanoid28.xml 一一对应)
  Layer 2 — 战斗语义 (ATTACK_PARTS / HITTABLE_PARTS / KEYPOINT_BODIES)
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np


class Humanoid28Meta:
    """Humanoid28 对抗仿真环境的静态元数据定义、校验与运行时关联。"""

    # ============================================================
    # Layer 1: MuJoCo 概念模型 — 与 battle_humanoid28.xml 一一对应
    # ============================================================

    # --- 环境 Geoms (不属于任何机器人) ---
    ENV_GEOMS: List[str] = [
        'ground', 'ceiling',
    ] + [f'wall_{i:02d}' for i in range(24)]

    # --- 机器人 Body Tree ---
    # 与 MimicKit humanoid.xml 的 body 结构一一对应；
    # 实际模型中每个 body/geom/joint name 带 _a 或 _b 后缀。
    # root joint (freejoint) 名称为 f"root{suffix}"，挂在 pelvis body 上。
    ROBOT_BODY_TREE: Dict[str, Dict[str, Any]] = {
        'pelvis': {
            'geoms': ['pelvis', 'upper_waist'],
            'joints': [],
            'children': ['torso', 'right_thigh', 'left_thigh'],
        },
        'torso': {
            'geoms': ['torso', 'right_clavicle', 'left_clavicle'],
            'joints': ['abdomen_x', 'abdomen_y', 'abdomen_z'],
            'children': ['head', 'right_upper_arm', 'left_upper_arm'],
        },
        'head': {
            'geoms': ['head'],
            'joints': ['neck_x', 'neck_y', 'neck_z'],
            'children': [],
        },
        'right_upper_arm': {
            'geoms': ['right_upper_arm'],
            'joints': ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
            'children': ['right_lower_arm'],
        },
        'right_lower_arm': {
            'geoms': ['right_lower_arm'],
            'joints': ['right_elbow'],
            'children': ['right_hand'],
        },
        'right_hand': {
            'geoms': ['right_hand'],
            'joints': [],
            'children': [],
        },
        'left_upper_arm': {
            'geoms': ['left_upper_arm'],
            'joints': ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
            'children': ['left_lower_arm'],
        },
        'left_lower_arm': {
            'geoms': ['left_lower_arm'],
            'joints': ['left_elbow'],
            'children': ['left_hand'],
        },
        'left_hand': {
            'geoms': ['left_hand'],
            'joints': [],
            'children': [],
        },
        'right_thigh': {
            'geoms': ['right_thigh'],
            'joints': ['right_hip_x', 'right_hip_y', 'right_hip_z'],
            'children': ['right_shin'],
        },
        'right_shin': {
            'geoms': ['right_shin'],
            'joints': ['right_knee'],
            'children': ['right_foot'],
        },
        'right_foot': {
            'geoms': ['right_foot'],
            'joints': ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
            'children': [],
        },
        'left_thigh': {
            'geoms': ['left_thigh'],
            'joints': ['left_hip_x', 'left_hip_y', 'left_hip_z'],
            'children': ['left_shin'],
        },
        'left_shin': {
            'geoms': ['left_shin'],
            'joints': ['left_knee'],
            'children': ['left_foot'],
        },
        'left_foot': {
            'geoms': ['left_foot'],
            'joints': ['left_ankle_x', 'left_ankle_y', 'left_ankle_z'],
            'children': [],
        },
    }

    # --- 受控关节 (policy 直接控制的 28 个 DOF) ---
    # 顺序 = XML 文档顺序（depth-first）= action vector 维度顺序，
    # 与 MimicKit kin char model / Isaac Gym DOF 顺序一致。
    # actuator 与 joint 同名, 即 actuator_name = f"{joint_name}_{suffix}"
    CONTROLLED_JOINTS: List[str] = [
        'abdomen_x', 'abdomen_y', 'abdomen_z',
        'neck_x', 'neck_y', 'neck_z',
        'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',
        'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow',
        'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee',
        'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
        'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee',
        'left_ankle_x', 'left_ankle_y', 'left_ankle_z',
    ]

    # --- 机器人实例化 ---
    ROBOT_SUFFIXES: Dict[str, str] = {
        'robot_a': '_a',
        'robot_b': '_b',
    }

    # --- 派生: 所有机器人 body name (不含后缀) ---
    ROBOT_BODY_NAMES: List[str] = list(ROBOT_BODY_TREE.keys())

    # --- 派生: 所有机器人 geom name (不含后缀) ---
    ROBOT_GEOM_NAMES: List[str] = []
    for _spec in ROBOT_BODY_TREE.values():
        ROBOT_GEOM_NAMES.extend(_spec['geoms'])
    del _spec

    # --- 派生: 所有机器人 joint name (不含后缀, 含 root) ---
    ROBOT_JOINT_NAMES: List[str] = ['root']
    for _spec in ROBOT_BODY_TREE.values():
        ROBOT_JOINT_NAMES.extend(_spec['joints'])
    del _spec

    # ============================================================
    # Layer 2: 战斗语义 — 建立在 MuJoCo body 之上
    # ============================================================

    # 可以攻击别人的部位
    ATTACK_PARTS: List[str] = [
        'right_hand', 'left_hand',
        'right_foot', 'left_foot',
    ]

    # 可以被攻击 (受击) 的部位
    # （pelvis 与 upper_waist geom 均归躯干类受击，与 humanoid21 的
    #   torso/waist/pelvis 归类一致）
    HITTABLE_PARTS: List[str] = [
        'head',
        'torso', 'pelvis',
    ]

    # --- 观测关键点 ---
    KEYPOINT_BODIES: List[str] = [
        'pelvis', 'torso', 'head',
        'right_foot', 'left_foot',
        'right_hand', 'left_hand',
    ]

    # --- 观测关键关节 (平衡分析用) ---
    # 三连踝铰链在同一 body 上、共享同一 anchor 点
    KEYPOINT_JOINTS: List[str] = [
        'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
        'left_ankle_x', 'left_ankle_y', 'left_ankle_z',
    ]

    # ============================================================
    # 仿真参数
    # ============================================================

    DT = 0.002          # 500Hz 物理（决策 #6）
    ACTION_DIM = 28

    # --- 初始姿态 ---
    # 来自 MimicKit deepmimic_humanoid_env.yaml 的 init_pose：
    #   root 高度 0.882416，root 旋转 expmap 为零向量（单位四元数），
    #   dof 除 right_shoulder_x=+1.5708 / left_shoulder_x=-1.5708 外全零
    #   （双臂从侧平举垂放到体侧的站立姿势）。
    # 注：MimicKit 把 expmap 值直接写入串联铰链 DOF（两者在单轴旋转下严格等价），
    #     本环境与其保持一致：joint_pos 直接是各 hinge 的弧度位置。
    INITIAL_POSES: Dict[str, Dict[str, np.ndarray]] = {
        'standing': {
            'root_pos': np.array([0, 0, 0.882416], dtype=np.float32),
            'root_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0, 0,                # abdomen x/y/z
                0, 0, 0,                # neck x/y/z
                1.5708, 0, 0, 0,        # right shoulder x/y/z, elbow
                -1.5708, 0, 0, 0,       # left shoulder x/y/z, elbow
                0, 0, 0, 0,             # right hip x/y/z, knee
                0, 0, 0,                # right ankle x/y/z
                0, 0, 0, 0,             # left hip x/y/z, knee
                0, 0, 0,                # left ankle x/y/z
            ], dtype=np.float32),
        },
    }

    # ============================================================
    # Actuation 数据加载 (robot_actuation.json, 脚本生成)
    # ============================================================

    _ACTUATION_JSON = Path(__file__).parent / 'robot_actuation.json'

    @classmethod
    def load_actuation(cls) -> Dict[str, np.ndarray]:
        """加载脚本生成的 actuation 参数，按 CONTROLLED_JOINTS 顺序排列为数组。

        返回:
            kp:    (28,) PD 比例增益
            kd:    (28,) PD 微分增益
            gear:  (28,) motor gear = 扭矩上限 (Nm)（ctrlrange [-1,1]）
            range: (28, 2) 关节机械限位 (rad)
            armature: (28,) 关节 armature
        校验 JSON 中的关节顺序与 CONTROLLED_JOINTS 完全一致（两者都应当等于
        上游 XML 文档顺序；不一致说明生成脚本或本文件的关节列表被改坏）。
        """
        with open(cls._ACTUATION_JSON, encoding='utf-8') as f:
            payload = json.load(f)
        joints = payload['joints']
        json_names = [j['name'] for j in joints]
        if json_names != cls.CONTROLLED_JOINTS:
            raise ValueError(
                'robot_actuation.json 关节顺序与 CONTROLLED_JOINTS 不一致。\n'
                f'json: {json_names}\nmeta: {cls.CONTROLLED_JOINTS}\n'
                '请重跑 generate_arena_xml.py 或修正 CONTROLLED_JOINTS。'
            )
        return {
            'kp': np.array([j['kp'] for j in joints], dtype=np.float32),
            'kd': np.array([j['kd'] for j in joints], dtype=np.float32),
            'gear': np.array([j['gear'] for j in joints], dtype=np.float32),
            'range': np.array([j['range'] for j in joints], dtype=np.float32),
            'armature': np.array([j['armature'] for j in joints], dtype=np.float32),
        }

    # ============================================================
    # Action bounds — 复刻 MimicKit CharEnv._build_action_bounds_pos
    # ============================================================

    # MimicKit 的 kin char model 把同一 body 上的三连铰链合并为 spherical
    # joint，其 action bounds 不按单关节 1.4x 而按球形 joint 1.2x max-norm
    # 计算（见 mimickit/envs/char_env.py）。本环境与其保持一致：
    SPHERICAL_GROUPS: List[List[str]] = [
        ['abdomen_x', 'abdomen_y', 'abdomen_z'],
        ['neck_x', 'neck_y', 'neck_z'],
        ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
        ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
        ['right_hip_x', 'right_hip_y', 'right_hip_z'],
        ['left_hip_x', 'left_hip_y', 'left_hip_z'],
        ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
        ['left_ankle_x', 'left_ankle_y', 'left_ankle_z'],
    ]
    # 单自由度关节（elbow/knee）按 mid ± 1.4*scale 计算；
    # zero_center_action 上游默认 False，即 mid = (upper+lower)/2。
    SPHERICAL_BOUND_SCALE = 1.2
    HINGE_BOUND_SCALE = 1.4

    @classmethod
    def compute_action_bounds(cls) -> Dict[str, np.ndarray]:
        """计算 action space 上下界（弧度制 PD 位置目标）。

        复刻 MimicKit 逻辑：
          * spherical 组：取组内全部 hinge 限位的最大绝对值 ×1.2，
            组内三个关节共用对称 bounds [-s, +s]；
          * 单 hinge 关节：mid=(hi+lo)/2, scale=max(|hi-mid|,|lo-mid|)×1.4，
            bounds = [mid-scale, mid+scale]。
        返回 {'low': (28,), 'high': (28,)}，按 CONTROLLED_JOINTS 顺序。
        """
        act = cls.load_actuation()
        ranges = act['range']
        idx = {name: i for i, name in enumerate(cls.CONTROLLED_JOINTS)}

        low = np.zeros(cls.ACTION_DIM, dtype=np.float32)
        high = np.zeros(cls.ACTION_DIM, dtype=np.float32)

        grouped = set()
        for group in cls.SPHERICAL_GROUPS:
            maxabs = 0.0
            for jname in group:
                lo, hi = ranges[idx[jname]]
                maxabs = max(maxabs, abs(lo), abs(hi))
            s = cls.SPHERICAL_BOUND_SCALE * maxabs
            for jname in group:
                low[idx[jname]] = -s
                high[idx[jname]] = s
                grouped.add(jname)

        for jname, i in idx.items():
            if jname in grouped:
                continue
            lo, hi = ranges[i]
            mid = 0.5 * (hi + lo)
            s = cls.HINGE_BOUND_SCALE * max(abs(hi - mid), abs(lo - mid))
            low[i] = mid - s
            high[i] = mid + s

        return {'low': low, 'high': high}

    # ============================================================
    # 校验
    # ============================================================

    @classmethod
    def validate(cls, model) -> List[str]:
        """校验 MuJoCo model 与代码定义是否一致, 返回错误列表 (空=通过)."""
        import mujoco
        errors: List[str] = []

        # 1. 校验环境 geom
        for name in cls.ENV_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                errors.append(f"Missing env geom: {name}")

        # 2. 校验每个机器人的 body / geom / joint
        for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
            for body_name, spec in cls.ROBOT_BODY_TREE.items():
                full_body = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_body)
                if bid < 0:
                    errors.append(f"[{robot_id}] Missing body: {full_body}")

                for geom_name in spec['geoms']:
                    full_geom = f"{geom_name}{suffix}"
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, full_geom)
                    if gid < 0:
                        errors.append(f"[{robot_id}] Missing geom: {full_geom}")

                for jnt_name in spec['joints']:
                    full_jnt = f"{jnt_name}{suffix}"
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_jnt)
                    if jid < 0:
                        errors.append(f"[{robot_id}] Missing joint: {full_jnt}")

            # 校验 root joint (freejoint)
            root_name = f"root{suffix}"
            root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_name)
            if root_jid < 0:
                errors.append(f"[{robot_id}] Missing root joint: {root_name}")

        # 3. 校验 controlled joints + actuators
        for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
            for jnt_name in cls.CONTROLLED_JOINTS:
                full_jnt = f"{jnt_name}{suffix}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_jnt)
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_jnt)
                if jid < 0:
                    errors.append(f"[{robot_id}] Missing controlled joint: {full_jnt}")
                if aid < 0:
                    errors.append(f"[{robot_id}] Missing actuator: {full_jnt}")

        # 4. 校验 keypoint bodies
        for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
            for body_name in cls.KEYPOINT_BODIES:
                full_body = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_body)
                if bid < 0:
                    errors.append(f"[{robot_id}] Missing keypoint body: {full_body}")

        # 5. 校验被动动力学已剥离（MimicKit 语义：stiffness/damping 是 PD 增益，
        #    不得以被动弹簧/阻尼形式留在模型里，否则 double-count）
        for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
            for jnt_name in cls.CONTROLLED_JOINTS:
                full_jnt = f"{jnt_name}{suffix}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_jnt)
                if jid < 0:
                    continue
                if float(model.jnt_stiffness[jid]) != 0.0:
                    errors.append(
                        f"[{robot_id}] Joint {full_jnt} has nonzero passive stiffness "
                        f"{float(model.jnt_stiffness[jid])}; XML 生成时应剥离")
                dof_adr = int(model.jnt_dofadr[jid])
                if float(model.dof_damping[dof_adr]) != 0.0:
                    errors.append(
                        f"[{robot_id}] Joint {full_jnt} has nonzero passive damping "
                        f"{float(model.dof_damping[dof_adr])}; XML 生成时应剥离")

        # 6. 校验 actuation JSON 与模型一致：gear 与关节限位
        try:
            act = cls.load_actuation()
        except Exception as e:  # noqa: BLE001 - 校验路径需要汇聚全部错误
            errors.append(f"load_actuation failed: {e}")
        else:
            for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
                for i, jnt_name in enumerate(cls.CONTROLLED_JOINTS):
                    full_jnt = f"{jnt_name}{suffix}"
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_jnt)
                    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_jnt)
                    if jid < 0 or aid < 0:
                        continue
                    gear_xml = float(model.actuator_gear[aid, 0])
                    if abs(gear_xml - float(act['gear'][i])) > 1e-4:
                        errors.append(
                            f"[{robot_id}] {full_jnt} gear mismatch: "
                            f"xml={gear_xml} json={float(act['gear'][i])}")
                    lo, hi = model.jnt_range[jid]
                    elo, ehi = act['range'][i]
                    if abs(lo - elo) > 1e-4 or abs(hi - ehi) > 1e-4:
                        errors.append(
                            f"[{robot_id}] {full_jnt} range mismatch: "
                            f"xml=[{lo},{hi}] json=[{elo},{ehi}]")

        return errors

    # ============================================================
    # 运行时构建
    # ============================================================

    @classmethod
    def build_runtime_tables(cls, model) -> Dict[str, Any]:
        """从代码定义 + MuJoCo model 构建运行时查找表.

        返回结构同 Humanoid21Meta.build_runtime_tables：
            robots / env_geom_ids / ground_geom_id / body_to_robot /
            geom_id_to_name / body_id_to_name / body_id_to_aff / geom_id_to_aff
        """
        import mujoco

        # --- 环境 geom IDs ---
        env_geom_ids: Set[int] = set()
        ground_geom_id = -1
        for name in cls.ENV_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                env_geom_ids.add(gid)
                if name == 'ground':
                    ground_geom_id = gid

        # --- body_id → robot_id 映射 ---
        body_to_robot: Dict[int, str] = {}

        # --- 静态查找表: ID → name, ID → aff ---
        geom_id_to_name: Dict[int, str] = {}
        body_id_to_name: Dict[int, str] = {}
        body_id_to_aff:  Dict[int, int] = {}
        geom_id_to_aff:  Dict[int, int] = {}

        # 环境 geom: aff = 0
        for gid in env_geom_ids:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            geom_id_to_name[gid] = gname
            geom_id_to_aff[gid] = 0
            bid = int(model.geom_bodyid[gid])
            body_id_to_aff[bid] = 0

        # --- per-robot 结构化数据 ---
        robots: Dict[str, Dict[str, Any]] = {}

        for robot_id, suffix in cls.ROBOT_SUFFIXES.items():
            r: Dict[str, Any] = {'suffix': suffix}

            # Root joint (freejoint)
            root_jnt_name = f"root{suffix}"
            root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_jnt_name)
            r['root_joint_name'] = root_jnt_name
            r['root_joint_id'] = root_jid
            r['root_qpos_adr'] = int(model.jnt_qposadr[root_jid])
            r['root_qvel_adr'] = int(model.jnt_dofadr[root_jid])

            # Root body (pelvis, 带 freejoint)
            root_body_name = f"pelvis{suffix}"
            root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
            r['root_body_name'] = root_body_name
            r['root_body_id'] = root_bid

            # Body 子树: 从 ROBOT_BODY_TREE 展开所有 body, 查 MuJoCo ID
            body_ids: List[int] = []
            body_names: List[str] = []
            for bname in cls.ROBOT_BODY_NAMES:
                full = f"{bname}{suffix}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full)
                body_ids.append(bid)
                body_names.append(full)
                body_to_robot[bid] = robot_id

            r['body_ids'] = set(body_ids)
            r['body_ids_sorted'] = np.asarray(sorted(body_ids), dtype=np.int32)
            r['body_names'] = body_names
            r['body_masses'] = np.asarray(
                [float(model.body_mass[bid]) for bid in sorted(body_ids)],
                dtype=np.float32,
            )
            # 体重 m*g (N) —— 观测中足底受力的归一化常量。
            r['total_mass'] = float(r['body_masses'].sum())
            r['body_weight'] = float(
                r['total_mass'] * float(np.linalg.norm(model.opt.gravity))
            )

            # 填充静态查找表: robot body/geom → name, aff
            aff_code = 1 if robot_id == 'robot_a' else 2
            for bid, bname in zip(body_ids, body_names):
                body_id_to_name[bid] = bname
                body_id_to_aff[bid] = aff_code
            for spec in cls.ROBOT_BODY_TREE.values():
                for gname_base in spec['geoms']:
                    full_geom = f"{gname_base}{suffix}"
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, full_geom)
                    if gid >= 0:
                        geom_id_to_name[gid] = full_geom
                        geom_id_to_aff[gid] = aff_code

            # 全部 joint: 从 ROBOT_BODY_TREE 展开 + root
            joint_names: List[str] = []
            joint_ids_by_name: Dict[str, int] = {}
            for jname in cls.ROBOT_JOINT_NAMES:
                full = f"{jname}{suffix}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
                if jid >= 0:
                    joint_names.append(full)
                    joint_ids_by_name[full] = jid
            r['joint_names'] = joint_names
            r['joint_ids_by_name'] = joint_ids_by_name

            # 受控关节: CONTROLLED_JOINTS + suffix
            ctrl_qpos, ctrl_qvel, ctrl_act, ctrl_ranges = [], [], [], []
            ctrl_joint_names: List[str] = []
            for jname in cls.CONTROLLED_JOINTS:
                full = f"{jname}{suffix}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
                if jid < 0:
                    raise ValueError(f"Joint {full} not found in model")
                ctrl_qpos.append(int(model.jnt_qposadr[jid]))
                ctrl_qvel.append(int(model.jnt_dofadr[jid]))
                ctrl_joint_names.append(full)

                if not model.jnt_limited[jid]:
                    raise ValueError(
                        f"Joint {full} has no limits. "
                        f"All joints must have finite limits for action bound computation."
                    )
                ctrl_ranges.append(model.jnt_range[jid].copy())

                # actuator 与 joint 同名
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
                if aid < 0:
                    raise ValueError(f"Actuator {full} not found in model")
                ctrl_act.append(aid)

            r['qpos_indices'] = np.array(ctrl_qpos, dtype=np.int32)
            r['qvel_indices'] = np.array(ctrl_qvel, dtype=np.int32)
            r['actuator_ids'] = np.array(ctrl_act, dtype=np.int32)
            r['jnt_ranges'] = np.array(ctrl_ranges, dtype=np.float32)
            r['controlled_joint_names'] = ctrl_joint_names

            # Keypoint bodies: body name → MuJoCo body ID
            keypoint_body_ids: Dict[str, int] = {}
            keypoint_body_names: Dict[str, str] = {}
            for body_name in cls.KEYPOINT_BODIES:
                full = f"{body_name}{suffix}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full)
                keypoint_body_ids[body_name] = bid
                keypoint_body_names[body_name] = full
            r['keypoint_body_ids'] = keypoint_body_ids
            r['keypoint_body_names'] = keypoint_body_names

            # Keypoint joints: 语义名 → MuJoCo joint full name
            keypoint_joint_names: Dict[str, str] = {}
            for jname in cls.KEYPOINT_JOINTS:
                full = f"{jname}{suffix}"
                keypoint_joint_names[jname] = full
            r['keypoint_joint_names'] = keypoint_joint_names

            robots[robot_id] = r

        return {
            'robots': robots,
            'env_geom_ids': env_geom_ids,
            'ground_geom_id': ground_geom_id,
            'body_to_robot': body_to_robot,
            'geom_id_to_name': geom_id_to_name,
            'body_id_to_name': body_id_to_name,
            'body_id_to_aff': body_id_to_aff,
            'geom_id_to_aff': geom_id_to_aff,
        }
