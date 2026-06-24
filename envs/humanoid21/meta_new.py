"""
Humanoid21 仿真环境元数据定义与管理

设计理念:
  以 MuJoCo 原生概念模型为基准, 在代码中显式定义环境的结构.
  用户读这份代码就能理解: 有哪些 body, 每个 body 上挂着什么 geom/joint,
  body 之间的父子关系是什么. 不需要学习自定义的编号体系.

  在 MuJoCo 概念之上, 按需添加战斗语义层 (ATTACK_PARTS, HITTABLE_PARTS, KEYPOINT_BODIES).

两层结构:
  Layer 1 — MuJoCo 概念 (与 battle_v1.xml 一一对应)
    ENV_GEOMS:        环境 geom 名称列表
    ROBOT_BODY_TREE:  机器人 body 树 (body → geoms, joints, children)
    CONTROLLED_JOINTS: policy 直接控制的 21 个关节 (顺序 = action vector 维度)
    ROBOT_SUFFIXES:   机器人实例化后缀

  Layer 2 — 战斗语义 (建立在 MuJoCo body 之上)
    ATTACK_PARTS:     可以攻击别人的 body name 列表
    HITTABLE_PARTS:   可以被攻击的 body name 列表
    KEYPOINT_BODIES:  需要观测的 body name 列表
    KEYPOINT_JOINTS:  需要观测的关节 name 列表

  校验: validate() 检查代码定义与加载的 MuJoCo model 是否一致
  运行时: build_runtime_tables() 从代码定义 + model 构建查找表
"""
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np


class Humanoid21Meta:
    """Humanoid21 对抗仿真环境的静态元数据定义、校验与运行时关联。"""

    # ============================================================
    # Layer 1: MuJoCo 概念模型 — 与 battle_v1.xml 一一对应
    # ============================================================

    # --- 环境 Geoms (不属于任何机器人) ---
    ENV_GEOMS: List[str] = [
        'ground', 'ceiling', 'southwall', 'northwall', 'westwall', 'eastwall',
    ]

    # --- 机器人 Body Tree ---
    # 每个 body 声明: 它包含哪些 geom、哪些 joint、它的子 body 是谁
    # 实际模型中每个 body/geom/joint name 会加上 _a 或 _b 后缀
    # root joint (freejoint) 名称为 f"root{suffix}", 挂在 torso body 上
    ROBOT_BODY_TREE: Dict[str, Dict[str, Any]] = {
        'torso': {
            'geoms': ['torso', 'waist_upper'],
            'joints': [],
            'children': ['head', 'waist_lower', 'upper_arm_right', 'upper_arm_left'],
        },
        'head': {
            'geoms': ['head'],
            'joints': [],
            'children': [],
        },
        'waist_lower': {
            'geoms': ['waist_lower'],
            'joints': ['abdomen_z', 'abdomen_y'],
            'children': ['pelvis'],
        },
        'pelvis': {
            'geoms': ['butt'],
            'joints': ['abdomen_x'],
            'children': ['thigh_right', 'thigh_left'],
        },
        'thigh_right': {
            'geoms': ['thigh_right'],
            'joints': ['hip_x_right', 'hip_z_right', 'hip_y_right'],
            'children': ['shin_right'],
        },
        'shin_right': {
            'geoms': ['shin_right'],
            'joints': ['knee_right'],
            'children': ['foot_right'],
        },
        'foot_right': {
            'geoms': ['foot1_right', 'foot2_right'],
            'joints': ['ankle_y_right', 'ankle_x_right'],
            'children': [],
        },
        'thigh_left': {
            'geoms': ['thigh_left'],
            'joints': ['hip_x_left', 'hip_z_left', 'hip_y_left'],
            'children': ['shin_left'],
        },
        'shin_left': {
            'geoms': ['shin_left'],
            'joints': ['knee_left'],
            'children': ['foot_left'],
        },
        'foot_left': {
            'geoms': ['foot1_left', 'foot2_left'],
            'joints': ['ankle_y_left', 'ankle_x_left'],
            'children': [],
        },
        'upper_arm_right': {
            'geoms': ['upper_arm_right'],
            'joints': ['shoulder1_right', 'shoulder2_right'],
            'children': ['lower_arm_right'],
        },
        'lower_arm_right': {
            'geoms': ['lower_arm_right'],
            'joints': ['elbow_right'],
            'children': ['hand_right'],
        },
        'hand_right': {
            'geoms': ['hand_right'],
            'joints': [],
            'children': [],
        },
        'upper_arm_left': {
            'geoms': ['upper_arm_left'],
            'joints': ['shoulder1_left', 'shoulder2_left'],
            'children': ['lower_arm_left'],
        },
        'lower_arm_left': {
            'geoms': ['lower_arm_left'],
            'joints': ['elbow_left'],
            'children': ['hand_left'],
        },
        'hand_left': {
            'geoms': ['hand_left'],
            'joints': [],
            'children': [],
        },
    }

    # --- 受控关节 (policy 直接控制的 21 个 DOF) ---
    # 顺序即 action vector 的维度顺序
    # actuator 与 joint 同名, 即 actuator_name = f"{joint_name}_{suffix}"
    CONTROLLED_JOINTS: List[str] = [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'hip_x_right', 'hip_z_right', 'hip_y_right',
        'knee_right', 'ankle_y_right', 'ankle_x_right',
        'hip_x_left', 'hip_z_left', 'hip_y_left',
        'knee_left', 'ankle_y_left', 'ankle_x_left',
        'shoulder1_right', 'shoulder2_right', 'elbow_right',
        'shoulder1_left', 'shoulder2_left', 'elbow_left',
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

    # --- 战斗语义: body name 列表 (不含 _a/_b 后缀) ---
    # 可以攻击别人的部位
    ATTACK_PARTS: List[str] = [
        'hand_right', 'hand_left',
        'foot_right', 'foot_left',
    ]

    # 可以被攻击 (受击) 的部位
    HITTABLE_PARTS: List[str] = [
        'head',
        'torso', 'waist_lower', 'pelvis',
    ]

    # --- 观测关键点 ---
    # 需要被观测 (位置/速度) 的 body
    KEYPOINT_BODIES: List[str] = [
        'torso', 'head', 'pelvis',
        'foot_right', 'foot_left',
        'hand_right', 'hand_left',
    ]

    # --- 观测关键关节 ---
    # 标记哪些 joint 需要被观测 (平衡分析用)
    KEYPOINT_JOINTS: List[str] = [
        'ankle_y_right', 'ankle_x_right', 'ankle_y_left', 'ankle_x_left',
    ]

    # ============================================================
    # 仿真参数
    # ============================================================

    DT = 0.002
    ACTION_DIM = 21

    KP = np.array([
        # 腹部 (abdomen_z, abdomen_y, abdomen_x)
        1000.0, 1000.0, 1000.0,
        # 右腿 (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 左腿
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 右臂 (shoulder1, shoulder2, elbow)
        150.0, 150.0, 100.0,
        # 左臂
        150.0, 150.0, 100.0,
    ], dtype=np.float32)

    KD = np.array([
        # 腹部
        100.0, 100.0, 100.0,
        # 右腿
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 左腿
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 右臂
        15.0, 15.0, 10.0,
        # 左臂
        15.0, 15.0, 10.0,
    ], dtype=np.float32)

    # --- 初始姿态配置 (来自 humanoid.xml 的 keyframes) ---
    INITIAL_POSES: Dict[str, Dict[str, np.ndarray]] = {
        'standing': {
            'root_pos': np.array([0, 0, 1.282], dtype=np.float32),
            'root_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4286, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32),
        },
        'squat': {
            'root_pos': np.array([0, 0, 0.596], dtype=np.float32),
            'root_quat': np.array([0.988015, 0, 0.154359, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.4, 0,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                0, 0, 0,
                0, 0, 0,
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4287, 0.0000,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32),
        },
        'stand_on_left_leg': {
            'root_pos': np.array([0, 0, 1.21948], dtype=np.float32),
            'root_quat': np.array([0.971588, -0.179973, 0.135318, -0.0729076], dtype=np.float32),
            'joint_pos': np.array([
                -0.0516, -0.202, 0.23,
                -0.24, -0.007, -0.34, -1.76, -0.466, -0.0415,
                -0.08, -0.01, -0.37, -0.685, -0.35, -0.09,
                0.109, -0.067, -0.7,
                -0.05, 0.12, 0.16,
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4285, 0.0001,
                0.4998, 0.2632, 0.7646, 0.9749, -0.0002, -0.0000,
                0.4999, 0.2632, 0.7646, 0.9752, -0.0001, -0.0000,
                0.1724, 0.1724, 0.3332, 0.1724, 0.1724, 0.3334,
            ], dtype=np.float32),
        },
        'prone': {
            'root_pos': np.array([0.4, 0, 0.0757706], dtype=np.float32),
            'root_quat': np.array([0.7325, 0, 0.680767, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.0729, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.56, -0.62, -1.752,
                0.186, -0.73, -1.73,
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4286, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.1725, 0.1723, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32),
        },
        'supine': {
            'root_pos': np.array([-0.4, 0, 0.08122], dtype=np.float32),
            'root_quat': np.array([0.722788, 0, -0.69107, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, -0.25, 0,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.186, -0.73, -1.73,
                0.186, -0.73, -1.73,
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4285, 0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.1725, 0.1722, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32),
        },
    }

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

        return errors

    # ============================================================
    # 运行时构建
    # ============================================================

    @classmethod
    def build_runtime_tables(cls, model) -> Dict[str, Any]:
        """从代码定义 + MuJoCo model 构建运行时查找表.

        返回:
            robots: {robot_id: {suffix, root_body_id, root_joint_id, ...}}
            env_geom_ids: set[int]  — 环境 geom 的 MuJoCo ID
            ground_geom_id: int     — 地面 geom 的 MuJoCo ID
            body_to_robot: Dict[int, str] — body_id → robot_id (None if env)
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

            # Root body (torso, 带 freejoint)
            root_body_name = f"torso{suffix}"
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
                        f"All joints must have finite limits for normalized control."
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
        }
