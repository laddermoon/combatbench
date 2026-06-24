"""
Humanoid21 仿真环境元数据定义与管理

概念体系 (bootstrip.md):

  AFF (3类): 归属
    0=environment  1=robot_a  2=robot_b

  GEOM_CAT (25类): 每个AFF下每个geom有唯一CAT, 不做语义合并
    环境(6):     0=ground  1=ceiling  2=southwall  3=northwall  4=westwall  5=eastwall
    机器人(19):  6=head  7=torso  8=waist_upper  9=waist_lower  10=butt
                 11=thigh_right  12=shin_right  13=foot1_right  14=foot2_right
                 15=thigh_left  16=shin_left  17=foot1_left  18=foot2_left
                 19=upper_arm_right  20=lower_arm_right  21=hand_right
                 22=upper_arm_left  23=lower_arm_left  24=hand_left

  CONTACT_PART (15类): 每个GEOM_CAT对应唯一CONTACT_PART
    0=ground       ← CAT: ground
    1=ceiling      ← CAT: ceiling
    2=wall         ← CAT: southwall, northwall, westwall, eastwall
    3=head         ← CAT: head
    4=torso        ← CAT: torso
    5=waist        ← CAT: waist_upper, waist_lower
    6=pelvis       ← CAT: butt
    7=right_leg    ← CAT: thigh_right, shin_right
    8=right_foot   ← CAT: foot1_right, foot2_right
    9=left_leg     ← CAT: thigh_left, shin_left
    10=left_foot   ← CAT: foot1_left, foot2_left
    11=right_arm   ← CAT: upper_arm_right, lower_arm_right
    12=right_hand  ← CAT: hand_right
    13=left_arm    ← CAT: upper_arm_left, lower_arm_left
    14=left_hand   ← CAT: hand_left

  CONTACT_GROUP (8类): 每个CONTACT_PART对应一个CONTACT_GROUP
    0=ground  ← PART: ground          [不可攻击, 不可受击]
    1=wall    ← PART: ceiling, wall   [不可攻击, 不可受击]
    2=head    ← PART: head            [不可攻击, 可受击]
    3=torso   ← PART: torso, waist, pelvis  [不可攻击, 可受击]
    4=arm     ← PART: right_arm, left_arm   [不可攻击, 不可受击]
    5=hand    ← PART: right_hand, left_hand [可攻击,   不可受击]
    6=leg     ← PART: right_leg, left_leg   [不可攻击, 不可受击]
    7=foot    ← PART: right_foot, left_foot [可攻击,   不可受击]

  GEOM: 与 MuJoCo geom 一一对应, 拥有 GEOM_CAT/AFF/ISKEYPOINT 属性
    共44个: 6环境 + 19(robot_a) + 19(robot_b)
    robot geom 命名: <base>_a / <base>_b, 共享同一 GEOM_CAT

  JOINT_CAT (22类/机器人): 每个AFF下每个joint有唯一JOINT_CAT
    0=root
    1=abdomen_z  2=abdomen_y  3=abdomen_x
    4=hip_x_right  5=hip_z_right  6=hip_y_right  7=knee_right  8=ankle_y_right  9=ankle_x_right
    10=hip_x_left  11=hip_z_left  12=hip_y_left  13=knee_left  14=ankle_y_left  15=ankle_x_left
    16=shoulder1_right  17=shoulder2_right  18=elbow_right
    19=shoulder1_left  20=shoulder2_left  21=elbow_left

  JOINT: 与 MuJoCo joint 一一对应, 拥有 JOINT_CAT/AFF 属性
    共44个: 22(robot_a) + 22(robot_b)
    joint 命名: <base>_a / <base>_b, 共享同一 JOINT_CAT

  层级关系链:
    GEOM → GEOM_CAT → CONTACT_PART → CONTACT_GROUP
    JOINT → JOINT_CAT → JOINT_SEMANTIC
    GEOM/JOINT → AFF
"""

from typing import Dict, List, Set, Tuple, Optional

import numpy as np


class Humanoid21Meta:
    """Humanoid21 对抗仿真环境的静态元数据定义、校验与运行时关联。"""

    # === AFF — 归属 (3类) ===
    AFF_ENV = 0
    AFF_ROBOT_A = 1
    AFF_ROBOT_B = 2

    AFF_ID_TO_NAME: Dict[int, str] = {0: 'environment', 1: 'robot_a', 2: 'robot_b'}
    AFF_NAME_TO_ID: Dict[str, int] = {v: k for k, v in AFF_ID_TO_NAME.items()}
    NUM_AFFS = 3

    # === GEOM_CAT — 几何体类别 (25类, 0~24) ===
    # 每个AFF下每个geom有唯一的CAT, 不做语义合并
    # 环境 (6)
    CAT_GROUND = 0
    CAT_CEILING = 1
    CAT_SOUTHWALL = 2
    CAT_NORTHWALL = 3
    CAT_WESTWALL = 4
    CAT_EASTWALL = 5
    # 机器人 (19, _a/_b共享同一CAT)
    CAT_HEAD = 6
    CAT_TORSO = 7
    CAT_WAIST_UPPER = 8
    CAT_WAIST_LOWER = 9
    CAT_BUTT = 10
    CAT_THIGH_RIGHT = 11
    CAT_SHIN_RIGHT = 12
    CAT_FOOT1_RIGHT = 13
    CAT_FOOT2_RIGHT = 14
    CAT_THIGH_LEFT = 15
    CAT_SHIN_LEFT = 16
    CAT_FOOT1_LEFT = 17
    CAT_FOOT2_LEFT = 18
    CAT_UPPER_ARM_RIGHT = 19
    CAT_LOWER_ARM_RIGHT = 20
    CAT_HAND_RIGHT = 21
    CAT_UPPER_ARM_LEFT = 22
    CAT_LOWER_ARM_LEFT = 23
    CAT_HAND_LEFT = 24

    CAT_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'ceiling', 2: 'southwall',
        3: 'northwall', 4: 'westwall', 5: 'eastwall',
        6: 'head', 7: 'torso', 8: 'waist_upper', 9: 'waist_lower',
        10: 'butt', 11: 'thigh_right', 12: 'shin_right',
        13: 'foot1_right', 14: 'foot2_right',
        15: 'thigh_left', 16: 'shin_left',
        17: 'foot1_left', 18: 'foot2_left',
        19: 'upper_arm_right', 20: 'lower_arm_right', 21: 'hand_right',
        22: 'upper_arm_left', 23: 'lower_arm_left', 24: 'hand_left',
    }
    CAT_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CAT_ID_TO_NAME.items()}
    NUM_CATS = 25

    # === CONTACT_PART — 碰撞部位分类 (15类, 0~14) ===
    # 每个GEOM_CAT对应唯一CONTACT_PART
    PART_GROUND = 0
    PART_CEILING = 1
    PART_WALL = 2
    PART_HEAD = 3
    PART_TORSO = 4
    PART_WAIST = 5
    PART_PELVIS = 6
    PART_RIGHT_LEG = 7
    PART_RIGHT_FOOT = 8
    PART_LEFT_LEG = 9
    PART_LEFT_FOOT = 10
    PART_RIGHT_ARM = 11
    PART_RIGHT_HAND = 12
    PART_LEFT_ARM = 13
    PART_LEFT_HAND = 14

    PART_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'ceiling', 2: 'wall',
        3: 'head', 4: 'torso', 5: 'waist', 6: 'pelvis',
        7: 'right_leg', 8: 'right_foot',
        9: 'left_leg', 10: 'left_foot',
        11: 'right_arm', 12: 'right_hand',
        13: 'left_arm', 14: 'left_hand',
    }
    PART_NAME_TO_ID: Dict[str, int] = {v: k for k, v in PART_ID_TO_NAME.items()}
    NUM_PARTS = 15

    # GEOM_CAT → CONTACT_PART (25 → 15, 多对一)
    CAT_TO_PART: Dict[int, int] = {
        0: PART_GROUND, 1: PART_CEILING,
        2: PART_WALL, 3: PART_WALL, 4: PART_WALL, 5: PART_WALL,
        6: PART_HEAD,
        7: PART_TORSO, 8: PART_WAIST, 9: PART_WAIST,
        10: PART_PELVIS,
        11: PART_RIGHT_LEG, 12: PART_RIGHT_LEG,
        13: PART_RIGHT_FOOT, 14: PART_RIGHT_FOOT,
        15: PART_LEFT_LEG, 16: PART_LEFT_LEG,
        17: PART_LEFT_FOOT, 18: PART_LEFT_FOOT,
        19: PART_RIGHT_ARM, 20: PART_RIGHT_ARM, 21: PART_RIGHT_HAND,
        22: PART_LEFT_ARM, 23: PART_LEFT_ARM, 24: PART_LEFT_HAND,
    }

    # === CONTACT_GROUP — 碰撞大组分类 (8类, 0~7) ===
    # 每个CONTACT_PART对应一个CONTACT_GROUP
    GROUP_GROUND = 0
    GROUP_WALL = 1
    GROUP_HEAD = 2
    GROUP_TORSO = 3
    GROUP_ARM = 4
    GROUP_HAND = 5
    GROUP_LEG = 6
    GROUP_FOOT = 7

    GROUP_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'wall', 2: 'head', 3: 'torso',
        4: 'arm', 5: 'hand', 6: 'leg', 7: 'foot',
    }
    GROUP_NAME_TO_ID: Dict[str, int] = {v: k for k, v in GROUP_ID_TO_NAME.items()}
    NUM_GROUPS = 8

    # CONTACT_GROUP 战斗属性: 可攻击 / 可受击
    # 规则: 仅 hand, foot 可作为有效攻击部位; 仅 head, torso 可作为有效受击部位
    GROUP_CAN_ATTACK: Dict[int, bool] = {
        GROUP_GROUND: False, GROUP_WALL: False,
        GROUP_HEAD: False, GROUP_TORSO: False,
        GROUP_ARM: False, GROUP_HAND: True,
        GROUP_LEG: False, GROUP_FOOT: True,
    }
    GROUP_CAN_BE_HIT: Dict[int, bool] = {
        GROUP_GROUND: False, GROUP_WALL: False,
        GROUP_HEAD: True, GROUP_TORSO: True,
        GROUP_ARM: False, GROUP_HAND: False,
        GROUP_LEG: False, GROUP_FOOT: False,
    }

    # CONTACT_PART → CONTACT_GROUP (15 → 8, 多对一)
    PART_TO_GROUP: Dict[int, int] = {
        0: GROUP_GROUND, 1: GROUP_WALL, 2: GROUP_WALL,
        3: GROUP_HEAD,
        4: GROUP_TORSO, 5: GROUP_TORSO, 6: GROUP_TORSO,
        7: GROUP_LEG, 8: GROUP_FOOT,
        9: GROUP_LEG, 10: GROUP_FOOT,
        11: GROUP_ARM, 12: GROUP_HAND,
        13: GROUP_ARM, 14: GROUP_HAND,
    }

    # GEOM_CAT → CONTACT_GROUP (组合映射, 等价于 CAT→PART→GROUP)
    CAT_TO_GROUP: Dict[int, int] = {
        0: GROUP_GROUND,
        1: GROUP_WALL, 2: GROUP_WALL, 3: GROUP_WALL, 4: GROUP_WALL, 5: GROUP_WALL,
        6: GROUP_HEAD,
        7: GROUP_TORSO, 8: GROUP_TORSO, 9: GROUP_TORSO, 10: GROUP_TORSO,
        11: GROUP_LEG, 12: GROUP_LEG, 13: GROUP_FOOT, 14: GROUP_FOOT,
        15: GROUP_LEG, 16: GROUP_LEG, 17: GROUP_FOOT, 18: GROUP_FOOT,
        19: GROUP_ARM, 20: GROUP_ARM, 21: GROUP_HAND,
        22: GROUP_ARM, 23: GROUP_ARM, 24: GROUP_HAND,
    }

    # === GEOM — MuJoCo geom → 元数据 ===

    ENV_GEOM_NAMES: Set[str] = {
        'ground', 'ceiling', 'southwall', 'northwall', 'westwall', 'eastwall',
    }

    ENV_GEOM_TO_CAT: Dict[str, int] = {
        'ground': CAT_GROUND, 'ceiling': CAT_CEILING,
        'southwall': CAT_SOUTHWALL, 'northwall': CAT_NORTHWALL,
        'westwall': CAT_WESTWALL, 'eastwall': CAT_EASTWALL,
    }

    ROBOT_GEOM_BASE_TO_CAT: Dict[str, int] = {
        'head': CAT_HEAD, 'torso': CAT_TORSO,
        'waist_upper': CAT_WAIST_UPPER, 'waist_lower': CAT_WAIST_LOWER,
        'butt': CAT_BUTT,
        'thigh_right': CAT_THIGH_RIGHT, 'shin_right': CAT_SHIN_RIGHT,
        'foot1_right': CAT_FOOT1_RIGHT, 'foot2_right': CAT_FOOT2_RIGHT,
        'thigh_left': CAT_THIGH_LEFT, 'shin_left': CAT_SHIN_LEFT,
        'foot1_left': CAT_FOOT1_LEFT, 'foot2_left': CAT_FOOT2_LEFT,
        'upper_arm_right': CAT_UPPER_ARM_RIGHT,
        'lower_arm_right': CAT_LOWER_ARM_RIGHT, 'hand_right': CAT_HAND_RIGHT,
        'upper_arm_left': CAT_UPPER_ARM_LEFT,
        'lower_arm_left': CAT_LOWER_ARM_LEFT, 'hand_left': CAT_HAND_LEFT,
    }
    ROBOT_GEOM_BASES: Set[str] = set(ROBOT_GEOM_BASE_TO_CAT.keys())

    KEYPOINT_GEOM_BASES: Set[str] = {
        'head', 'torso', 'butt',
        'hand_left', 'hand_right',
        'foot1_left', 'foot2_left', 'foot1_right', 'foot2_right',
    }

    # === GEOM Entity — 自定义实体 ID (25, 0~24) ===
    GEOM_ENTITY_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'ceiling', 2: 'southwall',
        3: 'northwall', 4: 'westwall', 5: 'eastwall',
        6: 'head', 7: 'torso', 8: 'waist_upper', 9: 'waist_lower',
        10: 'butt', 11: 'thigh_right', 12: 'shin_right',
        13: 'foot1_right', 14: 'foot2_right',
        15: 'thigh_left', 16: 'shin_left',
        17: 'foot1_left', 18: 'foot2_left',
        19: 'upper_arm_right', 20: 'lower_arm_right', 21: 'hand_right',
        22: 'upper_arm_left', 23: 'lower_arm_left', 24: 'hand_left',
    }
    GEOM_ENTITY_NAME_TO_ID: Dict[str, int] = {v: k for k, v in GEOM_ENTITY_ID_TO_NAME.items()}
    NUM_GEOM_ENTITIES = 25

    # GEOM Entity ID == CAT ID (both 0~24, 1:1 映射)
    GEOM_ENTITY_ID_TO_CAT_ID: Dict[int, int] = {i: i for i in range(25)}

    # MuJoCo geom name → GEOM Entity ID
    MUJOCO_GEOM_NAME_TO_ENTITY_ID: Dict[str, int] = {}
    for _en, _eid in GEOM_ENTITY_NAME_TO_ID.items():
        MUJOCO_GEOM_NAME_TO_ENTITY_ID[_en] = _eid
        if _en not in ENV_GEOM_NAMES:
            MUJOCO_GEOM_NAME_TO_ENTITY_ID[f'{_en}_a'] = _eid
            MUJOCO_GEOM_NAME_TO_ENTITY_ID[f'{_en}_b'] = _eid
    del _en, _eid

    # === JOINT_CAT — 关节类别 (22类/机器人, 0~21) ===
    JOINT_CAT_ID_TO_NAME: Dict[int, str] = {
        0: 'root',
        1: 'abdomen_z', 2: 'abdomen_y', 3: 'abdomen_x',
        4: 'hip_x_right', 5: 'hip_z_right', 6: 'hip_y_right',
        7: 'knee_right', 8: 'ankle_y_right', 9: 'ankle_x_right',
        10: 'hip_x_left', 11: 'hip_z_left', 12: 'hip_y_left',
        13: 'knee_left', 14: 'ankle_y_left', 15: 'ankle_x_left',
        16: 'shoulder1_right', 17: 'shoulder2_right', 18: 'elbow_right',
        19: 'shoulder1_left', 20: 'shoulder2_left', 21: 'elbow_left',
    }
    JOINT_CAT_NAME_TO_ID: Dict[str, int] = {v: k for k, v in JOINT_CAT_ID_TO_NAME.items()}
    NUM_JOINT_CATS = 22

    JOINT_SEMANTIC_ID_TO_NAME: Dict[int, str] = {
        0: 'root', 1: 'abdomen', 2: 'hip', 3: 'knee',
        4: 'ankle', 5: 'shoulder', 6: 'elbow',
    }

    JOINT_CAT_TO_SEMANTIC: Dict[int, int] = {
        0: 0, 1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2, 10: 2, 11: 2, 12: 2,
        7: 3, 13: 3,
        8: 4, 9: 4, 14: 4, 15: 4,
        16: 5, 17: 5, 19: 5, 20: 5,
        18: 6, 21: 6,
    }

    # === 静态期望值 ===
    EXPECTED_NUM_GEOMS = 44
    EXPECTED_NUM_BODIES = 33
    EXPECTED_NUM_JOINTS = 44
    EXPECTED_ENV_GEOMS = 6
    EXPECTED_ROBOT_GEOMS_PER_AFF = 19
    EXPECTED_ROBOT_JOINTS_PER_AFF = 22

    # === 仿真参数 ===
    DT = 0.002
    ACTION_DIM = 21
    
    KP = np.array([
        # 腹部 (abdomen_z, abdomen_y, abdomen_x) - 战斗中需维持上半身直立
        1000.0, 1000.0, 1000.0,
        # 右腿 (hip_x=roll, hip_z=yaw, hip_y=pitch, knee, ankle_y, ankle_x)
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 左腿
        150.0, 200.0, 200.0, 200.0, 100.0, 100.0,
        # 右臂 (shoulder1, shoulder2, elbow)
        150.0, 150.0, 100.0,
        # 左臂
        150.0, 150.0, 100.0
    ], dtype=np.float32)

    KD = np.array([
        # 腹部 - 高阻尼以减少过冲
        100.0, 100.0, 100.0,
        # 右腿 - 踝部较低增益以保持柔顺性
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 左腿
        15.0, 20.0, 20.0, 20.0, 10.0, 10.0,
        # 右臂
        15.0, 15.0, 10.0,
        # 左臂
        15.0, 15.0, 10.0
    ], dtype=np.float32)
    
    # 受控关节名称 (固定顺序)
    CONTROLLED_JOINTS = [
        'abdomen_z', 'abdomen_y', 'abdomen_x',
        'hip_x_right', 'hip_z_right', 'hip_y_right', 'knee_right', 'ankle_y_right', 'ankle_x_right',
        'hip_x_left', 'hip_z_left', 'hip_y_left', 'knee_left', 'ankle_y_left', 'ankle_x_left',
        'shoulder1_right', 'shoulder2_right', 'elbow_right',
        'shoulder1_left', 'shoulder2_left', 'elbow_left'
    ]

    # 初始姿态配置（来自 humanoid.xml 的 keyframes）
    # 每个姿态包含 root_pos, root_quat, joint_pos, action
    INITIAL_POSES = {
        'standing': {
            'root_pos': np.array([0, 0, 1.282], dtype=np.float32),
            'root_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0, 0,  # abdomen
                0, 0, 0, 0, 0, 0,  # right leg
                0, 0, 0, 0, 0, 0,  # left leg
                0, 0, 0,  # right arm
                0, 0, 0  # left arm
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4286, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.5000, 0.2632, 0.7647, 0.9753, -0.0000, -0.0000,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32)
        },
        'squat': {
            'root_pos': np.array([0, 0, 0.596], dtype=np.float32),
            'root_quat': np.array([0.988015, 0, 0.154359, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.4, 0,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                -0.25, -0.5, -2.5, -2.65, -0.8, 0.56,
                0, 0, 0,
                0, 0, 0
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4287, 0.0000,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.4998, 0.2630, 0.7642, 0.9747, -0.0003, 0.0002,
                0.1724, 0.1724, 0.3333, 0.1724, 0.1724, 0.3333,
            ], dtype=np.float32)
        },
        'stand_on_left_leg': {
            'root_pos': np.array([0, 0, 1.21948], dtype=np.float32),
            'root_quat': np.array([0.971588, -0.179973, 0.135318, -0.0729076], dtype=np.float32),
            'joint_pos': np.array([
                -0.0516, -0.202, 0.23,
                -0.24, -0.007, -0.34, -1.76, -0.466, -0.0415,
                -0.08, -0.01, -0.37, -0.685, -0.35, -0.09,
                0.109, -0.067, -0.7,
                -0.05, 0.12, 0.16
            ], dtype=np.float32),
            'action': np.array([
                -0.0000, 0.4285, 0.0001,
                0.4998, 0.2632, 0.7646, 0.9749, -0.0002, -0.0000,
                0.4999, 0.2632, 0.7646, 0.9752, -0.0001, -0.0000,
                0.1724, 0.1724, 0.3332, 0.1724, 0.1724, 0.3334,
            ], dtype=np.float32)
        },
        'prone': {
            'root_pos': np.array([0.4, 0, 0.0757706], dtype=np.float32),
            'root_quat': np.array([0.7325, 0, 0.680767, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, 0.0729, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.0077, 0.0019, -0.026, -0.351, -0.27, 0,
                0.56, -0.62, -1.752,
                0.186, -0.73, -1.73
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4286, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.5000, 0.2632, 0.7647, 0.9752, -0.0001, 0.0000,
                0.1725, 0.1723, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32)
        },
        'supine': {
            'root_pos': np.array([-0.4, 0, 0.08122], dtype=np.float32),
            'root_quat': np.array([0.722788, 0, -0.69107, 0], dtype=np.float32),
            'joint_pos': np.array([
                0, -0.25, 0,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.0182, 0.0142, 0.3, 0.042, -0.44, -0.02,
                0.186, -0.73, -1.73,
                0.186, -0.73, -1.73
            ], dtype=np.float32),
            'action': np.array([
                0.0000, 0.4285, 0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.5000, 0.2632, 0.7648, 0.9753, -0.0002, -0.0000,
                0.1725, 0.1722, 0.3329, 0.1725, 0.1722, 0.3329,
            ], dtype=np.float32)
        }
    }
    # === 内部分类工具 ===

    @classmethod
    def _classify_geom(cls, name: str) -> Tuple[Optional[int], Optional[int]]:
        """返回 (cat_id, aff_id) 或 (None, None)"""
        if name in cls.ENV_GEOM_NAMES:
            return cls.ENV_GEOM_TO_CAT.get(name), cls.AFF_ENV
        if name.endswith('_a'):
            base = name[:-2]
            aff = cls.AFF_ROBOT_A
        elif name.endswith('_b'):
            base = name[:-2]
            aff = cls.AFF_ROBOT_B
        else:
            return None, None
        return cls.ROBOT_GEOM_BASE_TO_CAT.get(base), aff

    @classmethod
    def _classify_joint(cls, name: str) -> Tuple[Optional[int], Optional[int]]:
        """返回 (joint_cat_id, aff_id) 或 (None, None)"""
        if name == 'root_a':
            return 0, cls.AFF_ROBOT_A
        if name == 'root_b':
            return 0, cls.AFF_ROBOT_B
        if name.endswith('_a'):
            base = name[:-2]
            aff = cls.AFF_ROBOT_A
        elif name.endswith('_b'):
            base = name[:-2]
            aff = cls.AFF_ROBOT_B
        else:
            return None, None
        return cls.JOINT_CAT_NAME_TO_ID.get(base), aff

    @classmethod
    def _is_keypoint_geom(cls, name: str) -> bool:
        if name in cls.ENV_GEOM_NAMES:
            return False
        base = name[:-2] if name.endswith('_a') or name.endswith('_b') else name
        return base in cls.KEYPOINT_GEOM_BASES

    # === 校验 ===

    @classmethod
    def validate(cls, model) -> List[str]:
        """校验 MuJoCo model 与静态元数据是否匹配。返回错误列表，空=通过。"""
        import mujoco
        errors: List[str] = []

        if model.ngeom != cls.EXPECTED_NUM_GEOMS:
            errors.append(f"Geom count: expected {cls.EXPECTED_NUM_GEOMS}, got {model.ngeom}")
        if model.nbody != cls.EXPECTED_NUM_BODIES:
            errors.append(f"Body count: expected {cls.EXPECTED_NUM_BODIES}, got {model.nbody}")
        if model.njnt != cls.EXPECTED_NUM_JOINTS:
            errors.append(f"Joint count: expected {cls.EXPECTED_NUM_JOINTS}, got {model.njnt}")

        aff_geom_counts = {0: 0, 1: 0, 2: 0}
        cats_found: Set[int] = set()
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if not name:
                errors.append(f"Geom id={gid} has no name")
                continue
            cat, aff = cls._classify_geom(name)
            if cat is None:
                errors.append(f"Geom '{name}' cannot be classified to CAT")
            else:
                cats_found.add(cat)
            if aff is None:
                errors.append(f"Geom '{name}' has unknown AFF")
            else:
                aff_geom_counts[aff] += 1

        aff_joint_counts = {1: 0, 2: 0}
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ''
            if not name:
                errors.append(f"Joint id={jid} has no name")
                continue
            jcat, aff = cls._classify_joint(name)
            if jcat is None:
                errors.append(f"Joint '{name}' cannot be classified to JOINT_CAT")
            if aff is not None and aff in aff_joint_counts:
                aff_joint_counts[aff] += 1

        for aff_id, expected, label in [
            (0, cls.EXPECTED_ENV_GEOMS, 'Env'),
            (1, cls.EXPECTED_ROBOT_GEOMS_PER_AFF, 'Robot_a'),
            (2, cls.EXPECTED_ROBOT_GEOMS_PER_AFF, 'Robot_b'),
        ]:
            if aff_geom_counts[aff_id] != expected:
                errors.append(f"{label} geom count: expected {expected}, got {aff_geom_counts[aff_id]}")

        for aff_id, expected, label in [
            (1, cls.EXPECTED_ROBOT_JOINTS_PER_AFF, 'Robot_a'),
            (2, cls.EXPECTED_ROBOT_JOINTS_PER_AFF, 'Robot_b'),
        ]:
            if aff_joint_counts[aff_id] != expected:
                errors.append(f"{label} joint count: expected {expected}, got {aff_joint_counts[aff_id]}")

        missing = set(range(cls.NUM_CATS)) - cats_found
        if missing:
            errors.append(f"Missing CATs: {[cls.CAT_ID_TO_NAME[c] for c in sorted(missing)]}")

        return errors

    # === 运行时关联 ===

    @classmethod
    def build_runtime_tables(cls, model) -> Dict:
        """构建 MuJoCo ID → 元数据的运行时查找表 (SoA 预映射)。"""
        import mujoco
        ngeom, njnt = model.ngeom, model.njnt

        geom_cat = np.full(ngeom, -1, dtype=np.int8)
        geom_aff = np.full(ngeom, 0, dtype=np.int8)
        geom_is_keypoint = np.zeros(ngeom, dtype=bool)
        geom_part = np.full(ngeom, -1, dtype=np.int8)
        geom_group = np.full(ngeom, -1, dtype=np.int8)
        geom_entity_id = np.full(ngeom, -1, dtype=np.int8)
        geom_names: List[str] = []

        for gid in range(ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            geom_names.append(name)
            cat, aff = cls._classify_geom(name)
            if cat is not None:
                geom_cat[gid] = cat
                part = cls.CAT_TO_PART[cat]
                geom_part[gid] = part
                geom_group[gid] = cls.PART_TO_GROUP[part]
            if aff is not None:
                geom_aff[gid] = aff
            geom_is_keypoint[gid] = cls._is_keypoint_geom(name)
            geom_entity_id[gid] = cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID.get(name, -1)

        joint_cat = np.full(njnt, -1, dtype=np.int8)
        joint_aff = np.full(njnt, 0, dtype=np.int8)
        joint_semantic = np.full(njnt, -1, dtype=np.int8)
        joint_names: List[str] = []

        for jid in range(njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ''
            joint_names.append(name)
            jcat, aff = cls._classify_joint(name)
            if jcat is not None:
                joint_cat[jid] = jcat
                joint_semantic[jid] = cls.JOINT_CAT_TO_SEMANTIC.get(jcat, -1)
            if aff is not None:
                joint_aff[jid] = aff

        return {
            'geom_cat': geom_cat,
            'geom_aff': geom_aff,
            'geom_is_keypoint': geom_is_keypoint,
            'geom_part': geom_part,
            'geom_group': geom_group,
            'geom_entity_id': geom_entity_id,
            'geom_names': geom_names,
            'joint_cat': joint_cat,
            'joint_aff': joint_aff,
            'joint_semantic': joint_semantic,
            'joint_names': joint_names,
        }

    # === 易用查询接口 ===

    @classmethod
    def aff_id_to_name(cls, aff_id: int) -> str:
        return cls.AFF_ID_TO_NAME[aff_id]

    @classmethod
    def aff_name_to_id(cls, name: str) -> int:
        return cls.AFF_NAME_TO_ID[name]

    @classmethod
    def cat_id_to_name(cls, cat_id: int) -> str:
        return cls.CAT_ID_TO_NAME[cat_id]

    @classmethod
    def cat_name_to_id(cls, name: str) -> int:
        return cls.CAT_NAME_TO_ID[name]

    @classmethod
    def geom_entity_id_to_name(cls, eid: int) -> str:
        return cls.GEOM_ENTITY_ID_TO_NAME[eid]

    @classmethod
    def geom_entity_name_to_id(cls, name: str) -> int:
        return cls.GEOM_ENTITY_NAME_TO_ID[name]

    @classmethod
    def joint_cat_id_to_name(cls, jcat_id: int) -> str:
        return cls.JOINT_CAT_ID_TO_NAME[jcat_id]

    @classmethod
    def joint_cat_name_to_id(cls, name: str) -> int:
        return cls.JOINT_CAT_NAME_TO_ID[name]

    @classmethod
    def part_id_to_name(cls, part_id: int) -> str:
        return cls.PART_ID_TO_NAME[part_id]

    @classmethod
    def part_name_to_id(cls, name: str) -> int:
        return cls.PART_NAME_TO_ID[name]

    @classmethod
    def cat_to_part(cls, cat_id: int) -> int:
        return cls.CAT_TO_PART[cat_id]

    @classmethod
    def part_to_group(cls, part_id: int) -> int:
        return cls.PART_TO_GROUP[part_id]

    @classmethod
    def group_id_to_name(cls, group_id: int) -> str:
        return cls.GROUP_ID_TO_NAME[group_id]

    @classmethod
    def group_name_to_id(cls, name: str) -> int:
        return cls.GROUP_NAME_TO_ID[name]

    @classmethod
    def cat_to_group(cls, cat_id: int) -> int:
        return cls.CAT_TO_GROUP[cat_id]

    @classmethod
    def geom_info(cls, mujoco_geom_name: str) -> Optional[Dict]:
        """查询单个 MuJoCo geom 的完整元数据。"""
        cat, aff = cls._classify_geom(mujoco_geom_name)
        if cat is None:
            return None
        eid = cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID.get(mujoco_geom_name, -1)
        part = cls.CAT_TO_PART[cat]
        group = cls.PART_TO_GROUP[part]
        return {
            'mujoco_name': mujoco_geom_name,
            'cat_id': cat,
            'cat_name': cls.CAT_ID_TO_NAME[cat],
            'aff_id': aff,
            'aff_name': cls.AFF_ID_TO_NAME.get(aff, 'unknown'),
            'entity_id': eid,
            'entity_name': cls.GEOM_ENTITY_ID_TO_NAME.get(eid, 'unknown'),
            'is_keypoint': cls._is_keypoint_geom(mujoco_geom_name),
            'part_id': part,
            'part_name': cls.PART_ID_TO_NAME[part],
            'group_id': group,
            'group_name': cls.GROUP_ID_TO_NAME[group],
            'can_attack': cls.GROUP_CAN_ATTACK[group],
            'can_be_hit': cls.GROUP_CAN_BE_HIT[group],
        }

    @classmethod
    def joint_info(cls, mujoco_joint_name: str) -> Optional[Dict]:
        """查询单个 MuJoCo joint 的完整元数据。"""
        jcat, aff = cls._classify_joint(mujoco_joint_name)
        if jcat is None:
            return None
        return {
            'mujoco_name': mujoco_joint_name,
            'joint_cat_id': jcat,
            'joint_cat_name': cls.JOINT_CAT_ID_TO_NAME[jcat],
            'aff_id': aff,
            'aff_name': cls.AFF_ID_TO_NAME.get(aff, 'unknown'),
            'semantic_id': cls.JOINT_CAT_TO_SEMANTIC.get(jcat, -1),
            'semantic_name': cls.JOINT_SEMANTIC_ID_TO_NAME.get(
                cls.JOINT_CAT_TO_SEMANTIC.get(jcat, -1), 'unknown'),
        }

    @classmethod
    def geoms_by_cat(cls, cat_id: int) -> List[str]:
        """返回属于指定 CAT 的所有 MuJoCo geom 名称。"""
        result = []
        for name in cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID:
            cat, _ = cls._classify_geom(name)
            if cat == cat_id:
                result.append(name)
        return sorted(result)

    @classmethod
    def geoms_by_aff(cls, aff_id: int) -> List[str]:
        """返回属于指定 AFF 的所有 MuJoCo geom 名称。"""
        result = []
        for name in cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID:
            _, aff = cls._classify_geom(name)
            if aff == aff_id:
                result.append(name)
        return sorted(result)

    @classmethod
    def keypoint_geoms(cls) -> List[str]:
        """返回所有 keypoint geom 名称。"""
        return sorted(n for n in cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID
                      if cls._is_keypoint_geom(n))

    @classmethod
    def geoms_by_group(cls, group_id: int) -> List[str]:
        """返回属于指定 CONTACT_GROUP 的所有 MuJoCo geom 名称。"""
        result = []
        for name in cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID:
            cat, _ = cls._classify_geom(name)
            if cat is not None and cls.CAT_TO_GROUP[cat] == group_id:
                result.append(name)
        return sorted(result)
