"""
Humanoid21 仿真环境元数据定义与管理

概念体系 (bootstrip.md):
  AFF: 归属 — environment(0), robot_a(1), robot_b(2)
  GEOM_CAT: 几何体类别 — 19类 (3环境 + 16机器人部位)
  SemanticCat: 语义大类 — 8类
  GEOM: 与 MuJoCo geom 一一对应, 拥有 CAT/AFF/ISKEYPOINT 属性
  JOINTCAT: 关节类别 — 22类/机器人
  JOINT: 与 MuJoCo joint 一一对应, 拥有 JOINTCAT/AFF 属性
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

    # === GEOM_CAT — 几何体类别 (19类, 0~18) ===
    CAT_ENV_GROUND = 0
    CAT_ENV_WALL = 1
    CAT_ENV_CEILING = 2
    CAT_HEAD = 3
    CAT_TORSO = 4
    CAT_PELVIS = 5
    CAT_HAND_LEFT = 6
    CAT_HAND_RIGHT = 7
    CAT_UPPER_ARM_LEFT = 8
    CAT_UPPER_ARM_RIGHT = 9
    CAT_LOWER_ARM_LEFT = 10
    CAT_LOWER_ARM_RIGHT = 11
    CAT_THIGH_LEFT = 12
    CAT_THIGH_RIGHT = 13
    CAT_SHIN_LEFT = 14
    CAT_SHIN_RIGHT = 15
    CAT_FOOT_LEFT = 16
    CAT_FOOT_RIGHT = 17
    CAT_WAIST = 18

    CAT_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'wall', 2: 'ceiling',
        3: 'head', 4: 'torso', 5: 'pelvis',
        6: 'hand_left', 7: 'hand_right',
        8: 'upper_arm_left', 9: 'upper_arm_right',
        10: 'lower_arm_left', 11: 'lower_arm_right',
        12: 'thigh_left', 13: 'thigh_right',
        14: 'shin_left', 15: 'shin_right',
        16: 'foot_left', 17: 'foot_right',
        18: 'waist',
    }
    CAT_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CAT_ID_TO_NAME.items()}
    NUM_CATS = 19

    # === Semantic Cat — 语义大类 (8类) ===
    SEMANTIC_GROUND = 0
    SEMANTIC_WALL = 1
    SEMANTIC_HEAD = 2
    SEMANTIC_TORSO = 3
    SEMANTIC_ARM = 4
    SEMANTIC_HAND = 5
    SEMANTIC_LEG = 6
    SEMANTIC_FOOT = 7

    SEMANTIC_ID_TO_NAME: Dict[int, str] = {
        0: 'ground', 1: 'wall', 2: 'head', 3: 'torso',
        4: 'arm', 5: 'hand', 6: 'leg', 7: 'foot',
    }

    CAT_TO_SEMANTIC: Dict[int, int] = {
        0: SEMANTIC_GROUND, 1: SEMANTIC_WALL, 2: SEMANTIC_WALL,
        3: SEMANTIC_HEAD, 4: SEMANTIC_TORSO, 5: SEMANTIC_TORSO, 18: SEMANTIC_TORSO,
        6: SEMANTIC_HAND, 7: SEMANTIC_HAND,
        8: SEMANTIC_ARM, 9: SEMANTIC_ARM, 10: SEMANTIC_ARM, 11: SEMANTIC_ARM,
        12: SEMANTIC_LEG, 13: SEMANTIC_LEG, 14: SEMANTIC_LEG, 15: SEMANTIC_LEG,
        16: SEMANTIC_FOOT, 17: SEMANTIC_FOOT,
    }

    # === GEOM — MuJoCo geom → 元数据 ===

    ENV_GEOM_NAMES: Set[str] = {
        'ground', 'ceiling', 'southwall', 'northwall', 'westwall', 'eastwall',
    }

    ENV_GEOM_TO_CAT: Dict[str, int] = {
        'ground': CAT_ENV_GROUND, 'ceiling': CAT_ENV_CEILING,
        'southwall': CAT_ENV_WALL, 'northwall': CAT_ENV_WALL,
        'westwall': CAT_ENV_WALL, 'eastwall': CAT_ENV_WALL,
    }

    ROBOT_GEOM_BASE_TO_CAT: Dict[str, int] = {
        'head': CAT_HEAD, 'torso': CAT_TORSO,
        'waist_upper': CAT_WAIST, 'waist_lower': CAT_WAIST,
        'butt': CAT_PELVIS,
        'thigh_right': CAT_THIGH_RIGHT, 'shin_right': CAT_SHIN_RIGHT,
        'foot1_right': CAT_FOOT_RIGHT, 'foot2_right': CAT_FOOT_RIGHT,
        'thigh_left': CAT_THIGH_LEFT, 'shin_left': CAT_SHIN_LEFT,
        'foot1_left': CAT_FOOT_LEFT, 'foot2_left': CAT_FOOT_LEFT,
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

    GEOM_ENTITY_ID_TO_CAT_ID: Dict[int, int] = {
        0: CAT_ENV_GROUND, 1: CAT_ENV_CEILING,
        2: CAT_ENV_WALL, 3: CAT_ENV_WALL, 4: CAT_ENV_WALL, 5: CAT_ENV_WALL,
        6: CAT_HEAD, 7: CAT_TORSO, 8: CAT_WAIST, 9: CAT_WAIST,
        10: CAT_PELVIS, 11: CAT_THIGH_RIGHT, 12: CAT_SHIN_RIGHT,
        13: CAT_FOOT_RIGHT, 14: CAT_FOOT_RIGHT,
        15: CAT_THIGH_LEFT, 16: CAT_SHIN_LEFT,
        17: CAT_FOOT_LEFT, 18: CAT_FOOT_LEFT,
        19: CAT_UPPER_ARM_RIGHT, 20: CAT_LOWER_ARM_RIGHT, 21: CAT_HAND_RIGHT,
        22: CAT_UPPER_ARM_LEFT, 23: CAT_LOWER_ARM_LEFT, 24: CAT_HAND_LEFT,
    }

    # MuJoCo geom name → GEOM Entity ID
    MUJOCO_GEOM_NAME_TO_ENTITY_ID: Dict[str, int] = {}
    for _en, _eid in GEOM_ENTITY_NAME_TO_ID.items():
        MUJOCO_GEOM_NAME_TO_ENTITY_ID[_en] = _eid
        if _en not in ENV_GEOM_NAMES:
            MUJOCO_GEOM_NAME_TO_ENTITY_ID[f'{_en}_a'] = _eid
            MUJOCO_GEOM_NAME_TO_ENTITY_ID[f'{_en}_b'] = _eid
    del _en, _eid

    # === JOINTCAT — 关节类别 (22类/机器人, 0~21) ===
    JOINTCAT_ID_TO_NAME: Dict[int, str] = {
        0: 'root',
        1: 'abdomen_z', 2: 'abdomen_y', 3: 'abdomen_x',
        4: 'hip_x_right', 5: 'hip_z_right', 6: 'hip_y_right',
        7: 'knee_right', 8: 'ankle_y_right', 9: 'ankle_x_right',
        10: 'hip_x_left', 11: 'hip_z_left', 12: 'hip_y_left',
        13: 'knee_left', 14: 'ankle_y_left', 15: 'ankle_x_left',
        16: 'shoulder1_right', 17: 'shoulder2_right', 18: 'elbow_right',
        19: 'shoulder1_left', 20: 'shoulder2_left', 21: 'elbow_left',
    }
    JOINTCAT_NAME_TO_ID: Dict[str, int] = {v: k for k, v in JOINTCAT_ID_TO_NAME.items()}
    NUM_JOINTCATS = 22

    JOINT_SEMANTIC_ID_TO_NAME: Dict[int, str] = {
        0: 'root', 1: 'abdomen', 2: 'hip', 3: 'knee',
        4: 'ankle', 5: 'shoulder', 6: 'elbow',
    }

    JOINTCAT_TO_SEMANTIC: Dict[int, int] = {
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
        """返回 (jointcat_id, aff_id) 或 (None, None)"""
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
        return cls.JOINTCAT_NAME_TO_ID.get(base), aff

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
                errors.append(f"Joint '{name}' cannot be classified to JOINTCAT")
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
        geom_semantic = np.full(ngeom, -1, dtype=np.int8)
        geom_entity_id = np.full(ngeom, -1, dtype=np.int8)
        geom_names: List[str] = []

        for gid in range(ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            geom_names.append(name)
            cat, aff = cls._classify_geom(name)
            if cat is not None:
                geom_cat[gid] = cat
                geom_semantic[gid] = cls.CAT_TO_SEMANTIC[cat]
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
                joint_semantic[jid] = cls.JOINTCAT_TO_SEMANTIC.get(jcat, -1)
            if aff is not None:
                joint_aff[jid] = aff

        return {
            'geom_cat': geom_cat,
            'geom_aff': geom_aff,
            'geom_is_keypoint': geom_is_keypoint,
            'geom_semantic': geom_semantic,
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
    def jointcat_id_to_name(cls, jcat_id: int) -> str:
        return cls.JOINTCAT_ID_TO_NAME[jcat_id]

    @classmethod
    def jointcat_name_to_id(cls, name: str) -> int:
        return cls.JOINTCAT_NAME_TO_ID[name]

    @classmethod
    def semantic_id_to_name(cls, sem_id: int) -> str:
        return cls.SEMANTIC_ID_TO_NAME[sem_id]

    @classmethod
    def cat_to_semantic(cls, cat_id: int) -> int:
        return cls.CAT_TO_SEMANTIC[cat_id]

    @classmethod
    def geom_info(cls, mujoco_geom_name: str) -> Optional[Dict]:
        """查询单个 MuJoCo geom 的完整元数据。"""
        cat, aff = cls._classify_geom(mujoco_geom_name)
        if cat is None:
            return None
        eid = cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID.get(mujoco_geom_name, -1)
        return {
            'mujoco_name': mujoco_geom_name,
            'cat_id': cat,
            'cat_name': cls.CAT_ID_TO_NAME[cat],
            'aff_id': aff,
            'aff_name': cls.AFF_ID_TO_NAME.get(aff, 'unknown'),
            'entity_id': eid,
            'entity_name': cls.GEOM_ENTITY_ID_TO_NAME.get(eid, 'unknown'),
            'is_keypoint': cls._is_keypoint_geom(mujoco_geom_name),
            'semantic_id': cls.CAT_TO_SEMANTIC[cat],
            'semantic_name': cls.SEMANTIC_ID_TO_NAME[cls.CAT_TO_SEMANTIC[cat]],
        }

    @classmethod
    def joint_info(cls, mujoco_joint_name: str) -> Optional[Dict]:
        """查询单个 MuJoCo joint 的完整元数据。"""
        jcat, aff = cls._classify_joint(mujoco_joint_name)
        if jcat is None:
            return None
        return {
            'mujoco_name': mujoco_joint_name,
            'jointcat_id': jcat,
            'jointcat_name': cls.JOINTCAT_ID_TO_NAME[jcat],
            'aff_id': aff,
            'aff_name': cls.AFF_ID_TO_NAME.get(aff, 'unknown'),
            'semantic_id': cls.JOINTCAT_TO_SEMANTIC.get(jcat, -1),
            'semantic_name': cls.JOINT_SEMANTIC_ID_TO_NAME.get(
                cls.JOINTCAT_TO_SEMANTIC.get(jcat, -1), 'unknown'),
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
    def geoms_by_semantic(cls, semantic_id: int) -> List[str]:
        """返回属于指定 Semantic Cat 的所有 MuJoCo geom 名称。"""
        result = []
        for name in cls.MUJOCO_GEOM_NAME_TO_ENTITY_ID:
            cat, _ = cls._classify_geom(name)
            if cat is not None and cls.CAT_TO_SEMANTIC[cat] == semantic_id:
                result.append(name)
        return sorted(result)
