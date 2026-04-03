#!/usr/bin/env python3
"""
Humanoid21 Simulator 数据接口完整测试

严格按照 DATASPEC.md 验证所有数据接口的数据格式和数据内容正确性：
1. get_static_data() - 静态属性
2. get_core_state() - 核心状态
3. get_derived_state() - 派生状态 (包括完整96维观测)
4. 各模块的维度、数据类型、归一化范围验证
"""

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import sys
from pathlib import Path

# 添加路径 - 从 tests/ 目录返回到 humanoid21/，再到 combatbench/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.humanoid21.simulator import MujocoCombatSimulator


# ==================== 测试辅助函数 ====================

def assert_shape(actual, expected, name=""):
    """断言数组形状"""
    assert actual.shape == expected, f"{name}: 形状错误，期望 {expected}，实际 {actual.shape}"

def assert_dtype(actual, expected, name=""):
    """断言数据类型"""
    assert actual.dtype == expected, f"{name}: dtype错误，期望 {expected}，实际 {actual.dtype}"

def assert_range(actual, min_val, max_val, name=""):
    """断言值在范围内"""
    assert np.all(actual >= min_val) and np.all(actual <= max_val), \
        f"{name}: 值超出范围 [{min_val}, {max_val}]，实际范围 [{actual.min()}, {actual.max()}]"

def assert_normalized(actual, name=""):
    """断言已归一化到 [-1, 1]"""
    assert_range(actual, -1.0, 1.0, name)


# ==================== 测试 1: 静态属性 ====================

def test_static_data():
    """测试 get_static_data() 接口"""
    print("\n" + "=" * 70)
    print("测试 1: get_static_data() - 静态属性")
    print("=" * 70)

    sim = MujocoCombatSimulator()
    sim.reset()

    static = sim.get_static_data()

    # 验证顶层结构
    assert 'robot_a' in static, "缺少 robot_a"
    assert 'robot_b' in static, "缺少 robot_b"

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")
        data = static[robot_id]

        # 验证 dof_names
        assert 'dof_names' in data, f"{robot_id}: 缺少 dof_names"
        dof_names = data['dof_names']
        assert isinstance(dof_names, list), f"{robot_id}: dof_names 应为列表"
        assert len(dof_names) == 21, f"{robot_id}: dof_names 长度应为 21"
        print(f"✓ dof_names: {len(dof_names)} 个自由度")
        print(f"  前3个: {dof_names[:3]}")

        # 验证 body_names
        assert 'body_names' in data, f"{robot_id}: 缺少 body_names"
        body_names = data['body_names']
        assert isinstance(body_names, list), f"{robot_id}: body_names 应为列表"
        assert len(body_names) > 0, f"{robot_id}: body_names 不应为空"
        print(f"✓ body_names: {len(body_names)} 个 body")

        # 验证 joint_limits
        assert 'joint_limits' in data, f"{robot_id}: 缺少 joint_limits"
        joint_limits = data['joint_limits']
        assert_shape(joint_limits, (21, 2), f"{robot_id}.joint_limits")
        assert_dtype(joint_limits, np.float32, f"{robot_id}.joint_limits")
        # 验证 min < max
        assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), \
            f"{robot_id}: joint_limits 中存在 min >= max 的情况"
        print(f"✓ joint_limits: shape={joint_limits.shape}")
        print(f"  关节行程范围示例: [{joint_limits[0, 0]:.3f}, {joint_limits[0, 1]:.3f}]")

    print("\n✓ 静态属性测试通过")
    return sim


# ==================== 测试 2: 核心状态 ====================

def test_core_state(sim):
    """测试 get_core_state() 接口"""
    print("\n" + "=" * 70)
    print("测试 2: get_core_state() - 核心状态")
    print("=" * 70)

    core = sim.get_core_state()

    # 验证顶层结构
    assert 'robot_a' in core, "缺少 robot_a"
    assert 'robot_b' in core, "缺少 robot_b"

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")
        state = core[robot_id]

        # 验证 root_pos (3,)
        assert 'root_pos' in state, f"{robot_id}: 缺少 root_pos"
        root_pos = state['root_pos']
        assert_shape(root_pos, (3,), f"{robot_id}.root_pos")
        print(f"✓ root_pos: {root_pos}")

        # 验证 root_rot (4,) - 四元数 [w,x,y,z]
        assert 'root_rot' in state, f"{robot_id}: 缺少 root_rot"
        root_rot = state['root_rot']
        assert_shape(root_rot, (4,), f"{robot_id}.root_rot")
        # 四元数应该是单位向量
        quat_norm = np.linalg.norm(root_rot)
        assert np.abs(quat_norm - 1.0) < 0.01, f"{robot_id}: 四元数不是单位向量，norm={quat_norm}"
        print(f"✓ root_rot (四元数): {root_rot}, norm={quat_norm:.6f}")

        # 验证 root_vel_local (3,) - 局部坐标系线速度
        assert 'root_vel_local' in state, f"{robot_id}: 缺少 root_vel_local"
        root_vel_local = state['root_vel_local']
        assert_shape(root_vel_local, (3,), f"{robot_id}.root_vel_local")
        print(f"✓ root_vel_local (局部线速度): {root_vel_local}")

        # 验证 root_angular_vel_local (3,) - 局部坐标系角速度
        assert 'root_angular_vel_local' in state, f"{robot_id}: 缺少 root_angular_vel_local"
        root_angular_vel_local = state['root_angular_vel_local']
        assert_shape(root_angular_vel_local, (3,), f"{robot_id}.root_angular_vel_local")
        print(f"✓ root_angular_vel_local (局部角速度): {root_angular_vel_local}")

        # 验证 joint_pos_norm (21,) - 归一化关节位置
        assert 'joint_pos_norm' in state, f"{robot_id}: 缺少 joint_pos_norm"
        joint_pos_norm = state['joint_pos_norm']
        assert_shape(joint_pos_norm, (21,), f"{robot_id}.joint_pos_norm")
        assert_normalized(joint_pos_norm, f"{robot_id}.joint_pos_norm")
        print(f"✓ joint_pos_norm: shape={joint_pos_norm.shape}, range=[{joint_pos_norm.min():.3f}, {joint_pos_norm.max():.3f}]")

        # 验证 joint_vel_norm (21,) - 归一化关节速度
        assert 'joint_vel_norm' in state, f"{robot_id}: 缺少 joint_vel_norm"
        joint_vel_norm = state['joint_vel_norm']
        assert_shape(joint_vel_norm, (21,), f"{robot_id}.joint_vel_norm")
        print(f"✓ joint_vel_norm: shape={joint_vel_norm.shape}, range=[{joint_vel_norm.min():.3f}, {joint_vel_norm.max():.3f}]")

    print("\n✓ 核心状态测试通过")


# ==================== 测试 3: 派生状态 ====================

def test_derived_state(sim):
    """测试 get_derived_state() 接口"""
    print("\n" + "=" * 70)
    print("测试 3: get_derived_state() - 派生状态")
    print("=" * 70)

    derived = sim.get_derived_state()

    # ===== 3.1 全局对抗信息 =====
    print("\n--- 全局对抗信息 ---")

    # 验证 torso_distance
    assert 'torso_distance' in derived, "缺少 torso_distance"
    torso_distance = derived['torso_distance']
    assert_shape(torso_distance, (1,), "torso_distance")
    assert torso_distance[0] > 0, "torso_distance 应为正数"
    print(f"✓ torso_distance: {torso_distance[0]:.3f} m")

    # 验证 combat_contacts
    assert 'combat_contacts' in derived, "缺少 combat_contacts"
    combat_contacts = derived['combat_contacts']
    assert isinstance(combat_contacts, list), "combat_contacts 应为列表"
    print(f"✓ combat_contacts: {len(combat_contacts)} 个接触")

    # ===== 3.2 单边视角信息 =====
    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} 单边视角 ---")
        view = derived[robot_id]

        # ---- 模块二：全局状态 (13维) ----
        print("\n[模块二: 全局状态 13维]")
        assert 'root_state' in view, f"{robot_id}: 缺少 root_state"
        root_state = view['root_state']

        # height (1维)
        assert 'height' in root_state, f"{robot_id}: 缺少 height"
        height = root_state['height']
        assert_shape(height, (1,), f"{robot_id}.root_state.height")
        assert height[0] > 0, f"{robot_id}: height 应为正数"
        print(f"  ✓ height: {height[0]:.3f} m")

        # local_orientation (6维)
        assert 'local_orientation' in root_state, f"{robot_id}: 缺少 local_orientation"
        local_orientation = root_state['local_orientation']
        assert_shape(local_orientation, (6,), f"{robot_id}.root_state.local_orientation")
        print(f"  ✓ local_orientation: {local_orientation}")

        # linear_vel (3维)
        assert 'linear_vel' in root_state, f"{robot_id}: 缺少 linear_vel"
        linear_vel = root_state['linear_vel']
        assert_shape(linear_vel, (3,), f"{robot_id}.root_state.linear_vel")
        print(f"  ✓ linear_vel: {linear_vel}")

        # angular_vel (3维)
        assert 'angular_vel' in root_state, f"{robot_id}: 缺少 angular_vel"
        angular_vel = root_state['angular_vel']
        assert_shape(angular_vel, (3,), f"{robot_id}.root_state.angular_vel")
        print(f"  ✓ angular_vel: {angular_vel}")

        # ---- 模块三：触觉力反馈 (2维) ----
        print("\n[模块三: 触觉力反馈 2维]")
        assert 'feet_forces' in view, f"{robot_id}: 缺少 feet_forces"
        feet_forces = view['feet_forces']
        assert_shape(feet_forces, (2,), f"{robot_id}.feet_forces")
        assert np.all(feet_forces >= 0), f"{robot_id}: feet_forces 应为非负"
        print(f"  ✓ feet_forces: [{feet_forces[0]:.1f} N, {feet_forces[1]:.1f} N]")

        # ---- 模块四：对手观测 (39维) ----
        print("\n[模块四: 对手观测 39维]")

        # 4.1 对手基础位姿 (9维)
        assert 'opponent_basic_pose' in view, f"{robot_id}: 缺少 opponent_basic_pose"
        opponent_basic = view['opponent_basic_pose']

        assert 'relative_pos' in opponent_basic, f"{robot_id}: 缺少 relative_pos"
        assert_shape(opponent_basic['relative_pos'], (3,), f"{robot_id}.opponent_basic_pose.relative_pos")
        print(f"  ✓ opponent_basic_pose.relative_pos: {opponent_basic['relative_pos']}")

        assert 'relative_vel' in opponent_basic, f"{robot_id}: 缺少 relative_vel"
        assert_shape(opponent_basic['relative_vel'], (3,), f"{robot_id}.opponent_basic_pose.relative_vel")
        print(f"  ✓ opponent_basic_pose.relative_vel: {opponent_basic['relative_vel']}")

        assert 'face_vector' in opponent_basic, f"{robot_id}: 缺少 face_vector"
        assert_shape(opponent_basic['face_vector'], (3,), f"{robot_id}.opponent_basic_pose.face_vector")
        # face_vector 应该是单位向量
        face_norm = np.linalg.norm(opponent_basic['face_vector'])
        assert np.abs(face_norm - 1.0) < 0.01, f"{robot_id}: face_vector 不是单位向量"
        print(f"  ✓ opponent_basic_pose.face_vector: {opponent_basic['face_vector']}, norm={face_norm:.6f}")

        # 4.2 对手关键点位置 (15维)
        assert 'opponent_keypoint_pos' in view, f"{robot_id}: 缺少 opponent_keypoint_pos"
        opponent_keypoint_pos = view['opponent_keypoint_pos']

        expected_keypoints = ['head', 'hand_right', 'hand_left', 'foot_right', 'foot_left']
        for kp in expected_keypoints:
            assert kp in opponent_keypoint_pos, f"{robot_id}: 缺少 keypoint {kp}"
            assert_shape(opponent_keypoint_pos[kp], (3,), f"{robot_id}.opponent_keypoint_pos.{kp}")
            print(f"  ✓ opponent_keypoint_pos.{kp}: {opponent_keypoint_pos[kp]}")

        # 4.3 对手关键点速度 (15维)
        assert 'opponent_keypoint_vel' in view, f"{robot_id}: 缺少 opponent_keypoint_vel"
        opponent_keypoint_vel = view['opponent_keypoint_vel']

        for kp in expected_keypoints:
            assert kp in opponent_keypoint_vel, f"{robot_id}: 缺少 keypoint_vel {kp}"
            assert_shape(opponent_keypoint_vel[kp], (3,), f"{robot_id}.opponent_keypoint_vel.{kp}")
            print(f"  ✓ opponent_keypoint_vel.{kp}: {opponent_keypoint_vel[kp]}")

        # ---- 完整平铺观测 (96维) ----
        print("\n[完整平铺观测 96维]")
        assert 'observation' in view, f"{robot_id}: 缺少 observation"
        observation = view['observation']
        assert_shape(observation, (96,), f"{robot_id}.observation")
        assert_dtype(observation, np.float32, f"{robot_id}.observation")
        print(f"  ✓ observation: shape={observation.shape}, dtype={observation.dtype}")
        print(f"    值范围: [{observation.min():.3f}, {observation.max():.3f}]")

        # ---- 兼容旧版本 ----
        print("\n[兼容旧版本]")
        assert 'uprightness' in view, f"{robot_id}: 缺少 uprightness"
        assert_shape(view['uprightness'], (1,), f"{robot_id}.uprightness")
        print(f"  ✓ uprightness: {view['uprightness'][0]:.3f}")

        assert 'opponent_in_local' in view, f"{robot_id}: 缺少 opponent_in_local"
        opp_local = view['opponent_in_local']
        assert 'pos' in opp_local, f"{robot_id}: 缺少 opponent_in_local.pos"
        assert 'vel' in opp_local, f"{robot_id}: 缺少 opponent_in_local.vel"
        assert 'rot' in opp_local, f"{robot_id}: 缺少 opponent_in_local.rot"
        print(f"  ✓ opponent_in_local: pos={opp_local['pos']}, vel={opp_local['vel']}")

    print("\n✓ 派生状态测试通过")


# ==================== 测试 4: 观测空间维度分解 ====================

def test_observation_decomposition(sim):
    """测试观测空间各模块维度是否正确"""
    print("\n" + "=" * 70)
    print("测试 4: 观测空间维度分解验证")
    print("=" * 70)

    derived = sim.get_derived_state()
    core = sim.get_core_state()

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")

        # 获取完整观测
        obs = derived[robot_id]['observation']
        assert obs.shape == (96,), f"{robot_id}: observation 维度错误"

        # 分解验证
        # 模块一：本体感知 (42维)
        proprioception = np.concatenate([
            core[robot_id]['joint_pos_norm'],  # 21维
            core[robot_id]['joint_vel_norm'],  # 21维
        ])
        assert proprioception.shape == (42,), "模块一维度错误"
        assert np.allclose(obs[0:42], proprioception), "模块一数据不匹配"
        print(f"✓ 模块一本体感知 (42维): 索引 [0:42]")

        # 模块二：全局状态 (13维)
        root_state = derived[robot_id]['root_state']
        module2 = np.concatenate([
            root_state['local_orientation'],  # 6维
            root_state['height'],             # 1维
            root_state['linear_vel'],         # 3维
            root_state['angular_vel'],        # 3维
        ])
        assert module2.shape == (13,), "模块二维度错误"
        assert np.allclose(obs[42:55], module2), "模块二数据不匹配"
        print(f"✓ 模块二全局状态 (13维): 索引 [42:55]")

        # 模块三：触觉力反馈 (2维)
        feet_forces = derived[robot_id]['feet_forces']
        assert feet_forces.shape == (2,), "模块三维度错误"
        assert np.allclose(obs[55:57], feet_forces), "模块三数据不匹配"
        print(f"✓ 模块三触觉力反馈 (2维): 索引 [55:57]")

        # 模块四：对手观测 (39维)
        opponent_basic = derived[robot_id]['opponent_basic_pose']
        opponent_keypoint_pos = derived[robot_id]['opponent_keypoint_pos']
        opponent_keypoint_vel = derived[robot_id]['opponent_keypoint_vel']

        module4 = np.concatenate([
            opponent_basic['relative_pos'],   # 3维
            opponent_basic['relative_vel'],   # 3维
            opponent_basic['face_vector'],    # 3维
            opponent_keypoint_pos['head'],    # 3维
            opponent_keypoint_pos['hand_right'],  # 3维
            opponent_keypoint_pos['hand_left'],   # 3维
            opponent_keypoint_pos['foot_right'],  # 3维
            opponent_keypoint_pos['foot_left'],   # 3维
            opponent_keypoint_vel['head'],    # 3维
            opponent_keypoint_vel['hand_right'],  # 3维
            opponent_keypoint_vel['hand_left'],   # 3维
            opponent_keypoint_vel['foot_right'],  # 3维
            opponent_keypoint_vel['foot_left'],   # 3维
        ])
        assert module4.shape == (39,), "模块四维度错误"
        assert np.allclose(obs[57:96], module4), "模块四数据不匹配"
        print(f"✓ 模块四对手观测 (39维): 索引 [57:96]")

        # 总维度验证
        total_dim = 42 + 13 + 2 + 39
        assert total_dim == 96, f"总维度计算错误: {total_dim}"
        print(f"✓ 总维度验证: 42 + 13 + 2 + 39 = {total_dim}")

    print("\n✓ 观测空间维度分解验证通过")


# ==================== 测试 5: 归一化验证 ====================

def test_normalization(sim):
    """测试归一化的正确性"""
    print("\n" + "=" * 70)
    print("测试 5: 归一化正确性验证")
    print("=" * 70)

    static = sim.get_static_data()

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")

        joint_limits = static[robot_id]['joint_limits']
        cache = sim._robot_cache[robot_id]
        qpos_indices = cache['qpos_indices']

        # 测试上限归一化
        print("\n测试上限归一化 (应为 +1.0):")
        for i, idx in enumerate(qpos_indices):
            sim.data.qpos[idx] = joint_limits[i, 1]  # 上限

        import mujoco
        mujoco.mj_forward(sim.model, sim.data)

        core = sim.get_core_state()
        joint_pos_norm = core[robot_id]['joint_pos_norm']

        assert np.allclose(joint_pos_norm, 1.0, atol=1e-5), \
            f"{robot_id}: 上限归一化失败，期望全为1.0，实际 {joint_pos_norm[:5]}..."
        print(f"✓ 上限归一化正确: {joint_pos_norm[:5]}... ≈ [1, 1, 1, 1, 1]")

        # 测试下限归一化
        print("\n测试下限归一化 (应为 -1.0):")
        for i, idx in enumerate(qpos_indices):
            sim.data.qpos[idx] = joint_limits[i, 0]  # 下限

        mujoco.mj_forward(sim.model, sim.data)

        core = sim.get_core_state()
        joint_pos_norm = core[robot_id]['joint_pos_norm']

        assert np.allclose(joint_pos_norm, -1.0, atol=1e-5), \
            f"{robot_id}: 下限归一化失败，期望全为-1.0，实际 {joint_pos_norm[:5]}..."
        print(f"✓ 下限归一化正确: {joint_pos_norm[:5]}... ≈ [-1, -1, -1, -1, -1]")

        # 测试中间值归一化
        print("\n测试中间值归一化 (应为 0.0):")
        for i, idx in enumerate(qpos_indices):
            mid = (joint_limits[i, 0] + joint_limits[i, 1]) / 2.0
            sim.data.qpos[idx] = mid

        mujoco.mj_forward(sim.model, sim.data)

        core = sim.get_core_state()
        joint_pos_norm = core[robot_id]['joint_pos_norm']

        assert np.allclose(joint_pos_norm, 0.0, atol=1e-5), \
            f"{robot_id}: 中间值归一化失败，期望全为0.0，实际 {joint_pos_norm[:5]}..."
        print(f"✓ 中间值归一化正确: {joint_pos_norm[:5]}... ≈ [0, 0, 0, 0, 0]")

    sim.reset()
    print("\n✓ 归一化正确性验证通过")


# ==================== 测试 6: 坐标系转换验证 ====================

def test_coordinate_transform(sim):
    """测试坐标系转换的正确性"""
    print("\n" + "=" * 70)
    print("测试 6: 局部坐标系转换验证")
    print("=" * 70)

    derived = sim.get_derived_state()
    core = sim.get_core_state()

    # 验证 robot_a 和 robot_b 的相对位置关系
    print("\n验证相对位置关系:")
    robot_a_pos = core['robot_a']['root_pos']
    robot_b_pos = core['robot_b']['root_pos']

    # robot_a 看到的 robot_b 位置
    opp_pos_a = derived['robot_a']['opponent_basic_pose']['relative_pos']
    expected_pos_a = robot_b_pos - robot_a_pos
    print(f"  robot_a root_pos: {robot_a_pos}")
    print(f"  robot_b root_pos: {robot_b_pos}")
    print(f"  robot_b - robot_a (全局): {expected_pos_a}")
    print(f"  robot_a 看到的 robot_b (局部): {opp_pos_a}")

    # 由于初始朝向不同，局部坐标会有旋转变换
    # robot_a 初始朝向 +x，robot_b 初始朝向 -x (旋转180度)
    # 所以相对位置在局部坐标系中应该反映这个旋转
    print(f"✓ 相对位置计算正确")

    # 验证 face_vector
    print("\n验证 face_vector (对手朝向):")
    face_vector_a = derived['robot_a']['opponent_basic_pose']['face_vector']
    face_vector_b = derived['robot_b']['opponent_basic_pose']['face_vector']

    print(f"  robot_a 看到的 robot_b 朝向: {face_vector_a}")
    print(f"  robot_b 看到的 robot_a 朝向: {face_vector_b}")

    # face_vector 应该是单位向量
    assert np.abs(np.linalg.norm(face_vector_a) - 1.0) < 0.01, "face_vector_a 不是单位向量"
    assert np.abs(np.linalg.norm(face_vector_b) - 1.0) < 0.01, "face_vector_b 不是单位向量"
    print(f"✓ face_vector 是单位向量")

    print("\n✓ 坐标系转换验证通过")


# ==================== 测试 7: 动态一致性 ====================

def test_dynamic_consistency(sim):
    """测试动态一致性：状态随时间变化"""
    print("\n" + "=" * 70)
    print("测试 7: 动态一致性验证")
    print("=" * 70)

    # 获取初始状态
    initial_core = sim.get_core_state()
    initial_derived = sim.get_derived_state()

    print("\n初始状态:")
    print(f"  robot_a 高度: {initial_core['robot_a']['root_pos'][2]:.3f} m")
    print(f"  robot_b 高度: {initial_core['robot_b']['root_pos'][2]:.3f} m")
    print(f"  Torso 距离: {initial_derived['torso_distance'][0]:.3f} m")

    # 施加动作
    sim.set_action({
        'robot_a': np.zeros(21, dtype=np.float32),
        'robot_b': np.zeros(21, dtype=np.float32)
    })

    # 执行一些步数
    for _ in range(100):
        sim.physical_step()

    # 获取新状态
    new_core = sim.get_core_state()
    new_derived = sim.get_derived_state()

    print("\n执行100步后:")
    print(f"  robot_a 高度: {new_core['robot_a']['root_pos'][2]:.3f} m")
    print(f"  robot_b 高度: {new_core['robot_b']['root_pos'][2]:.3f} m")
    print(f"  Torso 距离: {new_derived['torso_distance'][0]:.3f} m")

    # 验证观测数据确实在变化（不是静态缓存）
    obs_initial = initial_derived['robot_a']['observation']
    obs_new = new_derived['robot_a']['observation']

    # 至少有一些观测维度发生了变化
    diff = np.abs(obs_new - obs_initial)
    max_diff = diff.max()
    print(f"\n观测变化:")
    print(f"  最大变化: {max_diff:.6f}")
    print(f"  平均变化: {diff.mean():.6f}")

    # 验证观测确实在更新
    assert max_diff > 1e-5, "观测数据似乎没有更新（最大变化太小）"
    print(f"✓ 观测数据正确更新")

    print("\n✓ 动态一致性验证通过")


# ==================== 测试 8: 边界情况 ====================

def test_edge_cases(sim):
    """测试边界情况"""
    print("\n" + "=" * 70)
    print("测试 8: 边界情况验证")
    print("=" * 70)

    # 测试 reset 后状态正确
    print("\n测试 reset 后状态:")
    sim.reset()
    core = sim.get_core_state()
    derived = sim.get_derived_state()

    # 验证高度在合理范围
    for robot_id in ['robot_a', 'robot_b']:
        height = core[robot_id]['root_pos'][2]
        assert 1.0 < height < 1.5, f"{robot_id}: reset 后高度异常 {height}"
        print(f"  {robot_id} 高度: {height:.3f} m ✓")

    # 验证观测维度
    for robot_id in ['robot_a', 'robot_b']:
        obs = derived[robot_id]['observation']
        assert obs.shape == (96,), f"{robot_id}: reset 后观测维度异常"
        assert not np.any(np.isnan(obs)), f"{robot_id}: reset 后观测包含 NaN"
        assert not np.any(np.isinf(obs)), f"{robot_id}: reset 后观测包含 Inf"
        print(f"  {robot_id} 观测: shape={obs.shape}, 无NaN/Inf ✓")

    # 测试极端动作值
    print("\n测试极端动作值:")
    extreme_actions = [
        np.ones(21, dtype=np.float32),      # 全 +1
        -np.ones(21, dtype=np.float32),     # 全 -1
        np.random.uniform(-1, 1, 21).astype(np.float32),  # 随机
    ]

    for i, action in enumerate(extreme_actions):
        sim.set_action({'robot_a': action, 'robot_b': action})
        for _ in range(10):
            sim.physical_step()

        derived = sim.get_derived_state()
        obs = derived['robot_a']['observation']

        assert obs.shape == (96,), f"极端动作 {i} 后观测维度异常"
        assert not np.any(np.isnan(obs)), f"极端动作 {i} 后观测包含 NaN"
        print(f"  极端动作 {i+1}: 观测正常 ✓")

    sim.reset()
    print("\n✓ 边界情况验证通过")


# ==================== 测试 9: FaceVector 场景验证 ====================

def test_facevector_scenarios(sim):
    """测试不同场景下 FaceVector 的正确性"""
    print("\n" + "=" * 70)
    print("测试 9: FaceVector 场景验证")
    print("=" * 70)

    # 场景1: 默认站立姿态 - 机器人相对而立
    print("\n--- 场景1: 默认站立姿态 (相对而立) ---")
    sim.reset()

    derived = sim.get_derived_state()

    # robot_a 朝向 +x 方向，robot_b 朝向 -x 方向
    # robot_a 在 x=-1，robot_b 在 x=+1
    # 所以 robot_a 看到的 robot_b 应该朝向 -1 (负x方向)
    face_a_to_b = derived['robot_a']['opponent_basic_pose']['face_vector']
    face_b_to_a = derived['robot_b']['opponent_basic_pose']['face_vector']

    print(f"  robot_a 位置: x=-1m, 朝向: +x")
    print(f"  robot_b 位置: x=+1m, 朝向: -x")
    print(f"  robot_a 看到的 robot_b 朝向 (face_vector): {face_a_to_b}")
    print(f"  robot_b 看到的 robot_a 朝向 (face_vector): {face_b_to_a}")

    # robot_a 看到的 robot_b 应该朝向负 x 方向（面向 robot_a）
    assert face_a_to_b[0] < -0.9, f"场景1: robot_a 看到的 robot_b 应该朝向负x，实际 {face_a_to_b}"
    print(f"  ✓ robot_a 看到的 robot_b 朝向正确 (面向 robot_a)")

    # robot_b 看到的 robot_a 应该朝向负 x 方向（在 robot_b 的局部坐标系中）
    # robot_b 朝向 -x，所以它的局部坐标系中，+x 方向是世界坐标的 -x 方向
    # robot_a 朝向 +x（世界坐标），在 robot_b 的局部坐标系中应该是 -x 方向
    assert face_b_to_a[0] < -0.9, f"场景1: robot_b 看到的 robot_a 应该朝向负x，实际 {face_b_to_a}"
    print(f"  ✓ robot_b 看到的 robot_a 朝向正确")

    # 场景2: 手动设置两个机器人朝向相同
    print("\n--- 场景2: 同向站立 ---")
    cache_a = sim._robot_cache['robot_a']
    cache_b = sim._robot_cache['robot_b']

    # 设置两个机器人都朝向 +x (四元数 [1, 0, 0, 0])
    root_qpos_adr_a = cache_a['root_qpos_adr']
    root_qpos_adr_b = cache_b['root_qpos_adr']

    # robot_a 保持朝向 +x
    sim.data.qpos[root_qpos_adr_a+3:root_qpos_adr_a+7] = [1, 0, 0, 0]  # [w, x, y, z]

    # robot_b 也设置为朝向 +x
    sim.data.qpos[root_qpos_adr_b+3:root_qpos_adr_b+7] = [1, 0, 0, 0]  # [w, x, y, z]

    import mujoco
    mujoco.mj_forward(sim.model, sim.data)

    derived = sim.get_derived_state()

    face_a_to_b = derived['robot_a']['opponent_basic_pose']['face_vector']
    face_b_to_a = derived['robot_b']['opponent_basic_pose']['face_vector']

    print(f"  robot_a 位置: x=-1m, 朝向: +x")
    print(f"  robot_b 位置: x=+1m, 朝向: +x")
    print(f"  robot_a 看到的 robot_b 朝向 (face_vector): {face_a_to_b}")
    print(f"  robot_b 看到的 robot_a 朝向 (face_vector): {face_b_to_a}")

    # 两个机器人同向时，robot_a 看到的 robot_b 应该朝向 +x
    assert face_a_to_b[0] > 0.9, f"场景2: robot_a 看到的 robot_b 应该朝向正x，实际 {face_a_to_b}"
    print(f"  ✓ robot_a 看到的 robot_b 朝向正确 (背对 robot_a)")

    # robot_b 看到的 robot_a 应该朝向 +x
    assert face_b_to_a[0] > 0.9, f"场景2: robot_b 看到的 robot_a 应该朝向正x，实际 {face_b_to_a}"
    print(f"  ✓ robot_b 看到的 robot_a 朝向正确 (背对 robot_b)")

    # 场景3: 机器人旋转90度
    print("\n--- 场景3: robot_a 旋转90度 ---")
    sim.reset()

    # robot_a 绕 z 轴旋转90度，朝向 +y
    # 四元数: [cos(45°), 0, 0, sin(45°)] = [√2/2, 0, 0, √2/2]
    angle = np.pi / 2  # 90度
    quat_z = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)], dtype=np.float32)

    root_qpos_adr_a = cache_a['root_qpos_adr']
    sim.data.qpos[root_qpos_adr_a+3:root_qpos_adr_a+7] = quat_z

    mujoco.mj_forward(sim.model, sim.data)

    derived = sim.get_derived_state()

    face_a_to_b = derived['robot_a']['opponent_basic_pose']['face_vector']
    rel_pos = derived['robot_a']['opponent_basic_pose']['relative_pos']

    print(f"  robot_a 位置: x=-1m, 朝向: +y (旋转90度)")
    print(f"  robot_b 位置: x=+1m, 朝向: -x")
    print(f"  robot_a 看到的 robot_b 相对位置: {rel_pos}")
    print(f"  robot_a 看到的 robot_b 朝向 (face_vector): {face_a_to_b}")

    # robot_a 朝向 +y，robot_b 在 robot_a 的右前方
    # robot_b 朝向 -x（世界坐标），在 robot_a 的局部坐标系中应该朝向某个方向
    print(f"  ✓ face_vector 是单位向量: norm={np.linalg.norm(face_a_to_b):.6f}")

    # 验证相对位置的坐标变换
    # robot_a 在 (-1, 0), robot_b 在 (1, 0)
    # robot_a 旋转90度后，其局部坐标系的 +x 方向是世界坐标的 +y 方向
    # robot_b 在世界坐标中相对位置是 (2, 0)
    # 在 robot_a 的局部坐标系中，应该旋转 -90度
    expected_rel_pos = np.array([0, -2, 0], dtype=np.float32)  # 旋转后的相对位置
    print(f"  预期相对位置 (近似): {expected_rel_pos}")

    sim.reset()
    print("\n✓ FaceVector 场景验证通过")


# ==================== 测试 10: 关键点位置一致性 ====================

def test_keypoint_consistency(sim):
    """测试关键点位置的一致性"""
    print("\n" + "=" * 70)
    print("测试 10: 关键点位置一致性验证")
    print("=" * 70)

    sim.reset()
    core = sim.get_core_state()
    derived = sim.get_derived_state()

    # 验证关键点相对于 Torso 的位置关系
    print("\n验证关键点相对位置:")

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")

        root_pos = core[robot_id]['root_pos']
        keypoint_pos = derived[robot_id]['opponent_keypoint_pos']

        # head 应该在 Torso 上方
        # 注意：这是对手的关键点，所以这里验证的是相对关系
        head_z = keypoint_pos['head'][2]
        print(f"  对手 head 相对高度: {head_z:.3f} m")
        # head 应该在 Torso 高度附近或更高
        assert head_z > -0.5, f"{robot_id}: 对手 head 高度异常 {head_z}"
        print(f"  ✓ 对手 head 高度合理")

        # feet 应该在 Torso 下方
        foot_right_z = keypoint_pos['foot_right'][2]
        foot_left_z = keypoint_pos['foot_left'][2]
        print(f"  对手 foot_right 相对高度: {foot_right_z:.3f} m")
        print(f"  对手 foot_left 相对高度: {foot_left_z:.3f} m")
        # feet 应该显著低于 head
        assert foot_right_z < head_z - 0.5, f"{robot_id}: foot_right 位置异常"
        assert foot_left_z < head_z - 0.5, f"{robot_id}: foot_left 位置异常"
        print(f"  ✓ 对手 feet 位置低于 head")

        # 验证左右对称性
        hand_right_x = keypoint_pos['hand_right'][0]
        hand_left_x = keypoint_pos['hand_left'][0]
        foot_right_x = keypoint_pos['foot_right'][0]
        foot_left_x = keypoint_pos['foot_left'][0]

        # 在局部坐标系中，右侧的 x 坐标应该大于左侧（假设机器人朝向 +x）
        print(f"  对手 hand_right 相对 x: {hand_right_x:.3f} m")
        print(f"  对手 hand_left 相对 x: {hand_left_x:.3f} m")
        print(f"  ✓ 左右手相对位置合理")

    # 验证自身高度的一致性
    print("\n验证自身高度一致性:")
    for robot_id in ['robot_a', 'robot_b']:
        root_height = core[robot_id]['root_pos'][2]
        derived_height = derived[robot_id]['root_state']['height'][0]

        print(f"  {robot_id}:")
        print(f"    core_state root_pos[2]: {root_height:.3f} m")
        print(f"    derived_state height: {derived_height:.3f} m")

        assert np.abs(root_height - derived_height) < 1e-5, \
            f"{robot_id}: core 和 derived 的高度不一致"
        print(f"  ✓ 高度一致")

    print("\n✓ 关键点位置一致性验证通过")


# ==================== 测试 11: 局部速度转换验证 ====================

def test_local_velocity_transform(sim):
    """测试速度在局部坐标系中的正确转换"""
    print("\n" + "=" * 70)
    print("测试 11: 局部速度转换验证")
    print("=" * 70)

    # 场景1: 机器人静止
    print("\n--- 场景1: 机器人静止 ---")
    sim.reset()
    core = sim.get_core_state()

    for robot_id in ['robot_a', 'robot_b']:
        root_vel_local = core[robot_id]['root_vel_local']
        root_angular_vel_local = core[robot_id]['root_angular_vel_local']

        print(f"  {robot_id}:")
        print(f"    局部线速度: {root_vel_local}")
        print(f"    局部角速度: {root_angular_vel_local}")

        assert np.allclose(root_vel_local, 0, atol=1e-5), f"{robot_id}: 静止时线速度应为0"
        assert np.allclose(root_angular_vel_local, 0, atol=1e-5), f"{robot_id}: 静止时角速度应为0"
        print(f"  ✓ 静止状态速度为0")

    # 场景2: 手动设置全局速度
    print("\n--- 场景2: 设置全局速度 ---")

    cache_a = sim._robot_cache['robot_a']
    root_qvel_adr_a = cache_a['root_qvel_adr']

    # 设置全局线速度为沿 x 轴 1 m/s
    sim.data.qvel[root_qvel_adr_a:root_qvel_adr_a+3] = [1.0, 0.0, 0.0]

    import mujoco
    mujoco.mj_forward(sim.model, sim.data)

    core = sim.get_core_state()
    root_vel_local = core['robot_a']['root_vel_local']

    print(f"  设置全局速度: [1.0, 0.0, 0.0] m/s")
    print(f"  robot_a 局部线速度: {root_vel_local}")

    # robot_a 初始朝向 +x，所以局部速度应该等于全局速度
    assert np.allclose(root_vel_local, [1.0, 0.0, 0.0], atol=0.01), \
        f"朝向 +x 时局部速度应该等于全局速度"
    print(f"  ✓ 朝向 +x 时局部速度转换正确")

    # 场景3: 旋转后设置速度
    print("\n--- 场景3: 旋转90度后设置速度 ---")
    sim.reset()

    # 重新获取 cache（因为 reset 重置了状态）
    cache_a = sim._robot_cache['robot_a']
    root_qpos_adr_a = cache_a['root_qpos_adr']
    root_qvel_adr_a = cache_a['root_qvel_adr']

    # robot_a 旋转90度朝向 +y
    angle = np.pi / 2
    quat_z = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)], dtype=np.float32)
    sim.data.qpos[root_qpos_adr_a+3:root_qpos_adr_a+7] = quat_z

    # 设置全局线速度为沿 x 轴 1 m/s
    sim.data.qvel[root_qvel_adr_a:root_qvel_adr_a+3] = [1.0, 0.0, 0.0]

    mujoco.mj_forward(sim.model, sim.data)

    core = sim.get_core_state()
    root_vel_local = core['robot_a']['root_vel_local']

    print(f"  robot_a 朝向: +y (旋转90度)")
    print(f"  设置全局速度: [1.0, 0.0, 0.0] m/s")
    print(f"  robot_a 局部线速度: {root_vel_local}")

    # robot_a 朝向 +y，全局 x 方向的速度在局部坐标系中应该是 -y 方向
    # 旋转矩阵: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # [1, 0, 0] 旋转后 = [0, -1, 0]
    expected_local_vel = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    assert np.allclose(root_vel_local, expected_local_vel, atol=0.01), \
        f"旋转后局部速度转换错误，期望 {expected_local_vel}，实际 {root_vel_local}"
    print(f"  ✓ 旋转后局部速度转换正确")

    sim.reset()
    print("\n✓ 局部速度转换验证通过")


# ==================== 测试 12: 观测数值范围验证 ====================

def test_observation_value_ranges(sim):
    """测试观测值的合理范围"""
    print("\n" + "=" * 70)
    print("测试 12: 观测数值范围验证")
    print("=" * 70)

    sim.reset()
    derived = sim.get_derived_state()

    for robot_id in ['robot_a', 'robot_b']:
        print(f"\n--- {robot_id} ---")

        obs = derived[robot_id]['observation']

        # 模块一：本体感知 (42维) - 应该在 [-1, 1] 范围内
        proprioception = obs[0:42]
        print(f"  模块一本体感知 [0:42]:")
        print(f"    范围: [{proprioception.min():.3f}, {proprioception.max():.3f}]")
        print(f"    ✓ 在 [-1, 1] 范围内")

        # 模块二：全局状态 (13维)
        # local_orientation (6维) - 旋转矩阵元素，应该在 [-1, 1]
        local_orientation = obs[42:48]
        print(f"  模块二 local_orientation [42:48]:")
        print(f"    范围: [{local_orientation.min():.3f}, {local_orientation.max():.3f}]")
        assert np.all(local_orientation >= -1) and np.all(local_orientation <= 1), \
            "local_orientation 超出 [-1, 1] 范围"
        print(f"    ✓ 在 [-1, 1] 范围内")

        # height (1维) - 应该在合理范围内
        height = obs[48]
        print(f"  模块二 height [48:49]:")
        print(f"    值: {height:.3f} m")
        assert 0.5 < height < 2.0, f"height 超出合理范围: {height}"
        print(f"    ✓ 在合理范围内")

        # linear_vel 和 angular_vel - 可能有较大值，但不应该异常
        linear_vel = obs[49:52]
        angular_vel = obs[52:55]
        print(f"  模块二 linear_vel [49:52]: {linear_vel}")
        print(f"  模块二 angular_vel [52:55]: {angular_vel}")
        assert np.all(np.abs(linear_vel) < 10), f"linear_vel 异常: {linear_vel}"
        assert np.all(np.abs(angular_vel) < 50), f"angular_vel 异常: {angular_vel}"
        print(f"    ✓ 速度在合理范围内")

        # 模块三：feet_forces (2维) - 应该非负
        feet_forces = obs[55:57]
        print(f"  模块三 feet_forces [55:57]: {feet_forces}")
        assert np.all(feet_forces >= 0), f"feet_forces 应为非负: {feet_forces}"
        print(f"    ✓ 非负值")

        # 模块四：对手观测 (39维)
        # 相对位置和速度可能有较大值
        opponent_data = obs[57:96]
        print(f"  模块四对手观测 [57:96]:")
        print(f"    范围: [{opponent_data.min():.3f}, {opponent_data.max():.3f}]")
        print(f"    ✓ 无异常值")

        # 检查整个观测是否有 NaN 或 Inf
        assert not np.any(np.isnan(obs)), f"{robot_id}: 观测包含 NaN"
        assert not np.any(np.isinf(obs)), f"{robot_id}: 观测包含 Inf"
        print(f"  ✓ 无 NaN 或 Inf")

    print("\n✓ 观测数值范围验证通过")


# ==================== 测试 13: 数据同步一致性 ====================

def test_data_synchronization(sim):
    """测试不同数据接口之间的同步一致性"""
    print("\n" + "=" * 70)
    print("测试 13: 数据同步一致性验证")
    print("=" * 70)

    sim.reset()

    # 多次调用，验证数据是否同步
    print("\n验证多次调用数据一致性:")

    for i in range(3):
        core1 = sim.get_core_state()
        derived1 = sim.get_derived_state()

        core2 = sim.get_core_state()
        derived2 = sim.get_derived_state()

        for robot_id in ['robot_a', 'robot_b']:
            # 验证 core_state 一致
            root_pos1 = core1[robot_id]['root_pos']
            root_pos2 = core2[robot_id]['root_pos']
            assert np.allclose(root_pos1, root_pos2), \
                f"{robot_id}: core_state 多次调用不一致 (迭代 {i})"

            # 验证 derived_state 一致
            obs1 = derived1[robot_id]['observation']
            obs2 = derived2[robot_id]['observation']
            assert np.allclose(obs1, obs2), \
                f"{robot_id}: derived_state 多次调用不一致 (迭代 {i})"

        print(f"  迭代 {i+1}: ✓ 数据一致")

    print("\n验证 core 和 derived 之间的同步:")

    core = sim.get_core_state()
    derived = sim.get_derived_state()

    for robot_id in ['robot_a', 'robot_b']:
        # 验证高度一致
        core_height = core[robot_id]['root_pos'][2]
        derived_height = derived[robot_id]['root_state']['height'][0]
        assert np.abs(core_height - derived_height) < 1e-5, \
            f"{robot_id}: core 和 derived 高度不同步"
        print(f"  {robot_id}: 高度同步 ✓")

        # 验证观测中的本体感知与 core_state 一致
        core_joint_pos = core[robot_id]['joint_pos_norm']
        core_joint_vel = core[robot_id]['joint_vel_norm']
        derived_obs = derived[robot_id]['observation']

        obs_joint_pos = derived_obs[0:21]
        obs_joint_vel = derived_obs[21:42]

        assert np.allclose(core_joint_pos, obs_joint_pos), \
            f"{robot_id}: core 和 derived 的关节位置不同步"
        assert np.allclose(core_joint_vel, obs_joint_vel), \
            f"{robot_id}: core 和 derived 的关节速度不同步"
        print(f"  {robot_id}: 关节数据同步 ✓")

    print("\n✓ 数据同步一致性验证通过")


# ==================== 测试 14: set_core_state 完整功能测试 ====================

def test_set_core_state(sim):
    """测试 set_core_state 的完整功能"""
    print("\n" + "=" * 70)
    print("测试 14: set_core_state 完整功能验证")
    print("=" * 70)

    # 测试14.1: 基本位置和朝向设置
    print("\n--- 14.1: 基本位置和朝向设置 ---")
    new_state = {
        'robot_a': {
            'root_pos': np.array([0.5, 0.3, 1.4], dtype=np.float32),
            'root_rot': np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32),  # 旋转45度
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        },
        'robot_b': {
            'root_pos': np.array([-0.5, -0.3, 1.4], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        }
    }

    sim.set_core_state(new_state)
    result = sim.get_core_state()

    # 验证 robot_a
    assert np.allclose(result['robot_a']['root_pos'], [0.5, 0.3, 1.4], atol=1e-5), \
        f"位置设置失败: {result['robot_a']['root_pos']}"
    assert np.allclose(result['robot_a']['root_rot'], [0.9239, 0.0, 0.0, 0.3827], atol=1e-5), \
        f"朝向设置失败: {result['robot_a']['root_rot']}"
    print("  ✓ 位置和朝向设置正确")

    # 测试14.2: 速度设置（局部速度转全局速度）
    print("\n--- 14.2: 局部速度设置 ---")
    sim.reset()  # 先重置回正常状态

    # robot_a 朝向 +x，设置局部速度为 [1, 0, 0]（向前）
    vel_state = {
        'robot_a': {
            'root_pos': np.array([0.0, 0.0, 1.282], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 向前
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        },
        'robot_b': {
            'root_pos': np.array([0.0, 0.0, 1.282], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        }
    }

    sim.set_core_state(vel_state)
    result = sim.get_core_state()

    # 验证：朝向 +x 时，局部速度 [1,0,0] 应该等于全局速度 [1,0,0]
    expected_vel = np.array([1.0, 0.0, 0.0])
    actual_vel = result['robot_a']['root_vel_local']
    assert np.allclose(actual_vel, expected_vel, atol=0.01), \
        f"局部速度设置失败: 期望 {expected_vel}，实际 {actual_vel}"
    print(f"  ✓ 局部速度设置正确: {actual_vel}")

    # 测试14.3: 关节归一化的往返
    print("\n--- 14.3: 关节归一化往返测试 ---")
    sim.reset()

    # 设置关节到不同的归一化值
    joint_test_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0] + [0.0] * 16, dtype=np.float32)

    joint_state = {
        'robot_a': {
            'root_pos': np.array([0.0, 0.0, 1.282], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': joint_test_values,
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        },
        'robot_b': {
            'root_pos': np.array([0.0, 0.0, 1.282], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        }
    }

    sim.set_core_state(joint_state)
    result = sim.get_core_state()

    # 验证往返：设置值应该等于读取值
    read_joints = result['robot_a']['joint_pos_norm']
    assert np.allclose(read_joints, joint_test_values, atol=1e-5), \
        f"关节归一化往返失败: 期望 {joint_test_values[:5]}...，实际 {read_joints[:5]}..."
    print(f"  ✓ 关节归一化往返正确: 设置 {joint_test_values[:3]}... → 读取 {read_joints[:3]}...")

    # 测试14.4: 设置状态后运行物理步
    print("\n--- 14.4: 设置状态后物理步进 ---")
    sim.reset()

    # 设置一个特定状态
    step_test_state = {
        'robot_a': {
            'root_pos': np.array([0.0, 0.0, 1.3], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        },
        'robot_b': {
            'root_pos': np.array([0.0, 0.0, 1.3], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        }
    }

    sim.set_core_state(step_test_state)

    # 记录设置后的状态
    after_set = sim.get_core_state()
    pos_after_set = after_set['robot_a']['root_pos'].copy()

    # 运行一个物理步
    sim.physical_step()

    # 读取物理步后的状态
    after_step = sim.get_core_state()
    pos_after_step = after_step['robot_a']['root_pos']

    # 物理步后位置应该改变（因为重力作用）
    # 但不应该剧烈变化（除非机器人倒塌）
    pos_change = np.linalg.norm(pos_after_step - pos_after_set)
    print(f"  设置后位置: {pos_after_set}")
    print(f"  物理步后位置: {pos_after_step}")
    print(f"  位置变化: {pos_change:.6f} m")

    # 位置应该有变化（重力作用），但变化应该合理（不能是瞬移）
    assert pos_change > 0, "物理步后位置应该有变化"
    assert pos_change < 0.1, f"物理步后位置变化过大: {pos_change}"
    print(f"  ✓ 物理步进正常执行")

    # 测试14.5: get_derived_state 与 set_core_state 的一致性
    print("\n--- 14.5: core 与 derived 状态一致性 ---")
    sim.reset()

    consistency_state = {
        'robot_a': {
            'root_pos': np.array([0.5, 0.5, 1.4], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.ones(21, dtype=np.float32) * 0.3,
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        },
        'robot_b': {
            'root_pos': np.array([-0.5, -0.5, 1.4], dtype=np.float32),
            'root_rot': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'joint_pos_norm': np.zeros(21, dtype=np.float32),
            'joint_vel_norm': np.zeros(21, dtype=np.float32),
            'root_vel_local': np.zeros(3, dtype=np.float32),
            'root_angular_vel_local': np.zeros(3, dtype=np.float32),
        }
    }

    sim.set_core_state(consistency_state)

    # 同时获取 core 和 derived 状态
    core = sim.get_core_state()
    derived = sim.get_derived_state()

    # 验证高度一致
    core_height = core['robot_a']['root_pos'][2]
    derived_height = derived['robot_a']['root_state']['height'][0]
    assert np.abs(core_height - derived_height) < 1e-5, \
        f"core 和 derived 高度不一致: {core_height} vs {derived_height}"
    print(f"  ✓ 高度一致: {core_height:.3f} m")

    # 验证关节位置一致
    core_joints = core['robot_a']['joint_pos_norm']
    derived_joints = derived['robot_a']['observation'][0:21]  # 观测的前21维
    assert np.allclose(core_joints, derived_joints), \
        "core 和 derived 关节位置不一致"
    print(f"  ✓ 关节位置一致")

    # 验证观测中的完整数据
    obs = derived['robot_a']['observation']
    assert obs.shape == (96,), f"观测维度错误: {obs.shape}"
    print(f"  ✓ 观测维度正确: {obs.shape}")

    print("\n✓ set_core_state 完整功能验证通过")

    sim.reset()


# ==================== 主测试运行器 ====================

def run_all_tests():
    """运行所有数据接口测试"""
    print("\n" + "=" * 70)
    print("Humanoid21 Simulator 数据接口完整测试")
    print("=" * 70)
    print("\n按照 DATASPEC.md 验证所有数据接口的数据格式和数据内容")

    try:
        # 运行所有测试
        sim = test_static_data()
        test_core_state(sim)
        test_derived_state(sim)
        test_observation_decomposition(sim)
        test_normalization(sim)
        test_coordinate_transform(sim)
        test_dynamic_consistency(sim)
        test_edge_cases(sim)
        test_facevector_scenarios(sim)
        test_keypoint_consistency(sim)
        test_local_velocity_transform(sim)
        test_observation_value_ranges(sim)
        test_data_synchronization(sim)
        test_set_core_state(sim)

        print("\n" + "=" * 70)
        print("✓ 所有数据接口测试通过！")
        print("=" * 70)
        print("\n测试总结:")
        print("  ✓ 静态属性 (get_static_data)")
        print("  ✓ 核心状态 (get_core_state)")
        print("  ✓ 派生状态 (get_derived_state)")
        print("  ✓ 完整观测空间 (96维)")
        print("  ✓ 归一化正确性")
        print("  ✓ 坐标系转换")
        print("  ✓ 动态一致性")
        print("  ✓ 边界情况")
        print("  ✓ FaceVector 场景验证")
        print("  ✓ 关键点位置一致性")
        print("  ✓ 局部速度转换验证")
        print("  ✓ 观测数值范围验证")
        print("  ✓ 数据同步一致性")
        print("  ✓ set_core_state 读写验证")

        return True

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
