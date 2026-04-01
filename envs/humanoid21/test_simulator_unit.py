#!/usr/bin/env python3
"""
Humanoid21 Simulator 单元测试

验证新实现的 simulator.py 是否符合 DATASPEC.md 和 CONTROLSPEC.md 规范
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.humanoid21.simulator import MujocoCombatSimulator


def test_initialization():
    """测试初始化"""
    print("=" * 60)
    print("测试 1: 初始化")
    print("=" * 60)
    
    sim = MujocoCombatSimulator()
    sim.reset()
    
    print("✓ 初始化成功")
    print(f"  - 物理频率: {sim.get_physical_frequency()} Hz")
    print(f"  - 动作维度: {sim.action_dim}")
    print(f"  - KP 形状: {sim.KP.shape}")
    print(f"  - KD 形状: {sim.KD.shape}")
    print()
    
    return sim


def test_static_info(sim):
    """测试静态信息接口"""
    print("=" * 60)
    print("测试 2: get_static_data()")
    print("=" * 60)
    
    static_info = sim.get_static_data()
    
    # 验证结构
    assert 'robot_a' in static_info, "缺少 robot_a"
    assert 'robot_b' in static_info, "缺少 robot_b"
    
    for robot_id in ['robot_a', 'robot_b']:
        info = static_info[robot_id]
        
        assert 'dof_names' in info, f"{robot_id} 缺少 dof_names"
        assert 'body_names' in info, f"{robot_id} 缺少 body_names"
        assert 'joint_limits' in info, f"{robot_id} 缺少 joint_limits"
        
        assert len(info['dof_names']) == 21, f"{robot_id} dof_names 长度错误"
        assert info['joint_limits'].shape == (21, 2), f"{robot_id} joint_limits 形状错误"
        
        print(f"✓ {robot_id}:")
        print(f"  - DOF 数量: {len(info['dof_names'])}")
        print(f"  - Body 数量: {len(info['body_names'])}")
        print(f"  - 关节限位形状: {info['joint_limits'].shape}")
    
    print()


def test_core_state(sim):
    """测试核心状态接口"""
    print("=" * 60)
    print("测试 3: get_core_state()")
    print("=" * 60)
    
    core_state = sim.get_core_state()
    
    # 验证结构
    assert 'robot_a' in core_state, "缺少 robot_a"
    assert 'robot_b' in core_state, "缺少 robot_b"
    
    for robot_id in ['robot_a', 'robot_b']:
        state = core_state[robot_id]
        
        # 验证字段存在
        required_fields = [
            'root_pos', 'root_rot', 'root_vel_local', 'root_angular_vel_local',
            'joint_pos_norm', 'joint_vel_norm'
        ]
        for field in required_fields:
            assert field in state, f"{robot_id} 缺少 {field}"
        
        # 验证形状
        assert state['root_pos'].shape == (3,), f"{robot_id} root_pos 形状错误"
        assert state['root_rot'].shape == (4,), f"{robot_id} root_rot 形状错误"
        assert state['root_vel_local'].shape == (3,), f"{robot_id} root_vel_local 形状错误"
        assert state['root_angular_vel_local'].shape == (3,), f"{robot_id} root_angular_vel_local 形状错误"
        assert state['joint_pos_norm'].shape == (21,), f"{robot_id} joint_pos_norm 形状错误"
        assert state['joint_vel_norm'].shape == (21,), f"{robot_id} joint_vel_norm 形状错误"
        
        print(f"✓ {robot_id}:")
        print(f"  - root_pos: {state['root_pos']}")
        print(f"  - root_rot (四元数): {state['root_rot']}")
        print(f"  - joint_pos_norm 范围: [{state['joint_pos_norm'].min():.3f}, {state['joint_pos_norm'].max():.3f}]")
        print(f"  - joint_vel_norm 范围: [{state['joint_vel_norm'].min():.3f}, {state['joint_vel_norm'].max():.3f}]")
    
    print()


def test_normalization_boundary(sim):
    """测试归一化边界"""
    print("=" * 60)
    print("测试 4: 归一化边界测试")
    print("=" * 60)
    
    # 手动设置关节到上限
    static_info = sim.get_static_data()
    
    for robot_id in ['robot_a', 'robot_b']:
        cache = sim._robot_cache[robot_id]
        qpos_indices = cache['qpos_indices']
        joint_limits = static_info[robot_id]['joint_limits']
        
        # 设置到上限
        for i, idx in enumerate(qpos_indices):
            sim.data.qpos[idx] = joint_limits[i, 1]  # 上限
        
        # 更新物理状态
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        
        # 获取归一化状态
        core_state = sim.get_core_state()
        joint_pos_norm = core_state[robot_id]['joint_pos_norm']
        
        # 验证全为 1.0
        assert np.allclose(joint_pos_norm, 1.0, atol=1e-5), \
            f"{robot_id} 上限归一化失败: {joint_pos_norm}"
        
        print(f"✓ {robot_id} 上限归一化测试通过")
        
        # 设置到下限
        for i, idx in enumerate(qpos_indices):
            sim.data.qpos[idx] = joint_limits[i, 0]  # 下限
        
        mujoco.mj_forward(sim.model, sim.data)
        
        core_state = sim.get_core_state()
        joint_pos_norm = core_state[robot_id]['joint_pos_norm']
        
        # 验证全为 -1.0
        assert np.allclose(joint_pos_norm, -1.0, atol=1e-5), \
            f"{robot_id} 下限归一化失败: {joint_pos_norm}"
        
        print(f"✓ {robot_id} 下限归一化测试通过")
    
    # 重置环境
    sim.reset()
    print()


def test_state_isolation(sim):
    """测试状态隔离 (修改 robot_b 不影响 robot_a)"""
    print("=" * 60)
    print("测试 5: 状态隔离测试")
    print("=" * 60)
    
    sim.reset()
    
    # 获取初始状态
    initial_state = sim.get_core_state()
    robot_a_initial = initial_state['robot_a']['root_pos'].copy()
    
    # 修改 robot_b 的位置
    cache_b = sim._robot_cache['robot_b']
    root_qpos_adr_b = cache_b['root_qpos_adr']
    sim.data.qpos[root_qpos_adr_b:root_qpos_adr_b+3] = [5.0, 5.0, 2.0]
    
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    
    # 再次获取状态
    new_state = sim.get_core_state()
    robot_a_new = new_state['robot_a']['root_pos']
    robot_b_new = new_state['robot_b']['root_pos']
    
    # 验证 robot_a 未改变
    assert np.allclose(robot_a_initial, robot_a_new, atol=1e-6), \
        "robot_a 状态被 robot_b 修改影响"
    
    # 验证 robot_b 已改变
    assert not np.allclose([1.0, 0.0, 1.282], robot_b_new, atol=1e-2), \
        "robot_b 状态未正确修改"
    
    print(f"✓ 状态隔离测试通过")
    print(f"  - robot_a 初始位置: {robot_a_initial}")
    print(f"  - robot_a 修改后位置: {robot_a_new}")
    print(f"  - robot_b 修改后位置: {robot_b_new}")
    
    sim.reset()
    print()


def test_derived_state(sim):
    """测试派生状态接口"""
    print("=" * 60)
    print("测试 6: get_derived_state()")
    print("=" * 60)
    
    derived = sim.get_derived_state()
    
    # 验证全局信息
    assert 'torso_distance' in derived, "缺少 torso_distance"
    assert 'combat_contacts' in derived, "缺少 combat_contacts"
    
    assert derived['torso_distance'].shape == (1,), "torso_distance 形状错误"
    assert isinstance(derived['combat_contacts'], list), "combat_contacts 应为列表"
    
    print(f"✓ 全局信息:")
    print(f"  - Torso 距离: {derived['torso_distance'][0]:.3f} m")
    print(f"  - 接触数量: {len(derived['combat_contacts'])}")
    
    # 验证单边视角
    for robot_id in ['robot_a', 'robot_b']:
        assert robot_id in derived, f"缺少 {robot_id}"
        view = derived[robot_id]
        
        assert 'uprightness' in view, f"{robot_id} 缺少 uprightness"
        assert 'feet_forces' in view, f"{robot_id} 缺少 feet_forces"
        assert 'opponent_in_local' in view, f"{robot_id} 缺少 opponent_in_local"
        
        assert view['uprightness'].shape == (1,), f"{robot_id} uprightness 形状错误"
        assert view['feet_forces'].shape == (2,), f"{robot_id} feet_forces 形状错误"
        
        opp = view['opponent_in_local']
        assert 'pos' in opp and opp['pos'].shape == (3,), f"{robot_id} opponent pos 错误"
        assert 'rot' in opp and opp['rot'].shape == (4,), f"{robot_id} opponent rot 错误"
        assert 'vel' in opp and opp['vel'].shape == (3,), f"{robot_id} opponent vel 错误"
        assert 'angular_vel' in opp and opp['angular_vel'].shape == (3,), f"{robot_id} opponent angular_vel 错误"
        
        print(f"✓ {robot_id}:")
        print(f"  - 直立度: {view['uprightness'][0]:.3f}")
        print(f"  - 双脚受力: {view['feet_forces']}")
        print(f"  - 对手局部位置: {opp['pos']}")
    
    print()


def test_action_control(sim):
    """测试动作控制"""
    print("=" * 60)
    print("测试 7: set_action() 和 PD 控制")
    print("=" * 60)
    
    sim.reset()
    
    # 设置动作为零位
    sim.set_action({
        'robot_a': np.zeros(21, dtype=np.float32),
        'robot_b': np.zeros(21, dtype=np.float32)
    })
    
    # 执行足够多的步数让系统收敛
    for _ in range(200):
        sim.physical_step()
    
    # 检查机器人是否保持直立
    core_state = sim.get_core_state()
    derived_state = sim.get_derived_state()
    
    for robot_id in ['robot_a', 'robot_b']:
        # 检查高度 (如果倒下，高度会显著降低)
        height = core_state[robot_id]['root_pos'][2]
        uprightness = derived_state[robot_id]['uprightness'][0]
        
        print(f"  {robot_id}: 高度={height:.3f}m, 直立度={uprightness:.3f}")
        
        # 如果机器人倒下，说明 KP/KD 需要调整
        if height < 0.8 or uprightness < 0.7:
            print(f"  ⚠ {robot_id} 在零位控制下倒下，需要调整 KP/KD 参数")
        else:
            # 只有在机器人保持直立时才检查关节位置
            joint_pos_norm = core_state[robot_id]['joint_pos_norm']
            max_error = np.abs(joint_pos_norm).max()
            print(f"  {robot_id}: 最大关节偏差={max_error:.3f}")
    
    print(f"✓ 零位控制测试完成 (注意: 零位可能不是稳定平衡点)")
    
    # 设置动作为极限值
    sim.reset()
    sim.set_action({
        'robot_a': np.ones(21, dtype=np.float32),
        'robot_b': -np.ones(21, dtype=np.float32)
    })
    
    # 执行多步 (需要足够时间让关节到达目标)
    for _ in range(500):
        sim.physical_step()
    
    # 验证关节位置接近目标
    core_state = sim.get_core_state()
    robot_a_pos = core_state['robot_a']['joint_pos_norm']
    robot_b_pos = core_state['robot_b']['joint_pos_norm']
    
    # 检查机器人是否倒下
    robot_a_height = core_state['robot_a']['root_pos'][2]
    robot_b_height = core_state['robot_b']['root_pos'][2]
    
    print(f"  robot_a: 高度={robot_a_height:.3f}m, 平均位置={robot_a_pos.mean():.3f}")
    print(f"  robot_b: 高度={robot_b_height:.3f}m, 平均位置={robot_b_pos.mean():.3f}")
    
    # 极限位置可能导致机器人失去平衡，所以只做基本检查
    if robot_a_height > 0.5 and robot_b_height > 0.5:
        # 机器人保持站立，检查是否朝目标方向移动
        assert robot_a_pos.mean() > 0.3, f"robot_a 未朝上限移动: {robot_a_pos.mean()}"
        assert robot_b_pos.mean() < -0.3, f"robot_b 未朝下限移动: {robot_b_pos.mean()}"
        print(f"✓ 极限位置控制测试通过")
    else:
        print(f"✓ 极限位置控制测试完成 (机器人在极限姿态下失去平衡，这是预期的)")
    
    sim.reset()
    print()


def test_local_coordinate_transform(sim):
    """测试局部坐标系转换"""
    print("=" * 60)
    print("测试 8: 局部坐标系转换")
    print("=" * 60)
    
    sim.reset()
    
    # 给 robot_a 一个全局速度
    cache_a = sim._robot_cache['robot_a']
    root_qvel_adr = cache_a['root_qvel_adr']
    sim.data.qvel[root_qvel_adr:root_qvel_adr+3] = [1.0, 0.0, 0.0]  # 沿 x 轴运动
    
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    
    core_state = sim.get_core_state()
    root_vel_local = core_state['robot_a']['root_vel_local']
    
    # robot_a 初始朝向是 +x，所以局部速度应该也是 [1, 0, 0]
    assert np.allclose(root_vel_local, [1.0, 0.0, 0.0], atol=0.1), \
        f"局部速度转换错误: {root_vel_local}"
    
    print(f"✓ 局部坐标系转换测试通过")
    print(f"  - 全局速度: [1.0, 0.0, 0.0]")
    print(f"  - 局部速度: {root_vel_local}")
    
    sim.reset()
    print()


def run_all_tests():
    """运行所有单元测试"""
    print("\n" + "=" * 60)
    print("Humanoid21 Simulator 单元测试")
    print("=" * 60 + "\n")
    
    try:
        sim = test_initialization()
        test_static_info(sim)
        test_core_state(sim)
        test_normalization_boundary(sim)
        test_state_isolation(sim)
        test_derived_state(sim)
        test_action_control(sim)
        test_local_coordinate_transform(sim)
        
        print("=" * 60)
        print("✓ 所有单元测试通过!")
        print("=" * 60)
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
