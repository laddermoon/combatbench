#!/usr/bin/env python3
"""
Humanoid21 验收测试 (Acceptance Tests)

按照 ACCEPTANCE_CRITERIA.md 验证 KP/KD 参数是否满足所有指标
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.humanoid21.simulator import MujocoCombatSimulator


def test_tracking_error(sim: MujocoCombatSimulator) -> Dict[str, bool]:
    """
    测试 1: 跟踪误差与刚度
    
    方法:
    - 关闭重力
    - 输入 1Hz 正弦波指令
    - 运行 5 秒
    - 测量跟踪误差
    """
    print("=" * 70)
    print("验收测试 1: 跟踪误差与刚度")
    print("=" * 70)
    
    # 关闭重力
    original_gravity = sim.model.opt.gravity.copy()
    sim.model.opt.gravity[:] = 0.0
    
    sim.reset()
    
    # 测试参数
    freq = 1.0  # Hz
    duration = 5.0  # 秒
    dt = sim.dt
    steps = int(duration / dt)
    
    # 记录数据
    tracking_errors = {
        'robot_a': [],
        'robot_b': []
    }
    
    print(f"运行 {duration}s 正弦波跟踪测试...")
    
    for step in range(steps):
        t = step * dt
        
        # 生成正弦波指令 (幅度 1.0, 频率 1Hz)
        action_value = np.sin(2 * np.pi * freq * t)
        action = {
            'robot_a': np.full(21, action_value, dtype=np.float32),
            'robot_b': np.full(21, action_value, dtype=np.float32)
        }
        
        sim.set_action(action)
        sim.physical_step()
        
        # 每个周期采样一次
        if step % 10 == 0:
            core_state = sim.get_core_state()
            
            for robot_id in ['robot_a', 'robot_b']:
                # 目标位置 (rad)
                norm_params = sim._norm_params[robot_id]
                target_norm = action_value
                target_rad = target_norm * norm_params['scale'] + norm_params['reference']
                
                # 实际位置 (rad)
                cache = sim._robot_cache[robot_id]
                qpos_indices = cache['qpos_indices']
                actual_rad = sim.data.qpos[qpos_indices]
                
                # 跟踪误差
                error = np.abs(target_rad - actual_rad)
                tracking_errors[robot_id].append(error)
    
    # 恢复重力
    sim.model.opt.gravity[:] = original_gravity
    
    # 分析结果
    print("\n跟踪误差分析:")
    results = {}
    
    for robot_id in ['robot_a', 'robot_b']:
        errors = np.array(tracking_errors[robot_id])
        mean_error = errors.mean(axis=0)
        
        # 承重关节 (腿部、腰部): 索引 0-14
        heavy_joints = mean_error[:15]
        # 末端关节 (手臂): 索引 15-20
        light_joints = mean_error[15:]
        
        heavy_max = heavy_joints.max()
        light_max = light_joints.max()
        
        # 验收标准
        heavy_pass = heavy_max < 0.05  # < 0.05 rad
        light_pass = light_max < 0.02  # < 0.02 rad
        
        print(f"\n{robot_id}:")
        print(f"  承重关节最大误差: {heavy_max:.4f} rad ({np.degrees(heavy_max):.2f}°) - {'✓ PASS' if heavy_pass else '✗ FAIL'}")
        print(f"  末端关节最大误差: {light_max:.4f} rad ({np.degrees(light_max):.2f}°) - {'✓ PASS' if light_pass else '✗ FAIL'}")
        
        results[robot_id] = heavy_pass and light_pass
    
    overall_pass = all(results.values())
    print(f"\n{'='*70}")
    print(f"测试 1 结果: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")
    
    return {'pass': overall_pass, 'details': results}


def test_response_latency(sim: MujocoCombatSimulator) -> Dict[str, bool]:
    """
    测试 2: 响应延迟与过冲
    
    方法:
    - 从零位突然给出阶跃信号
    - 测量到达 90% 的时间
    - 测量过冲幅度
    """
    print("=" * 70)
    print("验收测试 2: 响应延迟与过冲")
    print("=" * 70)
    
    sim.reset()
    
    # 先让机器人稳定在零位
    for _ in range(100):
        sim.set_action({
            'robot_a': np.zeros(21, dtype=np.float32),
            'robot_b': np.zeros(21, dtype=np.float32)
        })
        sim.physical_step()
    
    # 记录初始位置
    initial_state = sim.get_core_state()
    
    # 突然给出阶跃信号
    target_action = np.ones(21, dtype=np.float32) * 0.5  # 使用 0.5 而不是 1.0 避免机器人失衡
    sim.set_action({
        'robot_a': target_action,
        'robot_b': target_action
    })
    
    # 记录响应过程
    max_steps = 500
    response_data = {
        'robot_a': [],
        'robot_b': []
    }
    
    print(f"执行阶跃响应测试 (目标: 0.5)...")
    
    for step in range(max_steps):
        sim.physical_step()
        
        core_state = sim.get_core_state()
        for robot_id in ['robot_a', 'robot_b']:
            pos_norm = core_state[robot_id]['joint_pos_norm']
            response_data[robot_id].append(pos_norm.copy())
    
    # 分析结果
    print("\n响应延迟分析:")
    results = {}
    
    for robot_id in ['robot_a', 'robot_b']:
        data = np.array(response_data[robot_id])  # (steps, 21)
        
        # 计算每个关节到达 90% 的时间
        target = 0.5
        threshold_90 = target * 0.9
        
        latencies = []
        overshoots = []
        
        for joint_idx in range(21):
            joint_data = data[:, joint_idx]
            
            # 找到第一次到达 90% 的时间
            reached_90 = np.where(joint_data >= threshold_90)[0]
            if len(reached_90) > 0:
                latency = reached_90[0]
                latencies.append(latency)
                
                # 计算过冲 (到达后的最大值)
                max_after = joint_data[latency:].max()
                overshoot_pct = (max_after - target) / target * 100
                overshoots.append(overshoot_pct)
            else:
                latencies.append(max_steps)
                overshoots.append(0)
        
        max_latency = max(latencies)
        max_overshoot = max(overshoots)
        
        # 验收标准
        latency_pass = max_latency < 100  # < 100 步 (0.2s)
        overshoot_pass = max_overshoot < 5  # < 5%
        
        print(f"\n{robot_id}:")
        print(f"  最大响应延迟: {max_latency} 步 ({max_latency * sim.dt:.3f}s) - {'✓ PASS' if latency_pass else '✗ FAIL'}")
        print(f"  最大过冲: {max_overshoot:.2f}% - {'✓ PASS' if overshoot_pass else '✗ FAIL'}")
        
        results[robot_id] = latency_pass and overshoot_pass
    
    overall_pass = all(results.values())
    print(f"\n{'='*70}")
    print(f"测试 2 结果: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")
    
    return {'pass': overall_pass, 'details': results}


def test_zero_oscillation(sim: MujocoCombatSimulator) -> Dict[str, bool]:
    """
    测试 3: 零震荡与控制努力
    
    方法:
    - 开启重力，机器人站立
    - 持续输入零指令
    - 分析力矩输出的震荡和幅度
    """
    print("=" * 70)
    print("验收测试 3: 零震荡与控制努力")
    print("=" * 70)
    
    sim.reset()
    
    # 持续输入零指令
    duration = 5.0  # 秒
    steps = int(duration / sim.dt)
    
    torque_history = {
        'robot_a': [],
        'robot_b': []
    }
    
    print(f"运行 {duration}s 静态站立测试...")
    
    for step in range(steps):
        sim.set_action({
            'robot_a': np.zeros(21, dtype=np.float32),
            'robot_b': np.zeros(21, dtype=np.float32)
        })
        sim.physical_step()
        
        # 记录控制力矩
        if step % 10 == 0:
            for robot_id in ['robot_a', 'robot_b']:
                cache = sim._robot_cache[robot_id]
                actuator_ids = cache['actuator_ids']
                
                # 获取 ctrl 值并转换为力矩
                ctrl_values = sim.data.ctrl[actuator_ids]
                gears = sim.model.actuator_gear[actuator_ids, 0]
                torques = ctrl_values * gears
                
                torque_history[robot_id].append(torques.copy())
    
    # 分析结果
    print("\n震荡与控制努力分析:")
    results = {}
    
    for robot_id in ['robot_a', 'robot_b']:
        torques = np.array(torque_history[robot_id])  # (samples, 21)
        
        # 计算力矩变化率 (一阶导数)
        torque_diff = np.diff(torques, axis=0)
        mean_change_rate = np.abs(torque_diff).mean()
        
        # 计算平均控制努力
        mean_torque = np.abs(torques).mean(axis=0)
        
        # 获取 ctrl_range 用于计算百分比
        cache = sim._robot_cache[robot_id]
        actuator_ids = cache['actuator_ids']
        ctrl_ranges = []
        for act_id in actuator_ids:
            ctrl_range = sim.model.actuator_ctrlrange[act_id]
            gear = sim.model.actuator_gear[act_id, 0]
            max_torque = max(abs(ctrl_range[0]), abs(ctrl_range[1])) * abs(gear)
            ctrl_ranges.append(max_torque)
        ctrl_ranges = np.array(ctrl_ranges)
        
        # 承重关节 (腿部、腰部)
        heavy_joints_idx = list(range(15))
        heavy_torque_pct = (mean_torque[heavy_joints_idx] / ctrl_ranges[heavy_joints_idx] * 100).max()
        
        # 验收标准
        oscillation_pass = mean_change_rate < 10.0  # 变化率阈值 (经验值)
        effort_pass = heavy_torque_pct < 30.0  # < 30%
        
        print(f"\n{robot_id}:")
        print(f"  力矩平均变化率: {mean_change_rate:.4f} - {'✓ PASS' if oscillation_pass else '✗ FAIL'}")
        print(f"  承重关节最大控制努力: {heavy_torque_pct:.2f}% - {'✓ PASS' if effort_pass else '✗ FAIL'}")
        
        results[robot_id] = oscillation_pass and effort_pass
    
    overall_pass = all(results.values())
    print(f"\n{'='*70}")
    print(f"测试 3 结果: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")
    
    return {'pass': overall_pass, 'details': results}


def test_absolute_stability(sim: MujocoCombatSimulator) -> Dict[str, bool]:
    """
    测试 4: 系统绝对稳定性
    
    方法:
    - 以 50Hz 频率输入随机白噪声
    - 运行 60 秒
    - 检查是否崩溃或发散
    """
    print("=" * 70)
    print("验收测试 4: 系统绝对稳定性")
    print("=" * 70)
    
    sim.reset()
    
    duration = 60.0  # 秒
    control_freq = 50  # Hz
    control_interval = int((1.0 / control_freq) / sim.dt)  # 每多少步更新一次动作
    total_steps = int(duration / sim.dt)
    
    print(f"运行 {duration}s 随机噪声稳定性测试...")
    print(f"控制频率: {control_freq} Hz, 总步数: {total_steps}")
    
    max_qpos = -np.inf
    max_qvel = -np.inf
    max_force = -np.inf
    
    crashed = False
    diverged = False
    
    try:
        for step in range(total_steps):
            # 每 control_interval 步更新一次动作
            if step % control_interval == 0:
                action = {
                    'robot_a': np.random.uniform(-1, 1, 21).astype(np.float32),
                    'robot_b': np.random.uniform(-1, 1, 21).astype(np.float32)
                }
                sim.set_action(action)
            
            sim.physical_step()
            
            # 每 1000 步检查一次数值
            if step % 1000 == 0:
                max_qpos = max(max_qpos, np.abs(sim.data.qpos).max())
                max_qvel = max(max_qvel, np.abs(sim.data.qvel).max())
                
                # 检查接触力
                for i in range(sim.data.ncon):
                    c_array = np.zeros(6, dtype=np.float64)
                    import mujoco
                    mujoco.mj_contactForce(sim.model, sim.data, i, c_array)
                    force = np.linalg.norm(c_array[:3])
                    max_force = max(max_force, force)
                
                # 检查是否发散
                if max_qpos > 1e10 or max_qvel > 1e10 or max_force > 1e10:
                    diverged = True
                    break
                
                # 检查 NaN
                if np.isnan(sim.data.qpos).any() or np.isnan(sim.data.qvel).any():
                    diverged = True
                    break
                
                # 进度显示
                if step % 5000 == 0:
                    progress = step / total_steps * 100
                    print(f"  进度: {progress:.1f}% - qpos_max={max_qpos:.2f}, qvel_max={max_qvel:.2f}, force_max={max_force:.2f}")
    
    except Exception as e:
        print(f"\n✗ 系统崩溃: {e}")
        crashed = True
    
    # 分析结果
    print("\n稳定性分析:")
    print(f"  最大 qpos: {max_qpos:.4f}")
    print(f"  最大 qvel: {max_qvel:.4f}")
    print(f"  最大接触力: {max_force:.4f} N")
    
    stability_pass = not crashed and not diverged
    
    if crashed:
        print(f"  ✗ 系统崩溃")
    elif diverged:
        print(f"  ✗ 数值发散")
    else:
        print(f"  ✓ 系统稳定")
    
    print(f"\n{'='*70}")
    print(f"测试 4 结果: {'✓ PASS' if stability_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")
    
    return {'pass': stability_pass, 'crashed': crashed, 'diverged': diverged}


def run_all_acceptance_tests():
    """运行所有验收测试"""
    print("\n" + "=" * 70)
    print("Humanoid21 验收测试套件")
    print("按照 ACCEPTANCE_CRITERIA.md 验证 KP/KD 参数")
    print("=" * 70 + "\n")
    
    sim = MujocoCombatSimulator()
    
    results = {}
    
    try:
        # 测试 1: 跟踪误差
        results['tracking'] = test_tracking_error(sim)
        
        # 测试 2: 响应延迟
        results['response'] = test_response_latency(sim)
        
        # 测试 3: 零震荡
        results['oscillation'] = test_zero_oscillation(sim)
        
        # 测试 4: 绝对稳定性
        results['stability'] = test_absolute_stability(sim)
        
        # 汇总结果
        print("\n" + "=" * 70)
        print("验收测试汇总")
        print("=" * 70)
        
        all_pass = True
        for test_name, result in results.items():
            status = "✓ PASS" if result['pass'] else "✗ FAIL"
            print(f"{test_name:20s}: {status}")
            all_pass = all_pass and result['pass']
        
        print("=" * 70)
        if all_pass:
            print("✓ 所有验收测试通过! KP/KD 参数满足要求。")
        else:
            print("✗ 部分测试未通过，需要调整 KP/KD 参数。")
        print("=" * 70)
        
        return all_pass
        
    except Exception as e:
        print(f"\n✗ 测试运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_acceptance_tests()
    sys.exit(0 if success else 1)
