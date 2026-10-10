#!/usr/bin/env python3
"""Humanoid21 仿真性能基准测量（手动脚本，不是 pytest 测试）。

按照 BENCHMARK.md 定义的方法测量仿真环境的性能指标，输出数字而非
pass/fail 判决：每项指标给出实测值与参考标线（reference bar）的对照，
并将全部结果写入 ``benchmark_output/benchmark_results_<timestamp>.json``。

指标一览：tracking_error / jump / response_latency / zero_oscillation /
absolute_stability / throughput（仿真吞吐 steps/s）。
指标定义与参考标线见 BENCHMARK.md；最新一次基准快照记录在该文档
「基准快照」一节。

运行：
    PYTHONPATH=. python3 envs/humanoid21/benchmark.py
"""

import json
import time
import datetime

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.humanoid21.simulator import Humanoid21Simulator


# 视频输出目录（本地产物；*.mp4 在 .gitignore 中）
OUTPUT_DIR = Path(__file__).parent / 'benchmark_output'
OUTPUT_DIR.mkdir(exist_ok=True)


def save_video(frames: list, filename: str, fps: int = 30):
    """保存帧列表为视频文件"""
    output_path = OUTPUT_DIR / filename
    # 降低分辨率以加快视频生成
    downsampled_frames = [frame[::2, ::2] for frame in frames]  # 360x640
    imageio.mimwrite(str(output_path), downsampled_frames, fps=fps, codec='libx264', quality=8)
    print(f"  视频已保存: {output_path}")
    return output_path


def measure_tracking_error(sim: Humanoid21Simulator, record_video: bool = True) -> Dict[str, bool]:
    """
    测试 1: 跟踪误差与刚度
    
    方法:
    - 关闭重力
    - 输入 1Hz 正弦波指令
    - 运行 5 秒
    - 测量跟踪误差
    """
    print("=" * 70)
    print("基准测量 1: 跟踪误差与刚度")
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

    # 视频帧
    frames = []
    video_fps = 30  # 输出视频帧率
    frames_to_capture = int(duration * video_fps)  # 总共要捕获的帧数
    capture_interval = max(1, steps // frames_to_capture)  # 捕获间隔

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

        # 录制视频帧
        if record_video and step % capture_interval == 0:
            frame = sim.get_broadcastview_image()
            if frame is not None:
                frames.append(frame)

        # 每个周期采样一次
        if step % 10 == 0:
            core_state = sim.get_core_state()
            
            for robot_id in ['robot_a', 'robot_b']:
                # 目标位置 (rad)
                norm_params = sim._norm_params[robot_id]
                target_norm = action_value
                target_rad = target_norm * norm_params['scale'] + norm_params['reference']
                
                # 实际位置 (rad)
                cache = sim._robot(robot_id)
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
    heavy_max_all = 0.0
    light_max_all = 0.0

    for robot_id in ['robot_a', 'robot_b']:
        errors = np.array(tracking_errors[robot_id])
        mean_error = errors.mean(axis=0)

        # 承重关节 (腿部、腰部): 索引 0-14
        heavy_joints = mean_error[:15]
        # 末端关节 (手臂): 索引 15-20
        light_joints = mean_error[15:]

        heavy_max = heavy_joints.max()
        light_max = light_joints.max()
        heavy_max_all = max(heavy_max_all, heavy_max)
        light_max_all = max(light_max_all, light_max)

        print(f"\n{robot_id}:")
        print(f"  承重关节最大误差: {heavy_max:.4f} rad ({np.degrees(heavy_max):.2f}°)  [标线 <0.05]")
        print(f"  末端关节最大误差: {light_max:.4f} rad ({np.degrees(light_max):.2f}°)  [标线 <0.02]")

        results[robot_id] = {'heavy_err_rad': float(heavy_max), 'light_err_rad': float(light_max)}

    # 保存视频
    if record_video and frames:
        save_video(frames, 'test1_tracking_error.mp4', fps=video_fps)

    return {'metrics': {
        'heavy_tracking_err_rad': float(heavy_max_all),
        'light_tracking_err_rad': float(light_max_all),
    }, 'details': results}


def measure_jump(sim: Humanoid21Simulator, record_video: bool = True) -> Dict[str, bool]:
    """
    测试机器人初始蹲姿，然后立即输入站姿。
    这个过程中的机器人的响应速度以及跳跃高度。
    """
    print("=" * 70)
    print("基准测量: 跳跃能力")
    print("=" * 70)

    # 使用蹲姿初始化
    sim.reset(options={'initial_pose_a': 'squat', 'initial_pose_b': 'squat'})

    # 获取站姿 action
    standing_action = sim.INITIAL_POSES['standing']['action']

    # 测试参数
    duration = 3.0  # 秒
    steps = int(duration / sim.dt)

    # 记录数据
    jump_data = {
        'robot_a': {'heights': [], 'velocities': []},
        'robot_b': {'heights': [], 'velocities': []}
    }

    # 视频帧
    frames = []
    video_fps = 30
    capture_interval = max(1, int(steps / (duration * video_fps)))

    print(f"运行跳跃测试 (蹲姿 -> 站姿, {duration}s)...")

    for step in range(steps):
        # 立即设置站姿 action
        sim.set_action({
            'robot_a': standing_action,
            'robot_b': standing_action
        })
        sim.physical_step()

        # 录制视频帧
        if record_video and step % capture_interval == 0:
            frame = sim.get_broadcastview_image()
            if frame is not None:
                frames.append(frame)

        # 记录数据 (每10步记录一次)
        if step % 10 == 0:
            core_state = sim.get_core_state()
            for robot_id in ['robot_a', 'robot_b']:
                root_pos = core_state[robot_id]['root_pos']
                root_vel = core_state[robot_id]['root_vel_local']
                jump_data[robot_id]['heights'].append(root_pos[2])
                root_vel_z = root_vel[2]  # 局部坐标系的 z 方向速度
                jump_data[robot_id]['velocities'].append(root_vel_z)

    # 分析结果
    print("\n跳跃测试分析:")
    results = {}
    jump_max = vel_max = resp_max = 0.0

    for robot_id in ['robot_a', 'robot_b']:
        heights = np.array(jump_data[robot_id]['heights'])
        velocities = np.array(jump_data[robot_id]['velocities'])

        # 初始高度 (蹲姿)
        initial_height = heights[0]

        # 最大高度
        max_height = heights.max()

        # 跳跃高度 (相对初始高度)
        jump_height = max_height - initial_height

        # 最大垂直速度
        max_velocity = velocities.max()

        # 起跳响应时间 (到达最大速度的时间)
        response_time = velocities.argmax() * sim.dt * 10  # *10 因为每10步记录一次

        jump_max = max(jump_max, jump_height)
        vel_max = max(vel_max, max_velocity)
        resp_max = max(resp_max, response_time)

        print(f"\n{robot_id}:")
        print(f"  初始高度 (蹲姿): {initial_height:.4f} m")
        print(f"  最大高度: {max_height:.4f} m")
        print(f"  跳跃高度: {jump_height:.4f} m  [标线 >0.01]")
        print(f"  最大垂直速度: {max_velocity:.4f} m/s")
        print(f"  响应时间: {response_time:.3f} s")

        results[robot_id] = {'jump_height_m': float(jump_height),
                             'max_vel_ms': float(max_velocity),
                             'response_s': float(response_time)}

    # 保存视频
    if record_video and frames:
        save_video(frames, 'test_jump.mp4', fps=video_fps)

    return {'metrics': {
        'jump_height_m': float(jump_max),
        'jump_max_vel_ms': float(vel_max),
        'jump_response_s': float(resp_max),
    }, 'details': results}


def measure_response_latency(sim: Humanoid21Simulator, record_video: bool = True) -> Dict[str, bool]:
    """
    测试 2: 响应延迟与过冲
    
    方法:
    - 从零位突然给出阶跃信号
    - 测量到达 90% 的时间
    - 测量过冲幅度
    """
    print("=" * 70)
    print("基准测量 2: 响应延迟与过冲")
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

    # 视频帧
    frames = []
    video_fps = 30
    # 阶跃响应只记录前 2 秒 (1s = 500 steps, so 2s = 1000 steps, but we only run 500 steps)
    # 记录所有步骤，视频会慢一点

    print(f"执行阶跃响应测试 (目标: 0.5)...")

    for step in range(max_steps):
        sim.physical_step()

        # 录制视频帧 (每5步录制一帧)
        if record_video and step % 5 == 0:
            frame = sim.get_broadcastview_image()
            if frame is not None:
                frames.append(frame)

        core_state = sim.get_core_state()
        for robot_id in ['robot_a', 'robot_b']:
            pos_norm = core_state[robot_id]['joint_pos_norm']
            response_data[robot_id].append(pos_norm.copy())
    
    # 分析结果
    print("\n响应延迟分析:")
    results = {}
    lat_all = 0
    ovs_all = 0.0

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
        lat_all = max(lat_all, max_latency)
        ovs_all = max(ovs_all, max_overshoot)

        print(f"\n{robot_id}:")
        print(f"  最大响应延迟: {max_latency} 步 ({max_latency * sim.dt:.3f}s)  [标线 <100步/0.2s]")
        print(f"  最大过冲: {max_overshoot:.2f}%  [标线 <5%]")

        results[robot_id] = {'latency_s': float(max_latency * sim.dt),
                             'overshoot_pct': float(max_overshoot)}

    # 保存视频
    if record_video and frames:
        # 视频时长约 1 秒 (500 steps @ 0.002s/step)
        save_video(frames, 'test2_response_latency.mp4', fps=30)

    return {'metrics': {
        'response_latency_s': float(lat_all * sim.dt),
        'overshoot_pct': float(ovs_all),
    }, 'details': results}


def measure_zero_oscillation(sim: Humanoid21Simulator, record_video: bool = True) -> Dict[str, bool]:
    """
    测试 3: 零震荡与控制努力

    方法:
    - 开启重力，机器人站立
    - 持续输入站立指令
    - 分析力矩输出的震荡和幅度
    """
    print("=" * 70)
    print("基准测量 3: 零震荡与控制努力")
    print("=" * 70)

    sim.reset()

    # 获取站立动作
    standing_action = sim.INITIAL_POSES['standing']['action']

    # 持续输入站立指令
    duration = 5.0  # 秒
    steps = int(duration / sim.dt)

    torque_history = {
        'robot_a': [],
        'robot_b': []
    }

    # 视频帧
    frames = []
    video_fps = 30
    frames_to_capture = int(duration * video_fps)
    capture_interval = max(1, steps // frames_to_capture)

    print(f"运行 {duration}s 静态站立测试...")

    for step in range(steps):
        sim.set_action({
            'robot_a': standing_action,
            'robot_b': standing_action
        })
        sim.physical_step()

        # 录制视频帧
        if record_video and step % capture_interval == 0:
            frame = sim.get_broadcastview_image()
            if frame is not None:
                frames.append(frame)

        # 记录控制力矩
        if step % 10 == 0:
            for robot_id in ['robot_a', 'robot_b']:
                cache = sim._robot(robot_id)
                actuator_ids = cache['actuator_ids']

                # 获取 ctrl 值并转换为力矩
                ctrl_values = sim.data.ctrl[actuator_ids]
                gears = sim.model.actuator_gear[actuator_ids, 0]
                torques = ctrl_values * gears

                torque_history[robot_id].append(torques.copy())
    
    # 分析结果
    print("\n震荡与控制努力分析:")
    results = {}
    rate_all = 0.0
    effort_all = 0.0

    for robot_id in ['robot_a', 'robot_b']:
        torques = np.array(torque_history[robot_id])  # (samples, 21)
        
        # 计算力矩变化率 (一阶导数)
        torque_diff = np.diff(torques, axis=0)
        mean_change_rate = np.abs(torque_diff).mean()
        
        # 计算平均控制努力
        mean_torque = np.abs(torques).mean(axis=0)
        
        # 获取 ctrl_range 用于计算百分比
        cache = sim._robot(robot_id)
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
        
        rate_all = max(rate_all, mean_change_rate)
        effort_all = max(effort_all, heavy_torque_pct)

        print(f"\n{robot_id}:")
        print(f"  力矩平均变化率: {mean_change_rate:.4f}  [标线 <10.0]")
        print(f"  承重关节最大控制努力: {heavy_torque_pct:.2f}%  [标线 <30%]")

        results[robot_id] = {'torque_rate': float(mean_change_rate),
                             'effort_pct': float(heavy_torque_pct)}

    # 保存视频
    if record_video and frames:
        save_video(frames, 'test3_zero_oscillation.mp4', fps=video_fps)

    return {'metrics': {
        'torque_change_rate': float(rate_all),
        'static_effort_pct': float(effort_all),
    }, 'details': results}


def measure_absolute_stability(sim: Humanoid21Simulator, record_video: bool = True) -> Dict[str, bool]:
    """
    测试 4: 系统绝对稳定性
    
    方法:
    - 以 50Hz 频率输入随机白噪声
    - 运行 60 秒
    - 检查是否崩溃或发散
    """
    print("=" * 70)
    print("基准测量 4: 系统绝对稳定性")
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

    # 视频帧 - 60秒测试只录制关键时间点
    frames = []
    video_fps = 30
    # 只录制前3秒和每10秒的片段
    capture_steps = set()
    # 前3秒，每0.1秒一帧
    capture_steps.update(range(0, int(3.0 / sim.dt), int(0.1 / sim.dt)))
    # 之后每10秒录制0.5秒
    for t in range(10, 61, 10):
        start_step = int(t / sim.dt)
        end_step = int((t + 0.5) / sim.dt)
        capture_steps.update(range(start_step, end_step, int(0.1 / sim.dt)))

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

            # 录制视频帧
            if record_video and step in capture_steps:
                frame = sim.get_broadcastview_image()
                if frame is not None:
                    frames.append(frame)

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

    stable = not crashed and not diverged

    if crashed:
        print(f"  结果: 系统崩溃  [标线: 无崩溃/发散]")
    elif diverged:
        print(f"  结果: 数值发散  [标线: 无崩溃/发散]")
    else:
        print(f"  结果: 系统稳定  [标线: 无崩溃/发散]")

    # 保存视频
    if record_video and frames:
        save_video(frames, 'test4_stability.mp4', fps=30)

    return {'metrics': {
        'stable': bool(stable),
        'max_qpos': float(max_qpos),
        'max_qvel': float(max_qvel),
        'max_contact_force_N': float(max_force),
    }, 'details': {'crashed': crashed, 'diverged': diverged}}


def measure_throughput(sim: Humanoid21Simulator) -> Dict:
    """
    指标 5: 仿真吞吐（steps/sec，墙钟）

    方法:
    - 站立动作输入，跑固定步数物理仿真
    - 报告 物理步/秒 与 折合决策步/秒（25 物理步 = 1 决策步）
    """
    print("=" * 70)
    print("基准测量 5: 仿真吞吐")
    print("=" * 70)

    sim.reset()
    warmup = 100
    for _ in range(warmup):
        sim.physical_step()

    n_steps = 2000
    t0 = time.perf_counter()
    for _ in range(n_steps):
        sim.physical_step()
    wall = time.perf_counter() - t0

    sps = n_steps / wall
    print(f"\n{n_steps} 物理步用时 {wall:.2f}s")
    print(f"  物理步/秒: {sps:.0f}")
    print(f"  决策步/秒 (x25 物理步): {sps / 25:.1f}")
    print(f"  实时倍率 (物理 500Hz): {sps / 500:.2f}x")

    return {'metrics': {
        'phys_steps_per_sec': float(sps),
        'action_steps_per_sec': float(sps / 25.0),
        'realtime_factor': float(sps / 500.0),
        'measured_steps': n_steps,
        'wall_s': float(wall),
    }, 'details': {}}


# 参考标线（与 BENCHMARK.md 一致；仅供速览对照，不构成判决）
REFERENCE_BARS = {
    'tracking_error': '承重<0.05rad / 末端<0.02rad',
    'jump': 'jump_height>0.01m',
    'response_latency': '<0.2s 且过冲<5%',
    'zero_oscillation': 'rate<10 且静态努力<30%',
    'absolute_stability': '无崩溃/发散',
    'throughput': '信息项（无标线）',
}


def run_benchmark():
    """运行全部基准测量，输出汇总表并写 JSON 快照。"""
    print("\n" + "=" * 70)
    print("Humanoid21 仿真性能基准")
    print("指标定义与参考标线见 BENCHMARK.md")
    print("=" * 70 + "\n")

    sim = Humanoid21Simulator()
    results = {}
    try:
        results['jump'] = measure_jump(sim)
        results['tracking_error'] = measure_tracking_error(sim)
        results['response_latency'] = measure_response_latency(sim)
        results['zero_oscillation'] = measure_zero_oscillation(sim)
        results['absolute_stability'] = measure_absolute_stability(sim)
        results['throughput'] = measure_throughput(sim)
    except Exception as e:
        print(f"\n基准运行错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 汇总表
    print("\n" + "=" * 70)
    print("基准汇总（实测值 vs 参考标线）")
    print("=" * 70)
    flat = {}
    for name, res in results.items():
        flat[name] = res.get('metrics', {})
        print(f"\n[{name}]  标线: {REFERENCE_BARS.get(name, '-')}")
        for k, v in flat[name].items():
            print(f"    {k} = {v}")

    # 写 JSON 快照
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = {
        'timestamp': ts,
        'sim': 'humanoid21',
        'results': {k: v.get('metrics', {}) for k, v in results.items()},
    }
    out = OUTPUT_DIR / f"benchmark_results_{ts}.json"
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(f"\n结果已写入: {out}")
    print("=" * 70)
    return True


if __name__ == '__main__':
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
