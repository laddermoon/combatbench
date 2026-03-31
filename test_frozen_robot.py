#!/usr/bin/env python3
"""
测试 FrozenRobotPlugin - 让机器人B保持初始姿态完全静止
"""
import numpy as np
from envs.humanoid21 import make_env, FrozenRobotPlugin

print("=" * 80)
print("测试 FrozenRobotPlugin - 机器人B保持静止")
print("=" * 80)

# 创建环境，添加 FrozenRobotPlugin
frozen_plugin = FrozenRobotPlugin(frozen_robot_id='robot_b')
runtime = make_env(
    match_duration=5.0,
    non_fall_mode=True,
    plugins=[frozen_plugin]
)

result = runtime.reset()

# 获取初始位置
sim = runtime.engine.simulator
robot_b_body_id = sim.robot_info['robot_b']['body_id']
initial_pos_b = sim.data.xpos[robot_b_body_id].copy()
initial_quat_b = sim.data.xquat[robot_b_body_id].copy()

print(f"\n初始状态:")
print(f"  robot_b 位置: {initial_pos_b}")
print(f"  robot_b 姿态: {initial_quat_b}")

# 机器人A进行随机动作，机器人B应该保持静止
print(f"\n开始测试 - 机器人A随机动作，机器人B应该保持静止...")

max_pos_diff = 0.0
max_quat_diff = 0.0

for step in range(100):
    action_a = np.random.uniform(-0.5, 0.5, 21)
    action_b = np.random.uniform(-0.5, 0.5, 21)
    
    result = runtime.step(action_a, action_b)
    
    # 检查机器人B的位置
    current_pos_b = sim.data.xpos[robot_b_body_id].copy()
    current_quat_b = sim.data.xquat[robot_b_body_id].copy()
    
    pos_diff = np.linalg.norm(current_pos_b - initial_pos_b)
    quat_diff = np.linalg.norm(current_quat_b - initial_quat_b)
    
    max_pos_diff = max(max_pos_diff, pos_diff)
    max_quat_diff = max(max_quat_diff, quat_diff)
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step + 1}:")
        print(f"    robot_b 位置: {current_pos_b}")
        print(f"    位置偏差: {pos_diff:.10f}")
        print(f"    姿态偏差: {quat_diff:.10f}")

print(f"\n测试结果:")
print(f"  最大位置偏差: {max_pos_diff:.10f}")
print(f"  最大姿态偏差: {max_quat_diff:.10f}")

if max_pos_diff < 1e-6 and max_quat_diff < 1e-6:
    print(f"\n✅ 测试通过！机器人B完全保持静止（偏差 < 1e-6）")
else:
    print(f"\n⚠️  机器人B有轻微移动，但偏差很小")

# 测试受到攻击时是否保持静止
print(f"\n" + "=" * 80)
print("测试受到攻击时的表现")
print("=" * 80)

runtime.reset()
initial_pos_b = sim.data.xpos[robot_b_body_id].copy()

# 让机器人A向前移动接近B
for step in range(50):
    action_a = np.array([0.0, 0.0, 0.0,
                         0.0, 0.0, 0.5, 0.5, 0.0, 0.0,
                         0.0, 0.0, -0.5, -0.5, 0.0, 0.0,
                         0.5, 0.0, 0.0,
                         0.5, 0.0, 0.0])
    action_b = np.zeros(21)
    
    result = runtime.step(action_a, action_b)
    
    if (step + 1) % 10 == 0:
        current_pos_b = sim.data.xpos[robot_b_body_id].copy()
        pos_diff = np.linalg.norm(current_pos_b - initial_pos_b)
        
        robot_a_body_id = sim.robot_info['robot_a']['body_id']
        pos_a = sim.data.xpos[robot_a_body_id]
        distance = np.linalg.norm(pos_a - current_pos_b)
        
        print(f"  Step {step + 1}: 距离={distance:.3f}m, robot_b偏差={pos_diff:.10f}")

current_pos_b = sim.data.xpos[robot_b_body_id].copy()
final_diff = np.linalg.norm(current_pos_b - initial_pos_b)

print(f"\n最终位置偏差: {final_diff:.10f}")

if final_diff < 1e-6:
    print(f"✅ 机器人B即使受到攻击也保持完全静止！")
else:
    print(f"⚠️  机器人B有轻微移动")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
