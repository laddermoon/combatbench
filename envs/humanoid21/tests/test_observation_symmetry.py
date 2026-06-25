#!/usr/bin/env python3
"""
测试观测数据的对称性

核心思想：当两个机器人都是站立姿态时，它们互相观测对方的数据应该是对称的。
利用这个对称性可以验证观测数据的正确性。

对称性检查：
1. 相对位置：robot_a 看到的 robot_b 的相对位置，应该与 robot_b 看到的 robot_a 的相对位置相反
2. 相对速度：两个机器人都是静止的，速度应该都是 0
3. FaceVector：两个机器人互相看到的朝向应该相等（因为它们都面向对方）
4. 关键点位置：应该对称（例如 robot_a 看到的 robot_b 的右手，应该与 robot_b 看到的 robot_a 的左手对称）
5. 关键点速度：两个机器人都是静止的，速度应该都是 0
"""

import sys
from pathlib import Path
import numpy as np
import mujoco

# 添加项目根目录到路径（确保能正确导入模块）
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from envs.humanoid21.simulator import Humanoid21Simulator


def test_observation_symmetry():
    """测试观测数据的对称性"""

    print("=" * 80)
    print("观测数据对称性测试")
    print("=" * 80)

    # 创建仿真器
    sim = Humanoid21Simulator()

    # 重置仿真（两个机器人都是站立姿态）
    sim.reset(seed=42)

    # 获取观测数据
    derived_state = sim.get_derived_state()

    # 获取 robot_a 和 robot_b 的对手观测数据
    robot_a_view = derived_state['robot_a']
    robot_b_view = derived_state['robot_b']

    print("\n" + "-" * 80)
    print("1. 对手基础位姿 (9维)")
    print("-" * 80)

    # 1.1 相对位置 (3维)
    a_sees_b_pos = robot_a_view['opponent_basic_pose']['relative_pos']
    b_sees_a_pos = robot_b_view['opponent_basic_pose']['relative_pos']

    print(f"\nrobot_a 看到的 robot_b 的相对位置: {a_sees_b_pos}")
    print(f"robot_b 看到的 robot_a 的相对位置: {b_sees_a_pos}")
    print(f"是否相等: {np.allclose(a_sees_b_pos, b_sees_a_pos, atol=1e-5)}")

    if not np.allclose(a_sees_b_pos, b_sees_a_pos, atol=1e-5):
        print("  ❌ 错误：相对位置应该相等！")
        print(f"  期望：两个机器人看对方的位置在各自局部坐标系中应该相等")
        print(f"  实际 a_sees_b_pos = {a_sees_b_pos}")
        print(f"  实际 b_sees_a_pos = {b_sees_a_pos}")
        print(f"  差异 = {a_sees_b_pos - b_sees_a_pos}")
    else:
        print("  ✅ 通过：相对位置相等")

    # 1.2 相对速度 (3维)
    a_sees_b_vel = robot_a_view['opponent_basic_pose']['relative_vel']
    b_sees_a_vel = robot_b_view['opponent_basic_pose']['relative_vel']

    print(f"\nrobot_a 看到的 robot_b 的相对速度: {a_sees_b_vel}")
    print(f"robot_b 看到的 robot_a 的相对速度: {b_sees_a_vel}")
    print(f"是否都接近 0: {np.allclose(a_sees_b_vel, 0, atol=1e-5) and np.allclose(b_sees_a_vel, 0, atol=1e-5)}")

    if not (np.allclose(a_sees_b_vel, 0, atol=1e-5) and np.allclose(b_sees_a_vel, 0, atol=1e-5)):
        print("  ❌ 错误：相对速度应该都是 0！")
        print(f"  robot_a 看到的速度: {a_sees_b_vel}")
        print(f"  robot_b 看到的速度: {b_sees_a_vel}")
    else:
        print("  ✅ 通过：相对速度都是 0")

    # 1.3 FaceVector (3维)
    a_sees_b_face = robot_a_view['opponent_basic_pose']['face_vector']
    b_sees_a_face = robot_b_view['opponent_basic_pose']['face_vector']

    print(f"\nrobot_a 看到的 robot_b 的朝向 (FaceVector): {a_sees_b_face}")
    print(f"robot_b 看到的 robot_a 的朝向 (FaceVector): {b_sees_a_face}")
    print(f"是否相等: {np.allclose(a_sees_b_face, b_sees_a_face, atol=1e-5)}")

    if not np.allclose(a_sees_b_face, b_sees_a_face, atol=1e-5):
        print("  ❌ 错误：FaceVector 应该相等！")
        print(f"  期望：两个 FaceVector 应该相等（都面向对方）")
        print(f"  差异 = {a_sees_b_face - b_sees_a_face}")
    else:
        print("  ✅ 通过：FaceVector 相等")

    print("\n" + "-" * 80)
    print("2. 对手关键点位置 (15维)")
    print("-" * 80)

    a_sees_b_keypoints = robot_a_view['opponent_keypoint_pos']
    b_sees_a_keypoints = robot_b_view['opponent_keypoint_pos']

    # 关键点对称性检查
    # 当两个机器人面对面站立时，它们应该是完全对称的：
    # - robot_a 看到的 robot_b 的左手，与 robot_b 看到的 robot_a 的左手，位置应该相等
    # - robot_a 看到的 robot_b 的右手，与 robot_b 看到的 robot_a 的右手，位置应该相等
    # 也就是说，相同的关键点在各自的局部坐标系中应该有相同的坐标

    symmetric_pairs = [
        ('head', 'head'),           # 头部 vs 头部
        ('hand_right', 'hand_right'),  # 右手 vs 右手
        ('hand_left', 'hand_left'),    # 左手 vs 左手
        ('foot_right', 'foot_right'),  # 右脚 vs 右脚
        ('foot_left', 'foot_left'),    # 左脚 vs 左脚
    ]

    all_keypoints_ok = True
    for a_key, b_key in symmetric_pairs:
        a_pos = a_sees_b_keypoints[a_key]
        b_pos = b_sees_a_keypoints[b_key]

        print(f"\nrobot_a 看到的 robot_b 的 {a_key}: {a_pos}")
        print(f"robot_b 看到的 robot_a 的 {b_key}: {b_pos}")

        # 应该完全相等
        match = np.allclose(a_pos, b_pos, atol=1e-5)
        print(f"  是否相等: {match}")

        if not match:
            print(f"  ❌ 错误：{a_key} 与 {b_key} 应该相等！")
            print(f"  差异: {a_pos - b_pos}")
            all_keypoints_ok = False
        else:
            print(f"  ✅ 通过：{a_key} 与 {b_key} 相等")

    print("\n" + "-" * 80)
    print("3. 对手关键点速度 (15维)")
    print("-" * 80)

    a_sees_b_vels = robot_a_view['opponent_keypoint_vel']
    b_sees_a_vels = robot_b_view['opponent_keypoint_vel']

    all_velocities_ok = True
    for key in ['head', 'hand_right', 'hand_left', 'foot_right', 'foot_left']:
        a_vel = a_sees_b_vels[key]
        b_vel = b_sees_a_vels[key]

        print(f"\nrobot_a 看到的 robot_b 的 {key} 速度: {a_vel}")
        print(f"robot_b 看到的 robot_a 的 {key} 速度: {b_vel}")
        print(f"  是否都接近 0: {np.allclose(a_vel, 0, atol=1e-5) and np.allclose(b_vel, 0, atol=1e-5)}")

        if not (np.allclose(a_vel, 0, atol=1e-5) and np.allclose(b_vel, 0, atol=1e-5)):
            print(f"  ❌ 错误：{key} 速度应该都是 0！")
            all_velocities_ok = False
        else:
            print(f"  ✅ 通过：{key} 速度都是 0")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    # 检查所有测试是否通过
    tests_passed = []
    tests_failed = []

    # 相对位置测试（应该相等）
    if np.allclose(a_sees_b_pos, b_sees_a_pos, atol=1e-5):
        tests_passed.append("相对位置")
    else:
        tests_failed.append("相对位置")

    # 相对速度测试
    if np.allclose(a_sees_b_vel, 0, atol=1e-5) and np.allclose(b_sees_a_vel, 0, atol=1e-5):
        tests_passed.append("相对速度")
    else:
        tests_failed.append("相对速度")

    # FaceVector 测试
    if np.allclose(a_sees_b_face, b_sees_a_face, atol=1e-5):
        tests_passed.append("FaceVector")
    else:
        tests_failed.append("FaceVector")

    # 关键点位置测试
    if all_keypoints_ok:
        tests_passed.append("关键点位置")
    else:
        tests_failed.append("关键点位置")

    # 关键点速度测试
    if all_velocities_ok:
        tests_passed.append("关键点速度")
    else:
        tests_failed.append("关键点速度")

    print(f"\n通过的测试 ({len(tests_passed)}):")
    for test in tests_passed:
        print(f"  ✅ {test}")

    if tests_failed:
        print(f"\n失败的测试 ({len(tests_failed)}):")
        for test in tests_failed:
            print(f"  ❌ {test}")
        print("\n⚠️  观测数据存在对称性问题，请检查实现！")
        return False
    else:
        print("\n✅ 所有测试通过！观测数据对称性验证成功！")
        return True


if __name__ == "__main__":
    success = test_observation_symmetry()
    sys.exit(0 if success else 1)
