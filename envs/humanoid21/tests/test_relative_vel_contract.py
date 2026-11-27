#!/usr/bin/env python3
"""
relative_vel 契约锁定测试 (R1)

契约（2026-11-26 起，见 OBSERVATION_zh.md 2.4.1）：
    relative_vel = R_ego^-1 @ (v_opp - v_self)
    双方速度同源：各自 torso 的 cvel 线速度（世界系），再旋转到 ego 机体系。

锁定三个关键性质：
1. 双方有不同速度时，字段精确等于世界系速度差转到 ego 系；
2. 双方同速运动时，relative_vel == 0（旧语义下该值非零，这是本次修正的核心行为差异）；
3. 对手静止、自身运动时，relative_vel == R_ego^-1 @ (-v_self)（旧语义下该值为 0）。
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from envs.humanoid21.simulator import Humanoid21Simulator


def _inject_root_linvel(sim, vel_a, vel_b):
    """直接注入双方 root 线速度（世界系）并刷新运动学。"""
    for rid, vel in (('robot_a', vel_a), ('robot_b', vel_b)):
        adr = sim._robot(rid)['root_qvel_adr']
        sim.data.qvel[adr:adr + 3] = vel
        sim.data.qvel[adr + 3:adr + 6] = 0.0  # 角速度清零，保证 cvel 线速度 == root 线速度
    mujoco.mj_forward(sim.model, sim.data)
    sim._data_cache.clear()


def _ego_rot_inv(sim, robot_id):
    torso_id = sim._robot(robot_id)['root_body_id']
    q = sim.data.xquat[torso_id]  # [w,x,y,z]
    return R.from_quat([q[1], q[2], q[3], q[0]]).inv()


def _cvel_lin(sim, robot_id):
    torso_id = sim._robot(robot_id)['root_body_id']
    return sim.data.cvel[torso_id, 3:6].copy()


def _views(sim):
    ds = sim.get_derived_state()
    return ds['robot_a'], ds['robot_b']


def test_relative_vel_exact_difference():
    """性质1：不同速度下的精确公式锁定。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    va, vb = np.array([1.0, 0.5, 0.0]), np.array([0.2, -0.3, 0.4])
    _inject_root_linvel(sim, va, vb)

    view_a, view_b = _views(sim)
    for self_rid, opp_rid, view in (('robot_a', 'robot_b', view_a),
                                    ('robot_b', 'robot_a', view_b)):
        rot_inv = _ego_rot_inv(sim, self_rid)
        expected = rot_inv.apply(_cvel_lin(sim, opp_rid) - _cvel_lin(sim, self_rid))
        actual = view['opponent_basic_pose']['relative_vel']
        assert np.allclose(actual, expected, atol=1e-5), \
            f"{self_rid}: relative_vel={actual}, expected={expected}"

        # 扁平观测一致性：relative_vel 位于第 60:63 维
        flat = view['observation']
        assert np.allclose(flat[60:63], actual, atol=1e-6), \
            f"{self_rid}: flat obs[60:63]={flat[60:63]} != dict field {actual}"
    print("PASS: 精确公式 + 扁平观测一致性")


def test_relative_vel_comoving_is_zero():
    """性质2：同速运动（含追击场景）relative_vel 必须为 0。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    v = np.array([0.8, 0.0, 0.0])
    _inject_root_linvel(sim, v, v.copy())

    view_a, view_b = _views(sim)
    for name, view in (('robot_a', view_a), ('robot_b', view_b)):
        rv = view['opponent_basic_pose']['relative_vel']
        assert np.allclose(rv, 0.0, atol=1e-5), \
            f"{name}: 同速运动 relative_vel 应为 0, 实际 {rv}"
    print("PASS: 同速运动 relative_vel == 0")


def test_relative_vel_stationary_opponent():
    """性质3：对手静止、自身运动时，relative_vel == ego 系下的 -v_self。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    va = np.array([0.0, 0.6, 0.0])
    _inject_root_linvel(sim, va, np.zeros(3))

    view_a, _ = _views(sim)
    rot_inv = _ego_rot_inv(sim, 'robot_a')
    expected = rot_inv.apply(-_cvel_lin(sim, 'robot_a'))
    actual = view_a['opponent_basic_pose']['relative_vel']
    assert np.allclose(actual, expected, atol=1e-5), \
        f"relative_vel={actual}, expected={expected}"
    assert np.linalg.norm(actual) > 0.3, "自身运动时相对静止对手的 relative_vel 必须非零"
    print("PASS: 静止对手 + 自身运动 -> relative_vel 非零且等于 -v_self (ego)")


KP_NAMES = ['head', 'hand_right', 'hand_left', 'foot_right', 'foot_left']


def test_keypoint_vel_exact_difference():
    """关键点速度契约：精确等于 (v_kp - v_self_torso) 转到 ego 系。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    va, vb = np.array([1.0, 0.5, 0.0]), np.array([0.2, -0.3, 0.4])
    _inject_root_linvel(sim, va, vb)

    view_a, view_b = _views(sim)
    for self_rid, opp_rid, view in (('robot_a', 'robot_b', view_a),
                                    ('robot_b', 'robot_a', view_b)):
        rot_inv = _ego_rot_inv(sim, self_rid)
        self_vel = _cvel_lin(sim, self_rid)
        kp_ids = sim._robot(opp_rid)['keypoint_body_ids']
        for name in KP_NAMES:
            kp_vel_world = sim.data.cvel[kp_ids[name], 3:6]
            expected = rot_inv.apply(kp_vel_world - self_vel)
            actual = view['opponent_keypoint_vel'][name]
            assert np.allclose(actual, expected, atol=1e-5), \
                f"{self_rid}/{name}: {actual} != {expected}"

        # 扁平观测一致性：头部速度不变换，位于第 81:84 维
        assert np.allclose(view['observation'][81:84],
                           view['opponent_keypoint_vel']['head'], atol=1e-6), \
            f"{self_rid}: flat obs[81:84] != keypoint_vel[head]"
    print("PASS: 关键点速度精确公式 + 扁平观测一致性")


def test_keypoint_vel_comoving_is_zero():
    """同速运动时所有关键点速度必须为 0（纯平移下 v_kp == v_root）。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    v = np.array([0.8, 0.0, 0.0])
    _inject_root_linvel(sim, v, v.copy())

    view_a, view_b = _views(sim)
    for rname, view in (('robot_a', view_a), ('robot_b', view_b)):
        for name in KP_NAMES:
            kv = view['opponent_keypoint_vel'][name]
            assert np.allclose(kv, 0.0, atol=1e-5), \
                f"{rname}/{name}: 同速运动关键点速度应为 0, 实际 {kv}"
    print("PASS: 同速运动关键点速度 == 0")


def test_keypoint_vel_stationary_opponent():
    """对手静止、自身运动：所有关键点速度 == ego 系下 -v_self（纯平移）。"""
    sim = Humanoid21Simulator()
    sim.reset(seed=42)
    va = np.array([0.0, 0.6, 0.0])
    _inject_root_linvel(sim, va, np.zeros(3))

    view_a, _ = _views(sim)
    rot_inv = _ego_rot_inv(sim, 'robot_a')
    expected = rot_inv.apply(-_cvel_lin(sim, 'robot_a'))
    for name in KP_NAMES:
        actual = view_a['opponent_keypoint_vel'][name]
        assert np.allclose(actual, expected, atol=1e-5), \
            f"{name}: {actual} != {expected}"
        assert np.linalg.norm(actual) > 0.3, \
            f"{name}: 自身运动时对静止对手的关键点速度必须非零"
    print("PASS: 静止对手 + 自身运动 -> 关键点速度非零且等于 -v_self (ego)")


if __name__ == '__main__':
    test_relative_vel_exact_difference()
    test_relative_vel_comoving_is_zero()
    test_relative_vel_stationary_opponent()
    test_keypoint_vel_exact_difference()
    test_keypoint_vel_comoving_is_zero()
    test_keypoint_vel_stationary_opponent()
    print("\n全部契约测试通过")
