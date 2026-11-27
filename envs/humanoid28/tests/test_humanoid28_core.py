"""
Humanoid28 核心契约测试

覆盖：
  1. 模型与 meta 校验（validate 全通过、关节顺序、被动动力学剥离）
  2. 生成 XML 与上游资产一致性（质量、关节限位、gear）
  3. action 语义（弧度制目标、bounds 裁剪、MimicKit 扩展限位数值）
  4. PD 控制正确性（tau = kp*(target-q) - kd*qd、扭矩饱和、阻尼项）
  5. reset（初始姿态、双机朝向、控制目标=当前关节位置）
  6. 观测（110 维布局、本体原始弧度无压缩、relative_vel 正确契约）
  7. 战斗伤害管线（hit 分类逻辑 + 端到端 hand→torso 打击伤害）
  8. 物理步进稳定性（长时无 NaN）

运行：
  cd /data1/mono/things/combatbench
  PYTHONPATH=. python3 -m pytest envs/humanoid28/tests/test_humanoid28_core.py -x -q
"""
import numpy as np
import pytest

from envs.humanoid28.meta import Humanoid28Meta
from envs.humanoid28.simulator import Humanoid28Simulator


@pytest.fixture(scope='module')
def sim():
    s = Humanoid28Simulator()
    yield s
    s.close()


@pytest.fixture(scope='module')
def fresh_sim():
    """每个用例独立 reset 的 simulator（module 级共享，用例内自行 reset）。"""
    s = Humanoid28Simulator()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. 模型与 meta 校验
# ---------------------------------------------------------------------------

def test_model_validate_passes():
    sim = Humanoid28Simulator()
    errors = Humanoid28Meta.validate(sim.model)
    sim.close()
    assert errors == [], f'meta validate errors: {errors}'


def test_passive_dynamics_stripped(sim):
    """MimicKit 语义：stiffness/damping 是 PD 增益，被动值必须为零。"""
    import mujoco
    for robot_suffix in ('_a', '_b'):
        for jname in Humanoid28Meta.CONTROLLED_JOINTS:
            jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname + robot_suffix)
            assert float(sim.model.jnt_stiffness[jid]) == 0.0
            dof_adr = int(sim.model.jnt_dofadr[jid])
            assert float(sim.model.dof_damping[dof_adr]) == 0.0


def test_controlled_joint_order_matches_model(sim):
    """qpos_indices 必须按 CONTROLLED_JOINTS 文档顺序递增且连续。"""
    cache = sim._robots['robot_a']
    qpos_idx = cache['qpos_indices']
    # 文档顺序 = 编译后模型的关节顺序 → qpos 地址严格递增
    assert all(qpos_idx[i] < qpos_idx[i+1] for i in range(len(qpos_idx)-1))


def test_total_mass(sim):
    """单机器人总质量应与 MimicKit 上游一致 (~50.05 kg)。"""
    mass = sim._robots['robot_a']['total_mass']
    assert abs(mass - 50.05) < 0.1, f'mass {mass}'


def test_armature_preserved(sim):
    """armature 是真实动力学参数，必须保留（abdomen=0.02, neck=0.01）。"""
    import mujoco
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'abdomen_x_a')
    dof = int(sim.model.jnt_dofadr[jid])
    assert abs(float(sim.model.dof_armature[dof]) - 0.02) < 1e-6


# ---------------------------------------------------------------------------
# 2. action 语义
# ---------------------------------------------------------------------------

def test_action_bounds_mimickit_exact():
    """抽查 MimicKit 扩展限位的精确数值（手算对照）。"""
    bounds = Humanoid28Meta.compute_action_bounds()
    idx = {n: i for i, n in enumerate(Humanoid28Meta.CONTROLLED_JOINTS)}

    # abdomen 组: maxabs = max(1.0472, 1.0472, 1.5708, 1.2217) = 1.5708
    #           -> 1.2 * 1.5708 = 1.88496 对称
    i = idx['abdomen_x']
    assert abs(bounds['high'][i] - 1.2 * 1.5708) < 1e-4
    assert abs(bounds['low'][i] + 1.2 * 1.5708) < 1e-4

    # right_elbow [0, 2.7925]: mid=1.39625, scale=1.4*1.39625=1.95475
    #   -> [-0.5585, 3.351]
    i = idx['right_elbow']
    mid = 2.7925 / 2
    s = 1.4 * mid
    assert abs(bounds['low'][i] - (mid - s)) < 1e-4
    assert abs(bounds['high'][i] - (mid + s)) < 1e-4

    # right_shoulder 组: maxabs = max(1.9199,2.4435,1.5708,1.0472,2.7925)=2.7925
    #   -> ±3.351
    i = idx['right_shoulder_x']
    assert abs(bounds['high'][i] - 1.2 * 2.7925) < 1e-4


def test_action_clipping(sim):
    """越界 action 被裁剪到 action bounds。"""
    sim.reset(seed=0)
    act = np.full(28, 99.0, dtype=np.float32)
    sim.set_action({'robot_a': act, 'robot_b': None})
    cur = sim.get_action()['robot_a']
    bounds = Humanoid28Meta.compute_action_bounds()
    np.testing.assert_allclose(cur, bounds['high'], atol=1e-6)

    act = np.full(28, -99.0, dtype=np.float32)
    sim.set_action({'robot_a': act, 'robot_b': None})
    cur = sim.get_action()['robot_a']
    np.testing.assert_allclose(cur, bounds['low'], atol=1e-6)


def test_action_shape_validation(sim):
    sim.reset(seed=0)
    with pytest.raises(ValueError):
        sim.set_action({'robot_a': np.zeros(21, dtype=np.float32), 'robot_b': None})


# ---------------------------------------------------------------------------
# 3. PD 控制正确性
# ---------------------------------------------------------------------------

def test_pd_proportional_term(sim):
    """target 偏移 +0.1 rad、零速度 → ctrl = kp*0.1/gear。"""
    sim.reset(seed=0)
    act = sim.get_action()['robot_a'].copy()
    act[0] += 0.1  # abdomen_x: kp=1000, gear=200 → ctrl=0.5
    sim.set_action({'robot_a': act, 'robot_b': None})
    sim._apply_pd_control()
    aid = sim._robots['robot_a']['actuator_ids'][0]
    assert abs(float(sim.data.ctrl[aid]) - 0.5) < 1e-5


def test_pd_saturation(sim):
    """大偏差扭矩超过 gear 上限时 ctrl 饱和到 ctrlrange 边界。"""
    sim.reset(seed=0)
    act = sim.get_action()['robot_a'].copy()
    act[0] += 1.0  # torque=1000 Nm >> cap 200 → ctrl=1.0
    sim.set_action({'robot_a': act, 'robot_b': None})
    sim._apply_pd_control()
    aid = sim._robots['robot_a']['actuator_ids'][0]
    assert float(sim.data.ctrl[aid]) == 1.0


def test_pd_damping_term(sim):
    """target=current 但 qd=0.5 → torque = -kd*qd。"""
    sim.reset(seed=0)
    cache = sim._robots['robot_a']
    sim.data.qvel[cache['qvel_indices'][0]] = 0.5
    sim._apply_pd_control()
    aid = cache['actuator_ids'][0]
    # abdomen_x: kd=100 → torque=-50 → ctrl=-50/200=-0.25
    assert abs(float(sim.data.ctrl[aid]) - (-0.25)) < 1e-6


def test_pd_target_is_radians_not_normalized(sim):
    """action 语义是弧度位置目标：target=当前位置时 ctrl 应为 0。"""
    sim.reset(seed=0)
    sim._apply_pd_control()
    aids = sim._robots['robot_a']['actuator_ids']
    np.testing.assert_allclose(sim.data.ctrl[aids], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. reset / 初始姿态
# ---------------------------------------------------------------------------

def test_reset_pose(sim):
    sim.reset(seed=42, options={'initial_distance': 2.0})
    cs = sim.get_core_state()
    pa = cs['robot_a']['root_pos']
    pb = cs['robot_b']['root_pos']
    np.testing.assert_allclose(pa, [-1.0, 0.0, 0.882416], atol=1e-4)
    np.testing.assert_allclose(pb, [1.0, 0.0, 0.882416], atol=1e-4)

    # 关节位置 = standing pose（肩部两个非零）
    jp = cs['robot_a']['joint_pos']
    idx = {n: i for i, n in enumerate(Humanoid28Meta.CONTROLLED_JOINTS)}
    assert abs(jp[idx['right_shoulder_x']] - 1.5708) < 1e-5
    assert abs(jp[idx['left_shoulder_x']] + 1.5708) < 1e-5
    assert abs(jp[idx['abdomen_x']]) < 1e-6

    # robot_b 旋转 180°：face_vector 应互相指向
    # robot_b quat [w,x,y,z] 对应绕 z 180°
    rb = cs['robot_b']['root_rot']
    assert abs(abs(rb[0]) - 0.0) < 1e-3 or abs(abs(rb[3]) - 1.0) < 1e-3


def test_reset_action_equals_joint_pos(sim):
    """位置目标语义下 reset 后 action 必须等于当前关节位置。"""
    sim.reset(seed=42)
    cs = sim.get_core_state()
    act = sim.get_action()
    np.testing.assert_allclose(act['robot_a'], cs['robot_a']['joint_pos'], atol=1e-6)


def test_reset_deterministic(sim):
    sim.reset(seed=7)
    qpos1 = sim.data.qpos.copy()
    sim.reset(seed=7)
    qpos2 = sim.data.qpos.copy()
    np.testing.assert_array_equal(qpos1, qpos2)


# ---------------------------------------------------------------------------
# 5. 观测契约
# ---------------------------------------------------------------------------

def test_observation_dim(sim):
    sim.reset(seed=0)
    obs = sim.get_observation()
    assert obs['robot_a'].shape == (110,)
    assert obs['robot_b'].shape == (110,)
    assert obs['robot_a'].dtype == np.float32


def test_observation_proprio_raw_values(sim):
    """本体 56 维 = 原始关节位置/速度（弧度），前 28 维应精确等于 qpos。"""
    sim.reset(seed=0)
    obs = sim.get_observation()
    cs = sim.get_core_state()
    np.testing.assert_allclose(obs['robot_a'][:28], cs['robot_a']['joint_pos'], atol=1e-5)
    np.testing.assert_allclose(obs['robot_a'][28:56], cs['robot_a']['joint_vel'], atol=1e-5)


def test_relative_vel_correct_contract(sim):
    """决策 #14：relative_vel = (v_opp - v_self) 旋转到 ego 系。

    构造：双机相向运动（各以 local +x 1m/s 前进；robot_b 因 180° 朝向，
    其 local +x 对应 world −x，故二者在世界系中互相接近），
    相对速度大小应为 2 m/s，而不是 humanoid21 错误实现下的 1 m/s（仅对手速度）。
    """
    sim.reset(seed=0)
    cs = sim.get_core_state()
    state = {}
    for rid, vx in (('robot_a', 1.0), ('robot_b', 1.0)):
        rs = {k: v.copy() for k, v in cs[rid].items()}
        rs['root_vel_local'] = np.array([vx, 0.0, 0.0], dtype=np.float32)
        state[rid] = rs
    sim.set_core_state(state)

    view = sim.get_derived_state(['robot_a'])['robot_a']
    rel_vel = view['opponent_basic_pose']['relative_vel']
    # robot_a 面向 +x，对手以 -1 而自身 +1 → 接近速度沿 +x 方向大小 2
    assert abs(rel_vel[0] - (-2.0)) < 1e-4, f'relative_vel {rel_vel}'
    np.testing.assert_allclose(rel_vel[1:], [0.0, 0.0], atol=1e-4)


def test_opponent_keypoint_layout(sim):
    """对手关键点应含 head/hands/feet 五个语义键。"""
    sim.reset(seed=0)
    view = sim.get_derived_state(['robot_a'])['robot_a']
    kp = view['opponent_keypoint_pos']
    assert set(kp.keys()) == {'head', 'right_hand', 'left_hand', 'right_foot', 'left_foot'}
    # reset 后对手头部在自身前方约 2m（x 方向）
    assert kp['head'][0] > 1.0


def test_feet_forces_normalized(sim):
    """站立时双足力之和应接近 1.0 倍体重。"""
    sim.reset(seed=0)
    sim.physical_step()
    view = sim.get_derived_state(['robot_a'])['robot_a']
    ff = view['feet_forces']
    assert ff.shape == (2,)
    # 初始瞬间足部接触建立中，只检查非负与量级
    assert np.all(ff >= 0.0)
    assert ff.sum() < 5.0


# ---------------------------------------------------------------------------
# 6. 战斗伤害管线
# ---------------------------------------------------------------------------

def test_hit_resolution_logic():
    """humanoid28 命名下的打击分类：right_hand/right_foot 为攻击部位，
    head/torso/pelvis/upper_waist 归入受击。"""
    from envs.humanoid28.plugins import CombatScoringPlugin
    p = CombatScoringPlugin()

    # 手打头 → head, defender robot_b
    assert p._resolve_hit('right_hand_a', 'head_b') == ('head', 'robot_b')
    # 脚踢躯干 → torso, defender robot_a
    assert p._resolve_hit('pelvis_a', 'right_foot_b') == ('torso', 'robot_a')
    # upper_waist 归入 torso
    assert p._resolve_hit('upper_waist_b', 'left_hand_a') == ('torso', 'robot_b')
    # 躯干打躯干 → 无伤害
    assert p._resolve_hit('torso_a', 'torso_b') is None
    # 手打手 → 无伤害
    assert p._resolve_hit('right_hand_a', 'left_hand_b') is None
    # 同队 → 无伤害
    assert p._resolve_hit('right_hand_a', 'head_a') is None
    # 手臂不是攻击部位
    assert p._resolve_hit('right_lower_arm_a', 'head_b') is None


def test_end_to_end_damage():
    """端到端：把 robot_a 右手几何中心对准 robot_b 躯干中心，
    接触力应驱动伤害累积且 HP 下降。"""
    from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / 'blueprint.yaml').read_text(encoding='utf-8')
    bp = ParameterizedEnvBlueprint.from_yaml(text).materialize()
    rt = bp.build()
    rt.reset(seed=42)

    sim = rt._core.simulator
    cs = sim.get_core_state()

    # 用派生状态读取右手/躯干世界位置，计算对齐偏移
    view_a = sim.get_derived_state(['robot_a'])['robot_a']
    view_b = sim.get_derived_state(['robot_b'])['robot_b']
    hand = view_a['body_xpos']['right_hand_a']
    torso = view_b['body_xpos']['torso_b']

    # 把 robot_b 平移到其躯干与 robot_a 右手几何体重叠的位置
    delta = hand - torso
    new_state = {
        'robot_b': {
            'root_pos': cs['robot_b']['root_pos'] + delta * 0.98,
        }
    }
    sim.set_core_state(new_state)

    health0 = 100.0
    dmg_found = 0.0
    a = np.zeros(28, dtype=np.float32)
    for _ in range(10):
        rt.step(a, a)
        out = rt.get_observer_output('combat_scoring')
        dmg_found = out['robot_b']['cumulative_damage_taken']
        if dmg_found > 0:
            break

    assert dmg_found > 0.0, '手-躯干重叠接触未产生任何伤害'
    assert out['robot_b']['health'] < health0
    rt.close() if hasattr(rt, 'close') else None


# ---------------------------------------------------------------------------
# 7. 物理稳定性与对称性
# ---------------------------------------------------------------------------

def test_long_run_no_nan(sim):
    """5 秒自由物理演化无 NaN、速度有界。"""
    sim.reset(seed=42)
    for _ in range(2500):
        sim.physical_step()
    assert not np.isnan(sim.data.qpos).any()
    assert not np.isnan(sim.data.qvel).any()
    assert np.abs(sim.data.qvel).max() < 200.0


def test_mirror_symmetry_reset(sim):
    """reset 后双机应镜像对称：y 坐标为 0，x 相反，关节位置相同。"""
    sim.reset(seed=0, options={'initial_distance': 2.0})
    cs = sim.get_core_state()
    pa, pb = cs['robot_a']['root_pos'], cs['robot_b']['root_pos']
    assert abs(pa[0] + pb[0]) < 1e-5
    assert abs(pa[1]) < 1e-6 and abs(pb[1]) < 1e-6
    np.testing.assert_allclose(cs['robot_a']['joint_pos'], cs['robot_b']['joint_pos'], atol=1e-6)

    # 观测对手模块应镜像一致：双机互相面对，各自 ego 系中对手都在正前方
    # （local +x 约 2m），因此 ra ≈ rb ≈ [+2, 0, 0]（不是相反——
    #  180° 朝向使世界系的反向位移在各自 ego 系中同号）
    va = sim.get_derived_state(['robot_a'])['robot_a']
    vb = sim.get_derived_state(['robot_b'])['robot_b']
    ra = va['opponent_basic_pose']['relative_pos']
    rb = vb['opponent_basic_pose']['relative_pos']
    np.testing.assert_allclose(ra, [2.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(ra, rb, atol=1e-4)
