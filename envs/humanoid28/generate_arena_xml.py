"""
humanoid28 仿真环境 XML 生成脚本（决策 #11：环境 XML 必须脚本生成，禁止手写）。

输入：
  assets/mimickit_humanoid.xml — MimicKit 单机器人角色资产（Apache-2.0，
      vendored 自 https://github.com/xbpeng/MimicKit
      data/assets/humanoid/humanoid.xml，勿手改；修改上游资产后重跑本脚本）。

输出：
  battle_humanoid28.xml  — 双机器人 + 圆形竞技场完整 MuJoCo 场景
  robot_actuation.json   — 逐关节 PD/执行器参数（kp/kd/gear/range/armature），
      按 XML 文档顺序排列，供 meta.py 运行时加载。

关键转换语义（每项都对应 bootstrip.md 中的一条决策）：

1. 【被动动力学剥离，决策 #2/#5】MimicKit XML 中 joint 的 stiffness/damping
   在 MimicKit 各引擎里被当作 *PD 控制增益* 使用（Newton 显式清零被动值，
   见 mimickit/engines/newton_engine.py _build_controls），不是被动弹簧。
   本脚本从 joint 上**删除** stiffness/damping 属性（MuJoCo 默认即为 0），
   并把它们写入 robot_actuation.json 作为 PD 增益 —— 若保留在 XML 里，
   MuJoCo 会把它们当成强被动弹簧/阻尼，动力学完全跑偏。

2. 【接触，用户决策 condim=3】geom condim 一律写 3（上游 XML 为 1，
   那是 MJCF-for-IsaacGym 的遗留产物；Isaac Gym 中该模型的接触本就有摩擦。
   condim=1 会导致 MuJoCo 中机器人间接触完全无摩擦，格斗推搡失效）。
   friction/solimp/solref 原样继承上游 class 默认值
   （friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"）。

3. 【环境，决策 #8】竞技场几何/材质与 humanoid21 的 battle_circular_v2.xml
   一致：圆形 24 墙（半径 3.4106m，首墙偏置 7.5°，间隔 15°）、地面半尺寸
   3.44m、天花板 z=6.10m、相同相机与贴图。地面摩擦 (1.0, 0.005, 0.0001)
   逐维不大于机器人 geom 摩擦 (1.0, 0.05, 0.05)，MuJoCo 按逐维 max 组合后
   有效摩擦恰好等于机器人自身的 MimicKit 值。

4. 【双机器人】机器人子树实例化两份，所有 body/geom/joint/freejoint 名称
   追加 _a / _b 后缀；root body（pelvis）pos 保持 "0 0 0" —— MuJoCo
   freejoint 的 qpos[0:3] 是绝对世界坐标（已实测验证），reset() 会写入
   绝对出生位置。robot_a 红色、robot_b 蓝色（humanoid21 配色惯例）。

5. 【求解器】option timestep=0.002 impratio=10 —— 与 humanoid21 及
   MimicKit Newton 引擎（SolverMuJoCo impratio=10）一致；iterations/
   ls_iterations 用 MuJoCo 默认值（100/50），与 Newton 配置相同。

用法：
  cd /data1/mono/things/combatbench
  python3 envs/humanoid28/generate_arena_xml.py
"""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
SOURCE_XML = HERE / 'assets' / 'mimickit_humanoid.xml'
OUT_XML = HERE / 'battle_humanoid28.xml'
OUT_JSON = HERE / 'robot_actuation.json'

# ---------------------------------------------------------------------------
# 竞技场参数 —— 与 humanoid21/battle_circular_v2.xml 逐项对应
# ---------------------------------------------------------------------------
ARENA = {
    'ground_half_size': 3.44,      # ground plane size="3.44 3.44 0.1"
    'wall_radius': 3.4106,         # 墙中心圆半径（由 circular_v2 wall_00 反解）
    'wall_count': 24,
    'wall_angle_offset_deg': 7.5,  # 首墙方位角
    'wall_half_width': 0.4490,     # 墙平面切向半宽（与 15° 弦长匹配）
    'wall_half_height': 3.05,      # 墙中心 z=3.05, 半高 3.05 → 覆盖 z∈[0, 6.1]
    'ceiling_z': 6.10,
    'ground_friction': '1.0 0.005 0.0001',
    'wall_friction': '0.3 0.005 0.0001',
    'env_solref': '0.02 1',
    'env_solimp': '0.9 0.95 0.001',
}

ROBOTS = {
    '_a': '1 0.2 0.2 1',   # robot_a 红色（humanoid21 惯例）
    '_b': '0.2 0.2 1 1',   # robot_b 蓝色
}


def parse_source():
    """解析 MimicKit 单机器人 XML，返回 (root_body_elem, joints_meta, motors)。

    joints_meta: 按文档顺序的受控关节参数列表。
    motors: {joint_name: gear}
    """
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    root_body = worldbody.find('body')
    assert root_body is not None and root_body.attrib.get('name') == 'pelvis', \
        '上游资产结构假设：worldbody 下唯一 root body 为 pelvis'

    # 按文档顺序遍历全部 hinge joint（freejoint 除外）
    joints_meta = []
    for j in root_body.iter('joint'):
        if j.attrib.get('type', 'hinge') != 'hinge':
            raise ValueError(f"非 hinge joint 不受支持: {j.attrib}")
        joints_meta.append({
            'name': j.attrib['name'],
            'axis': j.attrib['axis'],
            'range': j.attrib['range'],
            'stiffness': float(j.attrib['stiffness']),
            'damping': float(j.attrib['damping']),
            'armature': float(j.attrib.get('armature', '0')),
            'actuatorfrcrange': j.attrib.get('actuatorfrcrange'),
        })

    motors = {}
    actuator = root.find('actuator')
    for m in actuator.iter('motor'):
        motors[m.attrib['joint']] = float(m.attrib['gear'])

    # 每个受控关节都必须有同名 motor
    missing = [j['name'] for j in joints_meta if j['name'] not in motors]
    assert not missing, f'缺少 motor 的关节: {missing}'

    return root_body, joints_meta, motors


def instantiate_robot(root_body: ET.Element, suffix: str, rgba: str) -> ET.Element:
    """深拷贝机器人子树并实例化：名称加后缀、剥离被动刚度阻尼、condim=3、着色。"""
    elem = copy.deepcopy(root_body)

    # root body pos 归零（freejoint qpos 是绝对世界坐标，reset 时写入出生点）
    elem.attrib['pos'] = '0 0 0'

    for e in elem.iter():
        if e.tag in ('body', 'geom', 'joint', 'freejoint') and 'name' in e.attrib:
            e.attrib['name'] = e.attrib['name'] + suffix
        if e.tag == 'joint':
            # 决策：stiffness/damping 是 PD 增益而非被动动力学，从 XML 剥离
            # （值已存入 robot_actuation.json，由 simulator 作为 PD 增益使用）
            e.attrib.pop('stiffness', None)
            e.attrib.pop('damping', None)
        elif e.tag == 'geom':
            # 用户决策：condim=3（还原 Isaac Gym 中该模型的有摩擦接触行为），
            # friction/solimp/solref 不在 geom 上而在 class 默认值里，见 <default>。
            e.attrib['condim'] = '3'
            e.attrib['rgba'] = rgba
    return elem


def build_wall(index: int) -> ET.Element:
    """生成圆形围墙的一片（24 分之一），几何与 circular_v2 一致。"""
    import math
    theta_deg = ARENA['wall_angle_offset_deg'] + index * (360.0 / ARENA['wall_count'])
    theta = math.radians(theta_deg)
    r = ARENA['wall_radius']
    cx, cy = r * math.cos(theta), r * math.sin(theta)

    # 平面 x 轴沿切向 (sinθ, -cosθ, 0)，y 轴竖直 (0,0,1) → 法向朝内（-径向）
    tx, ty = math.sin(theta), -math.cos(theta)
    xyaxes = f'{tx:.4f} {ty:.4f} 0.0000 0.0000 0.0000 1.0000'

    g = ET.Element('geom')
    g.set('name', f'wall_{index:02d}')
    g.set('type', 'plane')
    g.set('pos', f'{cx:.4f} {cy:.4f} {ARENA["wall_half_height"]:.4f}')
    g.set('size', f'{ARENA["wall_half_width"]:.4f} {ARENA["wall_half_height"]:.4f} 0.1')
    g.set('xyaxes', xyaxes)
    g.set('condim', '3')
    g.set('friction', ARENA['wall_friction'])
    g.set('solref', ARENA['env_solref'])
    g.set('solimp', ARENA['env_solimp'])
    g.set('material', f'mat_wall_{index % 6}')
    return g


def build_xml(root_body: ET.Element) -> ET.Element:
    mujoco = ET.Element('mujoco', model='Humanoid28_Battle_Arena')
    ET.SubElement(mujoco, 'compiler', angle='radian')
    ET.SubElement(mujoco, 'option', timestep='0.002', impratio='10')
    # 离屏 framebuffer 需显式放大，否则默认 640x480 无法渲染 1280x720 广播视角
    visual = ET.SubElement(mujoco, 'visual')
    ET.SubElement(visual, 'global', offwidth='1920', offheight='1080')

    # --- default：与上游 MimicKit XML 相同的结构，geom condim 改为 3 ---
    default = ET.SubElement(mujoco, 'default')
    ET.SubElement(default, 'motor', ctrlrange='-1 1', ctrllimited='true')
    body_cls = ET.SubElement(default, 'default', {'class': 'body'})
    ET.SubElement(body_cls, 'geom', condim='3', friction='1.0 0.05 0.05',
                  solimp='.9 .99 .003', solref='.015 1')
    ET.SubElement(body_cls, 'joint', limited='true', solimplimit='0 .99 .01')

    # --- asset：材质贴图（复制自 circular_v2，文件在 textures/）---
    asset = ET.SubElement(mujoco, 'asset')
    ET.SubElement(asset, 'texture', name='tex_floor', type='2d',
                  file='textures/floor_circular.png')
    ET.SubElement(asset, 'material', name='mat_floor', texture='tex_floor',
                  texrepeat='1 1', texuniform='false', emission='1',
                  shininess='0', specular='0')
    ET.SubElement(asset, 'texture', name='tex_ceiling', type='2d',
                  file='textures/ceiling.png')
    ET.SubElement(asset, 'material', name='mat_ceiling', texture='tex_ceiling',
                  texrepeat='1 1', texuniform='false', emission='1',
                  shininess='0', specular='0')
    for i in range(6):
        ET.SubElement(asset, 'texture', name=f'tex_wall_{i}', type='2d',
                      file=f'textures/wall_{i}.png')
        ET.SubElement(asset, 'material', name=f'mat_wall_{i}',
                      texture=f'tex_wall_{i}', texrepeat='1 1',
                      texuniform='false', emission='1', shininess='0', specular='0')

    # --- worldbody ---
    worldbody = ET.SubElement(mujoco, 'worldbody')

    # 相机：与 circular_v2 完全一致
    cameras = [
        ('cam_diag_0', '2.2203 2.2203 4.0000', '-0.7071 0.7071 0.0000 -0.4404 -0.4404 0.7823'),
        ('cam_diag_1', '-2.2203 2.2203 4.0000', '-0.7071 -0.7071 0.0000 0.4404 -0.4404 0.7823'),
        ('cam_diag_2', '-2.2203 -2.2203 4.0000', '0.7071 -0.7071 0.0000 0.4404 0.4404 0.7823'),
        ('cam_diag_3', '2.2203 -2.2203 4.0000', '0.7071 0.7071 -0.0000 -0.4404 0.4404 0.7823'),
        ('cam_east', '3.1400 0.0000 3.0000', '0.0000 1.0000 -0.0000 -0.4310 0.0000 0.9023'),
        ('cam_south', '0.0000 3.1400 3.0000', '-1.0000 0.0000 0.0000 -0.0000 -0.4310 0.9023'),
        ('cam_west', '-3.1400 0.0000 3.0000', '-0.0000 -1.0000 0.0000 0.4310 -0.0000 0.9023'),
        ('cam_north', '-0.0000 -3.1400 3.0000', '1.0000 -0.0000 0.0000 0.0000 0.4310 0.9023'),
        ('ceilingcamera', '0.0000 0.0000 6.0000', '1.0000 0.0000 0.0000 -0.0000 1.0000 0.0000'),
    ]
    for name, pos, xyaxes in cameras:
        ET.SubElement(worldbody, 'camera', name=name, pos=pos, xyaxes=xyaxes)

    hs = ARENA['ground_half_size']
    ET.SubElement(worldbody, 'geom', name='ground', type='plane',
                  pos='0.0 0.0 0.0', size=f'{hs} {hs} 0.1',
                  xyaxes='1.0000 0.0000 0.0000 0.0000 1.0000 0.0000',
                  condim='3', friction=ARENA['ground_friction'],
                  solref=ARENA['env_solref'], solimp=ARENA['env_solimp'],
                  material='mat_floor')
    ET.SubElement(worldbody, 'geom', name='ceiling', type='plane',
                  pos=f'0.0 0.0 {ARENA["ceiling_z"]:.2f}', size=f'{hs} {hs} 0.1',
                  xyaxes='-1.0000 0.0000 0.0000 0.0000 1.0000 0.0000',
                  condim='3', friction=ARENA['ground_friction'],
                  solref=ARENA['env_solref'], solimp=ARENA['env_solimp'],
                  material='mat_ceiling')
    for i in range(ARENA['wall_count']):
        worldbody.append(build_wall(i))

    # --- 双机器人实例 ---
    for suffix, rgba in ROBOTS.items():
        worldbody.append(instantiate_robot(root_body, suffix, rgba))

    # --- actuator：motor 名 = joint 名（与 humanoid21 同约定），顺序 = 文档顺序 ---
    actuator = ET.SubElement(mujoco, 'actuator')
    for suffix in ROBOTS:
        for j in root_body.iter('joint'):
            jname = j.attrib['name'] + suffix
            gear = MOTORS[j.attrib['name']]
            ET.SubElement(actuator, 'motor', name=jname,
                          gear=repr(gear) if gear != int(gear) else str(int(gear)),
                          joint=jname)

    return mujoco


def indent(elem: ET.Element, level: int = 0):
    """ElementTree  Pretty-print（Python 3.9+ 可用 ET.indent，但保持自实现以免缩进风格意外）。"""
    ET.indent(elem, space='  ', level=level)


MOTORS = {}


def main():
    global MOTORS
    root_body, joints_meta, motors = parse_source()
    MOTORS = motors

    mujoco = build_xml(root_body)
    indent(mujoco)
    tree = ET.ElementTree(mujoco)
    header = (
        "AUTO-GENERATED by generate_arena_xml.py — DO NOT EDIT BY HAND.\n"
        "  Robot asset derived from MimicKit (Apache-2.0):\n"
        "  https://github.com/xbpeng/MimicKit data/assets/humanoid/humanoid.xml\n"
        "  Regenerate: python3 envs/humanoid28/generate_arena_xml.py\n"
    )
    with open(OUT_XML, 'w', encoding='utf-8') as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write(f'<!--{header}-->\n')
        tree.write(f, encoding='unicode', xml_declaration=False)
        f.write('\n')

    # --- actuation JSON：PD 增益 + 执行器参数，按文档顺序 ---
    payload = {
        'source': 'assets/mimickit_humanoid.xml (MimicKit, Apache-2.0)',
        'generator': 'generate_arena_xml.py',
        'convention': {
            'stiffness': 'PD 比例增益 kp（非被动弹簧）',
            'damping': 'PD 微分增益 kd（非被动阻尼）',
            'gear': 'motor gear，ctrlrange [-1,1] 下数值上等效扭矩上限 (Nm)',
            'order': 'XML 文档顺序（depth-first），与 MimicKit dof 顺序一致',
        },
        'joints': [
            {
                'name': j['name'],
                'range': [float(x) for x in j['range'].split()],
                'kp': j['stiffness'],
                'kd': j['damping'],
                'gear': motors[j['name']],
                'armature': j['armature'],
                'actuatorfrcrange': (
                    [float(x) for x in j['actuatorfrcrange'].split()]
                    if j['actuatorfrcrange'] else None
                ),
            }
            for j in joints_meta
        ],
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'written: {OUT_XML}')
    print(f'written: {OUT_JSON}')
    print(f'joints: {len(joints_meta)}')
    for j in payload['joints']:
        print(f"  {j['name']:>18}  range=[{j['range'][0]:+.4f},{j['range'][1]:+.4f}]"
              f"  kp={j['kp']:.0f} kd={j['kd']:.0f} gear={j['gear']:.0f}")


if __name__ == '__main__':
    main()
