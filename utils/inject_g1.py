import xml.etree.ElementTree as ET
import math
import os

def add_suffix_to_names(root, suffix, color_rgba):
    """
    为所有名称和引用添加后缀，并强制修改颜色
    """
    # 需要添加后缀的属性名
    ref_attrs = {'name', 'body', 'body1', 'body2', 'joint', 'target', 
                 'material', 'texture', 'mesh', 'class', 'childclass', 'tendon',
                 'site', 'objname', 'joint'}  # 添加site和sensor相关属性

    for elem in root.iter():
        # 1. 处理属性后缀
        for attr, value in list(elem.attrib.items()):
            if attr in ref_attrs:
                elem.set(attr, f"{value}{suffix}")
            # 特殊处理 actuator 里的 joint
            if elem.tag == 'motor' and attr == 'joint':
                elem.set(attr, f"{value}{suffix}")
            # 处理sensor中的site属性
            if elem.tag in ['force', 'framequat', 'gyro', 'accelerometer', 'framepos', 'framelinvel'] and attr == 'site':
                elem.set(attr, f"{value}{suffix}")

        # 2. 颜色覆盖逻辑
        if elem.tag in ['geom', 'material']:
            elem.set('rgba', color_rgba)
            # 如果 geom 引用了 material，确保引用的 material 名也带后缀
            if 'material' in elem.attrib:
                mat_name = elem.get('material')
                if not mat_name.endswith(suffix):
                    elem.set('material', f"{mat_name}{suffix}")

def get_yaw(pos, target):
    """
    计算朝向目标点的 Yaw 角度
    由于 G1 机器人的 compiler 设定是 angle="radian"，我们必须返回弧度值，而不是角度值。
    """
    return math.atan2(target[1] - pos[1], target[0] - pos[0])

def assemble_battle_scene(robot_xml, arena_xml, output_xml, robots_config):
    """
    主组装逻辑 - G1机器人版本
    robots_config: list of dicts, e.g., [{"pos": [1,0], "target": [0,0], "suffix": "_red", "color": "1 0 0 1"}]
    """
    # 加载主场景
    arena_tree = ET.parse(arena_xml)
    arena_root = arena_tree.getroot()
    
    # 确保compiler设置存在（G1需要meshdir）
    compiler = arena_root.find('compiler')
    if compiler is None:
        # 从机器人XML复制compiler设置
        robot_tree_temp = ET.parse(robot_xml)
        robot_compiler = robot_tree_temp.getroot().find('compiler')
        if robot_compiler is not None:
            arena_root.insert(0, robot_compiler)
    
    # 确保主场景有必要的顶级容器
    def get_or_create(parent, tag):
        element = parent.find(tag)
        if element is None:
            # 插入到 worldbody 之前比较规范
            element = ET.Element(tag)
            parent.insert(0, element)
        return element

    worldbody = get_or_create(arena_root, 'worldbody')
    asset_root = get_or_create(arena_root, 'asset')
    actuator_root = get_or_create(arena_root, 'actuator')
    tendon_root = get_or_create(arena_root, 'tendon')
    contact_root = get_or_create(arena_root, 'contact')
    default_root = get_or_create(arena_root, 'default')
    sensor_root = get_or_create(arena_root, 'sensor')  # G1有sensor

    for cfg in robots_config:
        # 每次重新加载机器人模板
        robot_tree = ET.parse(robot_xml)
        robot_root = robot_tree.getroot()

        # 1. 预处理：加后缀和变色
        add_suffix_to_names(robot_root, cfg["suffix"], cfg["color"])

        # 2. 移动 Assets (Texture, Material, Mesh)
        rob_assets = robot_root.find('asset')
        if rob_assets is not None:
            for item in list(rob_assets):
                asset_root.append(item)

        # 3. 移动 Actuators
        rob_actuators = robot_root.find('actuator')
        if rob_actuators is not None:
            for item in list(rob_actuators):
                actuator_root.append(item)

        # 4. 移动 Tendons
        rob_tendons = robot_root.find('tendon')
        if rob_tendons is not None:
            for item in list(rob_tendons):
                tendon_root.append(item)

        # 5. 移动 Contact Excludes
        rob_contacts = robot_root.find('contact')
        if rob_contacts is not None:
            for item in list(rob_contacts):
                contact_root.append(item)

        # 6. 核心修正：移动 Default Classes
        rob_default = robot_root.find('default')
        if rob_default is not None:
            for child in list(rob_default):
                if child.tag == 'default': # 这是一个 class 定义
                    default_root.append(child)
                elif child.tag in ['motor', 'joint', 'geom']:
                    # 如果全局容器里还没有基础定义，才添加
                    if default_root.find(child.tag) is None:
                        default_root.append(child)

        # 7. 移动 Sensors (G1特有)
        rob_sensors = robot_root.findall('sensor')
        if rob_sensors is not None:
            for rob_sensor in rob_sensors:
                for item in list(rob_sensor):
                    sensor_root.append(item)

        # 8. 放置机器人主体到 Worldbody
        rob_world = robot_root.find('worldbody')
        if rob_world is not None:
            main_body = rob_world.find('body')
            if main_body is not None:
                # 设置坐标和朝向
                yaw = get_yaw(cfg["pos"], cfg["target"])
                # G1机器人pelvis初始位置是z=0.793，这是站立高度
                main_body.set('pos', f"{cfg['pos'][0]} {cfg['pos'][1]} {cfg.get('z', 0.793)}")
                # 移除原有的quat属性，使用euler设置朝向
                if 'quat' in main_body.attrib:
                    del main_body.attrib['quat']
                main_body.set('euler', f"0 0 {yaw}")
                worldbody.append(main_body)

    # 导出
    arena_tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"成功生成对战场景: {output_xml}")

# --- 执行区 ---
if __name__ == "__main__":
    # 配置两个对战G1机器人
    battle_configs = [
        {
            "suffix": "_red",
            "color": "1 0.2 0.2 1",
            "pos": [-1.5, 0],
            "target": [0, 0],
            "z": 0.793  # G1站立高度
        },
        {
            "suffix": "_blue",
            "color": "0.2 0.2 1 1",
            "pos": [1.5, 0],
            "target": [0, 0],
            "z": 0.793
        }
    ]

    assemble_battle_scene(
        robot_xml="g1_29dof_no_hand.xml",   # G1机器人文件
        arena_xml="scene.xml",              # 环境文件
        output_xml="battle_g1.xml",          # 输出结果
        robots_config=battle_configs
    )
