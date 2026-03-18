import xml.etree.ElementTree as ET
import math
import os

def add_suffix_to_names(root, suffix, color_rgba):
    """
    Add suffix to all names and references, and force color change
    """
    # Attribute names that need suffix
    ref_attrs = {'name', 'body', 'body1', 'body2', 'joint', 'target', 
                 'material', 'texture', 'mesh', 'class', 'childclass', 'tendon',
                 'site', 'objname', 'joint'}  # Add site and sensor related attributes

    for elem in root.iter():
        # 1. Process attribute suffix
        for attr, value in list(elem.attrib.items()):
            if attr in ref_attrs:
                elem.set(attr, f"{value}{suffix}")
            # Special handling for joint in actuator
            if elem.tag == 'motor' and attr == 'joint':
                elem.set(attr, f"{value}{suffix}")
            # Process site attribute in sensor
            if elem.tag in ['force', 'framequat', 'gyro', 'accelerometer', 'framepos', 'framelinvel'] and attr == 'site':
                elem.set(attr, f"{value}{suffix}")

        # 2. Color override logic
        if elem.tag in ['geom', 'material']:
            elem.set('rgba', color_rgba)
            # If geom references material, ensure referenced material name also has suffix
            if 'material' in elem.attrib:
                mat_name = elem.get('material')
                if not mat_name.endswith(suffix):
                    elem.set('material', f"{mat_name}{suffix}")

def get_yaw(pos, target):
    """
    Calculate Yaw angle in degrees towards target point
    Since G1 compiler uses angle="radian", we must return radian values, not degree values.
    """
    return math.atan2(target[1] - pos[1], target[0] - pos[0])

def assemble_battle_scene(robot_xml, arena_xml, output_xml, robots_config):
    """
    Main assembly logic - G1 robot version
    robots_config: list of dicts, e.g., [{"pos": [1,0], "target": [0,0], "suffix": "_red", "color": "1 0 0 1"}]
    """
    # Load main scene
    arena_tree = ET.parse(arena_xml)
    arena_root = arena_tree.getroot()
    
    # Ensure compiler settings exist (G1 requires meshdir)
    compiler = arena_root.find('compiler')
    if compiler is None:
        # Copy compiler settings from robot XML
        robot_tree_temp = ET.parse(robot_xml)
        robot_compiler = robot_tree_temp.getroot().find('compiler')
        if robot_compiler is not None:
            arena_root.insert(0, robot_compiler)
    
    # Ensure main scene has necessary top-level containers
    def get_or_create(parent, tag):
        element = parent.find(tag)
        if element is None:
            # Inserting before worldbody is more standard
            element = ET.Element(tag)
            parent.insert(0, element)
        return element

    worldbody = get_or_create(arena_root, 'worldbody')
    asset_root = get_or_create(arena_root, 'asset')
    actuator_root = get_or_create(arena_root, 'actuator')
    tendon_root = get_or_create(arena_root, 'tendon')
    contact_root = get_or_create(arena_root, 'contact')
    default_root = get_or_create(arena_root, 'default')
    sensor_root = get_or_create(arena_root, 'sensor')  # G1 has sensors

    for cfg in robots_config:
        # Reload robot template each time
        robot_tree = ET.parse(robot_xml)
        robot_root = robot_tree.getroot()

        # 1. Pre-processing: add suffix and change color
        add_suffix_to_names(robot_root, cfg["suffix"], cfg["color"])

        # 2. Move Assets (Texture, Material, Mesh)
        rob_assets = robot_root.find('asset')
        if rob_assets is not None:
            for item in list(rob_assets):
                asset_root.append(item)

        # 3. Move Actuators
        rob_actuators = robot_root.find('actuator')
        if rob_actuators is not None:
            for item in list(rob_actuators):
                actuator_root.append(item)

        # 4. Move Tendons
        rob_tendons = robot_root.find('tendon')
        if rob_tendons is not None:
            for item in list(rob_tendons):
                tendon_root.append(item)

        # 5. Move Contact Excludes
        rob_contacts = robot_root.find('contact')
        if rob_contacts is not None:
            for item in list(rob_contacts):
                contact_root.append(item)

        # 6. Core fix: Move Default Classes
        rob_default = robot_root.find('default')
        if rob_default is not None:
            for child in list(rob_default):
                if child.tag == 'default': # This is a class definition
                    default_root.append(child)
                elif child.tag in ['motor', 'joint', 'geom']:
                    # Add only if global container doesn't have base definition yet
                    if default_root.find(child.tag) is None:
                        default_root.append(child)

        # 7. Move Sensors (G1 specific)
        rob_sensors = robot_root.findall('sensor')
        if rob_sensors is not None:
            for rob_sensor in rob_sensors:
                for item in list(rob_sensor):
                    sensor_root.append(item)

        # 8. Place robot body into Worldbody
        rob_world = robot_root.find('worldbody')
        if rob_world is not None:
            main_body = rob_world.find('body')
            if main_body is not None:
                # Set coordinates and orientation
                yaw = get_yaw(cfg["pos"], cfg["target"])
                # G1 robot pelvis initial position is z=0.793, this is standing height
                main_body.set('pos', f"{cfg['pos'][0]} {cfg['pos'][1]} {cfg.get('z', 0.793)}")
                # Remove original quat attribute, use euler to set orientation
                if 'quat' in main_body.attrib:
                    del main_body.attrib['quat']
                main_body.set('euler', f"0 0 {yaw}")
                worldbody.append(main_body)

    # Export
    arena_tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated battle scene: {output_xml}")

# --- Execution area ---
if __name__ == "__main__":
    # Configure two battling G1 robots
    battle_configs = [
        {
            "suffix": "_red",
            "color": "1 0.2 0.2 1",
            "pos": [-1.5, 0],
            "target": [0, 0],
            "z": 0.793  # G1 standing height
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
        robot_xml="g1_29dof_no_hand.xml",   # G1 robot file
        arena_xml="scene.xml",              # Environment file
        output_xml="battle_g1.xml",          # Output result
        robots_config=battle_configs
    )
