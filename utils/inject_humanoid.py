import xml.etree.ElementTree as ET
import math
import os

def add_suffix_to_names(root, suffix, color_rgba):
    """
    Add suffix to all names and references, and force color change
    """
    # Attribute names that need suffix
    ref_attrs = {'name', 'body', 'body1', 'body2', 'joint', 'target', 
                 'material', 'texture', 'mesh', 'class', 'childclass', 'tendon'}

    for elem in root.iter():
        # 1. Process attribute suffix
        for attr, value in list(elem.attrib.items()):
            if attr in ref_attrs:
                elem.set(attr, f"{value}{suffix}")
            # Special handling for joint in actuator
            if elem.tag == 'motor' and attr == 'joint':
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
    """Calculate Yaw angle in degrees towards target point"""
    return math.degrees(math.atan2(target[1] - pos[1], target[0] - pos[0]))

def assemble_battle_scene(robot_xml, arena_xml, output_xml, robots_config):
    """
    Main assembly logic
    robots_config: list of dicts, e.g., [{"pos": [1,0], "target": [0,0], "suffix": "_red", "color": "1 0 0 1"}]
    """
    # Load main scene
    arena_tree = ET.parse(arena_xml)
    arena_root = arena_tree.getroot()
    
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
        # We only copy sub-defaults (classes) under robot default to avoid duplicating global motor/joint
        rob_default = robot_root.find('default')
        if rob_default is not None:
            for child in list(rob_default):
                if child.tag == 'default': # This is a class definition
                    default_root.append(child)
                elif child.tag in ['motor', 'joint', 'geom']:
                    # Add only if global container doesn't have base definition yet
                    if default_root.find(child.tag) is None:
                        default_root.append(child)

        # 7. Place robot body into Worldbody
        rob_world = robot_root.find('worldbody')
        if rob_world is not None:
            main_body = rob_world.find('body')
            if main_body is not None:
                # Set coordinates and orientation
                yaw = get_yaw(cfg["pos"], cfg["target"])
                main_body.set('pos', f"{cfg['pos'][0]} {cfg['pos'][1]} 1.282")
                main_body.set('euler', f"0 0 {yaw}")
                worldbody.append(main_body)

    # Export
    arena_tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"Successfully generated battle scene: {output_xml}")

# --- Execution area ---
if __name__ == "__main__":
    # Configure two battle robots
    battle_configs = [
        {
            "suffix": "_red",
            "color": "1 0.2 0.2 1",
            "pos": [-1.5, 0],
            "target": [0, 0]
        },
        {
            "suffix": "_blue",
            "color": "0.2 0.2 1 1",
            "pos": [1.5, 0],
            "target": [0, 0]
        }
    ]

    assemble_battle_scene(
        robot_xml="humanoid.xml",   # Your robot file
        arena_xml="scene4.xml",      # Your environment file
        output_xml="battle_v1.xml", # Output result
        robots_config=battle_configs
    )