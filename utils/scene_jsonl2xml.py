import json
import math
import os

# --- Basic vector math ---
def sub(a, b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
def add(a, b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
def norm(a): return math.sqrt(dot(a, a))
def normalize(a):
    n = norm(a)
    return [x/n for x in a] if n > 0 else [0, 0, 0]
def mul(a, s): return [x*s for x in a]

def generate_mujoco_xml(jsonl_path, output_xml_path):
    assets = {}
    geoms = []
    lights = []
    cameras = []

    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            
            obj_type = data.get("type")
            name = data.get("name", "unnamed")
            
            if obj_type == "plane":
                corners = data["corners"]
                texture_path = data.get("texture")
                
                # Material resources
                mat_name = ""
                if texture_path:
                    tex_name = "tex_" + os.path.splitext(os.path.basename(texture_path))[0]
                    mat_name = "mat_" + os.path.splitext(os.path.basename(texture_path))[0]
                    if tex_name not in assets:
                        assets[tex_name] = texture_path
                
                # 1. Calculate geometric center
                center = [sum(x)/4 for x in zip(*corners)]
                
                # 2. Determine plane dimensions (assuming corners form a rectangle)
                edge1 = sub(corners[1], corners[0])
                edge2 = sub(corners[3], corners[0])
                size_x = norm(edge1) / 2.0
                size_y = norm(edge2) / 2.0

                # 3. Construct stable local coordinate system (force text upright)
                # Default Y axis always points up [0, 0, 1]，Unless it is floor/ceiling
                if abs(center[0]) < 0.01 and abs(center[1]) < 0.01: # Floor or ceiling
                    X_local = [1, 0, 0]
                    Y_local = [0, 1, 0]
                else: # Wall
                    # Make Y axis always vertical up
                    Y_local = [0, 0, 1]
                    # X axis is the vector parallel to wall on horizontal plane
                    # Obtained by cross product of normal and up vector
                    raw_N = normalize(cross(edge1, edge2))
                    X_local = normalize(cross(Y_local, raw_N))
                
                # 4. Normal direction calibration (pointing to scene center)
                N_final = cross(X_local, Y_local)
                to_center = normalize(sub([0, 0, 0], center))
                
                # If normal faces away from center, horizontally flip X axis (flips normal but text mirrors horizontally)
                # To keep text unmirrored and normal correct, we flip logical definition of N when flipping X
                if dot(N_final, to_center) < 0:
                    X_local = mul(X_local, -1)
                
                # Special handling for floor normal must point up
                if abs(center[2]) < 0.01:
                    X_local = [1, 0, 0]
                    Y_local = [0, 1, 0]

                geoms.append({
                    "name": name,
                    "pos": center,
                    "size": [size_x, size_y, 0.1],
                    "xyaxes": X_local + Y_local,
                    "material": mat_name
                })
                
            elif obj_type == "light":
                lights.append({"name": name, "pos": data["position"]})
            elif obj_type == "camera":
                pos = data["position"]
                look_at = data["look_at"]
                look_dir = normalize(sub(look_at, pos))
                Z_cam = mul(look_dir, -1)
                Z_up = [0, 0, 1]
                if abs(dot(Z_cam, Z_up)) > 0.999:
                    X_cam = [1, 0, 0]
                    Y_cam = cross(Z_cam, X_cam)
                else:
                    X_cam = normalize(cross(Z_up, Z_cam))
                    Y_cam = cross(Z_cam, X_cam)
                cameras.append({"name": name, "pos": pos, "xyaxes": X_cam + Y_cam})

    # --- XML assembly ---
    xml_output = [
        '<mujoco model="Laddermoon_Arena">',
        '    <visual>',
        '        <global offwidth="1920" offheight="1080" />',
        '        <headlight ambient="0.4 0.4 0.4" diffuse="0.4 0.4 0.4" />',
        '    </visual>',
        '    <asset>'
    ]
    
    for tex_name, tex_file in assets.items():
        xml_output.append(f'        <texture name="{tex_name}" type="2d" file="{tex_file}" />')
        xml_output.append(f'        <material name="mat_{tex_name.split("tex_")[1]}" texture="{tex_name}" '
                          f'texrepeat="1 1" texuniform="false" emission="1" shininess="0" specular="0" />')
    
    xml_output.append('    </asset>\n    <worldbody>')
    
    for l in lights:
        xml_output.append(f'        <light name="{l["name"]}" pos="{l["pos"][0]} {l["pos"][1]} {l["pos"][2]}" directional="false" />')
        
    for c in cameras:
        ax = c["xyaxes"]
        xml_output.append(f'        <camera name="{c["name"]}" pos="{c["pos"][0]} {c["pos"][1]} {c["pos"][2]}" '
                          f'xyaxes="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f} {ax[3]:.4f} {ax[4]:.4f} {ax[5]:.4f}" />')
        
    for g in geoms:
        ax = g["xyaxes"]
        xml_output.append(f'        <geom name="{g["name"]}" type="plane" pos="{g["pos"][0]} {g["pos"][1]} {g["pos"][2]}" '
                          f'size="{g["size"][0]} {g["size"][1]} {g["size"][2]}" '
                          f'xyaxes="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f} {ax[3]:.4f} {ax[4]:.4f} {ax[5]:.4f}" '
                          f'material="{g["material"]}" />')
        
    xml_output.append('    </worldbody>\n</mujoco>')
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_output))
    print(f"Success: {output_xml_path} generated.")

if __name__ == "__main__":
    generate_mujoco_xml("scene_elements.jsonl", "scene4.xml")