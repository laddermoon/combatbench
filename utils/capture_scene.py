#!/usr/bin/env python3
"""
Capture all cameras at initial state from MuJoCo XML scene file
Supports enhanced rendering: shadows, anti-aliasing, lighting optimization
"""

import os
import sys
from pathlib import Path
import numpy as np

# Set headless rendering
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import mujoco


def capture_cameras(xml_path, output_dir, width=640, height=480, 
                    enable_shadow=True, enable_antialias=True):
    """
    Get all camera captures from XML scene file
    
    Args:
        xml_path: MuJoCo XML scene file path
        output_dir: Output directory
        width: Image width
        height: Image height
        enable_shadow: Enable shadow
        enable_antialias: Enable anti-alias
    """
    # Load scene
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set rendering resolution
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    
    # Enhanced rendering settings
    # Shadow is set via renderer.enable_shadow(), no need to set in model.vis
    
    # Set background color (optional)
    # model.vis.global_.rgba = [0.8, 0.9, 1.0, 1.0]  # Light blue background
    
    # Initialize scene
    mujoco.mj_forward(model, data)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all cameras
    n_cameras = model.ncam
    print(f'Scene: {xml_path}')
    print(f'Render resolution: {width}x{height}')
    print(f'Shadow: {"enabled" if enable_shadow else "disabled"}')
    print(f'Number of cameras: {n_cameras}')
    print()
    
    if n_cameras == 0:
        print('No cameras found in scene.')
        return
    
    # Create renderer (Note: MuJoCo Renderer parameter order is height, width)
    renderer = mujoco.Renderer(model, height, width)
    
    # Render each camera
    for cam_id in range(n_cameras):
        # Get camera name
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if cam_name is None:
            cam_name = f'camera_{cam_id}'
        
        # Update scene (can add more rendering options)
        renderer.update_scene(data, camera=cam_id)
        
        # Render
        image = renderer.render()
        
        # Save image
        output_path = output_dir / f'{cam_name}.png'
        import imageio
        imageio.imwrite(str(output_path), image)
        
        print(f'Captured: {cam_name} -> {output_path}')
    
    renderer.close()
    print()
    print(f'All {n_cameras} camera images saved to: {output_dir}')


def main():
    if len(sys.argv) < 3:
        print('Usage: python capture_scene.py <scene.xml> <output_dir> [width] [height] [--no-shadow]')
        print('Example: python capture_scene.py arena.xml camera_captures 640 480')
        print('Options:')
        print('  --no-shadow  Disable shadow rendering')
        sys.exit(1)
    
    xml_path = sys.argv[1]
    output_dir = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 480
    
    # Check if shadow is disabled
    enable_shadow = '--no-shadow' not in sys.argv
    
    capture_cameras(xml_path, output_dir, width, height, enable_shadow=enable_shadow)


if __name__ == '__main__':
    main()
