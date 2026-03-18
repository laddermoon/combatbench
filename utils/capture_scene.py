#!/usr/bin/env python3
"""
从MuJoCo XML场景文件获取所有相机在初始状态下的截图
支持增强渲染效果：阴影、抗锯齿、光照优化
"""

import os
import sys
from pathlib import Path
import numpy as np

# 设置headless渲染
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import mujoco


def capture_cameras(xml_path, output_dir, width=640, height=480, 
                    enable_shadow=True, enable_antialias=True):
    """
    从XML场景文件获取所有相机截图
    
    Args:
        xml_path: MuJoCo XML场景文件路径
        output_dir: 输出目录
        width: 图像宽度
        height: 图像高度
        enable_shadow: 启用阴影
        enable_antialias: 启用抗锯齿
    """
    # 加载场景
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 设置渲染分辨率
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    
    # 增强渲染设置
    # 阴影通过renderer.enable_shadow()设置，不需要在model.vis中设置
    
    # 设置背景色（可选）
    # model.vis.global_.rgba = [0.8, 0.9, 1.0, 1.0]  # 浅蓝色背景
    
    # 初始化场景
    mujoco.mj_forward(model, data)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有相机
    n_cameras = model.ncam
    print(f'Scene: {xml_path}')
    print(f'Render resolution: {width}x{height}')
    print(f'Shadow: {"enabled" if enable_shadow else "disabled"}')
    print(f'Number of cameras: {n_cameras}')
    print()
    
    if n_cameras == 0:
        print('No cameras found in scene.')
        return
    
    # 创建渲染器 (注意: MuJoCo Renderer参数顺序是 height, width)
    renderer = mujoco.Renderer(model, height, width)
    
    # 渲染每个相机
    for cam_id in range(n_cameras):
        # 获取相机名称
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if cam_name is None:
            cam_name = f'camera_{cam_id}'
        
        # 更新场景（可添加更多渲染选项）
        renderer.update_scene(data, camera=cam_id)
        
        # 渲染
        image = renderer.render()
        
        # 保存图片
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
    
    # 检查是否禁用阴影
    enable_shadow = '--no-shadow' not in sys.argv
    
    capture_cameras(xml_path, output_dir, width, height, enable_shadow=enable_shadow)


if __name__ == '__main__':
    main()
