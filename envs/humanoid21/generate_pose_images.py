#!/usr/bin/env python3
"""
生成每种初始姿态的广播视角图片
"""

import numpy as np
import sys
from pathlib import Path
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.humanoid21.simulator import Humanoid21Simulator


def generate_pose_images():
    """生成每种初始姿态的图片"""
    print("=" * 80)
    print("生成初始姿态图片")
    print("=" * 80)

    poses = ['standing', 'squat', 'stand_on_left_leg', 'prone', 'supine']
    output_dir = Path(__file__).parent / 'pose_images'
    output_dir.mkdir(exist_ok=True)

    for pose_name in poses:
        print(f"\n生成姿态: {pose_name}")
        print("-" * 60)

        # 创建模拟器
        sim = Humanoid21Simulator(initial_pose_a=pose_name, initial_pose_b=pose_name)

        # 重置
        sim.reset()

        # 运行几步让机器人稳定
        for _ in range(10):
            sim.physical_step()

        # 生成图片
        try:
            image = sim.get_broadcastview_image()

            # 保存图片
            output_path = output_dir / f"{pose_name}.png"
            imageio.imwrite(str(output_path), image)

            print(f"  ✓ 保存图片: {output_path}")
            print(f"    图片尺寸: {image.shape}")

        except Exception as e:
            print(f"  ✗ 生成图片失败: {e}")

    print(f"\n{'='*80}")
    print(f"所有图片已保存到: {output_dir}")
    print(f"{'='*80}")

    # 生成一个对比图（所有姿态拼在一起）
    print(f"\n生成对比图...")

    images = []
    for pose_name in poses:
        image_path = output_dir / f"{pose_name}.png"
        if image_path.exists():
            images.append(imageio.imread(str(image_path)))

    if images:
        # 创建 2x3 的拼图（5个姿态 + 1个空白）
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (pose_name, image) in enumerate(zip(poses, images)):
            axes[i].imshow(image)
            axes[i].set_title(pose_name, fontsize=14, fontweight='bold')
            axes[i].axis('off')

        # 隐藏最后一个子图
        axes[5].axis('off')

        plt.tight_layout()
        comparison_path = output_dir / "comparison.png"
        plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 保存对比图: {comparison_path}")

    print(f"\n完成!")


if __name__ == '__main__':
    generate_pose_images()
