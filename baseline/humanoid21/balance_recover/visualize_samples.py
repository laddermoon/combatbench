#!/usr/bin/env python3
"""可视化采样器输出的采样点：为每个采样点生成冲量结束后的机器人状态图片。

加载 ImpulseSampler 从 NPZ 文件，采样 N 个点 (angle, force, duration)，
在内部 sim 中施加冲量，渲染冲量结束后的那一帧并保存为图片。
图片上标注采样参数（角度、力、持续时间）。

用法::

    PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/visualize_samples.py \
        --npz baseline/humanoid21/balance_recover/run_recovery_v2/sample_weights_gen0.npz \
        --policy baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml \
        --num-samples 20 \
        --output-dir /data1/dev/sample_vis

参数:
    --npz: 采样器 NPZ 文件路径
    --policy: 策略 blueprint 路径（用于内部 sim 中控制机器人，使扰动状态物理真实）
    --num-samples: 采样数量
    --output-dir: 图片输出目录
    --seed: 随机种子
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from envs.humanoid21.simulator import Humanoid21Simulator
from envs.framework.env_runtime import EnvRuntime
from envs.humanoid21.disturbance_plugins import ConstantForcePlugin
from envs.framework.policy import PolicyBlueprint

from baseline.humanoid21.balance_recover.sample_distribution import ImpulseSampler


def annotate_image(image: np.ndarray, angle: float, force: float,
                   duration: int, index: int) -> np.ndarray:
    """在图片上标注采样参数。"""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    lines = [
        f"Sample #{index}",
        f"Angle: {angle:.1f}°",
        f"Force: {force:.0f} N",
        f"Duration: {duration} steps",
    ]

    x, y = 10, 10
    for line in lines:
        draw.rectangle([x - 2, y - 2, x + 220, y + 24], fill=(0, 0, 0, 180))
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 26

    return np.array(img)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize sampler output: impulse end-state images")
    p.add_argument("--npz", required=True, help="Path to sample_weights NPZ file")
    p.add_argument("--policy", required=True, help="Path to policy_blueprint.yaml for internal sim")
    p.add_argument("--num-samples", type=int, default=20, help="Number of samples to visualize")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for images")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--agent-id", type=str, default="robot_a")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = ImpulseSampler(args.npz)
    rng = np.random.RandomState(args.seed)

    policy_bp = PolicyBlueprint.load(Path(args.policy))
    policy = policy_bp.build()

    sim = Humanoid21Simulator()
    phy_steps_per_action = 25

    print(f"Generating {args.num_samples} sample images...")

    for i in range(args.num_samples):
        sample = sampler.sample(rng)
        angle = sample["direction_angle"]
        force = sample["force"]
        duration = sample["duration_action_steps"]
        body = sample.get("body", "torso")

        force_plugin = ConstantForcePlugin(
            agent_id=args.agent_id,
            force=force,
            direction=angle,
            duration_action_steps=duration,
            body_name=body,
        )
        runtime = EnvRuntime(
            simulator=sim,
            plugins=[force_plugin],
            phy_steps_per_action=phy_steps_per_action,
        )

        runtime.reset()
        policy.reset()

        for _ in range(duration):
            obs = sim.get_observation()
            action, _ = policy.act(obs.get(args.agent_id))
            zeros = np.zeros(21, dtype=np.float32)
            if args.agent_id == "robot_a":
                runtime.step(action, zeros)
            else:
                runtime.step(zeros, action)

        image = sim.get_broadcastview_image()
        annotated = annotate_image(image, angle, force, duration, i)

        out_path = output_dir / f"sample_{i:03d}_a{angle:.0f}_f{force:.0f}_d{duration}.png"
        imageio.imwrite(str(out_path), annotated)
        print(f"  [{i+1}/{args.num_samples}] {out_path.name}  angle={angle:.1f} force={force:.0f} dur={duration}")

        runtime.close()

    print(f"\nAll {args.num_samples} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
