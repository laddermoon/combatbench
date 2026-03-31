#!/usr/bin/env python3
"""
使用 compareversion 运行站立测试并生成视频
"""
import sys
sys.path.insert(0, '/data1/mono/things/combatbench/compareversion')

from envs import RoundRunner
from policy.standing import StandingCombatPolicy

# 创建策略
policy_a = StandingCombatPolicy()
policy_b = StandingCombatPolicy()

print("Loading policy A: StandingCombatPolicy")
print("  Loaded: StandingCombatPolicy")
print("Loading policy B: StandingCombatPolicy")
print("  Loaded: StandingCombatPolicy")

# 创建 RoundRunner
runner = RoundRunner(
    policy_a=policy_a,
    policy_b=policy_b,
    render_mode="rgb_array",
    match_duration=5.0,
    control_frequency=20,
    non_fall_mode=True,
    non_fall_pitch_limit_deg=5.0,
    non_fall_roll_limit_deg=5.0,
    verbose=True
)

# 运行并保存视频
result = runner.run(save_video_path="test_original_standing.mp4")

# 打印结果
print()
print("=" * 60)
print("Round Summary")
print("=" * 60)
print(f"Policy A: {policy_a.__class__.__name__}")
print(f"Policy B: {policy_b.__class__.__name__}")
print(f"Winner: {result.winner or 'draw'}")
print(f"Steps: {result.steps}")
print(f"Final HP: A={result.scores['robot_a']:.1f}, B={result.scores['robot_b']:.1f}")
if hasattr(result, 'video_frames'):
    print(f"Video saved to: test_original_standing.mp4 ({result.video_frames} frames)")
print("=" * 60)
