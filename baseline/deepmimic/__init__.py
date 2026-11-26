"""DeepMimic 复现包 — 单 clip 运动跟踪训练。

自包含实现（不 import mocap.humanoid21.tracking 临时管线）：
- ``ref.CoreRefView``：CoreClip 的跟踪侧视图
- ``sim.DeepMimicTrackSimulator``：RSI + 参考观测扩展
- ``plugins.DeepMimicGuardPlugin``：摔倒/漂移/片段结束终止
- ``reward.DeepMimicRewarder``：加权模仿奖励
- ``experiment.DeepMimicTrack``：PPO 实验（注册壳在
  ``baseline/experiments_ppo/exp_deepmimic.py``）

与论文的差异见 DESIGN.md §5。
"""
