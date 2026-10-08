"""standup_dev — standup 的设备批量最大化版本（R2 加速候选臂）。

与 ``exp_standup.py`` 共享全部任务语义与 PPO 超参——**只改 rollout
规模**：episodes_per_update 512 → 12288（24×），配套设备 collector
一次 update 一波（6 worker × B=2048 = 12288 行满载）。

设计边界（R2 协议一致性）：
- env blueprint / reward / obs / 网络 / lr / target_kl / epochs /
  minibatch 全部继承——训练算法侧零改动；
- 唯一执行维度差异：collector=device + 每 update 批量放大
  （"已声明的执行维度"，见 ROADMAP M0 口径；样本效率对比按
  transition 计数而非 update 数）；
- success 定义不变：eval max_potential ≥ 0.9。

启动（6 空闲卡，0/1 为用户在跑训练保留）：

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \\
      python3 -B baseline/framework/train.py --experiment standup_dev \\
      --algo ppo --seed 42 \\
      --collector device --collector-devices 2,3,4,5,6,7 \\
      --collector-batch-size 2048 --background
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupDev(Standup):
    """standup 的设备批量版：rollout 规模 24×，任务/算法不变。"""

    name = "standup_dev"

    # --- Rollout schedule ---
    # 12288 episodes × 2 agents = 24576 trajectories per update
    # （CPU 版 512×2=1024）。每 update 一波满载（6×2048 行）。
    episodes_per_update: int = 12288
    # eval 走同一 device collector——批量加大无成本，顺势加密度
    eval_episodes: int = 512


EXPERIMENT_CLASS = StandupDev
