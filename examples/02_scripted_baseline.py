"""Example 02 — 手写一个 scripted baseline (Stage 1).

面向 (Audience) : 要做 RL 但先想搭一个非学习 baseline 当对照组的人。
阶段 (Stage)    : Policy 开发第一步，在上训练算法之前。
学到 (Takeaway) :
  - :class:`Policy` ABC 的最小实现（``act`` 必写，``reset`` 可选）。
  - 同一 seed 的 scripted policy 动作序列 bit-wise 一致 → 可用作训练
    期"确定性对手池"。
  - 两个 policy 如何同时挂到一个 :class:`EpisodeRunner`。

产物 (Outputs)  : 无文件产物，stdout only。
运行 (Run)      : python examples/02_scripted_baseline.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from _common import build_humanoid21_runtime
from envs.framework import EpisodeRunner, PostActionRecorder
from envs.framework.policy import Policy
from policy.random.policy import RandomCombatPolicy


@dataclass
class EpisodeResult:
    """Simple episode result container."""
    trajectories: Dict[str, Any] = field(default_factory=dict)
    shared_info_final: Dict[str, Any] = field(default_factory=dict)
    num_steps: int = 0


class SimpleRecorder(PostActionRecorder):
    """Simple recorder to capture trajectory data."""
    def __init__(self):
        self.trajectories = {
            "robot_a": {"actions": [], "rewards": []},
            "robot_b": {"actions": [], "rewards": []},
        }
        self.num_steps = 0

    def on_attach(self):
        pass

    def on_detach(self):
        pass

    def on_pre_episode(self, ctx):
        self.trajectories = {
            "robot_a": {"actions": [], "rewards": []},
            "robot_b": {"actions": [], "rewards": []},
        }
        self.num_steps = 0

    def on_post_action_step(self, ctx, observation, action, observer_outputs, action_extras):
        for agent_id in ("robot_a", "robot_b"):
            self.trajectories[agent_id]["actions"].append(action[agent_id])
            self.trajectories[agent_id]["rewards"].append(0.0)  # No reward extraction for now
        self.num_steps += 1


class SinusoidPolicy(Policy):
    """正弦波驱动所有关节 —— 确定性、可复现、肉眼可见的滑稽动作。

    实现要点：
    - ``act`` 必须实现（ABC 要求），返回一个 21 维 ndarray。
    - ``reset`` 可选：我们这里用它来保存一个随每 episode 变化的相位偏移。
    - 没有 ``__init__`` 契约要求——你怎么存状态都行。
    """

    ACTION_DIM = 21

    def __init__(self, frequency_hz: float = 1.0, amplitude: float = 0.4) -> None:
        self.frequency_hz = float(frequency_hz)
        self.amplitude = float(amplitude)
        self._step = 0
        self._phase_offset = 0.0

    def reset(self, seed: Optional[int] = None) -> None:
        # 用 episode 种子派生 phase，这样同 seed → 同动作序列；不同 seed →
        # 不同相位，适合做"同一 baseline，不同初始姿态"的对手池。
        self._step = 0
        rng = np.random.default_rng(seed)
        self._phase_offset = float(rng.uniform(0.0, 2.0 * np.pi))

    def act(self, observation: Any) -> np.ndarray:
        # 20Hz 决策频率 → 每步对应 dt = 1/20s。
        dt = 1.0 / 20.0
        t = self._step * dt
        self._step += 1
        # 每个关节的相位沿关节索引递增，产生类似"波浪手"的动作。
        joint_phase = np.arange(self.ACTION_DIM) * (np.pi / self.ACTION_DIM)
        action = self.amplitude * np.sin(
            2.0 * np.pi * self.frequency_hz * t + self._phase_offset + joint_phase
        )
        return action.astype(np.float32)


def _run(seed: int) -> EpisodeResult:
    """Run one episode with Sinusoid (A) vs Random (B), return the result."""
    runtime = build_humanoid21_runtime(match_duration=2.0)
    recorder = SimpleRecorder()
    runtime.attach_recorder(recorder)

    runner = EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": SinusoidPolicy(frequency_hz=1.5, amplitude=0.5),
            "robot_b": RandomCombatPolicy(scale=0.2),
        },
    )
    runner.run_episode(seed=seed)

    return EpisodeResult(
        trajectories=recorder.trajectories,
        shared_info_final=dict(runtime.ctx.metrics),
        num_steps=recorder.num_steps,
    )


def main() -> None:
    print("=" * 70)
    print("Example 02 — Scripted Baseline: Sinusoid vs Random")
    print("=" * 70)

    # --- (1) 决定论证明：同一 seed 跑两遍 → robot_a 动作序列逐帧相等 ---
    r1 = _run(seed=42)
    r2 = _run(seed=42)

    actions_a_1 = np.stack(r1.trajectories["robot_a"]["actions"])
    actions_a_2 = np.stack(r2.trajectories["robot_a"]["actions"])
    max_diff = float(np.abs(actions_a_1 - actions_a_2).max())
    print(f"\n[Determinism check] seed=42 run twice → max |action_a_1 - action_a_2| = {max_diff:.2e}")
    assert max_diff == 0.0, "Sinusoid policy is not deterministic under the same seed!"
    print("  ✓ Sinusoid actions are bit-wise identical. 可以作为训练期的确定性对手。")

    # --- (2) 一局对战摘要：累计 reward + 最终 HP ---
    result = _run(seed=7)
    ep_reward_a = float(np.sum(result.trajectories["robot_a"]["rewards"]))
    ep_reward_b = float(np.sum(result.trajectories["robot_b"]["rewards"]))
    hp = {
        "robot_a": result.shared_info_final.get("health_a", 0.0),
        "robot_b": result.shared_info_final.get("health_b", 0.0),
    }

    print("\n[Match summary] seed=7")
    print(f"  steps              : {result.num_steps}")
    print(f"  cumulative reward  : robot_a={ep_reward_a:+.4f}  robot_b={ep_reward_b:+.4f}")
    print(f"  final HP           : {hp}")

    # --- (3) 简单的动作范数曲线，看两个策略的"激烈程度" ---
    norms_a = np.linalg.norm(np.stack(result.trajectories["robot_a"]["actions"]), axis=1)
    norms_b = np.linalg.norm(np.stack(result.trajectories["robot_b"]["actions"]), axis=1)
    print("\n[Action-norm trajectory] (每个策略每一步 action 的 L2 范数)")
    print(f"  robot_a sinusoid : mean={norms_a.mean():.3f}  max={norms_a.max():.3f}  std={norms_a.std():.3f}")
    print(f"  robot_b random   : mean={norms_b.mean():.3f}  max={norms_b.max():.3f}  std={norms_b.std():.3f}")

    print("\nDone. 下一步建议：阅读 examples/03_training_aids.py 看训练辅助插件怎么写。")


if __name__ == "__main__":
    main()
