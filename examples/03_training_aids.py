"""Example 03 — 训练辅助插件三件套 (Stage 2).

面向 (Audience) : 准备开训的 RL 开发者。
阶段 (Stage)    : 训练算法之外，环境侧要为训练提供哪些挂载点。
学到 (Takeaway) :
  - ``BasePlugin`` vs ``BaseObserverPlugin`` 的分工。
  - ``ctx.accessor`` / ``ctx.mutator`` / ``ctx.metrics`` 三件套在训练辅助
    场景下的真实用法。
  - 什么 hook 能写状态、什么不能，违规时框架的反应。

覆盖三类典型训练辅助：
  1. ``CurriculumPushPlugin(BasePlugin)`` — 每 episode 施加横向推力，强度由
     **runner 通过 ``options["push_force"]`` 注入**，插件只负责执行。这是
     ``ctx.episode_options`` 通道的标准用法（envs/framework/RESET.md §4）。
  2. ``FallenEarlyTerminationPlugin(BasePlugin)`` — 摔倒即停，少浪费 sample
     (on_post_action_step, read-only)。
  3. ``ClosingDistanceRewardObserver(BaseObserverPlugin)`` — 把"插件算出来
     的 metric"包装成 reward，让 EpisodeRunner 自动吃。

产物 (Outputs)  : 无文件产物，stdout only。
运行 (Run)      : python examples/03_training_aids.py
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from _common import build_humanoid21_runtime
from envs.framework import EpisodeRunner
from envs.framework.episode_runner import AGENT_IDS, ObserverBinding
from envs.framework.plugin import BasePlugin
from envs.framework.runtime_plugin import BaseObserverPlugin
from policy.random.policy import RandomCombatPolicy


# ---------------------------------------------------------------------------
# (1) 课程化扰动：要写状态，挂在可写 hook 上。
# ---------------------------------------------------------------------------
class CurriculumPushPlugin(BasePlugin):
    """每 episode 施加一个横向推力。**强度由 runner 通过
    ``options["push_force"]`` 注入**——插件不持有课程进度状态。

    设计要点：
    - **课程进度由 runner 拥有**（``run_n_episodes(options_fn=...)``），
      插件只读 ``ctx.episode_options["push_force"]`` 并执行。这把"现在该
      多大力"和"怎么施力"解耦——你可以在不改插件的前提下换 schedule
      （linear / step / cosine / 完全外部 controller）。详见 RESET.md §4。
    - ``require_mutator=True``：要调用 ``apply_external_force``，必须申请。
    - 挂在 ``on_pre_phy_step`` 这个**可写 hook**，而不是 post_action_step
      （后者只读；见 framework/README.md 的权限表）。
    - RNG 在 ``set_episode_seed`` 里重建，遵守 ``SEED.md`` 约定。
    """

    DEFAULT_PUSH_FORCE = 0.0  # 没给 push_force 时 = 不推

    def __init__(self) -> None:
        self._rng: Optional[np.random.Generator] = None
        self._current_force = np.zeros(3, dtype=np.float64)
        self._pushed_this_episode = False

    @property
    def name(self) -> str:
        return "curriculum_push"

    @property
    def require_mutator(self) -> bool:
        return True

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def on_pre_episode(self, ctx):
        # 从 episode_options 拿本回合扰动强度；缺省 → 0（不推）。
        magnitude = float(
            ctx.episode_options.get("push_force", self.DEFAULT_PUSH_FORCE)
        )
        # 随机一个水平方向。
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._current_force = magnitude * np.array(
            [np.cos(theta), np.sin(theta), 0.0], dtype=np.float64
        )
        self._pushed_this_episode = False
        # 把决议出来的强度抄进 metrics，便于训练日志可视化。
        ctx.metrics["curriculum_push_magnitude"] = magnitude

    def on_pre_phy_step(self, ctx):
        # 只在 episode 第 30 个物理步推一次，避免持续施力。
        if self._pushed_this_episode or ctx.physics_step != 30:
            return
        # ``apply_external_force`` 是 MuJoCo simulator 暴露的能力；
        # 通过 ctx.mutator（此 hook 可用）调用。
        ctx.mutator.apply_external_force(
            body_name="torso",
            force=self._current_force,
            robot_id="robot_a",
        )
        self._pushed_this_episode = True


# ---------------------------------------------------------------------------
# (2) 早停插件：只读，不涉及 mutator。
# ---------------------------------------------------------------------------
class FallenEarlyTerminationPlugin(BasePlugin):
    """检测摔倒后立刻 ``ctx.request_termination`` —— 训练期少浪费 sample。

    决策点：
    - 挂在 ``on_post_action_step``：这里只读，而终止提案是纯信号，不需要
      mutator，完全吻合。
    - 用 ``ctx.request_termination(reason)`` 而不是抛异常——终止原因会
      进入 ``EpisodeResult.termination_reasons``，训练代码可据此做数据
      分类（丢弃摔倒局、或加权）。
    """

    def __init__(self, min_height: float = 0.6, grace_steps: int = 5) -> None:
        self.min_height = float(min_height)
        self.grace_steps = int(grace_steps)

    @property
    def name(self) -> str:
        return "fallen_early_termination"

    def on_post_action_step(self, ctx):
        if ctx.episode_step < self.grace_steps:
            return  # 初始几步允许调整，别误触。
        core = ctx.accessor.get_core_state()
        z = float(core["robot_a"]["root_pos"][2])
        if z < self.min_height:
            ctx.request_termination(f"fallen(height={z:.3f})")


# ---------------------------------------------------------------------------
# (3) 自定义 reward observer：把"距离"变成训练可吃的 reward。
# ---------------------------------------------------------------------------
class ClosingDistanceRewardObserver(BaseObserverPlugin):
    """Reward = 上一步 vs 当前步的距离缩短量（靠近为正，远离为负）。

    展示 **plugin → metrics → observer** 这条标准数据流的最短版本：
    - 这里我们直接在 observer 里读距离（两个 root_pos 的 L2 距离）；如果
      更重的计算想和 reward 解耦，应该拆成 "MetricPlugin 写 metrics →
      RewardObserver 读 metrics"。
    - observer 必须实现 ``get_output``；::class:`EpisodeRunner` 会按照
      :class:`ObserverBinding` 的约定把它取出来作为 reward。
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._last_distance: Optional[float] = None
        self._reward = 0.0

    def _distance(self, ctx) -> float:
        core = ctx.accessor.get_core_state()
        a = np.asarray(core["robot_a"]["root_pos"][:2], dtype=np.float64)
        b = np.asarray(core["robot_b"]["root_pos"][:2], dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def on_pre_episode(self, ctx):
        self._last_distance = self._distance(ctx)
        self._reward = 0.0

    def on_post_action_step(self, ctx):
        current = self._distance(ctx)
        # robot_a 想靠近 ⇒ 距离缩短得正 reward；robot_b 对称取反号。
        delta = (self._last_distance or current) - current
        self._reward = float(delta if self.agent_id == "robot_a" else -delta)
        self._last_distance = current

    def on_post_episode(self, ctx):  # 不再累加
        pass

    def on_manual_refresh(self, ctx):
        pass

    def get_output(self) -> float:
        return self._reward


# ---------------------------------------------------------------------------
# 串联 demo
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Example 03 — 三件套：课程扰动 + 早停 + 距离 reward")
    print("=" * 70)

    curriculum = CurriculumPushPlugin()
    early_stop = FallenEarlyTerminationPlugin(min_height=0.6, grace_steps=5)

    # Curriculum schedule lives **on the runner side**: linear ramp from 0 to
    # 200 N over the first 3 episodes, then plateau. Plugin reads the value
    # via ctx.episode_options, no internal counter needed.
    MAX_FORCE = 200.0
    RAMP_EPISODES = 3

    def push_schedule(episode_index: int) -> dict:
        progress = min(1.0, episode_index / max(1, RAMP_EPISODES))
        return {"push_force": progress * MAX_FORCE}

    runtime = build_humanoid21_runtime(
        match_duration=3.0,
        extra_plugins=[curriculum, early_stop],
        observer_plugins={
            "robot_a_reward": ClosingDistanceRewardObserver("robot_a"),
            "robot_b_reward": ClosingDistanceRewardObserver("robot_b"),
        },
    )
    # 这里用 default_bindings 就行（robot_{a,b}_reward 已注册）。
    runner = EpisodeRunner(
        runtime=runtime,
        policies={
            "robot_a": RandomCombatPolicy(scale=0.3),
            "robot_b": RandomCombatPolicy(scale=0.3),
        },
    )

    print("\n[Running 4 episodes to观察课程扰动与早停是否生效]\n")
    print(f"{'ep':>3} | {'seed':>12} | {'push_N':>8} | {'steps':>5} | "
          f"{'reward_a':>9} | {'reward_b':>9} | {'term_reasons'}")
    print("-" * 80)
    # run_n_episodes(options_fn=...) is the canonical curriculum entry — see
    # RESET.md §4 "options 通道语义".
    results = runner.run_n_episodes(
        4, base_seed=100, options_fn=push_schedule,
    )
    for ep, result in enumerate(results):
        ra = float(np.sum(result.trajectories["robot_a"].rewards))
        rb = float(np.sum(result.trajectories["robot_b"].rewards))
        push_mag = float(result.shared_info_final.get("metrics", {}).get(
            "curriculum_push_magnitude", 0.0
        ))
        print(
            f"{ep:>3} | {result.seed:>12} | {push_mag:>8.1f} | "
            f"{result.num_steps:>5} | {ra:>+9.3f} | {rb:>+9.3f} | "
            f"{result.termination_reasons}"
        )

    print("\n观察要点：")
    print("  - push_N 从 0 → 逐 episode 增长 → 200.0（课程到顶）。")
    print("  - 某些局提前 termination_reasons=['fallen(...)']，少浪费了 sample。")
    print("  - reward_a / reward_b 是距离缩减 observer 直接算出来的，训练 loop 可直接吃。")

    # ---------------------------------------------------------------------
    # 权限隔离小 demo：on_post_action_step 是只读 hook，尝试写入会被框架
    # 截断（ctx.mutator 为 None）。这里演示"你以为我能写，但框架不让"。
    # ---------------------------------------------------------------------
    print("\n[Permission isolation demo]")

    class IllegalWritePlugin(BasePlugin):
        def __init__(self) -> None:
            self._printed = False

        @property
        def name(self) -> str:
            return "illegal_writer"

        @property
        def require_mutator(self) -> bool:
            return True  # 我声明我要写

        def on_post_action_step(self, ctx):
            # 但这个 hook 是只读的：框架不会授予 mutator。
            assert ctx.mutator is None, "框架约定被破坏！"
            if not self._printed:
                print("  ✓ 挂在只读 hook 上时 ctx.mutator is None —— 权限被正确拒绝。")
                self._printed = True

    runtime2 = build_humanoid21_runtime(
        match_duration=0.5, extra_plugins=[IllegalWritePlugin()]
    )
    bindings = {
        a: ObserverBinding(obs_name=f"{a}_obs", reward_name=None) for a in AGENT_IDS
    }
    EpisodeRunner(
        runtime=runtime2,
        policies={
            "robot_a": RandomCombatPolicy(scale=0.1),
            "robot_b": RandomCombatPolicy(scale=0.1),
        },
        observer_bindings=bindings,
    ).run_episode(seed=0)

    print("\nDone. 下一步：examples/04_collect_rollouts.py 看怎么并行、可复现地灌数据。")


if __name__ == "__main__":
    main()
