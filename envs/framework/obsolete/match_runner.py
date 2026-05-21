from typing import Any, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from .round_runner import RoundRunner
from .common_plugins import VideoRecorderPlugin


@dataclass
class MatchResult:
    """比赛结果数据类"""
    total_rounds: int
    rounds_completed: int
    round_results: list = field(default_factory=list)
    final_winner: Optional[str] = None
    total_score: Dict[str, int] = field(default_factory=dict)
    ko_winner: Optional[str] = None  # 如果有KO获胜者


class MatchRunner:
    """
    运行多回合比赛的封装类，基于 RoundRunner 实现。

    比赛规则：
    1. 双方初始血量：100 点
    2. 获胜条件：
       - 2.1 将对方血量先降至 0，则比赛立即结束，KO获胜
       - 2.2 在比赛时间结束时，血量更高者获胜
       - 2.3 时间结束血量相同：判定为平局
    3. 比赛时长：每个回合时长 30 秒，进行 6 个回合
    4. 重置状态：每个回合开始时，都会从初始位置开始（但血量延续上一回合）

    使用方式 (env_factory 无参数，HP 延续走 ``ctx.episode_options``；
    详见 envs/framework/RESET.md §3 / §4)：

        from combatbench.envs.humanoid21 import make_env
        from envs.humanoid21.plugins import CombatScoringPlugin
        from envs.framework.common_plugins import VideoRecorderPlugin

        def runtime_factory():
            return make_env(
                match_duration=30.0,
                control_frequency=20,
                plugins=[CombatScoringPlugin(), VideoRecorderPlugin(fps=30)],
            )

        runner = MatchRunner(
            policy_a=policy_a,
            policy_b=policy_b,
            env_factory=runtime_factory,
            total_rounds=6,
            verbose=True,
        )
        result = runner.run(seed=42, video_dir="videos")
    """

    def __init__(
        self,
        policy_a: Any,
        policy_b: Any,
        env_factory: Callable[[], Any],
        total_rounds: int = 6,
        verbose: bool = True,
    ):
        """
        Args:
            policy_a: 机器人A的策略
            policy_b: 机器人B的策略
            env_factory: runtime 工厂函数，无参数。MatchRunner 只调用一次
                构建一个长周期 runtime，后续每回合通过
                ``runtime.reset(options={"initial_health_a": ..., "initial_health_b": ...})``
                复用；HP 由 ``CombatScoringPlugin`` 从 ``ctx.episode_options``
                读取。然后会在所有回合进行完后调用 ``runtime.close()``。
            total_rounds: 总回合数，默认 6 回合
            verbose: 是否打印详细信息
        """
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.runtime_factory = env_factory
        self.total_rounds = total_rounds
        self.verbose = verbose

    def _print_match_header(self):
        if not self.verbose:
            return
        print("=" * 60)
        print("CombatBench Match Started")
        print(f"Total Rounds: {self.total_rounds}")
        print("=" * 60)

    def _print_round_summary(self, round_num: int, result: Dict[str, Any]):
        if not self.verbose:
            return
        print("-" * 60)
        print(f"Round {round_num} Summary:")
        print(f"  Winner: {result['winner']}")
        print(f"  Final HP: robot_a={result['final_health'].get('robot_a', 0):.1f}, "
              f"robot_b={result['final_health'].get('robot_b', 0):.1f}")
        print(f"  Steps: {result['steps']}")
        print("-" * 60)

    def _print_match_summary(self, match_result: MatchResult):
        if not self.verbose:
            return
        print("=" * 60)
        print("Match Summary")
        print("=" * 60)
        print(f"Rounds Completed: {match_result.rounds_completed}/{match_result.total_rounds}")

        if match_result.ko_winner:
            print(f"KO Winner: {match_result.ko_winner} (opponent KO'd)")
        else:
            print(f"Final Winner: {match_result.final_winner}")

        # 打印每回合结果
        for i, r in enumerate(match_result.round_results, 1):
            winner = r['winner']
            hp_a = r['final_health'].get('robot_a', 0)
            hp_b = r['final_health'].get('robot_b', 0)
            print(f"  Round {i}: {winner} (HP: A={hp_a:.1f}, B={hp_b:.1f})")

        print("-" * 60)

    def run(
        self,
        seed: Optional[int] = None,
        video_dir: Optional[str] = None
    ) -> MatchResult:
        """
        运行完整比赛

        Args:
            seed: 随机种子
            video_dir: 视频保存目录，如果提供则每回合保存单独视频

        Returns:
            MatchResult: 比赛结果
        """
        self._print_match_header()

        round_results = []
        total_score = {"robot_a": 0, "robot_b": 0}
        ko_winner = None

        # 血量延续：初始血量
        current_health_a = 100.0
        current_health_b = 100.0

        # 按每回合预派生一批子种子（统一从一个 SeedSequence 上 spawn 出来，
        # 而不是 seed + round_num 这种算术推导——算术会引入隐性相关，见
        # envs/framework/SEED.md）。base_seed=None 由 EpisodeRunner 内部
        # 入口再做 None→uint32 的解析，这里保持 None 向下传递给 round_runner。
        if seed is not None:
            round_seeds = np.random.SeedSequence(int(seed)).generate_state(
                self.total_rounds, dtype=np.uint32
            )
        else:
            round_seeds = [None] * self.total_rounds

        # 构建一个长周期 runtime + RoundRunner，所有回合复用。HP 延续通过
        # ``runtime.reset(options={...})`` 注入。详见 RESET.md §3 / §7-G3。
        runtime = self.runtime_factory()
        try:
            round_runner = RoundRunner(
                policy_a=self.policy_a,
                policy_b=self.policy_b,
                runtime=runtime,
                verbose=self.verbose,
            )

            for round_num in range(1, self.total_rounds + 1):
                if self.verbose:
                    print(f"\n>>> Starting Round {round_num}/{self.total_rounds}")
                    print(
                        f">>> Current HP: robot_a={current_health_a:.1f}, "
                        f"robot_b={current_health_b:.1f}"
                    )

                # 每回合单独的视频路径；通过 RoundRunner.run(videosave_path=...)
                # 传给 runtime 中已 attach 的 VideoRecorderPlugin 实例。
                video_path = None
                if video_dir is not None:
                    video_path = str(Path(video_dir) / f"round_{round_num}.mp4")

                # 从预派生的 round_seeds 中取这一回合的种子（None 则一路透传，
                # 由 EpisodeRunner 入口再做 None→uint32 的解析）。
                raw_rs = round_seeds[round_num - 1]
                round_seed = None if raw_rs is None else int(raw_rs)
                round_options = {
                    "initial_health_a": float(current_health_a),
                    "initial_health_b": float(current_health_b),
                }
                result = round_runner.run(
                    seed=round_seed,
                    options=round_options,
                    videosave_path=video_path,
                )
                round_results.append(result)

                # 统计得分
                winner = result['winner']
                if winner == 'robot_a':
                    total_score['robot_a'] += 1
                elif winner == 'robot_b':
                    total_score['robot_b'] += 1

                # 更新当前血量（用于下一回合）
                current_health_a = result['final_health'].get('robot_a', 100)
                current_health_b = result['final_health'].get('robot_b', 100)

                self._print_round_summary(round_num, result)

                # 检查KO获胜条件：某方血量降至0
                if current_health_a <= 0:
                    ko_winner = 'robot_b'
                    if self.verbose:
                        print("\n!!! Robot B wins by KO !!!")
                    break
                elif current_health_b <= 0:
                    ko_winner = 'robot_a'
                    if self.verbose:
                        print("\n!!! Robot A wins by KO !!!")
                    break
        finally:
            # MatchRunner now owns the runtime lifecycle (RoundRunner.run no
            # longer closes it — see RESET.md §7-G3).
            close_fn = getattr(runtime, "close", None)
            if callable(close_fn):
                close_fn()

        # 计算最终获胜者
        if ko_winner:
            final_winner = ko_winner
        else:
            if total_score['robot_a'] > total_score['robot_b']:
                final_winner = 'robot_a'
            elif total_score['robot_b'] > total_score['robot_a']:
                final_winner = 'robot_b'
            else:
                final_winner = 'draw'

        match_result = MatchResult(
            total_rounds=self.total_rounds,
            rounds_completed=len(round_results),
            round_results=round_results,
            final_winner=final_winner,
            total_score=total_score,
            ko_winner=ko_winner
        )

        self._print_match_summary(match_result)
        return match_result


if __name__ == "__main__":
    from combatbench.envs.humanoid21 import make_env
    from .common_plugins import VideoRecorderPlugin

    def env_factory():
        return make_env(
            plugins=[VideoRecorderPlugin(fps=30, output_path="")],
            match_duration=30.0,
            control_frequency=20,
        )

    class DummyPolicy:
        def act(self, obs, info):
            return [0.0] * 21

    policy_a = DummyPolicy()
    policy_b = DummyPolicy()

    runner = MatchRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        env_factory=env_factory,
        total_rounds=6,
        verbose=True
    )
    result = runner.run(seed=42, video_dir="match_videos")

    print(f"\nFinal Result: {result.final_winner}")
    print(f"Score: A={result.total_score['robot_a']}, B={result.total_score['robot_b']}")
