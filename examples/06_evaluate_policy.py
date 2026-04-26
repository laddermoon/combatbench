"""Example 06 — 按比赛规则评测一个 policy (Stage 5).

面向 (Audience) : 要出一份"我的模型 vs baseline"评测报告的人。
阶段 (Stage)    : 训练完了，需要拿出一份按规则书走的权威评测。

**严格遵循 `docs/RULE_zh.md`**：
  - 双方初始血量 100。
  - **6 回合，每回合 30 秒**，20Hz 决策 / 500Hz 物理。
  - 每回合从初始位姿开始（血量延续上一回合）。
  - KO（某方血量 → 0）立即结束比赛；否则 6 回合后血量高者胜，相等为平。
  - 有效打击、伤害数值等全部由 humanoid21 内部按规则书判定。

这里用 ``StandingCombatPolicy`` 冒充"我的 policy"，``RandomCombatPolicy``
当 baseline。把自己的策略塞进去替换 ``policy_a`` 即可。

学到 (Takeaway) :
  - :class:`MatchRunner` 的 ``env_factory()`` 契约（无参数）——HP 延续走
    ``ctx.episode_options``，由 :class:`CombatScoringPlugin` 在
    ``on_pre_episode`` 读取，无需重建 runtime。
  - 回合级 seed 从 match base_seed 用 ``SeedSequence.spawn`` 派生。
  - 一份**按规则书走的 markdown 战报**模板。

产物 (Outputs)  : examples/out/06_evaluate_policy/match_report.md
                  examples/out/06_evaluate_policy/videos/round_N.mp4
运行 (Run)      : python examples/06_evaluate_policy.py
                  （约 30~90s，取决于 MuJoCo 速度）
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _common import example_out_dir
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.match_runner import MatchRunner, MatchResult
from envs.humanoid21 import make_env
from envs.humanoid21.plugins import CombatScoringPlugin
from policy.random.policy import RandomCombatPolicy
from policy.standing.policy import StandingCombatPolicy


# ---------------------------------------------------------------------------
# env_factory — called ONCE by MatchRunner; the same runtime is recycled
# across all rounds via ``runtime.reset(options={...})``. HP carry-over
# flows through ``ctx.episode_options`` (see envs/framework/RESET.md §4).
# ---------------------------------------------------------------------------
def _make_env_factory():
    """Return an ``env_factory()`` closure (no arguments).

    Builds the humanoid21 runtime exactly once with:
      - :class:`CombatScoringPlugin` reading per-round HP from
        ``ctx.episode_options``;
      - :class:`VideoRecorderPlugin` whose output path is retargeted per
        round by :class:`MatchRunner` via ``RoundRunner.run(videosave_path=...)``.
    """
    def env_factory():
        plugins: list[Any] = [
            CombatScoringPlugin(),         # HP read from ctx.episode_options
            VideoRecorderPlugin(fps=30),   # output_path is retargeted per-round
        ]
        # RULE §5: 物理 500Hz，决策 20Hz，单回合 30s。
        return make_env(
            match_duration=30.0,
            control_frequency=20,
            plugins=plugins,
        )

    return env_factory


# ---------------------------------------------------------------------------
# Report writer — render a rule-book-style markdown from ``MatchResult``.
# ---------------------------------------------------------------------------
def _write_report(
    result: MatchResult,
    policy_a_name: str,
    policy_b_name: str,
    elapsed_s: float,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# CombatBench Match Report")
    lines.append("")
    lines.append(f"- **Policy A**  : `{policy_a_name}`")
    lines.append(f"- **Policy B**  : `{policy_b_name}`")
    lines.append(f"- **Rules**     : `docs/RULE_zh.md` (HP=100, 6×30s, KO or higher-HP wins)")
    lines.append(f"- **Rounds run**: {result.rounds_completed}/{result.total_rounds}")
    lines.append(f"- **Elapsed**   : {elapsed_s:.1f}s")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    if result.ko_winner:
        lines.append(f"**KO winner**: `{result.ko_winner}` (opponent HP → 0)")
    else:
        lines.append(f"**Final winner**: `{result.final_winner}`  _(by round-win count)_")
    lines.append(f"- Round wins: A={result.total_score.get('robot_a', 0)}  "
                 f"B={result.total_score.get('robot_b', 0)}")
    lines.append("")
    lines.append("## Per-round detail")
    lines.append("")
    lines.append("| # | Winner | HP_A | HP_B | Steps | Termination |")
    lines.append("|---|--------|------|------|-------|-------------|")
    for i, r in enumerate(result.round_results, 1):
        hp_a = r["final_health"].get("robot_a", 0.0)
        hp_b = r["final_health"].get("robot_b", 0.0)
        term = ", ".join(r.get("termination_reasons") or ["-"])
        lines.append(
            f"| {i} | `{r['winner']}` | {hp_a:.1f} | {hp_b:.1f} | "
            f"{r.get('steps', 0)} | {term} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_How to reproduce with your own policy: replace `policy_a` in "
                 "`examples/06_evaluate_policy.py` with your `Policy` subclass._")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Example 06 — 按 docs/RULE_zh.md 规则评测（MatchRunner）")
    print("=" * 70)

    out_dir = example_out_dir("06_evaluate_policy")
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # "我的 policy" 占位——换成你自己的 Policy 子类即可。
    policy_a = StandingCombatPolicy()
    policy_b = RandomCombatPolicy(scale=0.3, seed=42)

    runner = MatchRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        env_factory=_make_env_factory(),
        total_rounds=6,           # RULE §2.3
        verbose=False,             # 自己打印更干净的 summary
    )

    print(f"\n对战开始: {type(policy_a).__name__} (A)  vs  {type(policy_b).__name__} (B)")
    print("规则: 100HP × 6 rounds × 30s，KO 或总血量胜")
    print("（约 30~90s 取决于 CPU/MuJoCo 速度）\n")

    t0 = time.perf_counter()
    # 固定 match base seed 便于复现；每回合的 seed 由 MatchRunner 内部
    # 用 SeedSequence.spawn 派生（见 envs/framework/SEED.md）。
    result = runner.run(seed=20260424, video_dir=str(video_dir))
    elapsed = time.perf_counter() - t0

    # 打印简洁终审报告
    print("\n" + "=" * 70)
    print(f"Match finished in {elapsed:.1f}s")
    print("-" * 70)
    for i, r in enumerate(result.round_results, 1):
        print(
            f"  Round {i}: winner={r['winner']:<10}  "
            f"HP A={r['final_health'].get('robot_a', 0):.1f}  "
            f"HP B={r['final_health'].get('robot_b', 0):.1f}  "
            f"steps={r.get('steps', 0)}"
        )
    print("-" * 70)
    if result.ko_winner:
        print(f"KO WINNER : {result.ko_winner}")
    else:
        print(f"MATCH WIN : {result.final_winner}  "
              f"(A={result.total_score.get('robot_a', 0)}  "
              f"B={result.total_score.get('robot_b', 0)})")
    print("=" * 70)

    # Write markdown report.
    report_path = out_dir / "match_report.md"
    _write_report(
        result,
        policy_a_name=type(policy_a).__name__,
        policy_b_name=type(policy_b).__name__,
        elapsed_s=elapsed,
        out_path=report_path,
    )
    print(f"\nReport : {report_path}")
    videos = sorted(video_dir.glob("round_*.mp4"))
    print(f"Videos : {len(videos)} file(s) under {video_dir}/")
    if videos:
        print("  - " + "\n  - ".join(str(v.name) for v in videos))
    print("\nDone. 想评测你自己的 policy：用 `Policy` 子类替换 policy_a 即可。")


if __name__ == "__main__":
    main()
