# RUNNERS — RoundRunner / MatchRunner CLI 参考

> 类型：指南

两个评估入口：**RoundRunner** 跑单回合、**MatchRunner** 编排多回合
match（内部逐回合调 RoundRunner）。均从蓝图启动。两文件用相对 import，
**必须 `python -m` 调用**（直接 `python3 xxx_runner.py` 会挂）：

```bash
PYTHONPATH=. python3 -m envs.framework.round_runner --help
PYTHONPATH=. python3 -m envs.framework.match_runner --help
```

## RoundRunner — 单回合

```bash
PYTHONPATH=. python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/fight_env.yaml \
  --policy-a-blueprint policy/blueprints/humanoid21/standing.yaml \
  --policy-b-blueprint policy/blueprints/random.yaml \
  --video out.mp4 --seed 42
```

| 参数 | 默认 | 语义 |
|---|---|---|
| `--env-blueprint` | 必填 | 环境蓝图（YAML/JSON） |
| `--policy-a-blueprint` / `--policy-b-blueprint` | 必填 | 双方策略蓝图 |
| `--video PATH` | 无 | 录像 mp4（自动挂 VideoRecorderPlugin，fps=30） |
| `--recorder SPEC` | 无 | 注入 PostActionRecorder，可重复；格式 `module:Class?key=value`（值按 JSON/布尔/None 解析） |
| `--seed` | 随机 | 回合种子 |
| `--health-a` / `--health-b` | 插件默认(100) | 初始 HP，并入 options 的 `initial_health_*` 键 |
| `--want-extras` | off | 调 `policy.act(want_extra=True)` 取侧信道 |
| `--score-log-file` | 无 | 逐物理步打分审计日志（入 options 的 `score_log_file` 键） |
| `--options-json` | 无 | JSON 文件提供 `episode_options`（如 impulse_params） |

**输出**：stdout 打印 JSON `{steps, termination_reasons, seed,
health_a, health_b}`。

## MatchRunner — 多回合 match

```bash
PYTHONPATH=. python3 -m envs.framework.match_runner \
  --env-blueprint ..._env.yaml --policy-a-blueprint ... --policy-b-blueprint ... \
  --total-rounds 6 --video-dir videos/ -o result.json
```

| 参数 | 默认 | 语义 |
|---|---|---|
| `--env-blueprint` | 必填 | 支持 ParameterizedEnvBlueprint（按默认值 materialize） |
| `--policy-a/b-blueprint` | 必填 | 策略蓝图 |
| `--total-rounds` | 6 | 回合数 |
| `--initial-health` | 100 | 双方初始 HP |
| `--score-log-dir` | 无 | 每回合 score 审计日志目录 |
| `--video-dir` | 无 | 每回合录像目录 |
| `--seed` | 随机 | 基础种子 |
| `--output`/`-o` | stdout | 结果 JSON 落盘路径 |

**MatchResult** 字段：`total_rounds`、`rounds_completed`、
`round_results[]`（每回合 RoundRunner 结果）、`final_winner`
（`robot_a`/`robot_b`/`draw`）、`total_score`、`ko_winner`（KO 提前
结束整场时置位）、`initial_health`。无 `-o` 时打印 JSON。

## 编程接口

```python
from envs.framework.round_runner import RoundRunner
from envs.framework.match_runner import MatchRunner, load_env_blueprint

with RoundRunner(blueprint=bp, policy_a=pa, policy_b=pb) as r:
    result = r.run(seed=42)          # -> dict（同上字段）

mr = MatchRunner(env_blueprint=bp, policy_a_bp=pa_bp, policy_b_bp=pb_bp,
                 total_rounds=6, initial_health=100)
match = mr.run(seed=42, video_dir="videos/")   # -> MatchResult
```

相关：`Episode` 级完整记录走 `baseline/framework/rollout/EpisodeRecorder`；
`episode_options` 键目录见 `EPISODE_OPTIONS.md`。
