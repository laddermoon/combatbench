# CombatBench Examples — 面向策略开发生命周期的示例集

> 本目录的 9 个示例（`01`–`09`）均已实现，按策略开发者真实会走的
> 阶段组织；每个例子一个 python 文件，产物落到 `examples/out/<name>/`。

## 立场

CombatBench 的定位是**一个格斗 benchmark 平台**：我们提供 Env + Framework，
让开发者在上面训练自己的策略。我们**不提供训练算法**（那是 PPO/GRPO/SAC 的事），
但要让开发者看到：**在策略开发的每个阶段，框架都能为他帮上什么忙。**

所以例子不按"框架有哪些类"组织，而按**策略开发者真实会走的那条路**组织：

```
阶段 0: 这个环境是什么？obs/action/reward 长什么样？
           ↓
阶段 1: 先来个 scripted baseline 看能不能跑，顺便校准感觉。
           ↓
阶段 2: 我要开始训练了。但我需要：课程化扰动、早停、从插件 metrics 造
        reward term ——  框架提供了哪些挂载点让我干这些？
           ↓
阶段 3: 训练 = 大规模采样 + 存轨迹。框架怎么让我并行、可复现、落盘？
           ↓
阶段 4: 训崩了/怀疑有 bug：回放录像、逐帧看状态、复算 observer。
           ↓
阶段 5: 训好了：跟 baseline 打，多 seed、多场，看胜率和置信度。
```

每个例子对应一个阶段，恰好**一个 python 文件，不带 CLI，直接 `python xxx.py`**，
产物全部落到 `examples/out/<example>/`。

---

## 例子列表

### `01_explore_env.py` — 阶段 0：认识这个环境
**角色**：刚拿到仓库的开发者。
**展示**：
- 用 `_common.build_humanoid21_runtime()` 拉起一个 runtime。
- 打印 **action 维度、phys freq、phy_steps_per_action、max_steps**。
- 跑一步 `EnvRuntime.step(zero_action, zero_action)`，打印 observer 输出的 shape / 数值范围 / 关键字段。
- dump 一份"最小 obs 示例"JSON 到 `out/01_explore_env/sample_obs.json`。

**读完知道**：
- 这个格斗环境的 I/O 边界是什么（不用看源码就能开始设计网络）。
- "observer 输出一个 dict"这个约定是怎么走的。

---

### `02_scripted_baseline.py` — 阶段 1：先写个 scripted policy 跑起来
**角色**：要做 RL 但先想搭个**非学习 baseline** 做对照组的开发者。
**展示**：
- 实现 `class SinusoidPolicy(Policy)`：正弦波驱动关节，纯确定性。
- 用 `EpisodeRunner` 跑 `sinusoid vs random`，打印累计 reward、每方动作范数的时间曲线。
- 同一 seed 跑两遍 → 逐帧 assert 动作 bit-wise 一致（证明"可作为训练期对手的确定对局"这点成立）。

**读完知道**：
- Policy ABC 的最小实现（`act` 必写、`reset` 可选）。
- Seed 管理让 scripted 对手可以当作训练课程里的**确定性 opponent pool**。
- 两个 policy 如何同时注入到一个 runtime。

---

### `03_training_aids.py` — 阶段 2：训练时我要用的插件三件套
**角色**：准备开训的 RL 开发者。核心例子，篇幅稍长。
**展示三个典型训练辅助插件**，分别对应训练中最常见的三类需求：

1. **课程扰动 `CurriculumPushPlugin(BasePlugin)`**
   - 在 `on_pre_episode` 根据"已完成 episode 数"调整 push 幅度 → 从平地训起，逐步上难度。
   - 演示**插件状态跨 episode 累计**，以及**和 `set_episode_seed` 配合**保持可复现。

2. **早停 `FallenEarlyTerminationPlugin(BasePlugin)`**
   - 在 `on_post_action_step` 检测摔倒，`ctx.request_termination("fallen")` 提前结束，少浪费 sample。
   - 演示 `ctx.termination_proposals` 机制，以及 early termination 如何进入 `EpisodeResult`。

3. **自定义 reward term `ClosingDistanceRewardObserver(BaseObserverPlugin)`**
   - 从 `ctx.accessor` 读两个机器人的距离，用 `on_post_action_step` 聚合，`get_output()` 返回可直接喂给训练 loop 的 reward。
   - 演示 **plugin（写/算）→ metrics（共享）→ observer（策略侧只读暴露）** 这条标准数据流。

最后，一个串联 demo：把三个插件一起挂上跑 3 局，打印每局的扰动幅度、是否早停、累计 reward 分解。

**读完知道**：
- Plugin 的生命周期 hook 具体应该选哪个（表格式说明）。
- 框架的**权限隔离**（accessor/mutator/metrics）在训练辅助场景下的真实用法。
- 如何把一个"想做的训练 trick"优雅地塞进框架而不是改环境代码。

---

### `04_collect_rollouts.py` — 阶段 3：可复现地、并行地采样
**角色**：要灌数据的训练工程师 / BC / offline RL 开发者。
**展示**：
- 用 `EpisodeRunner` 顺序跑 16 局（并行采样属 `baseline/framework/rollout`
  的 `ParallelRollouter`，不在本示例范围）。
- 以 `want_extras=True` 把 policy 返回的 `{"log_prob": ..., "value": ...}`
  一起透传给 recorder → **训练算法可以直接从这里吃数据**。
- 所有帧用 `BaseFrameRecorder` 同步落盘到 `out/04_collect_rollouts/rec/`；打印 `manifest.json` 里的 `base_seed` 字段。
- 用同一 `base_seed` 重跑一次，断言产出的轨迹完全一致（复现性保证）。

**读完知道**：
- per-episode seed 由 `SeedSequence.spawn` 派生的链路长什么样。
- `want_extras`/`action_extras` 是框架**专门为 on-policy RL 预留**的通道。
- 一条"可以直接喂给 PPO / BC 的数据管线"在框架层如何成立。

---

### `05_replay_and_inspect.py` — 阶段 4：出了 bug / 想看训练片段
**角色**：训练不 work 的人；想做可视化分析的人。
**展示**：
- 读 `04` 生成的录像目录，用 `ReplaySimulator` 把其中一局**重新跑一遍**。
- 挂**和在线训练时完全一样的 observer plugin**，断言在线 observer 输出 == 回放 observer 输出（bit-wise）。
- 同一条录像再挂 `VideoRecorderPlugin`，生成 MP4 到 `out/05_replay_and_inspect/episode.mp4`。
- 小彩蛋：遍历录像每帧，找到"第一个高度 < 0.6 的帧"，打印上下文 → 展示**逐帧 debug 的最短路径**。

**读完知道**：
- ReplaySimulator 是一个"非物理后端"，但对 observer / plugin **完全透明**。
- 训练复现 bug 的标准流程：seed → recorder → replay → 同一套 observer。
- Video 和 Frame recorder 如何叠加使用。

---

### `06_evaluate_policy.py` — 阶段 5：按比赛规则打一场
**角色**：要出一份"我的模型 vs baseline"评测报告的人。
**严格遵循 `docs/RULE_zh.md`**：
- 双方初始血量 100，**6 回合，每回合 30 秒**，20Hz 决策 / 500Hz 物理。
- 每回合从初始位姿开始（血量延续上一回合）。
- KO（对方血量 → 0）立即结束比赛；否则 6 回合结束后血量高者胜，相等为平。
- 有效打击、伤害数值等全部由 humanoid21 环境内部按规则书判定，例子不重写。

**展示**：
- 直接使用框架里已有的 `MatchRunner` —— 它的默认参数就是规则书的参数（`total_rounds=6`，`match_duration=30s`）。
- 一场 `StandingPolicy vs RandomPolicy` 的完整比赛，打印每回合 HP 变化、胜者、是否 KO。
- 把 `MatchResult` 序列化成一份规则书风格的 markdown 战报到
  `out/06_evaluate_policy/match_report.md`；同步把每回合视频存到
  `out/06_evaluate_policy/videos/round_N.mp4`（通过 `MatchRunner.run(video_dir=...)`）。

**读完知道**：
- `MatchRunner` 的 `env_factory()` 契约——无参调用一次构建 runtime，
  回合间 HP 延续经 `episode_options` 的 `initial_health_*` 传递
  （见 `envs/framework/RESET.md` G3）。
- 回合级 seed 是如何从 match base seed 用 `SeedSequence.spawn` 派生（见 `SEED.md`）。
- 把自己的 policy 塞进去代替 `StandingPolicy`，立刻就能按 benchmark 规则评测。

---

### `07_curriculum_recipe.py` — 阶段 2 延伸：课程化扰动 recipe

**展示**：用 `options_fn`（每回合下发不同 `episode_options`）表达课程
强度递增——扰动力/初始状态随 episode 序号变化，框架原生通道，无需
重建 runtime。

### `08_rollout_collection.py` — 阶段 3 延伸：rollout 模块

**展示**：新 rollout 模块——`EpisodeRecorder` + `ParallelRollouter`
的多进程采样与 episode 落盘。

### `09_episode_recorder_round_runner.py` — 阶段 4 延伸：单 episode 录制

**展示**：`RoundRunner` + `EpisodeRecorder` 做单回合录制，产出可直接
回放的轨迹文件。

---

## 目录结构

```
examples/
├── README.md                          # 本文档，同时也是用户入口
├── _common.py                         # 唯一共享工具：build_humanoid21_runtime(...)
├── 01_explore_env.py
├── 02_scripted_baseline.py
├── 03_training_aids.py
├── 04_collect_rollouts.py
├── 05_replay_and_inspect.py
├── 06_evaluate_policy.py
├── 07_curriculum_recipe.py
├── 08_rollout_collection.py
├── 09_episode_recorder_round_runner.py
└── out/                               # 所有产物（建议加到 .gitignore）
    ├── 01_explore_env/
    ├── 04_collect_rollouts/
    ├── 05_replay_and_inspect/
    └── 06_evaluate_policy/
```

依赖链：`04 → 05`（05 吃 04 的产物）；其余独立。

---

## 每个例子的统一"骨架"

为了让人一眼看出是同一套例子里的，所有文件遵循：

```python
"""Example N: <一句话标题>

面向 (Audience): <角色>
阶段 (Stage)   : <策略开发哪一步>
学到 (Takeaway): <1~3 句>
产物 (Outputs) : examples/out/<name>/...
运行 (Run)     : python examples/NN_xxx.py
"""
from __future__ import annotations
# ... 代码 ...

def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

---

## 设计原则（不变）

1. **一个文件一件事**，无 CLI 参数，直接 `python xxx.py`。
2. **产物进 `examples/out/<name>/`**，不污染别处。
3. **宁缺毋滥**（01–06 各对应阶段 0–5；07–09 是阶段 2/3/4 的延伸 recipe）。
4. **不重复 baseline/ 的训练脚本**，不做 Gym 适配演示，不写玩具 Simulator。

---

## 实现说明

本文最初是规划草案，示例已全部实现（01–09）。实现时的取舍：

- `env_factory()` 为无参契约——HP 延续走 `episode_options`（框架 G3 重构后）。
- 依赖链：`04 → 05`（05 吃 04 的产物）；其余独立。
- 所有例子直接 `PYTHONPATH=. python3 examples/NN_xxx.py` 运行。
