# R3-B 独立迁移试验 — 任务书与冻结口径

> 本文件分两部分：§1 是试验前冻结的协议（选例理由/验收层级/记录指标），
> §2 是可直接粘贴给空上下文 Agent 的任务书。
> 冻结日期：2026-10-09。依据：ROADMAP §11（M8 / R3 放行条件）。

## 1. 冻结的试验协议

### 选例：`step`（`step_mbs` 同环境族）

- 蓝图：`baseline/humanoid21/end2end/step_env.yaml`
- 覆盖 standup 未触达的扩展面（R3-B 要求"至少一种新差异"）：
  - `GaitClockSimulator` → 需要**新 simulator binding**（obs +3 维）
  - `FootStateObserver` ×2 → 需要**新设备 observer**
- 已 native 可复用：`RandomFallenStatePlugin`、`StandingBalance4StageRewarder` ×2

### 显式排除（已知框架缺口，登记即可，不要求实现）

- `step_mbs` 的 MoG actor（需要新 `PolicyExecutor` 类型）
- `exp_step` 的逐帧 callable `explore_factor`（采样 ctx 只支持 per-env 标量）

### 验收层级（达到哪级如实报告，不许跳级宣称）

| 级别 | 内容 |
|---|---|
| L1 | 全部蓝图单元登记/审计完成，产出结构化迁移计划 |
| L2 | 新设备单元通过跨后端 fixture/契约验证 |
| L3 | device collector `collect(jobs)` 产出合法 Episode |
| L4 | `--collector device` PPO smoke（2 updates）端到端跑通 |

### 记录指标

首次通过率（每级一次过/返工次数）、被迫人工介入次数与原因
（=文档/工具缺口）、新增代码量、到达层级、总耗时。
Agent 每次回来问问题都记为一次介入；**修复资产后需重新试验**才算数。

---

## 2. 任务书（粘贴给空上下文 Agent）

```text
任务：把 CombatBench 的 `step` 实验环境迁移到 batch framework
（设备端批量 rollout），并用它从已训练完成的 standup 策略热启动
做一次真实训练接入验证。

仓库：/data1/mono/things/combatbench（只在这个仓库内工作；
PYTHONPATH=<repo>）。

背景
----
仓库里有两个并行框架：
- CPU 参照框架：envs/framework（EnvRuntime + 插件体系，MuJoCo）
- GPU 批量框架：envs/batchframework（MuJoCo-Warp，BatchRuntime +
  DeviceRollouter/MultiDeviceRollouter），对外契约是
  `collect(jobs) -> List[Episode]`，训练入口 baseline/framework/train.py。

目标实验 baseline/experiments_ppo/exp_step.py（实验名 `step`）目前只
跑过 CPU collector。它的环境蓝图是
baseline/humanoid21/end2end/step_env.yaml。

先读（按序，全部读完后才开始写代码）
---------------------------------
envs/batchframework/README.md
envs/batchframework/PUBLIC_INTERFACE.md
envs/batchframework/MIGRATION_GUIDE.md
envs/batchframework/SEMANTICS.md
baseline/experiments_ppo/README.md（实验注册与启动方式）

硬规则
------
1. 严格按 MIGRATION_GUIDE 的流程与验证阶梯执行，每级留证据。
2. 不得改变任务语义（物理、观测维度/内容、奖励、终止、采样分布、
   episode_options 语义）。
3. 不支持的单元/能力必须显式拒绝或登记为缺口——禁止静默降级、
   禁止为绕过校验而改宽校验器。
4. 每完成一个代码修改立即 git commit 并 git push。
5. 遇到文档/工具覆盖不了的情况：记录"需要什么信息、文档缺什么"，
   停下并在最终报告里说明，不要凭猜测实现框架级机制。
6. 未注册的蓝图单元应该启动即失败——这是特性，不是 bug。

验收层级（逐级做到，如实报告到达哪级）
------------------------------------
L1  step_env.yaml 全部单元完成能力登记/审计，产出迁移计划
    （每个单元：native/需转换/不支持 + 依据）
L2  新设备单元（GaitClockSimulator 绑定、FootStateObserver）
    通过指南里的跨后端 fixture/契约验证
L3  DeviceRollouter/MultiDeviceRollouter.collect(jobs) 对 step 环境
    产出合法 Episode（trajectory/termination/bootstrap 字段完整）
L4  `--collector device` 的 PPO smoke 端到端跑通

已知范围外（遇到登记为缺口即可，不要实现）
----------------------------------------
- `step_mbs` 的 MoG actor（StateMixtureBoundedStdTruncatedNormalPolicy
  需要新 policy executor 类型）——验证用 `step`（TruncatedNormalPolicy）
- exp_step 的逐帧 callable explore_factor（采样 ctx 只支持 per-env
  标量）——如 rollout 需要 ef，记录"需要 per-frame ef 支持"为缺口

训练接入语义（warm-start，重要）
------------------------------
`step` 实验设计为从已收敛的 standup 策略热启动：actor 网络的观测维度
是 99（96 基础 + 3 gait-clock），`load_checkpoint` 会把 standup checkpoint
（obs_dim=96）第一层输入列零填充到 99，扩展输入初始惰性——恢复后的
策略初始行为与原 standup 策略一致。

现成的 standup checkpoint（一条 device collector 训出的收敛 run）：
  baseline/runs/train_standup_ppo_20261008_131639/checkpoints/checkpoint_u01500.pt

启动命令模板（验证用 smoke）：
  PYTHONPATH=. python3 -B baseline/framework/train.py \
      --experiment step --algo ppo \
      --collector device --collector-devices <空闲GPU,如 2,3> \
      --collector-batch-size 256 \
      --resume-from <上面的 checkpoint 路径> --reset-update --smoke

（`--reset-update` = "warm weights 的新 run"；L4 以 smoke 跑通为准，
不要求训练质量。）

最终报告要求
------------
- 迁移过程记录（每步做了什么、依据哪份文档）
- 每级验收的证据（命令 + 输出要点 + 产物路径）
- 文档/工具缺口清单：哪里文档没说清、哪个工具缺了、你做了什么猜测
- 卡住/失败的位置和原因（如果有）
- 新增/修改代码量统计（git diff --stat）
```
