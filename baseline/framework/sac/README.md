# SAC 训练框架 / 调试系统

> 类型：指南

独立、SAC 原生的多 critic off-policy 训练框架——目标是对齐 PPO
框架的工程深度（可验证、可诊断、可恢复），而不是"把 PPO 换一个
loss"。MVP 已实现并验证（`IMPLEMENTATION_SUMMARY.md`）。

## 🏆 这项成果的价值

- **SAC 原生而非移植**：off-policy 机制按 SAC 的需求建——
  `TaggedReplay`（轨迹连续存储、逐 channel 奖励、n-step return、
  `source_key` 全链路溯源 run→round→job→episode→frame）、
  env_step 时钟（eval/checkpoint 按环境步调度，与 UTD 无关）、
  逐 channel TD target。回放样本可溯源是它的调试原语。
- **多 critic 用 SAC 自己的方式**：每 channel 独立 twin-Q + 独立
  gamma + actor 门控/权重 + clipped double-Q + 自动温度——不是把
  PPO 的 advantage 归一化搬过来；action-gradient 归一化是主机制，
  `GradNormStats` 记录逐通道梯度占比供诊断。
- **诊断体系对齐 PPO 深度**：自有 `debugkit`/`debugserver`/
  `diagnostics`/`metric_catalog`/`DEBUG_PLAYBOOK`；训练中写哨兵
  文件即可在下一个 critic tick 捕获完整截面（按需 dump，非常驻开销）。
- **正确性优先的分层构建**：先退化到标准 SAC 正确性基线，再引入
  多通道与优化机制；checkpoint/resume 分段恢复与连续训练**逐位一致**
  （模型+优化器+replay+全部 RNG+实验状态）。
- **研究级路线设计**：Shannon 熵 SAC 保留为基线，uncertainty
  路线（U-bonus/U-floor）是用户批准的独立替代——actor/target/系数
  控制一致改写，不默认叠加（`DECISIONS.md` A4）。
- **复用成果而不反向污染**：共享 `EnvRuntime` 与环境数据契约，
  TruncNorm 八格策略族经 `tn_actor` 适配；按 PLAN 设计边界，
  需要借用的接口**复制后适配**，不改 PPO 迁就 SAC。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| `source_key` 全链路溯源的 TaggedReplay | 任意 buffer 样本可追到具体 episode/frame——off-policy 调试从"看统计猜"变成"看样本查" |
| 逐 channel twin-Q + 独立 gamma | 多目标格斗奖励的信用分配不互相缠住，与 PPO multi-critic 同构但走 TD 路径 |
| env_step 时钟与 UTD 解耦 | eval/checkpoint 语义不受采集-训练比影响，跨配置可比 |
| 哨兵文件触发按需 dump | 取证能力存在但零常驻开销——常驻全量记录的成本不付 |
| 复制后适配（不反向改造 PPO） | 两个算法框架独立演化，PPO 的成熟路径不被 SAC 实验拖累 |
| 先标准 SAC 基线再多通道 | 正确性有对照系——多通道 bug 不会伪装成"SAC 本身不 work" |

## 🔑 现状与边界

- **MVP 实现完成并验证**（`IMPLEMENTATION_SUMMARY.md`）；`experiments_sac/`
  已有 standup/balance 两个实验注册。
- **SAC 基线（成果 #7）尚未开始**——本项交付的是框架与调试系统，
  训练出的基线策略是下一项成果。
- PLAN 明确"不预设必须比 PPO 更快或更好"——验收标准是工程深度
  对齐，不是算法胜负。
- 深入阅读：`GUIDE.md`（实验作者用法）· `PLAN.md`（路线图）·
  `DECISIONS.md`（设计决策）· `DEBUG_PLAYBOOK.md`（故障排查）·
  `ARTIFACTS.md`（工件版本）。
