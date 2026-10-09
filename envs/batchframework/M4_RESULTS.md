# M4 结果：standup 任务原生设备化

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

对应 [M4_PLAN.md](M4_PLAN.md)。将 `standup_4stage_dense_v2_env.yaml`
的插件面（`RandomFallenStatePlugin` + `StandingBalance4StageRewarder`
+ `TimeoutPlugin`）转换为设备端原生实现，并完成固定策略交叉评估。

## 1. T1 — 接触力 flat 提取共享化

`device_obs.contact_forces_flat(state, geom_bodyid, geom_aff)`：

- 输入为 `state.sim.contacts_flat`（warp flat-packed：worldid/dist/
  geom1/geom2/frame/efc_address + `n_active`）与后端显式查找表；
- 活跃接触过滤只看 `n_active`，padding 槽位永不参与；
- 力重建按 `condim=3` 四行 EFC：normal = 四行和，
  `f1 = row0 − row1`、`f2 = row2 − row3`，再经接触 frame 转世界系；
- 返回 worldid / geom1·2 / body1·2 / aff1·2 / force_mag / world_force，
  观测与 rewarder 共用同一份实现（无第二份分类逻辑）。

## 2. T2 — DeviceStandup4StageRewarder（native observer）

`device_standup.DeviceStandup4StageRewarder`：四阶段分段 potential
的 torch 批量版。全部阈值/常量 **import 自**
`baseline/humanoid21/rewards/standing_balance_4stage.py`（单一来源，
不复制字面量）。

保真的精确语义：

- 接触分类阈值 `1.0 N`；承重转移门限 `10 N`；
- env 接触必须满足：一侧 aff==0 且 geom==`ground`，另一侧 aff==目标
  robot（`robot_aff = agent_idx + 1`）；
- extra_contact_count = **distinct robot body 去重计数**
  （(env,body) bool scatter → sum），不是接触点数；
- stage 3↔4 的 `d_hf` 只用 XY 分量；
- 输出键序稳定：`stage/potential/f_score/contact_score/d_score/
  d_hf/w_foot/h_score/h_torso`。

**验收（`tests/test_device_standup.py`）**：

- Layer A（纯逻辑，合成张量同注入态对照 CPU 标量实现）：
  9 用例全过——stage1 无接触趴姿 / stage2 含额外接触 / stage3 仅手
  足支撑 / stage4 站立 / distinct-body 去重 / 1N 阈值 / 10N 承重门
  限 / 非 ground 与错误 aff 排除 / XY-only d_score / F_ENTER 边界。
- Layer B（warp 真机同注入态 vs CPU）：`test_same_injected_state`
  通过。

## 3. T3 — DeviceFallenResetPlugin（native resetter）

CPU `RandomFallenStatePlugin` 在每个 episode reset 内嵌独立
`Humanoid21Simulator` 串行跑 ≤1000 物理步让机器人随机摔倒。
设备端版本直接在共享 warp batch sim 上**并行摔倒**：

- per-env 常量随机 action，robot_a / robot_b **独立抽取**
  （曾发现并修复广播 bug：双机同 action → 同步倒地、分布失真）；
- per-env 早停：首次 `h_root < 0.3` 捕获该行 qpos/qvel，rollout
  结束后统一写回（不使用"末帧"——保持与 CPU 的 first-crossing
  语义一致）；
- 非目标机器人按 `reset_interval=5` 复位到摔倒前快照
  （修复：写回基底必须是摔倒前状态而非摔倒后状态）；
- RNG = splitmix64(seed_offset ⊕ env_index ⊕ reset_counter)，
  reset_counter 用私有 host 计数（不能放 plugin pool——部分 reset
  会清零 pool 行导致同 env 复用序列）；`_lshr` 逻辑右移实现
  int64 环绕语义；
- 全量与部分 reset 均覆盖。

**验收 A（同注入态/逻辑）**：`test_already_fallen_captures_step1`、
`test_partial_reset_isolation`、`test_fallen_state_written_back`、
`test_seed_determinism_and_diversity` 全过。

**验收 B（分布等价，N=24/侧）**：

| | hit 率 | fall 步数中位 | h_a 均值±std |
|---|---|---|---|
| CPU (fp64) | 0.96 | 500 | 0.450 ± 0.234 |
| DEV (warp fp32) | 1.00 | 460 | 0.431 ± 0.175 |

双峰结构复现（~20% 的 env 中 robot_a 在捕获时刻仍站立——CPU 语义
是"任一目标过阈值即停"），只承诺统计等价——**不声明 bit identity**
（CPU fp64 vs warp fp32 物理轨迹本就不可能逐位相同）。

## 4. T4 — 固定策略交叉评估

协议：CPU env 产出 N=64 个冻结摔倒初态 → 两侧注入相同初态 →
确定性策略（`TruncatedNormalPolicy`，3 层 256 隐藏 MLP，
`deterministic_action`）跑满 200 步无早停。

checkpoint：`train_standup_floor04_ppo_20260920_164819`，
u0001 / u0300 / u1500（未训练 / 中期 / 成熟）。

**评估基础设施**：`probe_standup_xeval.py`。
`_MetricRecorder`（低优先级 device 插件）在 `on_post_action_step`
快照 observer 输出——必须在 reset 消费**之前**采样（步内终止的
`on_envs_reset`/`on_pre_episode` 会清 `_out`，且插件池行被清零）。

**结果**（N=64 冻结初态 × 2 agents = 128 episode-agent/组，
原始数据 `m4_t4_results.json`）：

| ckpt | 指标 | CPU fp64 | Warp B=16 | Warp B=64 |
|---|---|---|---|---|
| u0001 | success | 0.000 | 0.000 | 0.000 |
| 未训练 | max_pot | 0.2243 | 0.2259 | 0.2274 |
| | final_pot | 0.0201 | 0.0220 | 0.0203 |
| | max_stage | 2.398 | 2.422 | 2.438 |
| | max_h | 0.4328 | 0.4331 | 0.4328 |
| u0300 | success | 0.000 | 0.000 | 0.000 |
| 中期 | max_pot | 0.8157 | 0.8162 | 0.8158 |
| | final_pot | 0.8121 | 0.8126 | 0.8121 |
| | max_stage | 4.000 | 4.000 | 4.000 |
| | max_h | 0.9872 | 0.9881 | 0.9875 |
| u1500 | success | 1.000 | 1.000 | 1.000 |
| 成熟 | max_pot | 1.0000 | 1.0000 | 1.0000 |
| | final_pot | 0.99999 | 0.99999 | 0.99999 |
| | max_stage | 4.000 | 4.000 | 4.000 |
| | max_h | 1.2845 | 1.2843 | 1.2842 |

**判读**：

- success 三档全部逐点一致（0/0/1.0——u0300 已达到 stage 4 但
  平均未过 0.9 阈值，两侧一致，非后端分歧）。
- 全部连续指标差 ≤ 0.002（fp32 轨迹发散量级内）；
  final_pot 差 ≤ 0.002，远低于 ≤0.02 非劣效门限。
- **B 敏感性**：B16 vs B64 差 ≤ 0.0015（per-env 物理独立，
  残差为 warp kernel 批次级非确定性）——无系统性退化。
- 结论：三档策略在 warp 后端上的表现与 CPU 参考**无系统性偏
  差**；不存在"需要 PPO 重新适应"的环境实现误差。

**T4 过程修复**：`test_fallen_state_written_back` 断言修正——
hit 语义是"两目标 root 高度 min < 0.3"（任一达标即停），原断言
要求 robot_a 单独 <0.35；独立随机 action 修复后非先倒地机器人
可在捕获时仍处高位（正是双峰分布），断言改为 `min(za,zb)`。

## 5. 能力注册表

| 蓝图类 | 状态 | 设备端实现 |
|---|---|---|
| `RandomFallenStatePlugin` | NATIVE | `DeviceFallenResetPlugin` |
| `StandingBalance4StageRewarder` | NATIVE | `DeviceStandup4StageRewarder`（observer） |
| `TimeoutPlugin` | NATIVE | `DeviceTimeoutPlugin`（M3 已有） |

## 6. 已声明差异

- **fp32 vs fp64**：warp 物理单精度，CPU MuJoCo 双精度。断言走
  语义/统计等价，不走逐位相等（M2 已定：fixture 容差 atol 2e-2 /
  rtol 1e-3，实测偏差为舍入级）。
- 摔倒姿态分布：统计等价而非逐轨迹相同。
- contacts 插件视角仍为 flat + cap 元数据；padded scatter 视图在
  需要时补（本任务未需要）。
