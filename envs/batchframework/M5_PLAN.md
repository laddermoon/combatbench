# M5 计划：Rollout 接入 + PPO/Debug 数据链路验证（R1）

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

对应 [ROADMAP.md](ROADMAP.md) §8。目标：**训练框架能消费设备路
径产出的 Episode，且消费的是同一个训练问题**。本阶段不宣称训练
效果或加速达标——只证技术闭环。

## 0. 已审计的关键事实

| 事实 | 位置 | 结论 |
|---|---|---|
| 接入缝 | `ppo/loop.py:834` `rollouter.collect(jobs)` | 下游（build_trajectories→PPOBuffer→GAE→update→dump）只认 `List[Episode]`——`DeviceRollouter` 实现同签名即可，trainer 零改动 |
| Episode 组装 | `episode.py:from_buffer_frames` | 帧 dict（observation/action/action_extras/observer_outputs）→ 自动 stack；**直接复用**，不自写组装 |
| 采样契约 | `SamplingPolicy.act` → `sample(obs, ctx)` → extras | standup spec = `SamplingSpec(explore_factor=0.0)`，无 reference/delta；extras 每帧须含 `log_prob` + `explore_factor`（缺帧即整 agent 被 stacker 丢弃） |
| 批量采样 | `TruncatedNormalPolicy.sample_action(obs (B,96), ctx)` | 已天然批量：`(action (B,21), log_prob (B,))`；ef 支持 (B,) tensor |
| 策略版本 | `actor.to_blueprint()` → `policy_exports/uNNNNN` | rollout 消费**导出蓝图**而非活 actor；collector 从 `job.policy_*_bp` 重建 `TruncatedNormalPolicy`（self-play 双 agent 共享一份权重）——有 parity test 保证导出=训练侧 |
| per-env options | `mjx_simulator._compute_reset_state` | `initial_distance` 已支持标量或 (B,) 广播；**波次同步模式下 `dev_reset_rows` 的 per-env options 不需要**（全波同时 reset） |
| final_obs | `rt.step` 顺序 | obs 构建(step 8) 早于终止消费(step 10-13)——`io.obs_*` 在末步后即 obs_{T+1}，bootstrap 语义天然满足 |
| 终止记录 | Episode 要求 per-agent `(reason, step)` | standup 仅 timeout @200 双 agent；collector 从 `agent_terminated`/`term_reason` 合成 |
| 执行模式 | ROADMAP §6 | "第一个 collector 只用同步定长模式"——standup 全员 200 步 timeout 结束，天然 lockstep |

## 1. 架构

```
jobs (512) ──chunk(B)──▶ DeviceRollouter
                             │  per wave:
                             │    rt.reset(seeds, options={initial_distance:(B,)})
                             │    loop 200:
                             │      obs = io.obs_* (B,96)
                             │      a,logp = policy.sample_action(obs, ctx)
                             │      rt.step((a_a, a_b))
                             │      buffers[t] ← obs/act/logp/observer
                             │    final_obs ← io.obs_*（末步 obs）
                             │  per env → frames → Episode.from_buffer_frames
                             ▼
                        List[Episode]（与 jobs 同序）
```

- 单进程、常驻 sim+runtime（跨 wave/update 复用，无重建开销）
- 热路径零 sync；每个 wave 结束一次批量 `.cpu()` 组装
- **策略 RNG**：`torch.rand` on cuda——分布正确即可，不承诺与
  CPU rollout 的逐 seed 一致（语义等价边界，同 M4）

## 2. 支持边界（显式拒绝，不静默降级）

| Job 特性 | M5 处置 |
|---|---|
| `explore_factor` 标量 / `(B,)` | ✅ |
| `explore_factor` callable | ❌ 拒绝（须 host 逐帧求值） |
| reference / delta_factor≠0 / delta_mode | ❌ 拒绝（M5 范围外） |
| per-agent 早停（KO 类） | ❌ 拒绝——同步定长模式只支持 env 级终止；standup 不触发 |
| `stochastic=False` | ✅ deterministic_action 批量 |
| 非 `WarpHumanoid21Simulator` 后端 env_bp | ❌ 拒绝（blueprint simulator 字段校验） |
| 未注册/非 NATIVE 插件 | ❌ 拒绝（capability_registry） |

## 3. 工作包

### W1 — `device_rollouter.py`

- `DeviceRollouter(batch_size, device)`：`collect(jobs) -> List[Episode]`
- blueprint 校验 + `capability_registry.resolve_plugin/resolve_observer`
  装配（fallen reset / standup rewarders / timeout）
- 批量采样：policy bp → `TruncatedNormalPolicy` 重建 →
  `sample_action(obs, ctx)`；extras 记录 `{log_prob, explore_factor}`
- Episode 组装复用 `from_buffer_frames`；termination records 从
  `agent_terminated`+`term_reason`+步数合成；`episode_metrics`
  ← fallen 插件 pstate（init_steps/init_height/init_hit）+
  `{"backend": "warp-fp32"}` 来源标识

### W2 — 训练管线同数据验证（先于规模）

- 用 device collector 产出 N=32 eps → `build_trajectories` →
  PPOBuffer → GAE —— 与 CPU collector 同种子产出**逐项对照**
  （结构/键集/shape/dtype 必须同构；数值按分布级比较）
- 过渡矩阵计数核对：1024 trajs/update、204,800 transitions、
  无漏帧/重复/跨回合拼接；final_obs bootstrap 在位
- **log_prob 一致性**：rollout 记录 log_prob vs
  `evaluate_actions` 重算（同 obs/action/ctx）→ fp 容差内一致
  —— 这是"采样分布与训练分布相同"的可证伪检查

### W3 — 训练循环接入

- `train.py`/`loop.py`：collector 选择开关（`--collector device`|
  配置项），默认 CPU rollouter 不变
- eval 路径（stochastic=False jobs）走同一 collector
- 日志标注后端与 fp 精度；dump 的 Episode 带来源字段（
  "不能把 CPU 重跑当 warp 原轨迹"）

### W4 — 规模验收 + 性能分解

- 完整 update：512 eps 跑通 → 计数 + 指标分布 vs CPU 对照
- 每 wave 计时分解：物理步进 / obs 构建 / 策略前向 / host 组装 /
  reset（含摔倒）——证明 GPU 收益未被 Python 编排抵消
- CPU 回归测试不受影响

### W5 — Debug 链路 + 文档

- dump Episodes/trajectory/buffer 消费验证；metrics 语义不变
- M5_RESULTS.md：放行条件逐条核对 + 已声明差异

## 4. 放行条件（R1）

1. 512 eps → 1024 trajs → 204,800 transitions，结构核验全过
2. 同数据过训练管线结果一致（结构同构 + 分布等价）
3. log_prob 重算对齐（无过期策略混入）
4. timeout 的 final_obs bootstrap 语义保留
5. 性能分解：host 组装占比可量化、可接受
6. Episode 带来源标识；CPU 路径回归无损

## 5. 明确不做

- 异步/不等长 episode 调度（per-env options on `dev_reset_rows`
  仍 deferred）
- reference/delta 采样机制的批量实现（留给需要它的实验转换时）
- 训练效果验收（M6/M7 的事）
