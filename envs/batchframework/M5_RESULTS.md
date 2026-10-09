# M5 结果：Rollout 接入（R1）

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

日期：2026-09-30 · 分支 main · 关联提交 `d996a21b` `6c7963bb` `7c1cd021`

R1 目标：加速路径产出的数据能被原训练/分析体系消费，且消费的是**同一个训练问题**——
只证技术闭环，不宣称训练加速（见 M5_PLAN.md §边界）。

## 1. 交付物

| 工作包 | 内容 | 状态 |
|---|---|---|
| W1 | `device_rollouter.py`：`DeviceRollouter.collect(jobs) -> List[Episode]`，波次同步定长采集 | ✅ |
| W2 | 同数据管线验证（trajectories/PPOBuffer/GAE 可消费 + log_prob 重放对齐） | ✅ |
| W3 | `--collector {cpu,device}` + `--collector-batch-size`；CPU 默认不变；来源标识入 config.json + RAW_STATS | ✅ |
| W4 | 完整 512-ep update 端到端验收 + 性能分解 | ✅（性能结论为负——见 §4） |
| W5 | dump/序列化/拒绝路径验证 + 本文档 | ✅ |

## 2. W1/W2：契约等价验证

对照：`baseline/framework/rollout/parallel_rollouter.py::_run_job`（CPU）产出的
Episode vs `DeviceRollouter.collect` 产出（同 jobs：seed 100-103，u1500 权重重新导出
为当前格式——旧导出 pre-ctx API 无法跑 CPU 采样路径，属格式过期而非后端差异）。

**结构逐项一致**：num_frames=200、终止记录 `(('timeout',200),)` ×2 agents、
obs (200,96)、act (200,21)、extras 键 `{log_prob, explore_factor, sctx__delta_factor}`、observer 9 字段、final_observation (96,)、episode_options 透传、
explore_factors (T,)。唯一有意差异：`episode_metrics["backend"]="warp-fp32"`
（溯源标记，CPU 侧无此键）。

**关键可证伪检查（log_prob 重放）**：用同权重 `evaluate_actions(obs, act, ctx)`
重算 rollout 记录的 log_prob：

| 来源 | max \|Δlog_prob\| |
|---|---|
| CPU episodes（自身的重放误差） | 1.24e-05 |
| Device episodes | 1.20e-05（初测 7.6e-6） |

两侧误差同量级（fp32 噪声）→ 采样分布 = 训练分布，无 batching/clipping/ctx 串扰。

**管线消费**：`StandupFloor04.build_trajectories` 产出 8 trajs（4 ep × 2 agents），
ctx schema `{delta_factor, explore_factor}` 全字段齐全，
`PPOBuffer` 构建无异常且 buffer.log_probs 与记录值差 <1.3e-5；
`r_potential` 通道提取正常（is_terminated=False → `V(final_obs)` bootstrap 路径）。

**补充**：observer 标量叶子按 CPU 同构保留为 Python float（`_try_stack` → list，
与 CPU Episode 完全一致）；`stochastic=False` 波丢弃 extras（与 CPU eval 语义一致）。

测试：`tests/test_device_rollouter.py` 5/5（GPU-gated：契约/padding 顺序/log_prob
重放/PPOBuffer 兼容/拒绝路径）。

## 3. W3：接入方式

- `train.py --collector {cpu,device}`（默认 `cpu`，行为不变）+ `--collector-batch-size`。
- `train_ppo` 按开关构造 `DeviceRollouter` 或 `ParallelRollouter`——同一
  `collect(jobs)` 契约，`loop.py` 其余部分零改动；eval 波同样可走设备。
- 溯源：`[collector] device ...` 启动行 + `config.json["collector"]` +
  每 update `RAW_STATS.timing.collector`（reset/policy/step/assemble/n_waves）。
- 不支持即拒绝：非 file: policy bp、混合 env/policy bp、混合 stochastic、
  callable ef、reference ensemble、delta_factor≠0、未知 episode_options 键、
  注册表非 NATIVE 插件/observer（PENDING/UNSUPPORTED 按 UNSUPPORTED 拒绝）。

## 4. W4：完整 512-ep update —— 技术闭环 ✅，性能结论 ⚠️

命令：
```
train.py --experiment standup_floor04 --algo ppo --collector device \
  --collector-batch-size 256 --param max_updates=1 --param eval_interval=9999
```
（run `m5_w4_device_u1_b256`；B=512/cap48 曾在 warp collision scratch
处 OOM。**事后定位根因是两层叠加**：① `WarpHumanoid21Simulator`
继承 `MjxHumanoid21Simulator`，父类构造初始化 jax 后端时 XLA 默认
预分配 ~75% 显存（~18GB）——已修（ctor 加 `_init_jax=False`）；
② 更深的 bug：`put_data` 的 `nconmax` 是 **per-world** 语义
（`naconmax = nconmax × nworld`），代码误传 `B×48` 使总容量 = B²×48
——显存随 B **二次方**增长（B=1536 → ~20GB），且单 world 接触上限
放大后可在训练中触碰 `njmax` 溢出断言（u55 "nefc overflow" 实崩）。
修正为 `nconmax=48`（per-world）+ 显式 `njmax=512` 后实测：
B=512→3.0GB/58.8K、B=2048→3.3GB/237K、B=8192→4.4GB/**504K**
env-substeps/s（已超 CPU 池 ~472K）——"显存限制 B"的结论彻底
不成立，M2 probe 能跑 B=8192 正因为它用的就是 per-world 默认 48。
`nconmax_per_world=16` 的截断近似结论不变：采不采用仍需先测峰值
ncon，但不再出于显存理由。）

**结果**：512 episodes → 1024 trajs → 204,800 frames → PPO update 完整跑通
（KL early-stop、critic EV=0.489、uncertainty floor 链路均正常）。

**性能分解**（单次 update）：

| 阶段 | 设备 (B=256×2 波) | CPU 基线 (96 workers) |
|---|---:|---:|
| rollout 总计 | **220.5s** | **~6.0s** |
| ├ reset（含并行摔倒 init） | 30.2s | （含在 rollout 内，CPU 串行摔倒更贵） |
| ├ 物理+插件 step | 174.7s | — |
| ├ policy 推理 | 0.41s | — |
| ├ assemble（末波 D2H） | 3.4s | — |
| buffer+ppo | 1.2s | 同（共享路径） |

**结论：在当前任务规模下设备路径比 CPU 池慢 ~37×**。根因是 warp 物理步为
launch-latency 受限：25 子步块耗时 B=64→0.51s、B=256→0.26s、B=512→0.38s（局部
波动），单 warp step ~10-15ms 几乎不随 B 变化——吞吐随 B 线性（B=256 约 25K
env-substeps/s，B=512 约 34K）。M2 报告的 308K 是 **B=8192** 下的数字，而
512-ep update 物理上只有 512 个 env 可并行。~~且 B>256 已撞显存~~ ——显存
瓶颈是 jax 预分配 + `nconmax` 传值语义错误（均为本代码层 bug，已修）。
修复后 cap48 下 B=8192 实测 **504K env-substeps/s > CPU 池 472K**——
加速潜力的原始论证恢复成立。剩余问题是任务形状：512-ep/update 只
喂得饱 B=512 波次（~59K/s），要发挥 B≥2048 的吞吐需要更大 update
批量或多卡分波。要追平/反超 96-worker CPU，候选路径：
(a) episodes_per_update 提到 2K-8K 级（注意 ROADMAP 暂停条件：若加速
只在改批量后成立，不算等价迁移成功，但可另立优化实验）；
(b) 多卡分波；(c) warp solver 层面降延迟（CUDA graph、迭代数、
collision 配置）；(d) 部分 reset/摔倒初始化开销优化（reset 占 ~16%）。

**诚实含义**：R1 放行的"collector 性能分解"完成，同时暴露了本任务在单 4090 +
512-ep/update 的形状下 warp 无加速收益——ROADMAP 的"不宣称训练加速"原则
在此被数据强制执行。

## 5. W5：debug/溯源/拒绝路径

- **Episode.save/load roundtrip**：device episode 序列化/反序列化无损（obs、
  term records、sctx 字段全部还原）；`episode_metrics` 不序列化是 format v3
  的既有行为——CPU episode 同样丢失，非设备路径回归。
- **溯源链**：`config.json.collector` + `episode_metrics.backend="warp-fp32"` +
  `timing.collector` 分解 + code snapshot ——设备轨迹不会被误当 CPU 轨迹。
- **拒绝即失败**：注册表未注册/PENDING/UNSUPPORTED、spec 越界、options 白名单
  外键、混合 bp/stochastic——全部 collect 前显式 ValueError（有测试）。
- **历史坑记录**：旧 policy_exports（pre-ctx `sample(obs, *, explore_factor)` 签名）
  不能被 SamplingPolicy 消费——CPU 侧同样失败，属导出版本过期，评估旧 run 须先
  用当前代码重导出权重（W1 对照即如此处理）。

## 6. R1 放行核对

- [x] 512 场 / 1024 traj / 204,800 transitions，无漏帧/重复/跨回合拼接
  （ep_len 全 200，terms={timeout:1024}，顺序=job 序）
- [x] timeout 的 final_observation 保留 bootstrap 语义（is_terminated=False →
  `V(last_obs)` 被 GAE 消费）
- [x] collector 性能分解（§4）——结论是 GPU 收益**未能**兑现于当前规模，
  如实记录而非调参掩盖
- [x] CPU 原路径回归：默认 `--collector cpu`，ParallelRollouter 路径零改动
- [x] 未识别/未声明组件启动即拒绝，无静默回退
