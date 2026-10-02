# E3 计划：单卡设备采样引擎与批量 Episode 导出

**状态**：提案（2026-10-02）
**上游**：[discuss.md](./discuss.md) D9/D10/D11 | [E2_PLAN.md](./E2_PLAN.md)（已完成）
**ROADMAP 对应**：E3 —— "完成单卡设备采样引擎，保留 Episode 边界"

## 背景与定位

E2 落地后生命周期契约完整，但 `device_rollouter.py` 仍是**纵向原型**：

| 契约要求（discuss.md / ROADMAP） | 当前实现 | 差距 |
|---|---|---|
| collector 不构造具体后端 | `_build_runtime` 硬编码 `WarpHumanoid21Simulator` + `_SUPPORTED_SIM_CLS` 白名单 | 加新任务/后端必须改 collector |
| 输出 schema 来自绑定 | `obs_buf (T,B,96)` / `act_buf (T,B,21)` / `AGENT_IDS` 写死 | 换任务维度就错 |
| 策略适配器抽象 | `_load_policy` 直构 `TruncatedNormalPolicy`；`_policy_cache` 无界、以 bp JSON 字符串为键 | 无能力声明/版本 hash/缓存上界 |
| D10.1 RecordStore（runtime StepResult 的消费者） | `_WaveRecorder` 是 priority=-10000 的普通插件 | 记录归属/语义靠约定而非结构 |
| D10.2 批量导出 | 波末一次 `.cpu()` ✓，但 `_assemble_episode` 仍逐帧造 dict 走 `from_buffer_frames` | T×B 次 Python 开销，且这恰是导出契约的"参考实现"路径 |
| 无逐步 host 数据依赖 | `rt.step` 内含 `bool(.any())`×2 + `any_running()` 早退 + recorder 帧拷贝 | 无同步点计量，无法核查 |
| 采样块切分不改变 episode 语义 | 单波 T≥max_steps 全跑完 | 分块续采未建模（本阶段显式不做） |
| Job 分同构组 | 全 collect 要求统一 env_bp+policy+stochastic | 混合 job 被拒（E4 需分组分片，先在本阶段立分组） |

## 关键设计判断

**J1 — collector/backend/策略三层注入，而非 collector 懂一切。**
新增 `binding_registry`：`env_bp.simulator.cls` → `DeviceBinding`（生产 backend + obs_builder + task_tables + **io_schema**）。collector 只认识 `BatchRuntime`/`PhysicsBackend`/schema。`WarpHumanoid21Simulator` facade 保留为兼容 shim（现调用点不碎），新路径经 registry 构造 `WarpBackend`。

**J2 — RecordStore 是结构化对象，不是插件。**
波内记录（obs/act/extras/observer 树/final_obs/frame_valid/per-row 长度）归 `RecordStore` 所有——由 binding 声明的 schema 预分配；observer 叶子 schema 声明各叶 (dtype, shape-suffix)，不再"凡 (B,) 就收、其余丢"。`_WaveRecorder` 退化为把 `on_post_action_step`/`on_post_episode` 输出**搬运**进 RecordStore 的薄适配器（仍借 hook 序，但缓冲所有权归 store；partial reset 不动它——本就成立）。

**J3 — 波内语义与实现分离：WaveRunner 后端无关。**
把"rt.reset → 逐步 policy.act + rt.step + store 写 → 全 ENDED 早退 → 波末健康检查"的循环提成 `WaveRunner`，入参只有 runtime + PolicyExecutor + RecordStore + job 切片。**FakeBackend 可跑全流程**——混合长度波（不同 env 不同步终止）的导出契约因此可在无 GPU 下测。

**J4 — 导出向量化，`from_buffer_frames` 降级为参考实现。**
`Episode` 是 frozen dataclass、字段即 `Mapping[str, np.ndarray]`——可直接按行切片构造（`obs[:t_use,row]`），不经逐帧 dict。新增 `EpisodeExporter` 做这件事；`from_buffer_frames` 保留并在等价测试中对照同一批数据逐字段比对（它成为 golden，不是热路径）。

**J5 — 同步点显式计量，不做激进异步化。**
逐步 host sync 现状：屏障内 `bool(rr.any())` + `bool(newly.any())` + `pend.any()`（每 step ≥2），`any_running()` 早退（每 step 1）。本阶段：给 `BatchRuntime`/rollouter 加 `sync_stats` 计数器（分类记账）；`any_running` 早退改为每 8 步检查一次（纯性能检查——ended 行封存无数据影响）。**不**在 E3 把屏障改成延迟批量判定（会改变 `env_term_step`/`final_obs` 捕获语义）；异步 D2H/CUDA Graph 归 E7。

## 工作包拆分

```
W1 绑定注册表（sim→backend/obs/schema 解耦）
→ W2 PolicyExecutor + 有界版本化缓存
→ W3 RecordStore + WaveRunner（FakeBackend 可测）
→ W4 EpisodeExporter 向量化 + 参考实现等价测试
→ W5 同步点计量 + 波契约测试（混长/早退/padding）
→ W6 回归 + 冒烟 + 文档收尾
```

### E3-W1：绑定注册表与 IO schema

新增 `envs/batchframework/binding_registry.py`：

```python
class DeviceBinding(Protocol):
    def make_backend(self, batch_size, device) -> PhysicsBackend: ...
    def make_obs_builder(self, backend) -> ObsBuilder: ...
    def tables(self, backend) -> TaskTables: ...
    def io_schema(self) -> IoSchema: ...
        # agent_ids、每 agent obs_dim/action_dim、
        # observer 输出 schema {name: {leaf: (dtype, shape)}}
    def episode_metrics_schema(self) -> Mapping[str, ...]: ...
```

`Humanoid21Binding`（E1 已有，任务语义唯一实现）扩展出 `io_schema()`——从 `device_tables` 推 agent_ids/obs_dim(96)/action_dim(21)，observer schema 由 `DeviceStandup4StageRewarder` 等声明 `output_schema`（新增声明属性，装配校验白名单化"只收 (B,) 叶"为"收 schema 声明的叶"）。

`_SUPPORTED_SIM_CLS` → `binding_registry.resolve(env_bp.simulator.cls)`；未注册 → 显式拒绝（同 capability_registry 风格）。collector 不再 `import WarpHumanoid21Simulator`。

### E3-W2：PolicyExecutor 与有界缓存

新增 `envs/batchframework/policy_executor.py`：

```python
class PolicyCapabilities(NamedTuple):
    stochastic: bool            # sample_action 可用
    deterministic: bool
    ctx_fields: frozenset       # 支持的 SamplingContext 字段名
    stateful: bool              # 是否含需 reset_rows 的隐状态

class PolicyExecutor(Protocol):
    def capabilities(self) -> PolicyCapabilities: ...
    def act(self, obs_a, obs_b, ctx_a, ctx_b,
            policy_eval_mask) -> PolicyOutput: ...
    # PolicyOutput: action_a/b, log_prob_a/b (stochastic 才填),
    #               版本 hash
    def reset_rows(self, env_ids) -> None: ...
    def close(self) -> None: ...
```

`TruncatedNormalExecutor` 适配 `TruncatedNormalPolicy`：
- **能力检查**：`sampling spec` 要求的 ctx 字段 ⊆ `capabilities.ctx_fields`，否则显式拒绝（取代 `_check_spec` 白名单当永久能力边界——拒绝仍发生，但来源是声明而非硬编码表）；
- **版本**：`state_dict` 内容 hash（张量字节级）作缓存键与 provenance 项，不以 bp JSON/路径为版本保证；
- **缓存**：有界 LRU（默认 8），evict 调 `close()`；
- `policy_eval_mask` 入参：ended/hold 行不调用采样（E2 mask 语义贯通到策略层）。

### E3-W3：RecordStore + WaveRunner

`envs/batchframework/record_store.py`：

```python
class RecordStore:
    """按 io_schema 预分配 (T,B,·) 缓冲；拥有 frame_valid/per-row
    长度/final_obs/term_records 导出视图。生命周期跨 partial reset。"""
```

- `write_step(t, state)`：obs/action/log_prob/observer 树 → 设备缓冲；
- `seal_row(env_ids, step)`：记 env_term_step + final_obs 捕获；
- `frame_valid (T,B) bool`：该行该步是否产生 CPU 兼容记录帧（ENDED 行此后 False——`t_use` 截断已有等价语义，mask 使其显式化）；
- observer 输出写：按 `io_schema` 声明的叶集合直接 `buf[t] = v`（缺叶报错而非跳过——schema 是契约）；
- 计量：`n_bytes` 属性报设备缓冲总量（供 D10.3 预算）。

`envs/batchframework/wave_runner.py`：

```python
class WaveRunner:
    """后端无关波循环。FakeBackend 可注入（混合长度契约测试）。"""
    def run(self, rt, executor, store, jobs_slice, efs, T) -> None
```

`_WaveRecorder` 改为持有 `RecordStore` 的薄壳（on_post_action_step → store.write_step；on_post_episode → store.seal_row + 末帧覆写）。`finalize_wave` 移入 store/exporter。

### E3-W4：EpisodeExporter 向量化

`envs/batchframework/episode_exporter.py`：

```python
def export_episodes(store, wave_jobs, env_hash, efs, stochastic,
                    metrics_np, term_records) -> List[Episode]
```

- 一次 `torch.cat` 后 `.cpu()`（已存在，保留）→ numpy 按行切片；
- **直接构造 `Episode`**（frozen dataclass 字段直填），不经 `from_buffer_frames`；
- `action_extras`/`explore_factors`/`observer_outputs` 按 (T_use,·) 切片；
- 参考等价测试：同一 np_bufs 分别走 `from_buffer_frames`（逐帧 dict）与新 exporter，逐字段 `np.testing.assert_array_equal` + 记录/选项/metrics 字典相等——golden 对照而非口头等价。

### E3-W5：同步点计量 + 波契约测试

- `BatchRuntime.sync_stats: Counter`——每个 `bool(tensor)`/`item()`/`cpu()` 点自增并记 site；rollouter 同名。训练冒烟后打印计数（当前应 ≤ ~3/step + 波界常量次）；
- `any_running` 早退降为每 8 步一次（说明：ended 行封存后早退只是省算力，不影响数据）；
- FakeBackend 混合长度波测试：3 env 分别于 step 3/5/9 终止 → 导出 3 个不同 `num_frames` 的 Episode，records/frames/final_obs 各归其位；
- padding 行、全 ENDED 早退波、FAILED 行 collect 显式失败——契约测试补齐。

### E3-W6：回归 + 冒烟 + 文档

- 全套设备回归（含 `test_collect_episode_contract`/`test_logprob_replay_parity`/`test_ppo_pipeline_compat`——真 warp 端到端）；
- device collector 训练冒烟 2 updates；
- 报告：RecordStore 字节数 + host 导出峰值 + 同步计数；
- `ROADMAP.md` E3 状态、`discuss.md` H8/H9/H10 行更新。

## 风险与降级路径

| 风险 | 概率 | 降级 |
|---|---|---|
| io_schema 对"非 (B,) 叶"的扩展改变 observer 输出导出形态 | 中 | 首版仍只支持 (B,) 标量叶（standup 全集即此形态），多维叶在 schema 里声明但在导出端显式拒绝——不静默丢 |
| PolicyExecutor 抽离改变 log_prob 数值路径 | 低 | `test_logprob_replay_parity` 守住（rollout 记录 vs evaluate_actions 重算） |
| RecordStore 替代 _WaveRecorder 引帧序错位 | 中 | W3 的 FakeBackend 混长测试 + W4 参考实现逐字段等价测试守住 |
| FakeBackend WaveRunner 与真 warp 路径分叉 | 低 | WaveRunner 只依赖 BatchRuntime/PhysicsBackend 契约——契约测试矩阵已参数化两后端 |
| 版本 hash 对每次 collect 重算 state_dict 哈希的成本 | 低 | 缓存键命中后不重算；首载一次 sha256（256 维小网络 <10ms） |

## 放行条件

1. collector 源码无 `WarpHumanoid21Simulator`/`TruncatedNormalPolicy`/`96`/`21` 字面量（依赖方向测试扩展检查）；
2. `tests/test_device_rollouter.py` 现有 5 项全绿（Episode 契约/log_prob 回放/PPOBuffer 兼容/padding/拒绝路径）；
3. 新增：FakeBackend 混合长度波导出 + 参考实现逐字段等价 + 同步计数上限断言；
4. RecordStore 字节数与 host 峰值在 collect 结果 metrics 中可报告；
5. device collector 训练冒烟正常；
6. `Job` 混合 env_bp/policy/stochastic 按同构键分组执行（每波内仍统一），跨组 job 顺序与输入一致返回。

**明确不在本阶段**：多卡协调（E4）、CUDA Graph/异步 D2H（E7）、reference/delta/callable-ef 采样（能力声明拒绝，非本阶段实现）、分块续采（T<max_steps 跨波续 episode——显式推迟）、HOST plane lazy 物化。
