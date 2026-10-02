# E4 计划：单次训练的 1–8 卡 rollout

**状态**：提案（2026-10-02）
**上游**：[discuss.md](./discuss.md) D12/D13（coordinator/worker 协议、错误模型）| [E3_PLAN.md](./E3_PLAN.md)（已完成：同构分组、RecordStore、exporter、sync_stats）
**ROADMAP 对应**：E4 —— "单次训练的 1–8 卡 rollout"

## 背景与定位

E3 之后单卡采集引擎已完整：`DeviceRollouter.collect(jobs)` 内部按同构键分组、分波、向量化导出。E4 的目标是把**同一 collect 事务**分布到 1–8 个 GPU worker 进程，对外契约不变：

```python
class BatchRollouter:
    def collect(self, jobs) -> list[Episode]: ...   # 同序、同 schema
    def close(self) -> None: ...
```

learner 侧无感知——`loop.py` 仍消费 `List[Episode]`。

### 现状差距（对照 D12/D13）

| 契约要求 | 现状 | 差距 |
|---|---|---|
| coordinator 分配 JobRef、检查每个 job 恰好一次 | 无 coordinator 概念 | 全新组件 |
| worker = 独立进程，设备确定后才初始化 CUDA | DeviceRollouter 进程内直用 | 需 spawn 子进程协议 |
| 冻结策略快照集合按版本分发 | executor 已有 state_dict sha256 version | 经 manifest 下发 + worker 回报校验 |
| 静态确定性分片，结果按 job 身份重排 | E3 已有同构分组 | 需 shard 分配器 + 重排校验 |
| 任一 worker 失败 → collect 整体失败 | 单进程 FAILED 行检查已有 | 需 worker 故障检测/传播 |
| 有界队列/背压 | 无 | mp.Queue + maxsize |
| worker 显式 close 有序释放 | close() 已有 | worker join 序列 |

### 关键设计判断

**J1 — worker = `spawn` 子进程，不复用 fork。**
CUDA context 不可继承（D12.2.3，H12）。worker 进程入口：`device_id` 先写死 `torch.cuda.set_device` 再 init warp/torch。每个 worker 内部实例化**现有 `DeviceRollouter`**——单卡采样实现零分叉（discuss.md "单卡与多卡复用相同 worker 内采样实现"）。

**J2 — 传输首版用 pickle + `mp.Queue`，Episode 不做新序列化格式。**
`Episode`/`Job`/`EnvBlueprint` 都是 frozen dataclass、np.ndarray 载荷、可 pickle；`Episode.save`（npz）是给磁盘的，不进热路径。D12.3 明说"首版可先用简单正确的 CPU 序列化，再基于测量优化"。worker 上行消息带 `collect_id + job_refs + version + sync_stats/bytes 报告`；不发送活 GPU tensor。

**J3 — 分片确定性：同构组内按 job 索引连续块切分。**
E3 的 collect 已分组；E4 把**组**映射到 worker：组内 jobs 按输入序切成 `len(devices)` 个连续块（chunk），第 i 块给 worker i。行号/padding 纯属 worker 内部；JobRef = 全局 job 下标（int），结果经 JobRef 重排回输入序。**RNG 身份不变**：seed 在 Job 上携带，分片不改变每 job 的 `seed_offsets`——D9.1 天然满足（显式写入契约测试）。

**J4 — 失败模型：all-or-nothing collect。**
worker 上行 `{"status":"error", ...}` 或进程退出/超时 → collect 抛 `WorkerLost`/上游错误，不返回部分结果。coordinator 不自动重试（D13.1：上层决定）。`close()` 顺序：借用缓冲释放 → worker join → 超时未退则 `terminate` 并报告。

**J5 — 单设备时走进程内路径，不强制 spawn。**
`--collector device` 无 `--collector-devices`（或单卡）= 现有 in-process `DeviceRollouter`；多卡才进 coordinator。降低默认路径的进程开销与调试复杂度；多卡路径始终经过 worker 进程（即便只配 1 卡也可显式要求 worker 模式做故障隔离测试）。

## 工作包拆分

```
W0 spawn/CUDA-init/传输探针（H12 + pickle 带宽实测）
→ W1 分片协议（JobRef/同构组→worker 映射/结果重排校验）
→ W2 worker 进程（handshake/collect/close 命令通道）
→ W3 MultiDeviceRollouter facade（事务 + 失败传播 + 有序 close）
→ W4 train.py 接入（--collector-devices）
→ W5 契约测试（确定性/乱序重排/版本校验/故障注入/1-2 卡实跑）
→ W6 回归 + 冒烟 + 文档收尾
```

### E4-W0：启动与传输探针

探针脚本（不进主路径）：

- `spawn` 子进程内 `torch.cuda.set_device(k)` + warp init + 小 collect；
- 验证 `CUDA_VISIBLE_DEVICES` 不统一遮罩时，每进程绑定各自物理卡；
- 实测 Episode pickle 尺寸/序列化耗时（standup T=200×96×2+21×2+extras+observer ≈ 每 ep ~1.5MB，8 卡 512 ep 波 ~0.8GB——确认 mp.Queue pickle 吞吐 vs 每波时长）；
- worker 异常退出时 coordinator 侧检测延迟。

产出：`probe_worker_spawn.py` + 结论记入本文档附录。**H12 判定点**：若 spawn+per-device init 不可靠，改 csprng/容器级隔离另议。

### E4-W1：分片协议模块 `coordinator.py`

```python
@dataclass(frozen=True)
class JobRef: index: int            # 全局输入下标（唯一身份）

@dataclass(frozen=True)
class ShardPlan:
    collect_id: int
    group_key: str                  # 同构键 hash
    job_refs: Tuple[int, ...]       # 本 shard 的 JobRef
    jobs: Tuple[Job, ...]           # 序列化载荷

def plan_shards(jobs, n_workers) -> List[List[ShardPlan]]
def merge_results(shards_out, n_jobs) -> List[Episode]
    # 校验：每个 JobRef 恰好一次、collect_id 一致、policy 版本一致、
    # Episode schema 合法（num_frames>0、final_observation 存在）——
    # 违例即整个 collect 失败，不返回部分结果
```

- 分组逻辑**复用** E3 collect 的同构键函数（提为模块级 `homogeneous_key(job)`，DeviceRollouter 与 coordinator 共用单一实现）；
- shard 分配纯函数、可单测（无 GPU）。

### E4-W2：worker 进程 `worker.py`

```python
def worker_main(device_id, conn/queues, env_bp_dicts...) -> None
```

- 启动握手：上报 `{device, backend_descriptor, warp/torch 版本, capability 集合}`；coordinator 校验与 plan 兼容否则拒绝该 worker；
- 命令循环：`SHARD(collect_id, jobs)` → 内部 `DeviceRollouter.collect`（内部仍会再分波——粒度不变）→ 上行 `{collect_id, results:[(job_ref, Episode)], report}`；
- `CLOSE` → `rollouter.close()` → 进程退出；
- worker 内策略缓存沿用 E3 LRU——manifest 携带 executor.version，上行 report 含实际版本，coordinator 比对一致才算成功。

### E4-W3：`MultiDeviceRollouter` facade

```python
class MultiDeviceRollouter:
    def __init__(self, devices: Sequence[str], batch_size_per_worker: int):
        # spawn worker × len(devices)，握手自检
    def collect(self, jobs) -> List[Episode]: ...
    def close(self) -> None: ...
```

- collect 事务：plan → 分发 shard → 等齐 → merge+校验 → 按输入序返回；
- 有界：worker 命令队列 `maxsize=1`（背压：前次未完成不派新 shard——同步 on-policy 天然满足）；
- 超时/退出检测：`queue.get(timeout)` + `proc.is_alive()`；失败 → 标 `WorkerLost`，collect raise；
- `close()` 幂等；worker join 超时报资源清理失败（D13.3）。

### E4-W4：训练接入

- `train.py` 新增 `--collector-devices "0,1,..."`（仅 `--collector device` 有效）；
- `loop.py`：len>1 → `MultiDeviceRollouter`；否则现有 in-process；
- config.json 记录 devices 列表；`collector` 计时字段照旧（coordinator 侧增加 `shard_wait`/`merge` 分项）。

### E4-W5：契约测试

- `test_shard_plan.py`（无 GPU）：分组→分片确定性、奇数 job 余量分配、JobRef 全覆盖不重不漏、merge 重排、版本不符/重复/缺失 job_ref 拒绝；
- worker 故障注入：worker 进程 kill → collect 显式失败而非挂起；
- GPU-gated：2 卡 collect = 同 jobs 单卡 collect 的逐字段等价（同 seed → 同 episode 数据，分片不改变随机身份——D9.1 验收）；
- padding 不跨 shard 污染（末 shard 不满 B）；
- 固定全局 jobs 数（如 64）在 1 卡 vs 2 卡上各记一次耗时/内存（不承诺倍数）。

### E4-W6：回归 + 冒烟 + 文档

- tests/ 全量回归；
- `--collector device --collector-devices 0,1` 冒烟（2 updates）；
- `last_collect_report` 扩展：per-worker 缓冲字节数、sync_stats、版本集合；
- ROADMAP E4 状态 + 本计划放行核对；discuss.md H12 更新。

## 风险与降级路径

| 风险 | 概率 | 降级 |
|---|---|---|
| spawn+CUDA 初始化在 8 卡机器上有库冲突/驱动限制（H12） | 中 | W0 先探针；不行则每 worker 独立 CUDA_VISIBLE_DEVICES=1 卡 |
| Episode pickle 带宽成为瓶颈（~1.5MB/ep × 数百 ep/波） | 中 | 首版接受；E7 评估共享内存/npz-stream；不阻塞正确性 |
| worker 内存膨胀（policy 缓存泄漏） | 低 | executor LRU 已有界；W5 加重复 collect 内存稳定检查 |
| Job/EnvBlueprint pickle 含不可序列化成员 | 低 | W0 探针验证；必要时走 to_dict/from_dict 重建 |
| 分片改变 job 顺序影响训练随机性 | 低 | seed 在 Job 上携带与行号无关（已验证语义），W5 等价测试守住 |

## 放行条件

1. 1 卡 worker 模式与 in-process 单卡对同批 jobs 产出逐字段等价 Episode；
2. 2 卡 collect：每个 JobRef 恰好一次、顺序=输入序、各 shard 随机身份正确（同 job 在任卡上结果一致）；
3. worker kill → collect 显式失败（非挂起/部分返回）；close 幂等且 worker 全 join；
4. `--collector device --collector-devices` 冒烟训练正常，learner 侧无改动；
5. 报告 per-worker `record_store_bytes` + `sync_stats` + executor 版本集合；
6. padding 行不产出 Episode；shard 边界无缺漏重复。

**明确不在本阶段**：动态负载均衡/work stealing、多机、异步滞后策略、CUDA Graph、跨 collect 插件状态恢复（D13.2 pending 项）、learner 侧分布式。
