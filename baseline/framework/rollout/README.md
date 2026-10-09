# rollout — 采样/评估的回合数据管线

> 类型：指南

`baseline/framework/rollout/` 是**回合采集与数据模型层**：PPO 训练、
`bench_rollout.py` 基准工具、评估脚本共享同一套 `Job → collect() → Episode`
契约。设备端批量对等实现见 `envs/batchframework/`（同一 `collect(jobs) ->
List[Episode]` 契约）。

## 数据模型

```
Job ──► ParallelRollouter.collect(jobs) ──► List[Episode]
 │                                            │
 ├─ policy_a_bp / policy_b_bp (PolicyBlueprint)  ├─ per-step frames
 ├─ env_bp          (EnvBlueprint)               ├─ observer outputs（已堆叠）
 ├─ seed                                          ├─ action extras / sampling_ctx
 ├─ episode_options （仅环境配置，JSON 可序列化）     └─ termination proposals
 └─ sampling_a/b    (SamplingSpec)
```

- **`Job`**（`job.py`）——一回合的完整描述：双方策略蓝图、环境蓝图、
  seed、各自采样规格。**`collect()` 收 `Job` 对象列表，不是 tuple**。
- **`SamplingSpec`/`EfSpec`/`ReferenceSpec`**（`job.py`）——每智能体采样配置：
  explore_factor、参考策略 ensemble、delta 探索。`EfSpec` 的唯一权威定义
  就在 `job.py`（P-RO-3 已收敛）。
- **`Episode`**（`episode.py`）——回合记录：逐步帧、observer 输出堆叠、
  终止提案记录。`blueprint_hash()` 用于跨回合环境一致性校验。
- **`EpisodeCollection`**（`episode_collection.py`）——回合集合的落盘格式
  （`COLLECTION_FORMAT_VERSION` 版本锚）。

## 执行入口

### `ParallelRollouter`

```python
rollouter = ParallelRollouter(num_workers=4)        # spawn 进程池
episodes = rollouter.collect(jobs)                  # List[Episode]
```

- `num_workers <= 1`：本进程内跑（调试友好）。
- `num_workers > 1`（spawn）：**调用代码必须在真实 `.py` 文件里且
  `if __name__ == "__main__":` 守卫**——worker 会重新 import `__main__`，
  REPL/notebook 会 `BrokenProcessPool`。
- `rollout_inference="gpu"`：首用时拉起 UDS 推理服务器，策略包成
  `RemoteSamplingPolicy` 阻塞等 GPU batched forward；环境步进仍在 CPU。
  （`inference_server.py` / `remote_policy.py`）

### `EpisodeRecorder`（`episode_recorder.py`）

`PostActionRecorder` 实现：挂载到 `EnvRuntime.recorders`，每步收帧，
回合末生成 `Episode`。abandoned/closed 回合也会落盘并带终止提案记录
（P3-1/P-RO-2 契约）。

### `SamplingPolicy` / `ExploratoryPolicy`（`exploratory_policy.py`）

把裸策略包上采样语义（explore_factor、reference-delta）再交给
`EpisodeRunner`。`Job.stochastic=False` 时跳过包装，直接 `act()`
确定性评估（policy 自带采样逻辑的场景）。

## 消费方

| 消费方 | 用法 |
|---|---|
| `baseline/framework/ppo/` | 训练 collect → Episode → Trajectory |
| `bench_rollout.py` | 吞吐基准（serial + parallel + gpu inference） |
| `ppo/dumpkit/dump_rollout.py` | dump 采集 |
| 评估脚本 | `Episode`/`EpisodeCollection` 读取 |

## 测试

`test_episode_recorder.py`（abandoned/closed 契约）、`test_exploratory_policy.py`、
`test_remote_inference.py`。
