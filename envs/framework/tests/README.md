# Framework 测试

> 类型：指南

测试 CombatBench 框架的核心安全机制与运行契约。

## 测试文件一览

> 状态列标注当前实测结果（详见根目录 `AUDIT.md`）：

| 测试文件 | 重点 | 状态 |
|---------|------|------|
| `test_permission_control.py` | 权限授予/撤销机制（mutator 只在可写钩子暴露） | 通过 |
| `test_plugin_dispatch.py` | 插件调度顺序与异常隔离 | 通过 |
| `test_lifecycle.py` | 钩子调用顺序与终止传播 | 通过 |
| `test_observer_system.py` | Observer 去重与调度 | 通过 |
| `test_observer_dispatcher_ordering.py` | dispatcher 顺序 + 只读 ctx 强制 | 通过 |
| `test_edge_cases.py` | 边界情况与特殊条件 | 通过 |
| `test_episode_runner_behaviors.py` | 薄 runner 契约：返回 None、hold/policy 终止策略、want_extras 透传、duck-type 校验 | 通过 |
| `test_reset_chain.py` | RESET.md 不变式 I1–I6 | 通过 |
| `test_seed.py` | SEED.md 契约（派生树、None 解析） | 通过 |
| `test_blueprint.py` | EnvBlueprint/PolicyBlueprint 序列化 | 通过 |
| `test_policy.py` | Policy ABC 契约（`(action, extra)` 签名） | 通过 |
| `test_recorder_lifecycle.py` | Recorder 钩子顺序 + 落盘 schema | 通过 |
| `test_replay_simulator.py` | ReplaySimulator 回放往返 | 通过 |
| `test_sandbox.py` | accessor/mutator 沙箱白名单 | 通过 |
| `test_strict_mode.py` | strict 模式异常语义 | 通过 |
| `test_video_recorder.py` | VideoRecorderPlugin options 覆盖 + find_plugins | 通过 |
| `test_event_journal.py` | EventJournal 契约：episode 内 append-only、epoch 游标、reset 清账 | 通过 |
| `test_audit_*.py`（6 个） | 审计探针：`mutator_leak`/`terminal_frame`/`reset_recorder_gap`/`video_path_leak` 锁死已确认缺陷；`simulator_reach`/`fail_loud` 锁死已修复的回归（P-FW-4/P-FW-5） | 探针/回归 |

> 已删除：`test_episode_runner.py`、`test_parallel_runner.py`——两者测试的是
> runner 重构（`73fe8da3`）前的旧 API（`RolloutConfig`/`ObserverBinding`/
> `run_n_episodes`/`ParallelRunner`），新契约由 `test_episode_runner_behaviors.py`
> 覆盖；并行能力现属 `baseline/framework/rollout/`。

## 运行测试

```bash
# 运行所有框架测试
PYTHONPATH=. pytest envs/framework/tests/

# 运行特定文件
PYTHONPATH=. pytest envs/framework/tests/test_permission_control.py

# 显示详细输出
PYTHONPATH=. pytest envs/framework/tests/ -v
```

## 核心测试场景

### 1. 权限控制测试

验证只读生命周期真的无法写入：

```python
def test_mutator_revoked_in_readonly_lifecycle():
    # 插件尝试在 on_post_action_step 写入
    # 预期：mutator 是 None，写入失败
```

### 2. 异常隔离测试

验证单个插件异常不影响其他插件：

```python
def test_exception_in_one_plugin_doesnt_stop_others():
    # priority=100 的插件抛异常
    # 预期：priority=50 的插件仍然执行
```

### 3. 终止传播测试

验证终止后的钩子执行逻辑：

```python
def test_termination_in_pre_phy_step_stops_physical_step():
    # 在 on_pre_phy_step 请求终止
    # 预期：physical_step() 不执行，on_post_phy_step 不调用
```

### 4. Observer 去重测试

验证同一实例只被调用一次：

```python
def test_same_observer_instance_deduplicated():
    # 同一 observer 挂载到两个名称
    # 预期：on_post_action_step 只调用一次
```

## 设计理念

这些测试遵循以下原则：

1. **测试风险，不是覆盖代码** - 每个测试对应一个已识别的风险点
2. **使用 Mock** - 快速、可靠、可重复
3. **清晰的行为验证** - 每个测试有明确的预期行为
4. **独立性** - 测试之间无依赖，可并行运行

> `test_audit_*.py` 是例外：它们是审计期间写的**缺陷锁存探针**——
> 故意断言当前（有缺陷的）行为，修好后应翻转断言。
