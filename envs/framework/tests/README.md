# Framework 测试

测试 CombatBench 框架的核心安全机制。

## 测试重点

这些测试专注于框架的**关键风险点**，而不是简单的覆盖率：

| 测试文件 | 重点 | 风险等级 |
|---------|------|---------|
| `test_permission_control.py` | 权限授予/撤销机制 | P0 |
| `test_plugin_dispatch.py` | 插件调度顺序与异常隔离 | P0 |
| `test_lifecycle.py` | 钩子调用顺序与终止传播 | P0 |
| `test_observer_system.py` | Observer 去重与调度 | P1 |
| `test_edge_cases.py` | 边界情况与特殊条件 | P2 |

## 运行测试

```bash
# 运行所有框架测试
pytest envs/framework/tests/

# 运行特定文件
pytest envs/framework/tests/test_permission_control.py

# 显示详细输出
pytest envs/framework/tests/ -v

# 运行并显示覆盖率
pytest envs/framework/tests/ --cov=envs.framework --cov-report=html

# 只运行 P0 测试
pytest envs/framework/tests/ -m "not slow"
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
