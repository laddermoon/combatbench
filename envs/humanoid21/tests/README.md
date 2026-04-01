# Humanoid21 测试文档

## 测试结构

```
tests/
├── conftest.py              # 共享 fixtures 和 MockMuJoCoSimulator
├── test_plugins.py          # 插件测试
├── test_runtime_units.py    # Observer/Rewarder 测试
└── test_simulator.py        # 模拟器状态管理测试
```

## 测试覆盖

### 1. 插件测试 (test_plugins.py)

#### NonFallConstraintPlugin
- ✅ Pitch 限制裁剪
- ✅ Roll 限制裁剪
- ✅ 裁剪时清零水平速度
- ✅ 记录 clamp_count
- ✅ 在限制内不修改状态

#### CombatScoringPlugin
- ✅ 初始化血量和伤害指标
- ✅ 支持不同的初始血量
- ✅ 检测头部击中并计算伤害
- ✅ 检测躯干击中
- ✅ 血量归零时触发 KO
- ✅ 记录击中事件
- ✅ 忽略非攻击部位接触

#### FrozenRobotPlugin
- ✅ 捕获初始状态
- ✅ 在物理步后重置状态
- ✅ 只影响指定机器人
- ✅ 无初始状态时不执行

### 2. Runtime Units 测试 (test_runtime_units.py)

#### Humanoid21Observer
- ✅ 无效 agent_id 抛出异常
- ✅ 观测维度为 127
- ✅ 观测空间定义正确
- ✅ 动作空间定义正确
- ✅ step 后更新观测
- ✅ 观测值都是有限的
- ✅ 对手观测不同

#### Humanoid21Rewarder
- ✅ 无效 agent_id 抛出异常
- ✅ reset 返回 0
- ✅ post_step 返回 0
- ✅ post_episode 返回 0

#### build_shared_runtime_info
- ✅ 包含血量信息
- ✅ 包含伤害信息
- ✅ 血量默认值为 100
- ✅ 未终止时 winner 为 None
- ✅ KO 时判定获胜者
- ✅ 双方归零时判定平局
- ✅ 超时时判定获胜者
- ✅ 超时血量相同时判定平局
- ✅ 伤害默认值为 0

### 3. 模拟器测试 (test_simulator.py)

#### 状态管理
- ✅ get_core_state 返回所有必需键
- ✅ get_core_state 返回副本
- ✅ set_core_state 更新 qpos/qvel
- ✅ set_core_state 更新机器人位置
- ✅ set_core_state 更新机器人姿态
- ✅ set_core_state 更新机器人速度
- ✅ structured 到 array 的同步
- ✅ get_static_data 返回 robot_info
- ✅ robot_info 包含所有必需键
- ✅ get_derived_state 返回 contacts
- ✅ physical_step 增加时间
- ✅ reset 清零时间
- ✅ reset 清空碰撞
- ✅ reset 支持自定义初始距离
- ✅ get_physical_frequency 返回正确值
- ✅ 初始位置相向而立
- ✅ 初始姿态面朝对方

#### Data 属性
- ✅ data.qpos 可访问
- ✅ data.xpos 可访问
- ✅ data.time 可访问

## 运行测试

```bash
# 运行所有测试
pytest envs/humanoid21/tests/ -v

# 运行特定文件
pytest envs/humanoid21/tests/test_plugins.py -v

# 运行特定测试
pytest envs/humanoid21/tests/test_plugins.py::TestCombatScoringPlugin::test_detects_head_hit_and_deals_damage -v

# 查看覆盖率
pytest envs/humanoid21/tests/ --cov=envs/humanoid21 --cov-report=html
```

## 设计原则

1. **轻量级模拟**：使用 MockMuJoCoSimulator 避免依赖 MuJoCo
2. **隔离测试**：每个测试独立运行，不依赖顺序
3. **快速执行**：所有测试应在秒级完成
4. **清晰断言**：使用描述性断言消息
5. **边界覆盖**：重点测试边界条件和特殊情况

## 测试策略

### 单元测试
- 测试单个方法/函数的行为
- 使用 mock 隔离外部依赖
- 验证输入输出关系

### 集成测试
- 测试组件间的交互
- 使用真实场景数据
- 验证端到端行为

### 风险聚焦
- **P0**：状态一致性、插件逻辑、KO 判定
- **P1**：观测维度、奖励计算
- **P2**：性能优化、边缘情况
