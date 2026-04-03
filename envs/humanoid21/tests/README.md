# Humanoid21 Simulator 测试套件

本目录包含 Humanoid21 仿真器的完整测试套件，用于验证数据接口的正确性和符合 DATASPEC.md 规范。

## 测试文件

### test_data_interfaces.py
**数据接口完整测试** - 验证所有数据接口的数据格式和数据内容正确性

#### 测试覆盖范围

1. **静态属性测试 (get_static_data)**
   - 验证 dof_names (21个自由度名称)
   - 验证 body_names (15个body名称)
   - 验证 joint_limits (21×2 关节限位矩阵)

2. **核心状态测试 (get_core_state)**
   - 验证 root_pos (3维) - Torso 绝对世界坐标
   - 验证 root_rot (4维) - 四元数姿态
   - 验证 root_vel_local (3维) - 局部线速度
   - 验证 root_angular_vel_local (3维) - 局部角速度
   - 验证 joint_pos_norm (21维) - 归一化关节位置
   - 验证 joint_vel_norm (21维) - 归一化关节速度

3. **派生状态测试 (get_derived_state)**
   - 全局对抗信息：torso_distance, combat_contacts
   - 模块二：全局状态 (13维) - height, local_orientation, linear_vel, angular_vel
   - 模块三：触觉力反馈 (2维) - feet_forces
   - 模块四：对手观测 (39维)
     - 对手基础位姿 (9维) - relative_pos, relative_vel, face_vector
     - 对手关键点位置 (15维) - head, hand_right, hand_left, foot_right, foot_left
     - 对手关键点速度 (15维)
   - 完整平铺观测 (96维)

4. **观测空间维度分解验证**
   - 验证模块一本体感知 (42维): 索引 [0:42]
   - 验证模块二全局状态 (13维): 索引 [42:55]
   - 验证模块三触觉力反馈 (2维): 索引 [55:57]
   - 验证模块四对手观测 (39维): 索引 [57:96]

5. **归一化正确性验证**
   - 测试上限归一化 (应为 +1.0)
   - 测试下限归一化 (应为 -1.0)
   - 测试中间值归一化 (应为 0.0)

6. **坐标系转换验证**
   - 验证相对位置计算
   - 验证 face_vector 为单位向量

7. **动态一致性验证**
   - 验证观测数据随时间正确更新

8. **边界情况验证**
   - 验证 reset 后状态正确
   - 验证极端动作值处理

9. **FaceVector 场景验证** ⭐ 新增
   - 场景1: 默认站立姿态 (相对而立) - 验证 face_vector 是否正确反映相对朝向
   - 场景2: 同向站立 - 验证两个机器人朝向相同时的 face_vector
   - 场景3: 旋转90度 - 验证旋转后相对位置和朝向的正确性

10. **关键点位置一致性验证** ⭐ 新增
    - 验证对手关键点相对位置的合理性
    - 验证 head、hand、foot 的相对高度关系
    - 验证 core_state 和 derived_state 的高度一致性

11. **局部速度转换验证** ⭐ 新增
    - 场景1: 机器人静止时的速度
    - 场景2: 朝向 +x 时的速度转换
    - 场景3: 旋转90度后的速度转换

12. **观测数值范围验证** ⭐ 新增
    - 验证各模块观测值在合理范围内
    - 验证无 NaN 或 Inf 值
    - 验证归一化数据在 [-1, 1] 范围内

13. **数据同步一致性验证** ⭐ 新增
    - 验证多次调用数据的一致性
    - 验证 core 和 derived 之间的同步

## 运行测试

### 方法 1: 使用测试脚本
```bash
cd /data1/mono
./things/combatbench/envs/humanoid21/tests/run_tests.sh
```

### 方法 2: 直接运行
```bash
cd /data1/mono
PYTHONPATH=/data1/mono/things/combatbench python3 things/combatbench/envs/humanoid21/tests/test_data_interfaces.py
```

### 方法 3: 使用 pytest
```bash
cd /data1/mono/things/combatbench/envs/humanoid21
pytest tests/ -v
```

## 测试标准

所有测试基于以下文档规范：
- `DATASPEC.md` - 数据规范
- `CONTROLSPEC.md` - 控制规范
- `OBSERVATION_zh.md` - 观测空间设计

## 测试结果

所有测试应通过，输出示例：
```
======================================================================
✓ 所有数据接口测试通过！
======================================================================

测试总结:
  ✓ 静态属性 (get_static_data)
  ✓ 核心状态 (get_core_state)
  ✓ 派生状态 (get_derived_state)
  ✓ 完整观测空间 (96维)
  ✓ 归一化正确性
  ✓ 坐标系转换
  ✓ 动态一致性
  ✓ 边界情况
```

## 添加新测试

在添加新功能时，请在此目录添加相应的测试文件，并确保：
1. 测试文件以 `test_` 开头
2. 测试函数以 `test_` 开头
3. 包含清晰的测试说明
4. 验证数据格式和内容
