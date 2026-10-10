# Humanoid21 Simulator 测试套件

> 类型：指南

验证 Humanoid21 数据接口是否符合 `DATASPEC.md` 规范，以及控制层行为。

## 测试文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `test_data_interfaces.py` | 数据接口完整测试：static/core/derived state、96 维观测分解、归一化、坐标系转换、face_vector、关键点一致性、数值范围 | 通过 |
| `test_extended_schema.py` | 扩展 schema（contacts_vec、keypoint 表等） | 通过 |
| `test_observation_symmetry.py` | 双方观测的对称性（已改为真 assert） | 通过 |
| `test_balance_analysis.py` | 平衡分析的几何投影指标 | 通过 |
| `test_state_pool_plugins.py` | state-pool 插件（episode 末帧捕获等） | 通过 |
| `test_audit_combat_observer_events.py` | 审计探针：锁死 CombatScoringObserver 读错事件容器（P-H21-2） | 探针 |
| `test_audit_stale_contacts.py` | 审计回归：contacts 缓存跨物理步失效不变式（P-H21-1，已修复） | 回归 |

> `test_acceptance.py` 已移出本目录 → `../benchmark.py`（原 `acceptance_check.py`）
> （手动测量脚本，`measure_*` 函数不随 pytest 收集）。实测 3/5 项
> 在当前 KP/KD 下结构性不达标（跟踪误差/响应延迟/力矩振荡，
> 见 AUDIT.md P-H21-4）；`test_videos/` 产物目录已删（本就
> gitignored，未入库）。

## 运行测试

```bash
PYTHONPATH=. pytest envs/humanoid21/tests/ -x -q
```

## 核心覆盖（test_data_interfaces.py）

1. **静态属性 (get_static_data)** — dof_names(21)、body_names、joint_limits
2. **核心状态 (get_core_state)** — root_pos/rot/vel、joint_pos/vel_norm
3. **派生状态 (get_derived_state)** — contacts_vec、root_state(13)、
   feet_forces(2)、opponent(9+15+15)、完整观测(96)
4. **观测维度分解** — [0:42] 本体 / [42:52]+[54:57] 全局 /
   [52:54] 触觉 / [57:96] 对手
5. **归一化/坐标系/face_vector/关键点/数值范围/动态一致性**

## 参考文档

- `../DATASPEC.md` — 数据规范
- `../CONTROLSPEC.md` — 控制规范
- `../OBSERVATION_zh.md` — 观测空间设计
- `../BENCHMARK.md` — 仿真性能基准（指标定义+参考标线+实测快照）
- `../benchmark.py` — 配套手动测量脚本（`PYTHONPATH=. python3 envs/humanoid21/benchmark.py`）

## 添加新测试

1. 测试文件以 `test_` 开头
2. 测试函数以 `test_` 开头
3. 审计类探针用 `test_audit_*.py` 命名，故意锁存当前缺陷行为
4. 验证数据格式和内容，包含清晰的测试说明
