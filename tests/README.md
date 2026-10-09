# tests/ —— 顶层测试目录索引

> 类型：指南

> **注意**：本目录不在 `pyproject.toml` 的 `testpaths` 内，默认
> `pytest` 不收集这里。这些测试按归属迁移前暂存于此，需要显式
> 指定路径运行：`PYTHONPATH=. pytest tests/ -q`（多数需要 CUDA
> 设备，`CUDA_VISIBLE_DEVICES=<空闲卡>`）。

## 归属索引

| 归属 | 文件 |
|---|---|
| `envs/batchframework`（MJX/Warp 批量路径，多数需要 GPU） | `test_batch_validation.py`、`test_mjx_validation.py`、`test_warp_runtime.py`、`test_warp_validation.py`、`test_multi_rollouter.py`、`test_device_runtime.py`、`test_device_rollouter.py`、`test_device_balance.py`、`test_device_standup.py`、`test_device_lifecycle_contract.py`、`test_wave_contract.py`、`test_physics_contract.py`、`test_shard_plan.py`、`test_blackboard.py`、`test_dependency_direction.py`、`test_migration_manifest.py` |
| `baseline/humanoid21/curriculum` | `test_stage_seg_rewards.py` |
| fixture | `debug_fall_images/`（摔倒调试用图像资产目录） |

详细审计记录见 `AUDIT.md`（docs/scripts/tests 段）。
