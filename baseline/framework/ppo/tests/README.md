# ppo/tests — PPO 框架测试

> 类型：指南

| 文件 | 覆盖 |
|---|---|
| `test_trainer.py` | trainer 更新路径（GAE/multi-critic/confidence/dual-clip） |
| `test_dump.py` / `test_dump_delta.py` / `test_dump_analysis.py` | dumpkit 捕获/差分/分析 |
| `test_frame_access.py` | dump 帧访问器 |
| `test_viewer.py` | dumpkit viewer server |
| `test_resume_equivalence.py` | resume 续训与从头训的等价性（`_resume_driver.py` 辅助） |
| `test_post_update_artifacts.py` | 每 update 的 checkpoint/policy_exports 产物 |
| `test_s1_provenance.py` | dump 帧 provenance 字段（dump 专用，不入生产路径） |
| `test_param_overrides.py` | per-update 参数解析（resolve_update_params，含黑名单） |
| `test_minimal_example.py` | 最小实验冒烟 |

运行：`PYTHONPATH=. pytest baseline/framework/ppo/tests`。
