# blueprints/

环境蓝图与初始策略蓝图目录。文件清单与分类见
[`../README.md`](../README.md#blueprints)。

## 调试环境（录制 + 回放）

用任一训练环境蓝图跑一回合并录制帧数据：

```bash
PYTHONPATH=. python3 -m envs.framework.round_runner \
    --env-blueprint baseline/humanoid21/blueprints/basic_balance_env.yaml \
    --policy-a-blueprint policy/blueprints/random.yaml \
    --policy-b-blueprint policy/blueprints/random.yaml \
    --recorder "envs.framework.recorder:BaseFrameRecorder?output_dir=baseline/humanoid21/blueprints/out"
```

启动回放查看器（默认端口 8765）：

```bash
PYTHONPATH=. python3 -m envs.framework.recorder_viewer --no-browser baseline/humanoid21/blueprints/out
```

在浏览器打开 `http://localhost:8765/viewer.html` 查看逐帧数据。
