# RUNTIME — 运行环境说明

> 类型：指南 ｜ 日期：2026-10-10 ｜ 实测基线：GPU 服务器 `instance-1f1igpaq`

本文记录 CombatBench 的**已验证运行环境栈**、硬版本约束与
可复现的安装配方。与 `docs/ENVIRONMENT.md`（仿真物理环境：
场地/机器人/相机）正交——本文管"装什么能跑起来"。

## 1. 硬件与驱动基线（实测）

| 项 | 值 | 说明 |
|---|---|---|
| GPU | 8× NVIDIA RTX 4090 (24 GiB, sm_89) | device rollout 需要 CUDA GPU |
| Driver | 550.163.01 → CUDA driver API **12.4** | 决定 warp-lang 版本上限（§4）|
| OS | Linux 5.15 (Ubuntu) | |
| Python | **3.10.12** | |
| 系统 nvcc | CUDA 12.1 | **不影响运行**——torch/warp 自带 toolkit |
| pip 源 | `mirrors.baidubce.com/pypi/simple`（默认配置） | ⚠️ 缺新版 mujoco-warp，见 §5 |

## 2. 已验证生产栈（当前环境实测版本）

| 包 | 版本 | 用途 |
|---|---|---|
| mujoco | **3.8.0** | CPU 参照物理（FP64） |
| mujoco-warp | **3.8.0.3** | device 批量物理后端 |
| mujoco-mjx | 3.8.0 | jax-mjx 后备路径（probe 用） |
| warp-lang | **1.12.1** | mjw 运行时（自带 CUDA 12.9 toolkit） |
| torch | 2.7.1 (+cu12 系列) | policy 前向 + device 张量 |
| jax / jaxlib | 0.6.2 (+jax-cuda12-plugin) | mjx-jax 路径可选 |
| numpy / scipy | 2.2.6 / 1.15.3 | |
| gymnasium | 1.2.3 | 仅 spaces 类型注解 |
| imageio(+ffmpeg) / opencv | 2.37 / 4.12 | 视频导出（EGL 离屏渲染） |
| matplotlib / pytest | 3.10.8 / 8.4.2 | 分析 / 测试 |

## 3. 安装配方

```bash
pip install -r requirements.txt   # 完整钉版栈（含 device 依赖）
# 或：pip install -e .            # pyproject 同样钉死核心依赖
```

`requirements.txt` 为 `==` 钉版全集（含 mujoco-warp/warp-lang/
mjx/jax）；`pyproject.toml` 钉核心依赖（不含 jax/mjx 后备
路径）。宽松 `>=` 会让 pip 静默解析到未验证新版本——
**物理语义随版本漂移，禁止放宽**。

## 4. 硬版本约束（O5 spike 实测探明）

| 约束 | 边界 | 依据 |
|---|---|---|
| driver 12.4 | **warp-lang ≤1.17**（CUDA 12.9 toolkit） | warp 1.18 起切 CUDA 13.4，要求 driver≥13.0 |
| mujoco ↔ mujoco-warp | mjw 3.x.y 要求 `mujoco>=3.x`（如 mjw3.15→mj≥3.12） | pip 依赖声明 |
| CPU/device 版本对齐 | CPU 参照与 device 栈共用同一 `mujoco` 包——升级即两边同升 | 单一 site-packages |
| 物理一致性 | mujoco 3.8→3.15 CPU 轨迹漂移 ~1e-14（近 bit-identical）；mjw 3.8→3.15 device e2e −30% | `PERF_AUDIT.md` §8 |

**候选升级栈**（已在隔离 venv 验证，测试全绿，未投产）：
`mujoco==3.15.0 + mujoco-warp==3.15.0 + warp-lang==1.17.0`，
需一处 `xfrc_applied` vec6 适配。立项验证清单见
`envs/batchframework/PERF_AUDIT.md` §8.3。

## 5. 网络/镜像坑

- baidubce 镜像**没有** 3.9+ 的 mujoco-warp 与 1.13+ 的
  warp-lang——`pip index`/`pip install` 报
  "No matching distribution" 属镜像缺口，非包不存在；
- 装新版本需走官方源 + 代理：
  ```bash
  export https_proxy=http://192.168.16.76:18000
  pip install -i https://pypi.org/simple <pkg>
  ```

## 6. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PYTHONPATH` | — | **必须**指向 repo 根（训练/探针都依赖），`PYTHONPATH=/data1/mono/things/combatbench` |
| `CUDA_VISIBLE_DEVICES` | — | 选卡；多卡采集走 `--collector-devices` |
| `MUJOCO_GL` / `PYOPENGL_PLATFORM` | `egl`（代码内设） | 离屏渲染，无需手设 |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false`（warp_backend 内设） | jax 与 warp 共存，勿覆盖为 true |
| `CB_WARP_GRAPH` | `1` | `=0` 关 CUDA graph（A/B 对照/排障） |
| `CB_OBS_GRAPH` | `1` | `=0` 关观测构建图 |
| `CB_INFER_DEVICE` / `CB_INFER_CAPACITY` | `cuda` / `128` | CPU rollout 的推理服务 |
| `COMBATBENCH_FALL_DEBUG` / `_DIR` | `0` / `/tmp/fall_debug` | 摔倒调试导出 |

## 7. 验证 checklist

```bash
# 1) CPU 参照冒烟（mujoco 纯 CPU）
PYTHONPATH=. python3 -B baseline/framework/train.py \
  --experiment basic_balance --algo ppo --smoke

# 2) device 后端契约（需空闲 GPU）
CUDA_VISIBLE_DEVICES=<N> PYTHONPATH=. \
  python3 -B -m pytest tests/test_device_rollouter.py \
  tests/test_device_lifecycle_contract.py -x -q

# 3) 探测卡占用后选干净卡
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

## 8. 已知缺口

- **无 Dockerfile/bootstrap 脚本**——本文是"已验证快照"，
  非从零可重放配方；首次在新机部署需自行处理 EGL/系统库；
- jax 仅为 mjx-jax 探针路径所需，纯 warp device 路径不依赖。
