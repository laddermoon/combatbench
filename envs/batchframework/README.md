# batchframework — MuJoCo-Warp 批量采样框架

> 类型：指南

GPU（mujoco-warp）上的批量 rollout：CPU `ParallelRollouter`/`EnvRuntime`
的**设备端对等实现**——同一 `collect(jobs) -> List[Episode]` 契约、同一
Episode/PPO 接口、同一插件/observer 生命周期语义，1–8 卡采样同契约
Episode。CPU 实现始终是语义参照系；设备端是 fp32 近似后端，
不承诺 bit-identical。

## 🏆 这项成果的价值

- **消掉采样瓶颈而不破坏契约**：`collect(jobs) -> List[Episode]` 与
  CPU `ParallelRollouter` 同一签名——训练代码零改动换后端。实测
  B=2048 图化 **~480K env-substeps/s，单张 4090 打平 192-worker
  CPU 生产池**（E7_BASELINE，~472K 对照）。
- **契约优先的工程方法**：CPU 实现永远是语义参照系——golden 对拍
  测试把设备行为钉在 CPU 语义上；`PUBLIC_INTERFACE.md` 给公开面
  配版本锚（Episode npz v3 / blueprint v1），接口稳定性是定义出来
  的，不是默认的。
- **迁移显式且可审计**：`capability_registry` 是 blueprint 插件→
  设备实现的唯一裁决点——未注册即拒绝，不猜测映射、不静默近似；
  `migration_manifests` 给每个 CPU 蓝图产出可行性证据。
- **设备侧可调试**：wave 级 `debug_capture` + 三模式 `debug_replay`
  ——GPU 采样路径不是黑箱。
- **性能工程诚实**：E7_BASELINE 记录的是测量方法学+瓶颈归因
  （host 提交是总吞吐主约束，图化可再提数倍；8 卡 151K→166K
  说明当前扩展 host-bound），不只是峰值数字。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| 同一 `Job`/`Episode` 契约贯穿 CPU/device | 训练侧零改动切换后端——加速路径是替换不是分叉 |
| CPU 为语义参照系 + golden 契约测试 | fp32 近似后端不漂语义——"近似"被约束在数值层 |
| `capability_registry` 默认拒绝 | 迁移无静默语义缺口——能力缺口在启动期暴露 |
| ef 声明式程序（框架层设计，见 `baseline/framework/README.md`） | 逐帧探索干预在 device collector 上语义平价 |
| wave 批内生命周期复刻 | `episode_step`/`physics_step`/终止帧/退化帧语义与 CPU 一致，Episode 可直接进 PPO buffer |

> 在飞状态与已知限制（图化资格、容量规格、多卡 host-bound 等）
> 见下方「边界与限制」与 `STATUS_REVIEW.md`。

## 5 分钟跑通

```bash
cd /data1/mono/things/combatbench
export PYTHONPATH=/data1/mono/things/combatbench

# 单卡 collect 冒烟（B=512，200 步）
CUDA_VISIBLE_DEVICES=1 python3 envs/batchframework/probe_e7_baseline.py \
    --batch 512 --devices 0

# 无 GPU 的契约测试（FakeBackend 全流程）
python3 -m pytest tests/test_wave_contract.py \
    tests/test_device_lifecycle_contract.py -x -q

# GPU 契约套件（golden 对拍 + 生命周期）
CUDA_VISIBLE_DEVICES=1 python3 -m pytest \
    tests/test_wave_contract.py tests/test_device_lifecycle_contract.py \
    tests/test_device_rollouter.py tests/test_device_runtime.py -x -q
```

## 最小用法

```python
from envs.batchframework.device_rollouter import DeviceRollouter
# jobs: List[Job]（同 CPU 契约——env_bp + policy_bp + seed 等）
with DeviceRollouter(batch_size=B, device="cuda:0") as dr:
    episodes = dr.collect(jobs)           # -> List[Episode]（训练直接可消费）
report = dr.last_collect_report           # 分项计时/sync 记账/健康统计
```

多卡：`MultiDeviceRollouter(devices=[0,1,...], batch_size_per_worker=n)`
——jobs 按 shard 分配，各 worker 独立 collect 后合并。

## 跨后端行为验证（Identical 目测）

`dual_backend_video.py`：**同一导出策略、同一初始位姿**，分别在
GPU（MjWarp `BatchRuntime`）和 CPU（MuJoCo `EnvRuntime`）各跑一回合
确定性 rollout，两端帧都用 `Humanoid21Simulator` 的 broadcast-view
相机渲染（像素管线逐一致），输出左右对比视频。验证用临时工具，
不追求速度（B=1 单波，几十秒）。

```bash
PYTHONPATH=. python3 -m envs.batchframework.dual_backend_video \
    --policy baseline/runs/<run>/policy_exports/uNNNNN \
    --env-blueprint baseline/humanoid21/blueprints/standup_4stage_dense_v2_env.yaml \
    --seed 12345 --distance 2.0 --device cuda:0 \
    --out-dir /tmp/dualvid
```

输出 `gpu.mp4` / `cpu.mp4` / `compare.mp4`（左 GPU 右 CPU）+
`trajectory.npz`（GPU 逐帧 qpos/qvel）。

机制要点：两端 RNG 流不同，单靠 seed 对不齐摔倒位姿——脚本把
CPU 侧 post-reset 状态**覆写为 GPU 侧记录到的 post-reset qpos/qvel**，
保证严格同初始位姿；此后轨迹的任何肉眼可见差异都来自物理后端
本身（fp32 近似），不是采样/渲染差异。预期观感：同策略下两端
动作序列基本 identical，长时程可能出现微小发散（混沌敏感性，
属预期非缺陷）。

## 边界与限制

- **图化资格**：无子步回调的物理 `advance` 与 `obs_build` 走 CUDA
  Graph（默认开，`CB_WARP_GRAPH=0`/`CB_OBS_GRAPH=0` 可关）；
  **子步插件（pre/post_phy_step 回调）路径物理回退 eager**——可用
  但非图化（B=2048 实测 142K vs 582K env-substeps/s）。
- **容量规格**：`nconmax`/`njmax` 为每世界固定上限——接触数溢出
  是显式 FAILED 行（health scan），不静默截断。
- **终止语义**：`episode_step` 计"进入的 step() 调用"（含零物理
  退化帧），物理进度看 `physics_steps`；训练截取统一走
  `Episode.agent_frame_boundary`。
- **旧 npz 兼容**：v3 可选键缺失时按声明回退，不伪造旧切片。
- **未注册的 blueprint 单元**：启动即失败（`capability_registry`
  默认拒绝），不猜测映射、不静默回退。

## 扩展方法

| 要做什么 | 从哪里下手 |
|---|---|
| 接入新插件/observer | `capability_registry.register` 登记 cls→factory；参照 `device_balance.py`/`device_standup.py` 的设备实现范式 |
| 接入新任务（binding） | `binding_registry.py` + 参照 `envs/humanoid21/batch_binding.py` |
| 接入新后端 | `physics.py` 后端契约 + `fake_backend.py` 最小实现参照 |
| 迁移 CPU 实验 | `migration_audit.py` 审计 → manifest → 逐项转换（E5 流程） |
| 双后端行为对比 | `dual_backend_video.py`（同策略同位姿 GPU/CPU 对照视频） |
| 性能归因 | `probe_e7_baseline.py`（分项计时/sync 记账）；判读口径见 E7_BASELINE/E7_RESULTS |

## 文档索引

| 文档 | 内容 |
|---|---|
| `PUBLIC_INTERFACE.md` | 公开面/版本锚/弃用清单（**先看这个**） |
| `DESIGN.md` | 架构规格（分层/对象/数据平面/装配/优化面） |
| `SEMANTICS.md` | 语义规格（seed/reset/生命周期/与 CPU 差异登记） |
| `CONTEXT.md` | AI 速览 memo（入口 + 常见坑） |
| `MIGRATION_GUIDE.md` | CPU 实验→设备迁移 7 步流程 |
| `BLACKBOARD_DESIGN.md` | ctx.metrics/ctx.events 设备通道设计（E9 已落地） |
| `BATCHFRAMEWORK_AUDIT.md` | 系统审计（vs CPU 差距清单） |
| `E8_SUPPORT_MATRIX.md` | 能力逐项结算 + 证据指针 |
| `ROADMAP.md` | E0–E8 工程化路线 + 历史 M0–M8 |
| `E1–E7_PLAN.md`/`E7_RESULTS.md`/`E7_BASELINE.md` | 各阶段计划与实测结果 |
| `LIFECYCLE_TRACE.md` | 生命周期权威时序（CPU 参考） |
| `TERMINAL_FRAME_PLAN.md` | 终止帧语义契约 |
| `migration_manifests/*.json` | 已迁移实验的逐单元证据 |
