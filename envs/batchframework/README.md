# batchframework — MuJoCo-Warp 批量采样框架

GPU（mujoco-warp）上的批量 rollout：CPU `ParallelRollouter`/`EnvRuntime`
的**设备端对等实现**——同一 `collect(jobs) -> List[Episode]` 契约、同一
Episode/PPO 接口、同一插件/observer 生命周期语义，1–8 卡采样同契约
Episode。CPU 实现始终是语义参照系；设备端是 fp32 近似后端，
不承诺 bit-identical。

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
| 性能归因 | `probe_e7_baseline.py`（分项计时/sync 记账）；判读口径见 E7_BASELINE/E7_RESULTS |

## 文档索引

| 文档 | 内容 |
|---|---|
| `PUBLIC_INTERFACE.md` | 公开面/版本锚/弃用清单（**先看这个**） |
| `E8_SUPPORT_MATRIX.md` | 能力逐项结算 + 证据指针 |
| `ROADMAP.md` | E0–E8 工程化路线 + 历史 M0–M8 |
| `E1–E7_PLAN.md`/`E7_RESULTS.md`/`E7_BASELINE.md` | 各阶段计划与实测结果 |
| `LIFECYCLE_TRACE.md` | 生命周期权威时序（CPU 参考） |
| `TERMINAL_FRAME_PLAN.md` | 终止帧语义契约 |
| `migration_manifests/*.json` | 已迁移实验的逐单元证据 |
