# M2 结果：对等 simulator 修复、跨后端验证与性能探测

对应 [M2_PLAN.md](M2_PLAN.md)。本文是证据记录，不超出实测范围做推断。

## 1. 工作包 A：契约修复（W1）逐项关闭状态

| # | 差异 | 关闭方式 | 状态 |
|---|---|---|---|
| 1 | XML `battle_v1` vs `battle_circular_v2` | `ARENA_XML` 切换为 circular_v2（condim=3、impratio=10、24 段圆墙） | **关闭** |
| 2 | 96 维观测公式系统性错误 | `_get_robot_view_batch` 重写：sqrt 变换、projected_gravity、arena_center_local、机体系速度、对手 cvel 作 relative_vel、head 速度不变换 | **关闭**（实测 obs 与 CPU 0 差） |
| 3 | root_state 字段 | 对齐 height/projected_gravity/linear_vel(local)/angular_vel(local)/arena_center_local | **关闭** |
| 4 | 缺 body/joint 数组字段 | 已有批量 `_collect_body_joint_arrays` | **关闭** |
| 5 | 外力语义相反（持久 vs 每步清零） | pending buffer 累加，只注入首个子步后清零；**data.xfrc_applied 步后清零** | **关闭**（fixture `extforce-torso-push` 锁定） |
| 6 | set_core_state 无刷新 | 末尾 `mjx.forward`，且清零 warmstart/ctrl/xfrc | **关闭** |
| 7 | 角速度多转一次 | qvel[3:6] 本就是机体系，去掉多余旋转 | **关闭** |
| 8 | reset 不支持 per-env/seed | `reset(seeds=, options=)` 支持 per-env initial_distance/poses，seeds 记录 | **关闭** |
| 9 | contacts schema | `ncon` = per-env 活跃数 (B,)，padding + `capacity` + `active_mask` | **关闭** |
| 10 | set_action 无校验 | 加 (B,21) shape 校验 | **关闭** |

## 2. 验证中发现的实现级 bug（新发现，均已修复）

1. **scan 无历史时也堆叠完整 mjx.Data**：`physical_step(25)` 在 B=512 下需 ~132GiB → OOM。修复：`keep_history` 分支化，无历史 scan 不产 ys。B=512 稳态显存 19.8GiB。
2. **状态恢复残留求解偏置**：`set_integration_state`/`set_core_state` 只写 qpos/qvel，残留 `qacc_warmstart`/`ctrl`/`xfrc_applied` 把后续 forward 的接触力拉偏 ~4%，且随调用顺序漂移。修复：恢复契约定义为 `{qpos,qvel,ctrl=0,warmstart=0,xfrc=0}`，双侧（MJX 方法 + CPU oracle `_restore_raw`）同步清零。
3. **canonical 接触 dtype**：跨后端 int8/int32/int64 混排 + JSON 往返丢 dtype，导致标量精确类型误报。修复：`canonical_contacts` 统一 int64/float64；比较器标量改语义比较（int 比值、float 比容差），数组仍严格 dtype。

## 3. 工作包 B：跨后端验证（W2）结果

新增 `validation_mjx.py` 候选适配器与 6 个跨后端 case（CPU oracle 生成、digest 锁定、stale 检测有效）：

| case | 内容 | 结果 |
|---|---|---|
| dyn-standing-s1 / s25 | 站立态 + 固定动作，1/25 子步 | **pass**（qpos ~1e-13 级） |
| dyn-moving-2x5 | 中间姿态 + 双动作 ×5 子步 | **pass** |
| extforce-torso-push | 施加力+力矩 → 3 子步 → 残差校验 | **pass**（residual_xfrc=0） |
| state-io-write-read | 非穿透写 → 立即读 derived | **pass** |
| batch-isolation-2 | B=2 不同初态/动作逐 env 对照 | **pass**（无跨 env 泄漏） |

28 个 M1 case 按声明返回 `unsupported`（reward/trajectory 为宿主侧逻辑；mjSTATE blob 无 MJX 对应物）——不支持即显式上报，符合规范。

**fixture 容差**（冻结，`atol=rtol=1e-5`，只作用于 frames/envs 数值子树；结构字段仍精确）：远小于任何已见语义偏差（外力未清零 ~1e6、深穿透分歧 ~1e6、action 未生效），实测差值集中在 1e-16~1e-7。

**记录的语义边界**：深穿透姿态（如传送至与对手重叠）下接触力可差至 4 个数量级（CPU 332N vs MJX 3.7e6N）——两求解器在深穿透下本质分歧，geom 对/位置/法向一致。状态写入必须避免深穿透；该边界是后端固有差异，不是实现 bug。

## 4. 工作包 C：吞吐探测（W3，初步，机器负载 ~115）

| 配置 | reset+jit | 首个 25 子步（scan jit） | 稳态 25 子步 | 吞吐 |
|---|---|---|---|---|
| jax / fp64 / B=512 | 53.3s | 62.9s | 15.9s | **804 env-substeps/s**，19.8GiB |
| jax / fp32 / B=512 | 48.9s | 60.5s | 4.1s | 3124 env-substeps/s |
| jax / fp32 / B=128 | — | — | 0.24s | **13438 env-substeps/s（峰值）** |
| jax / fp32 / B=32 | — | — | 0.08s | 10028 env-substeps/s |

CPU 参照：单 env 0.17ms/substep（5895 substeps/s）；生产 rollout 实测 ≈ **472K env-substeps/s**（10.85s/update ÷ 5.12M substeps）。

**结论**：mjx-jax 后端在本模型（双 humanoid + condim=3 + impratio=10 + 64 geoms）× RTX 4090 上，FP32 峰值仍比 CPU 生产吞吐慢 **~35×**，FP64 慢 **~590×**。B=512 反而劣化（调度/显存压力），峰值在 B≈128。ojax 后端**不满足**"至少一个语义可接受配置有加速潜力"的放行条件——按 ROADMAP 触发暂停评估。

**mujoco-warp**：已安装 `warp-lang 1.12.1` + `mujoco-warp 3.8.0.3`（版本匹配 mujoco 3.8.0，未动共享环境；需 PyPI 代理，百度镜像无此包）。探测发现 warp 后端**不是 drop-in**：它使用自有 `NWORLDS` batching 布局（非 vmap），`contact__dim` 等字段形状约定不同，接入需要独立的批量架构 + 版本兼容 shim（GraphMode/`warp_type_to_np_dtype` 已打补丁）。**尚未评估其吞吐**——这是当前唯一可能翻盘的候选路径。

## 5. 放行标准逐项核对

- [x] 差异 1–10 全部关闭（无"保留近似"项）
- [x] CPU 自重放全 pass；MJX 候选 FP64 下 6/6 跨后端 case 通过（1e-5 容差内）
- [x] 不支持能力（mjSTATE blob、reward/trajectory、policy_eval）显式 `unsupported`
- [x] 未跑训练/性能验收；吞吐标注为初步探测
- [ ] **"至少一个语义可接受后端有加速潜力"——mjx-jax 不满足；mujoco-warp 未评估**

## 6. 建议的下一步（供决策）

1. **mujoco-warp 专用批量路径**：按 NWORLDS 布局重写步进层（不是给现有 vmap 换 impl）；若 warp 吞吐 ≥ ~1M env-substeps/s 则值得继续。
2. **或接受 MJX 定位为"GPU 上正确但非加速"**：jax 后端保留为跨后端验证的候选实现与回归基线。
3. 在 warp 结论出来前，不建议启动 M3（插件框架）建立在 jax 后端上。

*测试入口：`pytest tests/test_mjx_validation.py`（新增 `test_m2_cross_backend_fixtures`：6 pass + 3 unsupported）。*
