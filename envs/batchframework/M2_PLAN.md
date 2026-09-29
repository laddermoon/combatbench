# M2 计划：对等 simulator 与后端可行性

对应 [ROADMAP.md](ROADMAP.md) §5 的 M2 阶段。输入契约见 [M0_BASELINE.md](M0_BASELINE.md)，验证工具见 [M1_VALIDATION.md](M1_VALIDATION.md)。本文先给出**已完成审计的差异清单**（工作包 A 的输入），再列执行顺序与放行标准。

## 1. 审计结果：现有 `mjx_simulator.py` 与 CPU 契约的差异

审计方法：逐方法对照 `envs/humanoid21/simulator.py`（CPU 参考）与 `envs/batchframework/mjx_simulator.py`（现有原型），并以 `battle_circular_v2.xml` 验证 MJX 3.8.0 可加载性。

**已验证可行**：`mjx.put_model` 接受目标模型（condim=3、impratio=10、64 geoms 含 24 段圆墙、33 bodies、nq=56/nv=54/nu=42）；`mjx.forward`、`xfrc_applied`、`xanchor`、`xipos`、`cvel` 在 mjx.Data 中均存在；contact 容量 2461。

| # | 差异 | CPU 契约 | 现有 MJX 实现 | 严重度 |
|---|---|---|---|---|
| 1 | XML | `battle_circular_v2.xml`（condim=3 摩擦、impratio=10、圆形墙） | `battle_v1.xml`（condim=1 无摩擦、方墙） | **阻断**：接触模型本质不同 |
| 2 | 96 维观测 | sqrt 速度变换、projected_gravity、arena_center_local、机体系速度、对手 cvel 作 relative_vel、head 速度不变换 | 线性 proprioception、多余 local_orientation(6)、世界系速度、relative_vel 用**自身**速度、kp_vel 无变换 | **阻断**：观测语义系统性错误 |
| 3 | `root_state` 字段 | height/projected_gravity/linear_vel(local)/angular_vel(local)/arena_center_local | height/local_orientation/linear_vel(world)/angular_vel(world) | 高 |
| 4 | per-body/joint 数组 | `body_xipos`/`body_xquat`/`body_linvel_world`/`body_angvel_world`/`joint_world_anchor`（`_collect_body_joint_arrays`） | 只有 `body_xpos` | 高：缺字段 |
| 5 | `apply_external_force` | `+=` 累加进 `xfrc_applied`，**每次 physical_step 后清零**（力只作用下一步） | 覆盖赋值 + 跨步持久 | **高**：语义相反 |
| 6 | `set_core_state` 后刷新 | 写完 `mj_forward`，derived 立即可读且一致 | 只替换 qpos/qvel，xpos/contacts 陈旧至下一步 | 高 |
| 7 | 角速度坐标系 | qvel[3:6] 本就是机体系，get/set 均**不做旋转** | get/set 都多乘一次 R/R⁻¹（双向对称 bug，roundtrip 自洽但与 CPU 不符） | 高 |
| 8 | `reset` options | scalar initial_distance/poses | 只支持标量，忽略 `seeds`，不支持 per-env | 中：standup 需要逐 env initial_distance |
| 9 | contacts schema | `ncon`=活跃数，数组长度 ncon（无 padding） | `ncon`=容量，padding + contact_count/active_mask | 中：schema 需统一定义 |
| 10 | `set_action` | clip 到 [-1,1] + shape 校验报错 | 只 clip 无校验 | 低 |
| 11 | warmup/积分内部态 | `mjSTATE_INTEGRATION` 含 warmstart 等 | MJX 无 warmstart 概念 | 固有偏差：跨后端 case 用显式 qpos/qvel，不用 mjSTATE blob |
| 12 | `get_sensor_data` | `{}` | `{}` | 一致，无需改 |
| 13 | `get_static_data` | 字段集 | 相同字段集 | 一致（自动随新 XML 更新） |
| 14 | `get_broadcastview_image` | CPU renderer | 未实现（默认 None） | 允许延后：渲染不走 GPU 热路径，M2 标记为可选缺口 |

## 2. 工作包与执行顺序

### W1：修正 `mjx_simulator.py`（对应差异 1–8、10）

1. `ARENA_XML` 切到 `battle_circular_v2.xml`。
2. **重写 `_get_robot_view_batch`**：严格按 CPU `_get_robot_view` 的 96 维拼装顺序与变换（`sign*sqrt(|v|)/2` 关节速度、`sign*sqrt(|v/2|)` 角速度、`projected_gravity = -R[2,:]`、`arena_center_local`、机体系 linear/angular vel、`relative_vel = R_self⁻¹ @ opp_cvel_linear`、head 关键点速度不变换、其余 4 个 kp_vel sqrt 变换）。root_state 字段对齐 CPU 五项。
3. **补 `_collect_body_joint_arrays` 批量版**：xipos/xquat/cvel 拆 linvel/angvel/xanchor。
4. **角速度 bug**：`get_core_state`/`set_core_state`/`view` 中 qvel[3:6] 均为机体系，去掉多余旋转；`root_vel_local` 世界→机体系转换保留。
5. **外力语义**：维护 pending 累加 buffer；`physical_step` 将其注入**第一个**子步后清零（scan carry 内置零化），与 CPU "set→作用一步→清零" 一致。
6. **`set_core_state` 末尾 `mjx.forward`** 刷新 derived。
7. **`reset` 支持 per-env options**：`initial_distance`/`initial_pose_a/b` 接受标量或 (B,) 序列；`seeds` 接受并记录（当前初始姿态确定性，seed 不改变状态——如实记录该事实而非假装用到）。
8. `set_action` 加 shape 校验；contacts 输出调整为：`ncon` 改为 per-env 活跃数 (B,) int32，padding 数组保留 + `capacity` 字段，文档化。

### W2：扩展验证用例与比较器（M1 框架上增量）

9. **跨后端物理 case**：新 operation，输入为显式 `{qpos, qvel, initial_action, actions[], substeps}`（不用 mjSTATE blob——MJX 无 warmstart）；CPU 与 MJX 两侧各自恢复→推进→导出。
10. **接触 canonical 化**：两侧适配器把接触整理为**仅活跃项、按 (geom1,geom2,量化 pos) 排序**的定长列表再进 expected；比较器保持通用严格比较（数量不一致→length failure；排序歧义→pos 容差内视为同项，文档化局限）。
11. 新 case：`set_core_state` write-then-read（写后立即读 derived，不需 step）、`apply_external_force`（施力→1 步→验证清零语义+位移方向）、部分 `set_core_state(env_ids)`、batch>1 隔离。
12. **`validation_mjx.py`**：候选适配器实现 `Adapter` 协议（backend/version/dependencies/execute），只调 `MjxHumanoid21Simulator` 公共接口；对尚不支持的 operation 抛 `Unsupported`。
13. **容差冻结**：FP64 下先实测再冻结；初值目标 qpos/qvel `atol≤1e-6`（已知 MJX 求解器 ~1e-5 级偏差，实测后如实记录）；观测 float32 容差放宽到转换噪声级。容差只审批数值差，不用于掩盖字段缺失。

### W3：轻量可行性探测（受限执行）

14. GPU 空闲卡上测：JIT 编译耗时、batch=1/64/256/512 的 `physical_step(25)` 稳态吞吐与显存。CPU 侧只取已有 `time.rollout` 参照量级，不做正式计时（高负载，正式性能基线仍按 §5.2 协议延后）。结果记入结果文档并标注"初步探测"。

### W4：文档与收尾

15. `M2_RESULTS.md`（或并入本文结果章节）：差异表逐项关闭状态、实测容差、吞吐初值、未通过项与原因。
16. 若 fixture 依赖清单需要纳入 MJX 侧文件——不纳入参考 manifest（MJX 是候选侧，其指纹记录在报告的 `target` 字段）。

## 3. 放行标准（与 ROADMAP §5 一致）

- 上述 1–10 项差异全部关闭或逐项标注"保留+理由"（不允许静默近似）。
- 新旧 case 在 CPU 自重放 pass；MJX 候选在 FP64 下对冻结容差通过，或每个失败项有定位结论。
- `policy_eval`、外力插件、RandomFallen 等非本阶段能力以 `Unsupported` 显式上报。
- 吞吐数据标注为初步探测，不作为 M6 正式基线。

**暂停条件**：condim=3+priority 的接触在 MJX 中语义不保、imx.step 外力注入不可行、或 FP64 吞吐低到加速无意义（4090 实测为准）。

## 4. 明确不做

- 不做 RandomFallen/disturbance 插件迁移（M3）。
- 不做 BatchRuntime/插件体系（M3）。
- 不接 PPO/rollout（M5），不做训练验证（M6）。
- 不做 warp 后端选型实验（先记录候选，M2 只验证 MJX-JAX 可行性）。
- CPU 复评/成本分解仍延后（机器负载高，GPU 探测不受 CPU 负载影响可先做）。
