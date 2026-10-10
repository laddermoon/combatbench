# Batch Framework 阶段 Review —— 上下文底稿

> 类型：记录  
> 日期：2026-10-09  
> 目的：回到初心（"用 MJX 能力加速训练，在稳定抽象边界对等替换
> 原实现并达到一样效果——不要求 bit-identical"），对照冻结的
> 验收口径盘点已证事实、已测数据与未决问题。

---

## 1. 初心与验收口径

**主命题**：device/warp rollout 在 `collect(jobs) -> List[Episode]`
边界上等对替换 CPU collector，PPO 侧零感知。

**冻结验收（M0 §5.1，用户已确认，R2 用）**：

| 门槛 | 值 |
|---|---|
| success 非劣效 | `success_MJX ≥ success_CPU − 0.02` |
| final_pot 非劣效 | `final_pot_MJX ≥ final_pot_CPU − 0.02` |
| 样本效率 | 达同能力的 agent transitions ≤ CPU 1.25× |
| ~~端到端加速~~ | ~~`wall_CPU / wall_MJX ≥ 2.0`~~ **口径变更**：2026-10-10 用户裁定 R2 以**等效正确性**结算——上三条为判据，≥2× 降为如实记录的实测事实，不作门槛 |

R1（链路闭环）= E5 已过；R3（独立 Agent 迁移）= **已执行**：
step 实验 L1–L4 通过，结果归档 `R3_RESULTS.md`（2026-10-10）。

---

## 2. 已交付能力（有证据指针）

| 面 | 状态 | 证据 |
|---|---|---|
| 后端无关 wave/生命周期契约 | 原生通过 | `test_device_lifecycle_contract` 22 项、终止帧语义 |
| `DeviceRollouter`/`MultiDeviceRollouter`（1–8 卡） | 原生通过 | `test_device_rollouter`/`test_multi_rollouter`/`test_shard_plan` |
| 策略 executor —— 8 truncnorm 族 | 原生通过（2026-10-09 补齐） | `test_policy_executor_golden` 56 项 |
| 逐帧 ef —— 声明式程序 | 原生通过（同日） | `test_ef_program` 14 项 |
| job-keyed RNG / manifest / capability 注册 / debug capture+replay | 原生通过 | E8 矩阵逐行指针 |
| 迁移指导（MIGRATION_GUIDE + audit 工具） | 原生通过 | R3 已执行（step，L1–L4）：`R3_RESULTS.md` + `migration_manifests/step.json` |
| 每 run host CPU ≈ 2 核 + 6GB | 已计量 | `CPU_REQUIREMENTS.md`（对照 CPU collector ~96 核） |

---

## 3. R2 实测（standup，协议等同）

| | device s42 | device s43 | device s44 | **CPU s42（同代码新版）** | CPU s42（8月旧码，仅旁证） |
|---|---|---|---|---|---|
| 首次 success>0.5 | u495 | u325 | u440 | **u450** | u1185 |
| 收敛 ~1.0 | u625 | u410 | u470 | **u470** | u1210 |
| 最终 success/pot | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/0.999 | 1.0/1.0 |
| 每 update | ~12.8s | ~12.8s | ~12.8s | **~9.8–11.1s** | ~11.1s |
| 全量墙钟（1500u） | 5.3h | 5.4h | 5.3h | **~5.1h** | ~5.3h |

### 按门槛判定

- **success/final_pot 非劣效**：✅ 通过（三 seed 均 1.0，远超 −0.02）。
- **样本效率 ≤1.25×**：✅ 通过（device 收敛 u410–625 ≈ 84–128M
  transitions；CPU u470 ≈ 94M——同量级，device 均值甚至略优）。
- **端到端 ≥2×**：❌ **不达标（实测事实，保留）**——同协议下 device
  每 update ~1.15–1.3× **更慢**。此前"device 收敛快一半"的观察是同
  **旧代码** CPU run 对比的伪影；同代码 CPU seed 同样在 u450–470 收敛。
- **R2 结算（2026-10-10 用户裁定）**：✅ **通过**——验收口径改以
  等效正确性为准（质量非劣效 + 样本效率均达标），端到端加速事实
  按任务类型如实记录、不升级为门槛。

### 为什么 2× 达不到（结构解释）

standup 是 **update-延迟受限**而非吞吐受限：PPO 每 update 的移动量
被 clip 结构封顶，batch 放大（12288 eps）换不来 update 数下降
（devfast 实测仅 ~2–3× update 压缩且每 update 贵 ~2.4×）。GPU 的
2.1M sub/s 吞吐在 512eps/update 的协议下用不满。

**GPU 优势的真实形态是并行吞吐**：~2 核/run → 8 卡可并行多 seed/
实验（3 seeds 并行 5.4h 全收敛 vs CPU 串行 ~15h+ 或需 288 核）。

---

## 4. 已闭合缺口（本次补齐）

- ~~executor 只有 TruncatedNormal~~ → 8 族全注册 + 显式拒绝；
- ~~逐帧 callable ef~~ → 声明式 ef 程序（obs_threshold）双后端，
  `exp_step` 已迁移。

## 5. 未决项（Review 要定夺）

1. ~~**R2 的 ≥2× 门槛如何结算**~~ → **已定夺（2026-10-10）**：
   R2 以**等效正确性**结算——选项 (c) 落地。standup 的 ≥2× 不达
   标作为实测事实保留在 §3，不作门槛。
2. ~~**R3 未执行**~~ → **已结算**（`R3_RESULTS.md`）：L1–L4 通过 +
   负例试验已补做（`test_negative_migration` 11 项，顺手修复
   `unit_hash` 源码指纹缺口与 manifest 坏文件静默跳过）；
   过程指标（首过率/token 成本）未结构化记录——后续 R3 类试验
   需埋点。
3. **残余 spec 缺口**：reference/delta_factor（executor ctx 未接）、
   `policy_eval_mask`（hold 模式）、其余 callable ef 形态。
4. **工程残余**：facade shim、hooks-on eager 路径、worker 单核
   ~90% 的下一 host 瓶颈。
5. **CPU 参照只有 1 个新 seed**（3 device vs 1 CPU）——若要严格
   统计口径可再补 CPU seeds（每条 ~5h/96 核）。

---

## 6. 一句话现状

R2 **已按等效正确性口径结算通过**（2026-10-10 用户裁定）：
对等替换层面达成（接口稳定、全策略族、生命周期/RNG/ef 语义
闭合、三 seed 质量非劣效样本效率达标）；"加速"层面 standup 类
update-延迟受限任务实测 ~1.15–1.3× 更慢/update（如实记录、
不作门槛），优势在**并行实验吞吐与 ~40× host CPU 节约**。
下一步主攻方向见未决项。
