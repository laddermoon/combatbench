# E5 计划：CPU 实验迁移成为有约束的工程流程

> 类型：记录

**状态**：已完成（W0–W4 全部落地，验收见下）
**上游**：[discuss.md](./discuss.md) D14/D15.2/D16 | [E3_PLAN.md](./E3_PLAN.md)/[E4_PLAN.md](./E4_PLAN.md)（已完成）
**ROADMAP 对应**：E5 —— "把 CPU 实验迁移变成有约束的工程流程"

## 背景与定位

前几个阶段把**机制**建完了（backend 契约、生命周期、采集、多卡）。E5 把"迁移一个 CPU 实验"从手工搬运变成**有审计、有证据、可失效追踪**的流程，并用第二个真实案例（`basic_balance`）驱动框架扩展。

### 现状差距

| 契约要求（D14） | 现状 | 差距 |
|---|---|---|
| 解析单元级能力表（原生/兼容/待转换/不支持+原因） | `capability_registry` 只查单个 cls | 无 blueprint 级整体审计工具 |
| 未使用配置项也必须解释 | 无 | 需 config 处置表 |
| 来源/配置/验证证据 manifest + 失效追踪 | 无 | 需 manifest schema |
| 逐 agent 早停（per-agent termination）真实案例 | standup 只走 env 级 timeout | basic_balance 的 `DualImbalanceTerminationPlugin` 恰好是逐 agent（A 终 B 续跑）——首个真实不对称终止用例 |
| 第二个迁移案例验证 Episode/训练接入 | 无 | basic_balance 端到端 |

### basic_balance 依赖面（源码核查）

`basic_balance_v2_phi_dual_env.yaml`：`phy_steps=25, max_steps=600, standing 初态` +

| 单元 | 类型 | 读取面 | 迁移性质 |
|---|---|---|---|
| `DualImbalanceTerminationPlugin` | plugin | `contacts_vec`（aff/body/geom 筛选非足-地接触力）+ root_pos 高度门 + per-agent 计数器→`imbalance_robot_a/b/both` | **逐 agent 终止**——新生命周期路径；contact 管线 warp 已具备（standup contact_score 同构） |
| `CrossSupportBalanceRewarder` | observer ×2 | contacts + static + per-robot derived | 中量（415 行 CPU 参考） |
| `PostureRewarder` | observer ×2 | joint_pos/vel_norm + uprightness + keypoint xpos | 小（task_tables 已供 norm/xpos） |
| `HeightPhiObserver` | observer ×2 | root_pos.z + uprightness | 极小 |

四个单元全部未注册 → 当前走设备路径**启动即拒绝**（正确的显式失败，审计要覆盖到）。

## 关键设计判断

**J1 — 审计先行，工具化而非手工列表。**
`migration_audit.py`：输入 blueprint → 输出每单元的能力状态、所需 accessor/ctx 面（静态声明 `declared_reads`/`require_mutator`/`rng_salt`/`output_schema`）、config 键处置表（每个键标 `consumed`/`ignored-with-reason`/`unsupported`）。审计输出同时是 manifest 的输入。

**J2 — manifest 是可失效的证据记录，不是描述文档。**
`migration_manifest.py`：`{experiment_name, source_bp_hash, units: [{cls, device_cls, capability, config_disposition, evidence: {level, input_hash, pass_ts}}]}`。源 blueprint/常量变了 → hash 不匹配 → 相关 evidence 标 `stale`（D14 显式失效语义）。

**J3 — 逐单元等价测试走"同输入状态回放"。**
不从零编测试：构造同一物理态（固定 qpos/qvel/接触）分别在 CPU 插件和设备插件上跑 hook，输出逐字段对照（observer 输出、termination 请求、计数器演化）。`DeviceFallenResetPlugin` 已有此类先例。

**J4 — 逐 agent 终止是本次的"差异案例"锚点。**
`imbalance_robot_a` 只终 A：world_running 维持、A 在 `policy` 模式下继续被采样驱动（CPU `post_termination_action` 默认语义）、B 的 records/帧长不受影响、A 后续仍可能积累新 reason——这正是 E2 建的 `agent_done`/term_history/`policy_eval_mask` 首次被真实任务行使。

## 工作包拆分

```
W0 审计工具 + manifest schema（机制层，不依赖具体迁移）
→ W1 basic_balance 单元迁移（imbalance 插件 + 3 observer 类型 ×2）
→ W2 逐单元状态回放等价测试（CPU 参考 vs 设备实现）
→ W3 端到端：固定策略 CPU/device episode 契约对照 + build_trajectories
      兼容 + device collector 冒烟（单卡 + 2 卡）
→ W4 回归 + 文档（capability 矩阵/manifest 落盘/ROADMAP）
```

### E5-W0：审计与 manifest 机制

- `envs/batchframework/migration_audit.py`：`audit_blueprint(env_bp) -> AuditReport`
  - 每单元：cls → capability 状态、声明能力面（`declared_reads`/`declared_writes`/`output_schema`/`plane`/`rng_salt`/`per_hook_mutator`）、config 键处置（对照构造签名：消费的参数 vs 透传未用的键——未用键必须有 `note`）；
  - blueprint 级字段：`simulator.config` 每键处置、`phy_steps_per_action`/`max_steps` 是否被设备路径使用；
  - 输出机器可读 dict + 人读 markdown 表。
- `envs/batchframework/migration_manifest.py`：`MigrationManifest` dataclass + `to_dict`/`from_dict` + `validate_freshness(env_bp)`（hash 变 → 标 stale 不静默沿用）。
- 对 `standup_4stage_dense_v2` + `basic_balance_v2_phi_dual` 各跑一次审计，产出首版 manifest。

### E5-W1：basic_balance 单元迁移

新设备原生实现（`envs/batchframework/device_balance.py` 或按模块分工）：

1. `DeviceDualImbalancePlugin`：contacts_flat → 非足-地接触力>阈值 → per-agent 计数器 → `request_termination(ids, reason, agents=[i])`（逐 agent！）；min_height 门、counter 衰减语义对齐 CPU。
2. `DeviceCrossSupportRewarder` / `DevicePostureRewarder` / `DeviceHeightPhiObserver`：obs/derived 字段经 `task_tables()` + `views()` 取；`output_schema` 声明每叶。
3. `capability_registry` 注册 4 个 NATIVE 条目；`binding_registry` 不需改（同 sim_cls，io_schema 相同——observer 扩展不改 IO）。

### E5-W2：逐单元等价测试

`tests/test_device_balance.py`（GPU-gated）+ FakeBackend 子集：

- 同输入（构造的 contacts/state）下 CPU `DualImbalanceTerminationPlugin` 与 `DeviceDualImbalancePlugin` 的终止判定、reason 串、计数器演化逐帧一致；
- 三个 observer 的输出逐字段与 CPU 版同输入一致（容差记 fp32）；
- 逐 agent 终止的波契约：A 终 B 续 → term records 不对称、`num_frames` = env 末步（world 存活到全 agent done）、A 的后续帧仍记录（"policy" 语义）、各自 reason 归位。

### E5-W3：端到端接入验证

- 固定策略导出 + basic_balance blueprint：device collect 产出合法 Episode（`test_collect_episode_contract` 同构断言 + basic_balance 专属字段）；
- `BasicBalance.build_trajectories` 无差别消费 device episode（两奖励通道 + phi）；
- 冒烟训练 `--collector device`（单卡）+ `--collector-devices 0,1`（多卡）各 2 updates。

### E5-W4：回归 + 文档

- tests/ 全量；`migration_manifest` 随审计落盘（`envs/batchframework/migration_manifests/`）；
- `capability_registry`/`binding_registry` 矩阵更新进 ROADMAP 支持状态；
- ROADMAP E5 + discuss.md D14 验证状态行更新。

## 风险与降级路径

| 风险 | 概率 | 降级 |
|---|---|---|
| `contacts_vec` 的 aff/body/geom 编码在 warp 端布局不同，imbalance 判定漂移 | 中 | W0 审计先对 contact 语义建小探针（CPU vs warp 同布局断言）；不一致则该单元标 `compat`/pending 而非冒充 NATIVE |
| CrossSupport 规则复杂，逐字段等价失败 | 中 | 拆分叶子逐个对齐；实在有分叉则该 observer 登记 PENDING 并写清哪叶差异——不静默近似 |
| 逐 agent 终止暴露 E2 生命周期路径的真实 bug | 低-中 | 那正是本阶段要暴露的——修 runtime 而非绕开 |
| 审计工具覆盖不全（动态 config 消费路径看不见） | 中 | 处置表以构造签名为下限，未用键必须显式 `ignored:` 注解——漏注解视为审计失败 |

## 放行条件

1. `audit_blueprint` 对 standup + basic_balance 产出无 `unknown` 处置的审计表；
2. 迁移的 4 个单元各有同输入状态回放等价证据（manifest evidence=pass）；
3. basic_balance device collect Episode 契约合法 + `build_trajectories` 无差别消费；
4. 单卡 + 2 卡冒烟训练正常；
5. manifest 落盘且源变更可使相关 evidence 失效（测试覆盖 stale 判定）。

**明确不在本阶段**：一次迁完所有实验；HOST_SLOW/compat 适配层的全面实现（仅在迁移确实需要时按需补）；跨 collect 状态链；observer 多端输出的新形态（schema 已支持，用例驱动再加）；reference/delta 采样迁移。

---

## 执行结果（2026-10-02）

全部工作包完成，放行条件逐项核对：

| 条件 | 结果 |
|---|---|
| 1. 审计无 unknown | ✅ standup/basic_balance 两审计 0 unknown 键；`test_audit_no_unknown_keys` |
| 2. 逐单元等价证据 | ✅ `test_device_balance` 13 项同注入态对拍（CUDA），manifest evidence=unit_replay |
| 3. e2e collect + build_trajectories | ✅ `TestBasicBalanceE2E`：4 ep 契约合法，两通道 trajectory 正常；逐 agent 终止真实行使（terms={imbalance_robot_a/b}） |
| 4. 单/双卡冒烟 | ✅ `--collector device` 与 `--collector-devices 0,1` 各 2 updates，两通道 conf/aw 正常 |
| 5. manifest 失效追踪 | ✅ `test_manifest_freshness_and_stale`：config 漂移 → 仅该单元 stale；落盘 `migration_manifests/*.json`（gitignore 已放行） |

**执行中抓到并修掉的 bug**：
- `DeviceCrossSupportObserver` 初版让 WAIT→TRACKING 迁移当步又跑了一遍 tracking 逻辑（CPU 是早退）——counter 多 +1，由 `test_short_segment_penalty` 对拍暴露；
- `validate_freshness` 初版按 cls 索引 live spec——同名 observer 多实例（cross_support_a/b）互相覆盖误报 stale，改为 observer 按 name、plugin/simulator 按 cls；
- `manifests` 被 `*.json` gitignore 拦掉——加例外后入库。

**残余偏差记录**：
- `cross_support_*` 的 Episode `observer_outputs` 叶结构是 `{reward: list}`（CPU 是裸 scalar list）——`extract_per_step_scalar` 对两者兼容，记录于此而非伪装 bit-identical；
- `DeviceDualImbalancePlugin` 的 contact 判定是 fp32 并行快照语义，不等同 CPU 逐点串行迭代——单测已锁语义等价边界。

提交：`c324c914`（W0–W2 实现+测试）→ `3ad283a1`（gitignore+manifest 落盘）→ 本次收尾。
