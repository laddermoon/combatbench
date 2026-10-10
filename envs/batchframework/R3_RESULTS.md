# R3 迁移试验结果 —— step 实验（GaitClockSimulator + FootStateObserver）

> 类型：产物 ｜ 日期：2026-10-09（迁移执行）/ 2026-10-10（本文归档）
> 对应任务书：`R3_TASK_BRIEF.md`（冻结协议 §1）
> 到达级别：**L4**（L1–L4 全部通过，证据见 §2）

## 1. 迁移过程记录

执行者：AI agent（未参与 batchframework 实现，按 brief 仅使用
项目文档 + `migration_audit`/capability_registry 工具链）。

| 步 | 动作 | 依据 |
|---|---|---|
| 1 | `audit_yaml('baseline/humanoid21/end2end/step_env.yaml')` 前置审计 | MIGRATION_GUIDE §0 |
| 2 | 6 个蓝图单元全部判 native 可迁（sim/摔倒 reset/4 段奖励 ×2/foot_state ×2），产出结构化计划 | L1 |
| 3 | 发现框架缺口：blueprint `simulator.config`（含 `gait_period: 40`）无设备端接收方 → 贯通 `make_sim(batch_size, device, sim_config)`，两个调用点同步 | 见 §3 缺口 1 |
| 4 | 实现 `WarpGaitClockSimulator`（组合 `WarpHumanoid21Simulator`，obs 96→99 追加 cmd_L/cmd_R/progress，由 `episode_steps` 推导——CPU 帧索引对齐，partial reset 后各行独立发散） | MIGRATION_GUIDE §3.3 |
| 5 | 实现 `DeviceFootStateObserver`（h/sole_clear/contact 六叶，常量引用 CPU 模块单一来源） | MIGRATION_GUIDE §1 映射 |
| 6 | 注册 `_GaitClockWarpBinding`（`sim_config_keys` 白名单多消费 `gait_period`）+ capability 条目（`debug_torque` 经 `config_notes` 声明忽略） | §3 缺口 1 约定 |
| 7 | 写 Layer A（同注入态对拍）+ Layer B（真机 warp-vs-CPU 注入同 `qpos/qvel` 对拍）对照测试 | MIGRATION_GUIDE §4 阶梯 |
| 8 | L3 collect 契约测试 + L4 device collector 冒烟训练 | brief 验收层级 |

## 2. 每级验收证据

| 级 | 通过判据 | 证据（命令 + 产物） |
|---|---|---|
| L1 | 全单元登记/审计，零 unknown | `step.json` manifest：6 单元 native，`config_disposition` 逐键有处置（`gait_period`=consumed，`debug_torque` 等 4 键=declared） |
| L2 | 新设备单元跨后端契约验证 | `tests/test_device_step.py`：步态时钟公式等价（Layer A 伪 accessor 对拍 + Layer B `test_foot_state_warp_vs_cpu` 真机注入同态对拍，`assert_array_equal` 严格比接触 bool）；`TestStepDeviceWarp.test_gait_sim_assembly` 99 维装配 |
| L3 | `collect(jobs)` 产出合法 Episode | `TestStepDeviceCollect`：obs (T,99) 步态时钟列逐帧对 CPU 公式（跨右窗 t=20、周期回卷 t=40）、终止记录/final_obs/ef-program ctx 完整；`Step.build_trajectories` + `PPOBuffer` 无改动消费（log_prob parity <1e-4）；`MultiDeviceRollouter` 分片保序 |
| L4 | `--collector device` PPO smoke | `train_step_ppo_20261009_173147`：`--collector-devices 2,3` 从 standup `ckpt_u01500` 热启动（96→99 零填充），2 updates 全绿，eval success=1.000 |

复跑确认（归档日）：`pytest tests/test_device_step.py` = **18 passed,
1 skipped**（multi-device 用例需 ≥2 卡）。

## 3. 文档/工具缺口清单

| # | 缺口 | 处置 |
|---|---|---|
| 1 | **框架缺口**：`make_sim` 拿不到 `simulator.config`——`gait_period` 等蓝图配置无设备端接收方 | 已修（本试验内）：`sim_config` 贯通 `make_sim`/device_rollouter/dual_backend_video；`sim_config_keys` 白名单消费，余键须 `config_notes` 声明否则 `ValueError`——与 audit unknown 判定同一约定 |
| 2 | `episode_steps` "步尾自增、obs 构造紧随"的精确次序此前只在源码可考 | 源码注释已写明（`device_runtime.py`）；未另立文档 |
| 3 | binding 对 sim config 的处置语义 MIGRATION_GUIDE 未写（本实现是首个实例） | **已回填**（`3f17986e`）：§3.3 三去向表 + §6 陷阱清单 + §7 索引 |

## 4. 假设与偏差记录

- **接触语义**：设备 `_foot_ground` 与 CPU `_detect_contact` 逐字段
  同式（geom 对 × foot-body ∩ ground-geom，无阈值）——Layer B
  真机对拍实证；
- **bool 叶保真**：`RecordStore` 按 `spec.dtype` 分配，Episode 层
  bool 保真；归档核查发现迁移报告称 `extract_per_step_field(dtype=bool)`——
  **该函数无 dtype 参数**，`coerce_per_step` 恒转 float32（CPU 侧
  对所有 observer 叶的既有统一行为，值保真、dtype 在 extract 层
  不保真）。属报告措辞偏差，非功能缺陷；
- **过程指标**：brief 要求的首过率/人工介入轮次/token 成本**未在
  迁移时结构化记录**——可考的是 3 commit 分轮交付（L1-L2 → L3 →
  manifest）+ 归档核查发现 1 文档缺口 + 1 措辞偏差、0 功能缺陷；
  后续 R3 类试验应在任务执行中埋点记录，不能事后补造。

## 5. 代码量统计（`git diff` 三 commit 合计）

| commit | 内容 | diffstat |
|---|---|---|
| `2fe58ec6` | L1-L2：binding/config_notes/WarpGaitClockSimulator/DeviceFootStateObserver/manifest/测试 | 7 文件 +1212/−8 |
| `55c34f01` | L3：TestStepDeviceCollect 契约测试 | 1 文件 +142 |
| `81fae0e8` | manifest L3/L4 证据 | 1 文件 +90/−3 |
| `3f17986e`（归档期补） | MIGRATION_GUIDE sim_config 语义回填 | 1 文件 +24/−3 |

合计 ~+1468/−14，新增 `device_step.py`(259 行） + `test_device_step.py`(808 行）。

## 6. 负例试验（工作包 C，2026-10-10 补做）

`tests/test_negative_migration.py` 11 项全绿。执行中**发现并修复
两个真缺口**：

1. **`unit_hash` 不含源码指纹**（负例核心发现）：v1 只哈希
   {cls 名, config}——改类源码（同名同 config）旧证据照样新鲜，
   与 MIGRATION_GUIDE §4 的声明和 discuss D14 的设计意图都不符。
   已升级 v2 = {cls, config, **CPU 模块文件**, **device_cls 模块
   文件**}：`CapabilityEntry.device_cls` 声明设备实现类（顺带修复
   `from_audit` 从不填 `device_cls` 的问题）；指纹口径升级使三份
   旧 manifest 全部判 stale——**机制自证**——已重跑 pytest 证据级
   并重新盖章（train_smoke 沿用原 run：collect 行为路径未变，仅
   audit/manifest 元数据变更）。
2. **畸形 manifest 静默跳过**：`find_manifest_for` 的
   `except: continue` 让 stale 保护静默失效（E8 踩过的缺字段坑
   的同类）——改为 `UserWarning` + 按无关处理。

覆盖矩阵：

| 工作包 C 条款 | 测试 | 结果 |
|---|---|---|
| 不支持插件拒迁 | `test_audit_flags_unregistered_plugin_unsupported` + `test_collect_rejects_unregistered_plugin` | audit 判 unsupported+unknown；collect `ValueError` |
| 改参数/代码判过期 | `test_collect_rejects_config_drifted_blueprint`（config）+ `test_freshness_marks_source_edited_unit_stale`（源码）+ `test_unit_hash_tracks_*` | stale→`RuntimeError`；源码/模块常量/设备实现漂移均判 stale |
| 错误注入定位 | `test_observer_missing_leaf_names_unit_and_leaf` + `test_observer_wrong_shape_names_leaf` + `test_malformed_manifest_warns_and_skips` | 报错点名单元/叶；坏 manifest warn 不静默 |
| 指纹非空转 | `test_native_units_source_resolvable` | 全 NATIVE 条目 cpu/dev 源码指纹可解析 |

**残留（如实）**：`unit_hash` 指纹粒度=模块文件——跨模块依赖
（基类、他模块 import 的常量）变化不触发 stale；
"能力扩展逐项验收"条款的打包形态未单独立项（现有各套件已覆盖
其内容）。

## 7. 未覆盖项（如实记录）

- L4 仅 smoke（2 updates），不含训练质量判定（brief 明确允许）；
- 结论不能外推：本次仅覆盖 step 实验形态，reference/delta_factor
  等未支持 spec 的迁移路径未验证；
- 过程指标（首过率/token 成本）未结构化记录——§4 已说明。
