# E8 计划：综合工程验收与接口稳定

> 类型：记录

> 定位：收官包。**不写新机制**——E1–E7 已把契约层、生命周期、
> 多卡、迁移、故障管理、执行优化全部落地；E8 是清点、结算、
> 定版本、留证据。产出以文档/矩阵/小改动为主，凡涉及行为变更
> 必须单独标注并经契约测试锁定。
>
> 路线依据：`ROADMAP.md` E.4「E8：综合工程验收与接口稳定」与
> E.5 完成标准。

## 0. 现状盘点（起草时已确认的资产）

| 资产 | 位置 | 状态 |
|---|---|---|
| 能力注册表 | `capability_registry.py` | 14 条：NATIVE 10 / COMPAT 1 / PENDING 1 / UNSUPPORTED 3；未注册类 blueprint 启动即拒 |
| 版本字段 | `episode.py` `EPISODE_FORMAT_VERSION=3`；`blueprint.py` `BLUEPRINT_VERSION=1` | 已存在但无统一"公开 schema 版本表" |
| 组合测试 | `tests/` + `envs/framework/tests/` + `envs/humanoid21/tests/` | 414 passed（E7 收官态） |
| 验收文档 | E1–E7 PLAN/RESULTS、LIFECYCLE_TRACE、TERMINAL_FRAME_PLAN | 散置，无统一入口 |
| 已知缺口 | `WarpHumanoid21Simulator` facade shim；旧 host 原型（`host_compat.py`/`batch_plugin.py`/`coordinator.py`/`batch_context.py`）与正式入口并存 | 弃用路径未声明 |
| 已知既有失败 | `test_m2_cross_backend_fixtures`（fixture stale，自 `17d031fd`）；`test_stage_seg_rewards.py` collection 错误 | 待结算为"修复"或"显式拒绝" |

## 1. 工作包

### W0 — 支持矩阵逐项结算

- 以 `capability_registry.REGISTRY` 为骨架，补齐三列：**证据**
  （哪个测试/探针锁定）→ **E8 结算状态**（原生通过 / 兼容通过 /
  未实现 / 拒绝 / 过期）→ **处置**（保留 / 计划移除 / 文档拒绝）。
- 结算对象不止 14 条 registry 项——还要覆盖矩阵维度：后端
  （warp/fake）、collector（单卡/多卡）、迁移（standup 已迁移 +
  basic_balance E5 已验证）、observer/record/export、debug/
  replay、checkpoint/resume、故障路径、CUDA Graph 资格。
- 每条 UNSUPPORTED/PENDING 必须有一句"为什么 + 怎么解"。
- 产出：`E8_SUPPORT_MATRIX.md`（表格化，逐项引用证据文件/测试名）。

### W1 — 组合验收矩阵

按 ROADMAP E8 §2 的组合清单逐项给证据，**优先复用既有测试**，
缺口的补最小测试而非重写：

| 组合 | 现证据 | 缺口 |
|---|---|---|
| standalone WarpBackend（不经 runtime 直接用） | `test_warp_runtime.py`/backend 契约测试 | 待核 |
| FakeBackend runtime（无 GPU 全流程） | wave/lifecycle 测试主体 | 待核 |
| 单卡 collect | probe + GPU 套件 | 已证（E7 矩阵） |
| 多卡 collect（1/2/8） | `test_multi_rollouter.py` + E7 规模表 | 已证 |
| CPU→设备迁移 | `test_migration_manifest.py` + E5 验收 | 待核（审计输出是否仍新鲜） |
| Debug 捕获/回放 | `test_device_*` debug 项（E6-W2/W3） | 待核 |
| checkpoint/resume | `test_device_resume.py`（E6-W4） | 待核 |
| 故障矩阵 | E6-W5 逐项断言 | 待核 |
| 短程训练（device→PPO loop 接通） | ？ | **重点核实项**——若无端到端证据需补 smoke |
| hooks-on（子步插件）路径 | E7 矩阵 eager 格 | 已证（142K，如实标注限制） |

产出：验收证据表填入 `E8_SUPPORT_MATRIX.md`；发现的真实缺口
单列"W1.5 补缺清单"逐项关闭。

### W2 — 公开接口稳定

- 划定**公开面**（建议，待确认）：`DeviceRollouter`、`MultiDeviceRollouter`
  `collect(jobs)`、`Episode`/`Job`/`Trajectory` 数据契约、
  `EnvBlueprint`/`ParameterizedEnvBlueprint`、
  `capability_registry`（迁移结算入口）、`binding_registry`、
  `debug_capture`/`debug_replay`、探针脚本（probe_*）。
- 每面一条**版本锚**：Episode v3、blueprint v1 已有；registry/
  manifest/wave-record schema 若无版本则补字段或显式声明"内部
  schema 不承诺兼容"。
- 弃用路径：`WarpHumanoid21Simulator` facade、`host_compat`、
  `coordinator.py`/`batch_plugin.py`/`batch_context.py` 旧批量
  原型——判定各自为"保留兼容（标注弃用）"还是"移除"，写进
  支持矩阵而非静默留着。
- 不新造 `__init__.py` 大门面或未来 CLI（E.6 禁令）。

### W3 — 文档与示例收口

- `envs/batchframework/` 入口文档（README 或 CONTEXT.md 级别）：
  公开面一页图 + "5 分钟跑通"命令（单卡 collect 冒烟 / wave
  契约测试 / probe）+ 边界（图化资格、容量规格、hooks-on 限制、
  npz 兼容规则）+ 扩展方法（新 binding/新插件/新后端各指向一个
  现存范例）。
- 各阶段文档索引：E1–E7 PLAN/RESULTS 一页目录，避免考古。
- `CAPABILITY_LEDGER.md` 与 E8 矩阵的关系：ledger 是全仓视角，
  E8 矩阵是 batchframework 结算视角——ledger 增补一行指针，不重复
  记账。

### W4 — 回归与触发矩阵

- 全量回归复跑（414 项基线 + 本包新增）。
- 结算两个既有失败：
  - `test_m2_cross_backend_fixtures`：fixture 重录（依赖哈希
    刷新）或显式标记 expected-stale——二选一写进矩阵；
  - `test_stage_seg_rewards.py` collection 错误：修 import 或
    移出收集路径。
- 变更触发评估：E5 以来改动（终止帧语义、E7 图化）是否触发
  V0–V3 复验或短训练 smoke——按 ROADMAP「数据语义/生命周期/采样
  变化才触发」逐条判，默认不重跑长训练；若判定需要短程训练，
  单独与用户确认后执行。

## 2. 放行标准（对应 ROADMAP E.5 完成标准）

1. `E8_SUPPORT_MATRIX.md` 每一项有结算状态 + 证据指针，无
   "靠 standup 成功代替通用清单"的模糊格；
2. 组合验收九项（含短程训练判定）全部有证据或显式拒绝理由；
3. 公开面/版本/弃用表成文；新用户按入口文档可独立跑通单卡
   collect 与契约测试；
4. 全量回归绿（既有失败项全部结算）；
5. CPU 主路径无回归；触发矩阵的判定逐条留痕。

## 3. 明确不做

- 不启动 8 卡长训练（ROADMAP E.6 与 E8 放行均不要求）；
- 不重写/合并旧 host 原型实现——只做弃用判定与标注；
- 不承诺旧 npz 的向前兼容新语义（v3 可选键回退语义已在
  Episode 侧声明，矩阵中如实记录）；
- 不扩展到新任务/新后端（humanoid21 之外的任务接入属下一阶段）。
