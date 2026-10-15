# WP1 实施计划 — 契约审计引擎 + 切口 ⓪①②③④⑤⑥⑧⑨ 数学闭合

> 类型：计划
>
> 状态：待执行
> 上游设计：`DESIGN_dump_cuts.md` §2 切口定义、§4 断言表、§5 缺口地图
> 范围边界：**本包只做轻量/中等计算的核验与分解**（batch 256–1024 行上的
> 张量代数、单层 actor/critic 前向、参数 diff）。明确排除：逐样本梯度
> （WP2）、Episode 重跑与画面渲染（WP3）、跨 checkpoint 回放（WP3）、
> viewer 工具页与视频编辑器（WP4）。WP1 交付 CLI + HTTP 端点 + dump 首页
> Contract Ledger 卡（最小前端改动）。

---

## 0. WP1 交付物与验收标准

### 交付物

1. `analysis.py`：契约审计引擎 `verify_dump_contracts()` + 八个切口分析函数；
2. `experiments_sac/contract_verifiers.py`（新文件）：奖励/门控语义的
   逐样本复算器注册表，按 `versions.reward_semantics` 分发；
3. `debugkit.py`：新子命令 `verify` / `rollout-eps` / `slice-verify` /
   `batch-demo` / `twin` / `target-recompute` / `critic-fit` /
   `actor-contract` / `update-checks` / `post-detail`；
4. `debugserver.py`：上述全部对应的 `/api/dump/<name>/<endpoint>` 路由；
5. `sac_viewer.html`：dump 首页顶部 Contract Ledger 卡（状态灯 + 断言表），
   各模块卡角标绿/黄/红——**本包不动工具页布局**；
6. `transition.py` / `debugkit.py`（capture 侧）：`spec.json` 增补
   `slice_contract` 段 + `versions` 元数据硬编码修复；
7. `tests/test_contracts_wp1.py`（新）：合成 dump 契约测试 + 真实 dump 回归。

### 验收标准（可执行判据）

| # | 判据 | 验证方式 |
|---|---|---|
| A1 | `debugkit verify <dump>` 对 u_floor 真实 dump 与 shannon 真实 dump 各输出一张完整 Ledger，全部断言有确定状态（无 unhandled exception） | 在 `reg_balance_ufloor_s42/critic_tick_00212684` 与一个 shannon dump 上实跑 |
| A2 | Ledger 抓出已知真实问题：`versions.regularizer_mode="entropy"` vs `spec.sac_params.regularizer_mode="u_floor"` 必须 FAIL | 实跑确认 |
| A3 | 数学闭合断言在真实 dump 上实际通过：`bellman_decomp_closure`、`critic_mse_exact`、`polyak_soft_update_exact`、`integration_weights_sum_one`、`candidate_replay_exact` 全部 PASS（若有 FAIL → 是真 bug，升级为独立问题） | 实跑 + 人工复核 |
| A4 | 反事实重算：对 u_floor dump 改 `u_floor`/`reg_lambda` 重算 target，逐样本误差界 `atol=2e-5` 内闭合，并输出越界率 | `target-recompute --u-floor 0.3` 实跑 |
| A5 | 单元测试：合成最小 dump（shannon + u_floor 各一），注入已知违例（terminate+bootstrap=1、reward 篡改 1%、pair_index 翻转）能被对应断言精确捕获 | pytest |
| A6 | 修复后新采集的 slice `versions.regularizer_mode` 正确反映 `sac_params` 实际值 | 修改 `build_slices` + 单测 |
| A7 | `pytest baseline/framework/sac/tests/` 全绿；CLI 与 HTTP 同源同数据（抽查 3 端点比对 JSON） | 实跑 |

---

## 1. 关键设计决定（先于代码固化）

### D1. 奖励/门控语义的验证器注册表（切口①的核心）

**问题**：`build_slices()` 的奖励公式归实验所有（`sac_balance` 的
`per_step_phi_coef`、`_base_actor_weights`、`sac_standup` 的 `(1−γ)·potential`
缩放），但 dump 里目前**没有记录这些常量**——`config.json` 的
`experiment.knobs` 只有 5 个键，`spec.json` 只含 `sac_params`/`channels`。
用类默认值复算无法区分"类默认值"与"被 `--set` 覆盖的值"。

**决定**：capture 侧自描述化。

1. `CombatExperimentSACBase` 新增可选钩子 `slice_contract_spec() ->
   Dict[str, Any]`，返回实验的奖励/门控公式常量（如
   `{"reward_semantics": "basic_balance_phi_cross_v1", "per_step_phi_coef":
   0.01, "base_actor_weights": [3.0, 1.0], "r_cross_source":
   "cross_support"}`）。
2. `finish_critic_tick_dump` 将其写入 `spec.json["slice_contract"]`
   （additive，向后兼容）。
3. `contract_verifiers.py` 注册 `REWARD_SEMANTICS_VERIFIERS`：
   `reward_semantics` 字符串 → `(batch_row_arrays, slice_contract) ->
   violations[]`。未知 semantics → `WARN unverifiable`，**绝不静默 PASS**。
4. 旧 dump 无 `slice_contract` → 验证器用实验类当前默认值并标注
   `contract_source: "class_default_fallback"` + WARN。

### D2. `versions` 元数据硬编码修复（A6/A2 的另一半）

`exp_sac_standup.py` / `exp_sac_balance.py` 的 `build_slices()` 把
`versions.regularizer_mode`/`objective_mode` 写死为 `"entropy"`/`"shannon"`。
改为 `self.regularizer_mode` 动态值（`objective_mode` 同理映射）。
**不做迁移回填**——旧 dump 的差异本身就是 Ledger 要抓的证据。

### D3. U(s)/U(s') 的计算路径（切口④⑥的实体）

`_state_regularizer` 要求 actor 有 `uncertainty(obs, u_kind)`。
`_build_actor(spec, trainer_pre)` 重建 actor 后对 batch 的
`obs`/`next_obs` 各做一次 forward（256×96 输入，毫秒级，非"重计算"），
得到逐样本 `U(s)` / `U(s')` → 越界率 `P(U < u_floor)`、分位数、以及与
`reg_actor`/`reg_next` 的逐样本残差核验。actor 无 `uncertainty`
（shannon dump）→ 对应检查降级为 `SKIP` 并标注原因。

### D4. Contract Ledger 数据结构

```python
{
  "dump_dir": ..., "critic_tick": ...,
  "checks": [
    {"id": "C4.1", "cut": "target", "name": "bellman_decomp_closure",
     "status": "pass|warn|fail|skip",   # skip 带原因
     "severity": "contract|statistical",  # contract=算错即bug; statistical=病态域
     "metric": {"max_abs_err": 1.2e-7, "threshold": 2e-5},
     "violating_samples": [sample_id, ...],   # 最多回传 Top-20 违约行
     "detail": "human-readable zh summary"}
  ],
  "summary": {"pass": n, "warn": n, "fail": n, "skip": n}
}
```

阈值常量集中在 `analysis.py` 顶部 `CONTRACT_THRESHOLDS`，不进 spec
（阈值是分析侧口径，不是被测对象的属性）。

### D5. 性能纪律

- 全部 WP1 计算在 CPU 上完成，单次 `<200ms` 上限（batch 256）——
  actor/critic rebuild + forward 已验证可接受（`_build_actor` 现有路径）；
- 除 `target-recompute`/`post-detail` 的 rebuild+forward 外，一律纯
  `batch.pt`/`forward.pt`/`trainer_*.pt`/`analysis.json` 张量代数；
- 所有函数返回 JSONable dict；大数组（逐样本）默认返回分位数+Top-K，
  `full=True` 时才回全量（与 `dump_samples` 现有惯例一致）。

---

## 2. 工作分解（按依赖序）

### WP1.1 — capture 侧自描述化（先于一切验证）

| 改动 | 文件 |
|---|---|
| `versions.regularizer_mode/objective_mode` 动态化 | `exp_sac_standup.py`、`exp_sac_balance.py`（`"regularizer_mode": self.regularizer_mode`，objective 保持实验实际模式） |
| `slice_contract_spec()` 钩子 + 两个实验实现 | `experiments_sac/base.py`、`exp_sac_*.py` |
| `spec.json` 写 `slice_contract` 段 | `debugkit.py::_trainer_spec` 签名增补可选参数（`finish_critic_tick_dump` 传入 `experiment` 或预取 dict） |
| replay 侧顺带 | `buffer_stats()` 已含 `collection_round_counts`/`draw_counts`/`policy_fingerprint_counts`——②所需数据已齐，不改 |

### WP1.2 — 契约审计引擎骨架（`analysis.py`）

- `CONTRACT_THRESHOLDS` 常量表；
- `verify_dump_contracts(dump_dir, *, checks=None)`：注册表驱动的断言
  执行器（每个断言是 `(dump_ctx) -> CheckResult`），聚合为 Ledger；
- `_dump_ctx` 共享加载：batch/forward/spec/pre/post/analysis 一次性
  `load_dump` + lazy 属性（actor rebuild 仅在需要 U(s) 的检查时做）。

### WP1.3 — 切口⓪ realized exploration（`rollout_exploration`）

直接张量代数（无模型重建）：

```
eps = batch["actions"] - batch["policy_action"]          # [B,D]
```

输出：`eps_l2` 分布（mean/q50/q95/max）、每维 `eps_std[21]`、
`sat_frac = P(|a_env| > 1−1e-6)`、按 `metadata.behavior.explore_factor`
分组的均值；`policy_action` 缺失/全零 → SKIP。
断言：`C0.1 action_bounds`、`C0.2 realized_eps_present`（确定性采集
且 explore_factor=0 时 `eps_l2_max` 应 ≈0；`random_start` 样本的
actions 应近似 Uniform——KS 检验告警级）。

### WP1.4 — 切口① slice 契约复算（`slice_verification`）

- 按 `versions.reward_semantics` 分发到
  `REWARD_SEMANTICS_VERIFIERS`：
  - `basic_balance_phi_cross_v1`：逐行验
    `rewards[r_fall] ?= coef·clip(phi_post_ref,0,1)`、
    `rewards[r_cross] ?= reward_features.cross_support`、
    `actor_weight`/`actor_weight_next` ?= `softmax-free gate 归一化
    (3.0, 1.0·φ²)`；
  - `standup_4stage_dense_v2`：`rewards[r_potential] ?= (1−γ)·clip(potential)`；
- 通用断言：`C1.1 bootstrap_terminal_exclusivity`（升级现有
  `_batch_consistency`，并入 Ledger 输出违约 sample_id）、
  `C1.2 reward_formula_recompute`、`C1.3 actor_weight_gate_recompute`、
  `C1.4 provenance_version_match`（versions vs spec.sac_params）、
  `physics_delta>0`、`policy_action` 形状一致。

### WP1.5 — 切口② batch 人口学（`batch_demography`）

- batch vs `analysis.json.replay_stats` 对比：`collection_round_counts`
  归一化比例差（L1 distance）、`draw_counts` 分布 vs 几何分布近似
  （uniform 采样下 E[draws] ≈ tick×B/live_span）、终态样本占比
  （`terminated_rows`、`truncated_rows`）；
- 断言：`C2.1 live_slot_integrity`、`C2.2 sampling_coverage`
  （如终态样本占比 <1% → WARN，因为终态价值传播可能不足）。

### WP1.6 — 切口③④ target 链核验（`twin_analysis` + `target_recompute`）

- `candidate_twin_analysis(dump)`：
  - `C3.1`：`|Σ integ_w − 1|` 逐样本；
  - `C3.3`：`target_pair1_frac`、`actor_pair1_frac`、逐通道 twin gap
    `|q1−q2|` 分位数；
  - 联合-vs-独立 min 差异：独立 `min(q1_c, q2_c)` vs 联合选中
    `q_sel_c` 的逐通道差分布（量化耦合选择的实际影响）；
- `recompute_targets(dump, overrides={gamma, reward_scale, alpha,
  u_floor, reg_lambda})`：
  - **公式级**（无需模型）：shannon 模式下 `reg' = −α'·Σw·logp`，
    `target' = r·scale' + γ'·b·(softq + reg')` —— 全部来自已捕获张量；
  - **模型级**（u 模式或 f/λ 覆盖）：`_build_actor(spec, pre)` +
    `uncertainty(next_obs)` → `B'(s')`，重算 `target'`；
  - 断言：`C4.1 bellman_decomp_closure`（默认参数时闭合）、
    `C4.2 u_reg_next_exact`、终态 `bootstrap=0` 行 target==r·scale；
  - 返回新旧 target 分布对照 + 逐样本 Δtarget（供 WP4 滑杆 UI）。

### WP1.7 — 切口⑤⑥ critic/actor 契约（`critic_fit` + `actor_contract`）

- `critic_fit(dump)`：逐通道 EV（`1 − Var(y−Q)/Var(y)`，q1/q2 分开）、
  Q_pred vs target 散点数据（分位数 + 采样 256 点全量）、TD² Top-K
  集中度（top5% 样本占 Σδ² 比例）、`C5.1`/`C5.2`；
- `actor_contract(dump)`：
  - `actor_cand_q` 语义核验：用 **post** critic 前向 `obs`/`new_actions`
    与 `forward["actor_cand_q"]` 比对（证伪 pre/post 混用）；
  - `U(s)` 分布 + `P(U(s)<u_floor)` 越界率（`C6.2`）；
  - 门控链：`phi_pre → actor_weight → |w_c·q_c|` 逐通道贡献占比；
  - `C6.1 actor_decomp_closure`（复用 `objective_decomp` 的 check）。

### WP1.8 — 切口⑧⑨ 更新契约（`update_checks` + `post_detail`）

- `update_checks(dump)`：
  - `C8.1` Polyak：逐张量 `‖q_targ_post − ((1−τ)q_targ_pre + τ·q_post)‖∞`
    （per-channel q1/q2，报告 max）；
  - `C8.2` alpha：非 auto_alpha 时 `log_alpha` 逐位不变；auto_alpha 时
    符号方向核验（`Δlogα` 与 `−(logp̄+H̄)` 同号）；
  - 分层相对步长：`param_delta` 已有 L2 —— 补 `‖Δθ‖/‖θ‖` 比值与
    per-layer 排名（actor mu/std head vs trunk，critic 各通道）；
- `post_detail(dump)`（`policy_drift` 的逐样本版 + 反向核验）：
  - 逐样本 `Δdet_action`、`Δlogp`、`ΔU(s)`、各通道 `ΔQ(s,a)`、
    `ΔTD²`（post critic 在固定 pre-target 下的 TD² 变化 → `C9.1`
    同批 loss 是否下降）；
  - actor 侧：`weighted_q` 在 θ_post 下重估 → `C9.2` 单步目标改进检验。

### WP1.9 — CLI/HTTP/最小前端 + 测试

- `debugkit.py` 子命令（全部 `--pretty` + JSON）：
  `verify`、`rollout-eps`、`slice-verify`、`batch-demo`、`twin`、
  `target-recompute [--gamma --scale --alpha --u-floor --reg-lambda]`、
  `critic-fit`、`actor-contract`、`update-checks`、`post-detail`；
- `debugserver.py`：`/api/dump/<name>/` 下同名端点；
- `sac_viewer.html`：dump 首页顶部 Contract Ledger 卡（断言表格 +
  状态徽章 + 违约 sample_id 可点击设为共享游标）；八模块卡角标按
  所属切口的断言状态着色；
- `tests/test_contracts_wp1.py`：
  - 合成最小 dump 生成器（复用现有 test fixture 构建
    batch/forward/trainer_pre/post）；
  - 注入违例：terminated+bootstrap=1 / reward×1.01 / pair_index 翻转 /
    target+ε / Polyak 残差注入 → 断言逐一命中；
  - 真实 dump 回归：对 `reg_balance_ufloor_s42` 既有 dump 断言
    `C1.4` FAIL（硬编码 bug 证据）、数学闭合项 PASS；
  - HTTP↔CLI 一致性抽查。

---

## 3. 文件改动清单

| 文件 | 动作 |
|---|---|
| `framework/sac/analysis.py` | +`CONTRACT_THRESHOLDS`、`verify_dump_contracts`、`rollout_exploration`、`slice_verification`、`batch_demography`、`candidate_twin_analysis`、`recompute_targets`、`critic_fit`、`actor_contract`、`update_checks`、`post_detail`；`_dump_ctx` 共享加载器 |
| `framework/sac/debugkit.py` | +CLI 子命令；`_trainer_spec`/`finish_critic_tick_dump` 增 `slice_contract`；`load_dump` 兼容字段 |
| `framework/sac/debugserver.py` | +10 个 `/api/dump/<name>/...` 路由注册进 `_dump_endpoint` |
| `framework/sac/sac_viewer.html` | +Contract Ledger 卡 + 模块卡状态角标（最小改动，工具页留给 WP4） |
| `experiments_sac/base.py` | +`slice_contract_spec()` 钩子（默认 `None`） |
| `experiments_sac/exp_sac_balance.py` | +`slice_contract_spec` 实现 + versions 动态化 |
| `experiments_sac/exp_sac_standup.py` | 同上 |
| `experiments_sac/contract_verifiers.py` | 新建：`REWARD_SEMANTICS_VERIFIERS` 注册表 + 两个语义验证器 |
| `framework/sac/tests/test_contracts_wp1.py` | 新建 |
| `framework/sac/ARTIFACTS.md` | `sac_dump_v3` spec.json 增 `slice_contract` 字段说明 |
| `framework/sac/DEBUG_PLAYBOOK.md` | 新增“契约审计”一节：`verify` 入口 + 各 FAIL 的处置路径 |

## 4. 风险与对策

| 风险 | 对策 |
|---|---|
| `slice_contract` 只在**新** dump 里有；旧 dump 的验证只能用类默认值 | D1.4 fallback + WARN 显式标注 `contract_source`，允许 `--set` 覆盖过的 run 被标记为"不可证伪"而非假 PASS |
| 浮点容差：fused Adam / TF32 / CPU-GPU 差异导致恒等式微超阈值 | 阈值分两档：数学恒等式 `2e-5`（沿用 `recompute_dump` 口径）、契约布尔项严格相等；所有 err 同时报绝对值便于复判断 |
| `_build_actor` 在 `uncertainty()` 不存在的 actor（shannon s01）上 | 逐检查 `hasattr` 降级 SKIP，原因入 Ledger |
| WP1 函数与 WP2 的 `grad_attribution` 共享 rebuild 开销 | `_dump_ctx` 做 lazy cache；WP2 复用同一加载器 |
| `finish_critic_tick_dump` 签名变更影响既有调用 | 新参数可选（`slice_contract: Optional[Mapping]`），旧调用不变 |

## 5. 执行顺序与检查点

1. WP1.1（capture 自描述）→ 先提交，保证从此刻起的新 dump 都自带契约；
2. WP1.2 引擎骨架 + WP1.3/1.4/1.5（纯张量代数切口⓪①②）→ 提交；
3. WP1.6 target 链（③④，含首个模型级路径）→ 提交；
4. WP1.7 critic/actor（⑤⑥）→ 提交；
5. WP1.8 update/post（⑧⑨）→ 提交；
6. WP1.9 CLI/HTTP/前端卡 + 测试 + 真实 dump dogfood → 提交收口；
7. 收口径：对两个真实 dump 打印 Ledger 结果表给用户审阅，顺带用
   `target-recompute --u-floor` 出一份"如果 f 设为 0.3/0.7 会怎样"的
   反事实证据。
