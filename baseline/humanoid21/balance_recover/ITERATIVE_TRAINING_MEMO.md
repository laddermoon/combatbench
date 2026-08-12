# 迭代平衡恢复训练实验 Memo

## 目标

通过迭代「探测边界 → 生成采样分布 → 训练」循环，逐步提升策略在冲量扰动下的平衡恢复能力。

## 初代策略

- **来源**: `baseline/runs/fixaw_survonly_crossphi2_s42`
- **最新 export**: `u00460`（460 updates）
- **实验名**: `v2_basic_balance_v2_phi_dual_fixaw_survonly_crossphi2`
- **policy_blueprint**: `baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml`

## 迭代流程

每轮迭代 3 步：

1. **Probe** — `probe_boundary.py`
   - 用当前最新策略跑全量 (direction × force × duration) 扫描
   - 输出: `boundary_genN.csv` / `boundary_genN.json`
   - 命令:
     ```bash
     PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/probe_boundary.py \
         --policy-blueprint-path <latest_export>/policy_blueprint.yaml \
         --output baseline/humanoid21/balance_recover/boundary_genN.csv \
         --json-output baseline/humanoid21/balance_recover/boundary_genN.json \
         --workers 96
     ```

2. **Sample** — `sample_distribution.py`
   - 从边界数据生成训练用采样权重分布
   - 输出: `sample_weights_genN.npz` / `sample_distribution_genN.json` / `samples_genN.csv` + 热力图
   - 每轮保留历史，不覆盖
   - 命令:
     ```bash
     PYTHONPATH=/data1/mono/things/combatbench python3 baseline/humanoid21/balance_recover/sample_distribution.py \
         --input baseline/humanoid21/balance_recover/boundary_genN.csv \
         --output-dir baseline/humanoid21/balance_recover/
     ```
   - 训练时用 `--set weight_npz_path=.../sample_weights_genN.npz`

3. **Train** — `train.py --experiment v2_weighted_impulse`
   - 用采样分布训练，warm-start from 上一轮 checkpoint
   - 框架自动导出 `policy_exports/uXXXXX`
   - 每轮最多 ~2000 updates，或超过 100 updates 没有提升则提前停止
   - 命令:
     ```bash
     PYTHONPATH=/data1/mono/things/combatbench python3 baseline/framework/train.py \
         --experiment v2_weighted_impulse --algo ppo --background \
         --set policy_blueprint_path=<latest_export>/policy_blueprint.yaml \
         --set weight_npz_path=baseline/humanoid21/balance_recover/sample_weights_genN.npz \
         --resume-from <previous_run_dir> \
         --run-name weighted_impulse_genN
     ```

## 当前进度

### Gen 0（从零开始）

- [ ] Probe
- [ ] Sample
- [ ] Train

## 关键文件

| 文件 | 说明 |
|------|------|
| `probe_boundary.py` | 边界探测脚本 |
| `sample_distribution.py` | 采样分布生成 |
| `weighted_impulse_env.yaml` | 训练环境蓝图 |
| `relative_impulse_plugin.py` | 相对角度冲量插件 |
| `exp_weighted_impulse.py` | 训练实验类（在 `baseline/experiments_v2/`） |

## 注意事项

- `policy_blueprint_path` 同时用于内部 sim（`RelativeImpulsePlugin`）和 warm-start resume
- `sample_weights` 每轮带 genN 后缀保留历史，不覆盖
- 训练自动导出 policy_exports，无需手动 export
- 每轮训练最多 ~2000 updates，或超过 100 updates 没有提升则提前停止
- 中间可能有 Bug，需持续监控训练日志
- 每次操作后、发现问题后、修复后，都在「操作日志」section 追加记录，不要覆盖以前的记录
- `exp_weighted_impulse.py` 中 `posture_key` 是死代码（已从 YAML 删除 observer），待清理

---

## 操作日志

（追加记录，不要覆盖以前的记录）

### 2026-08-12 13:15 — 实验启动

- 创建 `ITERATIVE_TRAINING_MEMO.md`
- 初代策略: `fixaw_survonly_crossphi2_s42/u00460`
- Gen 0 从零开始: probe → sample → train
