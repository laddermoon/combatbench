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

- [x] Probe: `boundary_gen0.csv` / `.json`（1920 episodes）
- [x] Sample: `sample_weights_gen0.npz`
- [x] Train: `weighted_impulse_gen0_v2`（update 455→773，best@655 survived=101 sr=78.9%）

### Gen 1

- [x] Probe: `boundary_gen1.csv` / `.json`（F=40N mean_cd=22.9，大幅提升）
- [x] Sample: `sample_weights_gen1.npz`（duration mean=11.5）
- [x] Train: `weighted_impulse_gen1`（update 771→1328，ep_len 113→156）
  - **问题**: `_best_survived=101` 从 Gen 0 checkpoint 继承，Gen 1 eval 用更难的分布无法超越
  - **修复**: 添加 `reset_best=True` 参数，新代重置 `_best_survived=-1`

### Gen 2

- [x] Probe: `boundary_gen2.csv` / `.json`（F=40N mean_cd=38.3，接近饱和）
- [x] Sample: `sample_weights_gen2.npz`（duration mean=13.9）
- [x] Train: `weighted_impulse_gen2`（update 1325→1728，best@1620 survived=62 sr=48.4%）

### Gen 3

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
| `plot_boundary_comparison.py` | 各代 boundary 对比可视化（静态 PNG + 动画 GIF/MP4） |

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

### 2026-08-12 13:26 — Gen 0 Probe + Sample

- Probe: 16方向 × 3力 × 40 duration = 1920 episodes，22.8秒
- 结果: F=40N mean_cd=1.4, F=100N mean_cd=0.4, F=200N mean_cd=0.1
- Sample: `sample_weights_gen0.npz` 生成，角度 mean=176° std=106°, duration mean=2.8

### 2026-08-12 13:27 — Gen 0 Train 启动

- `--resume-from checkpoint_u00455.pt` (fixaw_survonly_crossphi2_s42)
- run_dir: `baseline/runs/weighted_impulse_gen0_v2`
- PID=1973623
- 首次尝试 `--resume-from` 传目录报错 IsADirectoryError，改为传 `.pt` 文件成功

### 2026-08-12 13:27~14:10 — Gen 0 Train 监控

- update 455: survived=26, sr=20.3% (resume 初始)
- update 475: survived=45, sr=35.2% [new_best]
- update 525: survived=53, sr=41.4% [new_best]
- update 550: survived=62, sr=48.4% [new_best]
- update 580: survived=68, sr=53.1% [new_best]
- update 595: survived=76, sr=59.4% [new_best]
- update 650: survived=80, sr=62.5% [new_best]
- update 655: survived=101, sr=78.9% [new_best] ← 最终 best
- update 655→770: 115 updates 无新 best，满足 100 updates 无提升停止条件
- update 773 停止时: ep_len_mean=169, timeout=1208/2048 (大部分跑满)

### 2026-08-12 14:10 — Gen 0 Train 停止

- kill PID 1973623
- best policy: `weighted_impulse_gen0_v2/policy/` (update 655, survived=101, sr=78.9%)
- 总训练: 318 updates (455→773)，约 45 分钟
- 准备 Gen 1: 用 best policy 做 probe

### 2026-08-12 14:15 — Gen 1 Probe + Sample

- Probe: 用 Gen 0 best policy (u655) 探测边界
- 结果: F=40N mean_cd=22.9 (Gen 0: 1.4), F=100N mean_cd=5.5 (Gen 0: 0.4), F=200N mean_cd=1.8 (Gen 0: 0.1)
- Sample: `sample_weights_gen1.npz`，duration mean=11.5 (Gen 0: 2.8)，更难的分布

### 2026-08-12 14:17 — Gen 1 Train 启动

- `--resume-from checkpoint_u00770.pt` (gen0_v2)
- run_dir: `baseline/runs/weighted_impulse_gen1`
- PID=2766193
- 用 Gen 1 sample weights (更难分布)

### 2026-08-12 14:17~15:50 — Gen 1 Train 监控

- update 771: ep_len=113, survived=47 (初始)
- update 1085: survived=66 (Gen 1 最高)
- update 1175: survived=68 (Gen 1 最高)
- ep_len_mean 从 113→156，r_fall return=0.91，策略确实在进步
- 但 `_best_survived=101` 从 Gen 0 继承，Gen 1 eval 用更难分布无法超越 → 无 new_best
- update 1175→1325: 150 updates 无新 best，停止

### 2026-08-12 15:50 — Gen 1 Train 停止

- kill PID 2766193
- 无 `policy/` 目录 (从未触发 new_best)
- 用最新 export `u01329` 作为 Gen 2 probe 的 policy
- **发现 Bug**: 视频渲染未传 `impulse_params` → 修复 `round_runner.py` + `ppo_loop_v2.py`
- **发现问题**: 跨代 resume `_best_survived` 不重置 → 添加 `reset_best` 参数
- 准备 Gen 2: 用 Gen 1 最新 policy + `reset_best=True`

### 2026-08-12 16:20 — Gen 2 Probe + Sample

- Probe: 用 Gen 1 最新 policy (u01329) 探测边界
- 结果: F=40N mean_cd=38.3 (Gen1: 22.9), F=100N mean_cd=7.8 (Gen1: 5.5), F=200N mean_cd=3.3 (Gen1: 1.8)
- F=40N 13/16 方向达到 40（满分），接近饱和
- 持续弱点: 247.5°~270° (侧后方) 在所有力度下最弱
- Sample: `sample_weights_gen2.npz`，duration mean=13.9 (Gen1: 11.5)

### 2026-08-12 16:25 — Gen 2 Train 启动

- `--resume-from checkpoint_u01325.pt` (gen1)
- `--set reset_best=True`
- run_dir: `baseline/runs/weighted_impulse_gen2`
- PID=4143802
- reset_best 生效: update 1325 survived=42 立即触发 [new_best]

### 2026-08-12 16:25~17:10 — Gen 2 Train 监控

- update 1325: survived=42 (初始, new_best)
- update 1350: survived=52 (new_best)
- update 1435: survived=54 (new_best)
- update 1465: survived=55 (new_best)
- update 1565: survived=56 (new_best)
- update 1605: survived=58 (new_best)
- update 1620: survived=62 sr=48.4% (最终 best)
- update 1620→1725: 105 updates 无新 best，超过 100 阈值

### 2026-08-12 17:10 — Gen 2 Train 停止

- kill PID 4143802 (旧代码无 early stop，手动停止)
- best policy: `weighted_impulse_gen2/policy/` (update 1620, survived=62, sr=48.4%)
- 总训练: 403 updates (1325→1728)，约 45 分钟
- Gen 2 有 policy/ 目录（多次触发 new_best）
- 准备 Gen 3: 用 Gen 2 best policy 做 probe
