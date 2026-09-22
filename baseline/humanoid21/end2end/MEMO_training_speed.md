# 训练速度优化 Memo

> 主线任务：在 `standup_floor04` 基线上持续优化策略训练速度。
> 工作方式：**分析 → 提议 → 实现 → 训练验证 → 再分析** 的双重迭代——
> 既提升对训练动力学的理解，也迭代 Debug 系统本身。
> 本文档持续追加：每次尝试、证据、结论都记录在此。

## 任务定义

- **基线实验**: `baseline/experiments_ppo/exp_standup_floor04.py`
  （4 阶段 dense potential 站立；uncertainty_floor=0.4, coef=1.0）
- **参照 Run**: `verify_resume_A`（seed=42，u380 逃逸，1500 updates 跑满）
- **策略锁定**: `TruncatedNormalPolicy`（tanh_gaussian_mlp），不可更换。
  策略架构类实验（statesig/low-rank/MoG/RealNVP）不属于本主线。
- **Debug 系统**: `baseline/framework/ppo/dumpkit/` + viewer（localhost:8873 /
  180.76.152.227:8873）。所有分析走 dump/debug 数据流，不另建管线。

## 速度的度量（操作化指标）

| 指标 | 定义 | verify_resume_A 锚点 |
|---|---|---|
| 逃逸点 | eval.success ≥0.5 首次出现 | ~u380 |
| 收敛点 | eval.success ≥0.9 稳定 | ~u500 |
| 收敛后效率 | 早停率 / 有效 minibatch 占比 | 66–82% 早停（浪费） |

wall-clock 暂不作为主指标：rollout 占每 update 26.4s 中的 24.5s（93%），
update 数学仅 1.3s；除非改动影响 rollout 成本，优化空间在 sample
efficiency。episode 数改动除外——它直接压缩 rollout 壁钟。

## 杠杆池（已确认边界）

| 杠杆 | 参数 | 现状 | 备注 |
|---|---|---|---|
| 探索机制 | `uncertainty_floor`, `uncertainty_coef`, `explore_factor` | floor=0.4/coef=1.0；hinge 从 init U≈0.46 起咬合 | 本任务历史核心变量 |
| Episode 数 | `episodes_per_update` | 512→204K帧/update | 直接速度杠杆；与 epochs 耦合为"数据预算" |
| ADV 变换 | `adv_norm`, `adv_winsorize_sigma`, `dual_clip`, `actor_weight` | zscore 基线 | 帧级梯度方向结构 |
| KL/LR 预算 | `lr`, `critic_lr`, `target_kl`, `clip_eps`, early-stop 窗口 | lr=3e-4, tkl=0.05 | 收敛后早停浪费的主要回收口 |
| 数据复用 | `update_epochs`, `minibatch_size` | 4 epoch × 50 mb | 与 episode 数是同一数据预算的两维 |

**排除**: 策略架构替换。

## 已确立的事实（证据库）

1. **逃逸是离散事件**：平台期 max_pot ~0.66 停滞 → 某个 update 附近
   success 阶跃。非渐进爬升。
2. **Seed 方差压过臂间差异（逃逸序）**：
   - s42/s1: base > winz4 > grank
   - s2: grank > winz4 > base（完全倒转）
   → "裁 adv 尾部放慢逃逸"不是稳健效应；单 seed A/B 不可信。
3. **收敛后早停差异跨 seed 一致**：
   - gaussrank: s42=23%, s1=0%，且梯度范数维持 base 的 2–3 倍
   - base: s42=82%, s1=66%
   → ADV 秩变换让 update 持续"满负荷"，但学的是不是有用信号待定。
4. **平台期不撞 KL**：三组对比 run 逃逸前早停率均 ~1%。
   早停纯粹是收敛后现象。
5. **梯度相干度极低**：u452 截面 coherence=0.008——204K 帧逐帧
   梯度互相抵消，聚合 ‖G‖=0.544 是 mean‖g‖≈71 的残差。
   加数据买的可能主要是噪声。
6. **三组 run 代码完整性已核实**：s2 组快照树一致、配置正确、
   启动后无运行时代码污染。

## 验证协议（吸取 s2 倒转教训）

- **初筛**: 候选臂跑 seed 42，对比 verify_resume_A 曲线。
  u500 仍未逃逸 → 杀（基线 u380，给 ~30% 余量）。
- **确认**: 初筛通过的臂补 seed {1,2}；3 seed 中 ≥2 一致改进才算数。
- 8 GPU 并行，初筛可同时铺 2–3 臂。
- 新 run 必须带 `--dump-at` 排程——逃逸是瞬态，稳态截面抓不到。

## Debug 系统已知缺口

- 逃逸瞬态未被抓到（现有 dump 全是稳态截面）
- 跨 update 演化视角缺失（dump 是单 update 截面；
  verify_resume_A 的 gradsig/u*.npz 旧格式有全量 per-update 数据可先用）
- 逃逸归因：trajectory 级 ADV 贡献视图（trace_frame 是帧级）

## 进展日志

### 2026-09-22 任务启动

- 基线/边界/指标/协议确立（见上）。
- 资源：s2 对比组三条 run 已停（结论已收），GPU 0/1/2 空出。
  GPU2/3 上有两个非主线 run 在跑（step resume、statesig）。
- **下一步**: gradsig 时间序列取证——verify_resume_A 全量 per-update
  gradsig 扫 u380 逃逸点前后的 coherence/‖G‖/proj 变化；
  对照 base_from0_s2（同臂未逃逸）隔离 seed 路径差异。
  产出逃逸点梯度画像 → 定首发臂。
