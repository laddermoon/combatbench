# STEPTVAR V1 — 随机时长步令实验方案

> 状态：**已按本会话决策定稿**
> 实验名：`v2_steptvar`，首轮 `v2_steptvar_r1`

## 0. 一句话

在 stepcmd 基础上把指令周期从固定 10f 改为逐令随机
**U(1,20) 帧**（0.05s–1.0s），时长**不进入观测**。以 stepcmd
最新 ckpt 热启动，检验残差编码是否学出 ASAP 语义、半空中换令
的鲁棒性、以及到位后的驻留能力。

## 1. 动机

- **ASAP 验证**：固定周期下"掐点完成"与"尽快完成"不可分；
  随机且不可见的时长使"永远尽快"成为唯一鲁棒策略。
- **半空换令**：1f 令必完不成 → 晋升脚型常态化为动脚/轴心脚
  悬空；preview 通道从辅助变为短令唯一可操作通道。
- **驻留验证**：长令（~1s）到位后残余积分惩罚要求脚停在
  目标上——"到达"与"保持"分离检验。

## 2. 任务定义（相对 stepcmd 的增量）

- `dur_f ~ U(1,20)` 逐令在 `_issue`（晋升）时采样，存
  `sst["dur_f"]`；到期判定 `age_f >= dur_f`。
- 时长**不写 obs**（唯一鲁棒策略=ASAP）。
- 锚定语义不变：`_issue` 按实际脚位（含半空）锚世界系目标；
  preview 仍在 P̂ 系内表达。
- 调度/生成器/撞墙/保持令 P(0.1) 全部不变；5 通道与权重原样。
- 边界帧结算不变：到期帧奖励按活跃令（即新令）残差计费。

## 3. 新增诊断字段（report → observer → eval）

| 字段 | 含义 |
|---|---|
| dur | 本令时长（帧），eval 分桶用 |
| res0 | 发令时刻动脚初始残差（preview 预摆位效果度量） |
| st_z | 发令时刻轴心脚 z（保持令=双脚较大者）；>0.20 → 半空发令 |

## 4. Eval（stepcmd 全字段 +）

- 非保持令按时长分桶：`arr_s/dmv_s`（dur≤5）、`arr_m/dmv_m`
  （6-15）、`arr_l/dmv_l`（≥16）。
- `hold_err`：长令且已首达者的终末 d_mv 均值（驻留质量）。
- `air_frac`：st_z>0.20 的事件占比（半空发令频率）。
- `res0`：发令残差均值（对比 stepcmd 的固定期水位）。

## 5. 配置与看板

- env = stepcmd env 换 `V2StepTvarSimulator` + `period_range` +
  `steptvar_report` observer；5 通道原样。
- warm：stepcmd 最新 ckpt + reset-update（critic 热启）。
- **预期瞬态**：短令（≤5f）arr 必然低（物理不可达占大头），
  总 arr_rate 相对 stepcmd 的 0.90 会**结构性下移**——看
  分桶而非总值；res0 应逐降（preview 预摆位学习中）；
  长令 hold_err 应 →~0.05；falls ≤0.05。
- GPU2。

## 6. 实现清单

| 件 | 内容 |
|---|---|
| `v2_steptvar_simulator.py` | `V2StepTvarSimulator(V2StepCmdSimulator)`：`_issue` 采 dur+记 issue 诊断；`physical_step` 用 st["dur_f"]；`_settle` 报告加 dur/res0/st_z |
| `steptvar_report_observer.py` | stepcmd_report 同构 + 三个新字段 |
| `v2_steptvar_env.yaml` | stepcmd env + period_range + 新 observer |
| `exp_v2_steptvar.py` | `V2StepTvar(V2StepCmd)`：eval 加分桶/驻留/半空指标 |
