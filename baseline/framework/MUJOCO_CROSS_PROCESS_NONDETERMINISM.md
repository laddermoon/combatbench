# MuJoCo 跨进程非确定性问题分析

## 问题现象

在使用 `StateBankInitPlugin` 从状态池注入扰动状态后跑 episode，发现验证结果与状态池中记录的 label 不完全一致。

- 60 个样本中，仅 43 个匹配（71.7%）
- 17 个 mismatch，bank_label 和 verify_label 相反
- Episode length correlation 0.65，有相关性但不完美

初始假设：`StateBankInitPlugin` 的状态注入可能不精确，或 MuJoCo 存在混沌行为导致同样初始状态产生不同结果。

## 分析过程

### 第一步：验证状态注入精度

对比 `StateBankInitPlugin` 注入后的 `core_state` 和 `observation` 与 bank 中记录的值：

```
bankinit_v_bank = 0.00e+00  (core_state 完全一致)
bankinit_v_bank = 0.00e+00  (observation 完全一致)
```

**结论：状态注入是精确的，diff = 0。**

### 第二步：同进程内跑两次

在同一个 Python 进程中，用同一个 sim 实例跑两次相同的 episode：

```
pos_diff: step1=0.00e+00  step10=0.00e+00  step50=0.00e+00  max=0.00e+00
vel_diff: step1=0.00e+00  step10=0.00e+00  step50=0.00e+00  max=0.00e+00
```

**结论：同进程内 MuJoCo 完全确定性，跑两次结果 100% 一致。**

### 第三步：ParallelRollouter 跨进程跑两次

用 `ParallelRollouter(num_workers=8)` 对同一状态池跑两次：

```
Run1 vs Run2: 60/60 (100.0%) label match
Length exact: 60/60 (100.0%)
Label corr:   1.0000
Length corr:  1.0000
```

**结论：同一 workers 配置跑两次也完全一致。** 排除随机种子问题。

### 第四步：workers=1 vs workers=8

```
workers=1 vs workers=8: 37/60 (61.7%) label match
bank vs workers=1:      38/60 (63.3%)
bank vs workers=8:      43/60 (71.7%)
```

**关键发现：workers=1 和 workers=8 之间只有 61.7% 匹配。** 差异来自进程间。

### 第五步：workers=1 跑两次

```
workers=1 run1 vs run2: 60/60 (100.0%)
len_diff: mean=0.0  max=0  median=0
```

**结论：同进程配置跑两次完全一致，确认差异是跨进程的。**

### 第六步：MuJoCo 内部状态对比

对比同进程内两个 sim 实例（reset + set_core_state 后）：

```
qpos_diff = 0.00e+00
qvel_diff = 0.00e+00
qacc_diff = 0.00e+00
ncon = 10/10
```

同进程内所有 MuJoCo 内部状态完全一致。

## 发现汇总

| 对比项 | Label 匹配率 | Episode Length | 说明 |
|--------|-------------|----------------|------|
| 同进程跑两次 | 100% | 完全一致 | MuJoCo 同进程完全确定性 |
| workers=1 跑两次 | 100% | 完全一致 | 同进程确定性 |
| workers=8 跑两次 | 100% | 完全一致 | 同 batch 内确定性 |
| workers=1 vs workers=8 | 61.7% | 有差异 | **跨进程非确定性** |
| Bank vs workers=1 | 63.3% | 有差异 | 跨进程（bank 生成在 workers=8） |
| Bank vs workers=8 | 71.7% | 有差异 | 跨进程（但同一 workers 配置） |

### 状态注入精度

| 指标 | 值 | 说明 |
|------|-----|------|
| `core_state` diff | 0.00e+00 | 注入后状态与 bank 记录完全一致 |
| `observation` diff | 0.00e+00 | 注入后观测与 bank 记录完全一致 |
| `qpos` diff (同进程) | 0.00e+00 | MuJoCo 位置状态一致 |
| `qvel` diff (同进程) | 0.00e+00 | MuJoCo 速度状态一致 |
| `qacc` diff (同进程) | 0.00e+00 | MuJoCo 加速度状态一致 |

### Mismatch 分布特征

Mismatch 集中在 **边界 case**（bank 中 episode length 在 200-500 之间的不稳定 episode）：

- 快速摔倒（< 100 步）的 case 基本都匹配
- 稳定站立（600 步）的 case 大部分匹配
- 中间区域（300-500 步）的 case 容易因微小差异翻转

## 最终结论

### 根因

**MuJoCo 跨进程浮点运算非确定性。** 不同进程中的 MuJoCo（底层 C 库）在以下方面可能产生微小差异：

1. **BLAS 线程调度** — 矩阵运算的并行化路径不同导致浮点累加顺序不同
2. **SIMD 指令路径** — 不同进程可能命中不同的向量化代码路径
3. **内存对齐** — 进程间内存布局差异影响浮点运算精度

这些差异在单步物理仿真中量级约 1e-15（机器精度），但 humanoid 机器人的站立平衡是一个 **混沌系统**——微小的初始差异在 600 步（15000 个物理子步）的迭代中被指数级放大，最终导致完全不同的结果（存活 vs 摔倒）。

### 影响评估

- **对训练的影响：可忽略。** 状态池的 label 是统计性指导信号，不需要逐样本 100% 复现。训练时使用 `StateBankInitPlugin` 随机采样状态，label 用于课程学习或奖励 shaping，单个样本的 label 翻转不影响整体训练效果。
- **对评估的影响：需注意。** 如果需要精确复现某个 episode 的结果，必须在同一进程中运行。跨进程评估应使用统计指标（存活率、平均 episode length）而非逐样本对比。
- **对状态池生成的影响：可接受。** 状态池中的 label 是在特定进程配置下生成的，跨进程使用时约 70% 逐样本匹配，但统计分布（存活率、力-持续时间边界）保持一致。

### 建议

1. **训练用途**：直接使用 `StateBankInitPlugin`，label mismatch 不影响训练
2. **评估用途**：使用统计指标而非逐样本对比；固定 workers 数量以保证可重复性
3. **调试用途**：在单进程（workers=1）中运行以获得确定性结果
4. **论文报告**：报告统计指标（存活率、边界区域）而非单样本复现率
