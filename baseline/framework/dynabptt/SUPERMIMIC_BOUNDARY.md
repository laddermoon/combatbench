# SuperMimic 方法解析与适用边界地图

> 目标（用户定义）：**必须攻克 SuperMimic 式训练**。核心交付物是搞清楚它的
> **适用边界**——什么条件下有效、什么条件下失效——从而能正确利用它，
> 不再闭门造车。

资料来源：
- 论文 arXiv:2610.28480v3（CoRL 2026 Oral，Berkeley Hybrid Robotics）
- 官方实现 `github.com/HybridRobotics/supermimic`（Apache-2.0, mjlab/MuJoCo-Warp）
- 独立复现 `CuboiLeo/SuperMimic-Reproduced`（4×RTX5090 7.1h, BONES-SEED 0.930/0.11%失败）
- 其他复现：`zita-ch/supermimic_vibed`（ETH, Isaac Lab 3）、`Breezewrf/supermimic_mjlab`
  （A6000, LAFAN 40/40）、`Baojch/SuperMimic_Dyn`、`kangzejian1896/supermimic-g1-reproduction`
- 本地实测：`hybrid_ppo/` 实验 + `HYBRID_PPO_DETAIL.md`

## 一、官方实现的真实机制（读源码确认，非仅凭论文）

### 1.1 状态表示工程 —— 最被低估的核心

官方代码 `representations.py` 提供两种表示，二者共享同一设计哲学：

**网络只预测速度级增量，其余全部解析积分。**

GenCoord（官方默认对照配置）：

```
full state : [root_pos(3), root_quat(4), qj(N), v_b(3), w_b(3), vj(N)]
rep state  : [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]     # 航向不变
prediction : [Δv_b(3), Δw_b(3), Δvj(N)]                      # 仅 35 维
```

MaxCoord（仿 ControlVAE，逐 body 笛卡尔）：

```
rep state  : [local_pos(3B), local_rot6d(6B), local_vel(3B),
              local_avel(3B), root_h(1), up_dir(3)]           # 全部在根 body 系
prediction : [Δvel(3B), Δavel(3B)]                            # 仅速度增量
```

积分规则（`apply_prediction`，半隐式）：

```
v_{t+1}  = v_t + Δv                         # 网络输出，全量更新
qj_{t+1} = qj_t + (vj_t + c·Δvj)·dt         # 位置由速度解析积分
root pose由 twist 解析积分（quat_integrate）
z, g_b   由积分后位姿解析重导出              # 不预测
c = substep_coeff = (decimation+1)/(2·decimation)  # 对齐仿真子步合成
```

**这意味着：**

- 网络只学"冲量/加速度映射" Δv —— 接触冲击这类最难学的量被隔离在
  **有界的速度通道**里（冲量/质量·dt），不会污染位置预测
- 位置、旋转、关节角的下一时刻值**零模型误差**（解析积分精确）
- 位置误差只能通过速度误差单级传导，不存在位置预测自身的独立复合误差
- **帧不变性**：GenCoord 丢掉 x,y,yaw；MaxCoord 一切量在根 body 系——
  模型永不在世界系预测，yaw 漂移不会造成无界输入

对照我们的实现：`SelfWorldModel` 直接预测全部 55 维下一状态（残差式），
位置误差与速度误差在 learned map 内互相污染。

### 1.2 损失设计

- 损失**计算在积分后的状态块上**（root pose、joint pos、joint vel / body
  pos、rot、vel、avel），不是原始 delta
- **Bryson's rule 分组加权** 1/nominal²：root_pos 0.01m、root_ori 0.02、
  joint_pos 0.1、joint_vel 3.14、body_pos 0.05、body_vel 1.0、body_avel 3.14
- 论文报告的"8 步预测误差 0.007–0.011"就是按这些 nominal error 归一后的值
- k=8 自回归窗口，10 批 × 2048 窗口/update；经验归一化器每 iteration
  用新采集数据更新（只归一 rep_state 与 proc(action)）

### 1.3 策略与想象

- 想象 H=32 > 拟合 k=8：k 步拟合训练模型"自我纠错"，因此长程稳定
- 想象内 policy **确定性**，但给预测 obs 重加传感器噪声 η~ρ_η（观测噪声
  是已知外源——模型只预测无噪 obs，噪声在外面采样加回）
- 软生存权重 ĉ_t=∏(1-d̂_t) 替代硬 done，保持可微
- critic 无 EMA target，GAE λ=0.95 只在想象轨迹内计算，5 epoch × 4 mb/update
- 动作噪声 σ=0.1 **只为拓宽模型拟合的动作覆盖**，replay 存的是加噪后的
  执行动作（模型在真实执行分布上拟合）
- reward 头**解析**（跟踪奖励从参考运动可算），termination 头学习；
  obs 头学习无噪 obs

### 1.4 消融里已有的对照（论文 Fig.5b）

PPO-WM（模型 rollout + PPO 式更新）**可以达到同样最好回报，但需要
~64× 更多想象步数**。即用户的混合路线本身被官方背书可行——前提是
模型足够准。

## 二、适用边界地图

### ✅ 有效条件（论文+复现实测）

| # | 条件 | 证据 |
|---|---|---|
| B1 | 任务动力学可被"平滑加速度映射"近似：冲击要么温和，要么在分布内规律重复（步态触地在固定相位） | BONES-SEED 含侧滚/翻腾等 contact-rich 动作，仍 0.007-0.011 误差——但都是**训练分布内的规律冲击** |
| B2 | 每步有稠密奖励信号（逐帧全身跟踪误差） | 论文："dense reward that motion tracking gives at every step"是他们认为训练极稳的原因 |
| B3 | 数据分布贴近当前策略（FIFO on-policy replay；模型跟策略走） | replay 仅 1000 步/env，UTD 0.4 |
| B4 | 状态可表示为 (s,a) 的马尔可夫映射——所有不连续源必须要么在输入里（ξ条件化），要么不存在 | ξ 显式喂摩擦/标定/质心 |
| B5 | 海量 on-policy 数据吞吐支撑模型追策略 | 4096 GPU envs vs 我们 ~120 CPU worker |
| B6 | 学习目标状态在**根 body 系/航向不变系**表示 | CuboiLeo 实测：纯本体观测策略 yaw 漂移 ~1.7°/s，9s 后世界系奖励全失效 |

### ❌ 失效条件（实测+论文自认）

| # | 失效条件 | 证据 |
|---|---|---|
| F1 | **外生状态不连续**（推力/传送不在 (s,a) 输入中） | 论文显式移除 BeyondMimic 的 root-velocity 扰动，原话 "such pushes inject state discontinuities that the deterministic world model cannot capture" |
| F2 | **冲击主导且不可预测的动力学**——每步转移的 |Δv| 大方差且不能从 (s,a) 推断 | 我们 standup 实测：67% 转移 \|Δv\|>0.3m/s，85% 平方误差集中于此；4×容量+20×训练后留出 NRMSE 仍 0.50（阈值 0.20） |
| F3 | **物体/地形交互** | 论文 limitation 自认 "No terrain or object interaction"；BONES-SEED 过滤掉所有楼梯/箱子/椅子 clip |
| F4 | **长时程无全局反馈的漂移** | CuboiLeo 实测：LAFAN1 整 clip 220s，纯本体策略 xy 漂移 6.3m/yaw 82°——不是模型问题，是观测设计边界 |
| F5 | 稀疏奖励 + 信用期 > H | 我们 standup stage-4：J 平坦→BPTT 梯度退化为噪声→REJ 冻结（dynabptt 实测） |
| F6 | 多智能体耦合（对手动作影响自身状态但不在输入） | 我们正是为此做了碰撞位掩码隔离；格斗交互场景本质是 F2+F6 叠加 |

### 边界机理的统一解释

```
模型可学性 ∝ 每步转移中"可从 (s,a,ξ) 平滑推断的成分"占比
```

- 冲击**不可怕**，可怕的是**不可推断的冲击**：步态触地在同一相位反复出现，
  有条件规律可学；倒地起身中被碰撞点/时机主导的冲击近乎随机——
  这就是 F2 与 B1 的分界线，也是我们和 SuperMimic 的分界线
- 速度增量输出+解析积分把这个问题切成两块：**不可学的留在 Δv（有界），
  可学的（位置/几何）零误差**——他们最多只能错在"冲量估计"上，
  我们错在所有维度上
- 所以适用边界不是"有没有接触"，而是 **"每步不确定冲量占动力学信号的比例"**

## 三、对我们的迁移清单

按杠杆效率排序（越大越优先）：

| # | 改动 | 对付的问题 | 代价 |
|---|---|---|---|
| T1 | **模型输出改 Δv-only + 解析积分**（GenCoord 式 rep：z+g_b+body-frame 速度+关节状态） | 位置误差复合、误差跨通道污染 | 中：重写 model.py forward + transitions 状态布局 |
| T2 | **动作平滑层**（低增益 PD setpoint 或动作滤波 a'=λa+(1-λ)a_prev） | 单步冲击幅度 | 低：blueprint/obs 加 prev_action，ctrl 映射改 setpoint |
| T3 | **更细控制步** 30ms→15-20ms | 摊薄每步不连续 | 低-中：env 配置 + 序列步数×2 |
| T4 | **帧不变表示**（去 x,y,yaw；速度进 body 系） | 无界输入、yaw 漂移 | 随 T1 一并做 |
| T5 | **Bryson 分组损失权重** | 损失被大量纲通道主导 | 低：改 fit_model |
| T6 | **ξ 条件化**（摩擦/质心等 episode 参数入模型） | 域随机化摊平 | 中：blueprint 需暴露参数 |
| T7 | 缩小动作噪声到 σ≈0.1-0.15（探索另想办法） | 模型在过宽动作分布上稀释 | 低 |
| T8 | 稠密化奖励（若可能给跟踪式逐帧目标） | BPTT 信号厚度 | 取决于任务 |

### 开放问题（边界待实测）

1. Δv-only + 解析积分后，standup 的冲击帧留出误差能降多少？
   （我们测的是全维直接预测；新架构只错在 21+6 维）
2. 我们的任务（倒地起身）本质上是不是 F2 区域——即冲击在 (s,a) 下
   是否条件可推断？这决定该任务是否**原则上**可学
3. 若 standup 不可学：转换到"可学区"的最近任务是什么
   （如单腿平衡、有参考轨迹的起身分解）

## 四、外部资源清单

| 资源 | 价值 | 链接 |
|---|---|---|
| 官方实现 | representations.py 是表示工程的完整参考（FK/IK 解析积分、帧变换） | github.com/HybridRobotics/supermimic |
| CuboiLeo 复现 | 最有价值的是 docs/EXPERIMENTS.md：漂移诊断、e5/e8 全程曲线、bug 清单——含大量论文没写的实操细节 | github.com/CuboiLeo/SuperMimic-Reproduced |
| supermimic_vibed | ETH 复现，Isaac Lab 3 无 mjlab 依赖，明确写"只预测 35D 加速度增量+解析积分" | github.com/zita-ch/supermimic_vibed |
| supermimic_mjlab | A6000 单卡 LAFAN 40/40；其 state_delta 变体（预测 70D 位姿+速度增量）可做架构对照 | github.com/Breezewrf/supermimic_mjlab |
| mjlab 本体 | MuJoCo-Warp GPU 批量仿真的封装层，吞吐是关键差异源 | github.com/mujocolab/mjlab |
| 改进方向 | ControlVAE（MaxCoord 表示来源）、SHAC（硬终止→ĉ 的出处）、PWM/MAAC（同族方法） | 见论文 related work |

## 五、当前判定

混合 PPO+WM 线在 standup 的失败**不再归因于"学习型世界模型不可行"**，
而精确归因于：**我们的状态表示+预测目标设计把不可学的冲击误差扩散到了
所有通道**。SuperMimic 的答案不是"模型更大"，而是**把可学与不可学的
部分在表示层切开**。T1-T5 都是廉价实现，值得在 hybrid_ppo 框架上重做
一版验证——若 Δv-only 表示后留出误差进入可用区，整条混合路线立刻复活；
若仍不行，则为"standup 属 F2 区"的直接证据，按边界地图换任务或换方法。
