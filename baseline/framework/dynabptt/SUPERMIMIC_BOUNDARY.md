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
- reward **解析**（`compute_wm_reward_from_reference`：预测状态 vs 参考
  clip 逐帧算跟踪项），termination 也**解析**——`_compute_term_prob`
  用任务自身阈值规则（anchor 高度差 / tilt 差）经温度 sigmoid 软化：
  `σ((normalized_error - 1) / softness)`。官方**没有任何学习式
  reward/done 头**——世界模型只学 35 维 Δv 动力学
- obs 头学习无噪 obs（sensor 模式）；actor 也可用 rep_state 直读
  （proprio 模式）
- critic 输入 = [rep_state, wm_goal_obs, sim_ctx]，V 读**模型状态**
  而非重建 obs——这是我们 RepCritic 的直接出处，实测把配对
  advantage 相关性从 0.35（obs critic）提到 0.95
- 策略超参（policy_learner.py 默认值）：lr 2e-4、Adam β2=0.95、
  imag batch 4096、H=32、γ=0.99、λ=0.95、norm-clip 1.0 后
  elem-clip 0.1、每 update 一个想象批一步 actor step
- 训练节流（supermimic_runner）：`should_train = Ratio(num_envs /
  (wm_batches×wm_batch×k) × replay_ratio=16)` → 每 env step 0.4 轮
  "10×2048×k8 模型批 + 1×4096×H32 策略批"

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

## 五、迁移实施结果（2026-11-26 更新）

T1-T5 已全部在 hybrid_ppo 落地并实测（GenCoord port + dt=20ms +
Bryson 损失 + 帧不变 rep；T2 动作平滑未做——动作本身已是 PD 目标）：

| 杠杆 | 效果 |
|---|---|
| dt 50ms→20ms | 留出 s1_lv NRMSE 0.59→0.40（最大单一杠杆；冲击 \|Δv\| 中位数 0.37→0.27） |
| Δv-only+解析积分 | 中性（本身不增益，但结构上必需） |
| 数据量 320k→1M | 0.40→0.35 |
| 容量 512³→1024³ | 0.40→0.35（2048³→0.31，递减） |
| **rep-space critic** | **配对 adv_corr 0.35→0.95、sign 0.98、rel_rmse 0.32**——辅助路径直接判据达标 |

**关键机理复核**：瓶颈不是状态精度而是 critic 读哪——V(ô) 在预测 obs
上的 bootstrap 噪声 ≈ adv_std 本身；V(rep) 读模型隐状态直接绕过。
这正是官方 critic 输入 [rep,goal,ctx] 的设计意图。

**当前判定**：
1. **P2 A/B 收官（standup, dt20, 45 update, seed 42）**：assist 门控自
   u1 即准入（paired adv_corr 0.94-0.98），但 eval max_pot 上 B 臂
   中段持续落后 A 臂 ~0.03-0.05，终点持平（B u44: 0.346/0.363 vs
   A u39: 0.343/0.356）——**模型辅助在 F2 区任务上为中性略负**：
   advantage 估计本身准确，但 α=0.25 的辅助信号稀释了 PPO 自身梯度。
   结论：不是机制错误，是**任务错配**——这正是"standup 属 F2 区"
   的行为级证据，混合 PPO+WM 实验线到此为止。
2. 主攻线 = **smbptt**（SuperMimic 原生 BPTT 复刻）在 balance 任务上：
   想象 J、alive、真实 episode len 同步爬升（u0→u50：len 55→~70），
   官方参数对齐（batch 4096、H=32、γ=0.99 λ=0.95、β2=0.95、clip
   1.0+0.1）。与官方的两个有意偏差：done 用**学习头**（官方为解析
   阈值 sigmoid——balance 的"非足接触"无解析式，学习头已校准：
   alive≈0.56 vs 数据隐含 1-32/70≈0.54）；obs 为 55 维解析重构 +
   2 维学习足部力。

## T6 决定性对照：PPO@同环境 vs BPTT@同环境（u280 结论）

**实验设计**：`basic_balance_dt20` PPO 实验 = 与 smbptt 完全同环境
（同蓝图、phy_steps=10/dt=20ms、tolerance=1、max_steps=1500、同
r_fall/r_cross 奖励结构；唯一差异是双智能体 obs=96 vs self-only 57 —
— 若说有影响，双机碰撞风险使 PPO 侧**更难**，不影响结论方向）。

**结果**（run `ppoc_dt20_balance`）：

| update | u1 | u21 | u140 | u170 | … |
|---|---|---|---|---|---|
| ep_len_mean | 74 | 83 | 166 | 359 | 持续线性爬升，未减速 |

vs **smbptt BPTT**（v5–v9 五种配置、~270 update）：len 56→~80 后
平台 150+ update 不动。尝试过：critic 索引修正、r_cross 通道、
噪声想象（σ 对齐 0.37）、tail-frac 窗口过采样、brink seeds、
官方规格 4096 seeds/H32/imag8——全部落在同一 80 平台。

**结论**：任务本身完全可学（PPO 真实-rollout 信号无障碍通过 80 并
冲向 400+），瓶颈**不是任务难度而是世界模型的想象保真度**。

## T7 机理定位：悲观终止循环（pessimistic-termination loop）

对 v9 ckpt 的配对诊断（512 brink seeds × H=32 想象）：

- 真实数据：φ<0.85 的轨迹中 **14.6%** 能再存活 ≥20 步（存在真实恢复）
- 模型想象：dip 进 φ<0.85 后 **73%** 被 done head 判死，仅 **0.5%** 恢复到 φ>0.95

模型**没有伪造恢复**（先前假设证伪）——它是**悲观**的。机理：

> 学习式终止头从"当前策略的死亡统计"学到「dip→死」，而非「不可恢复
> →死」。策略在恢复区越弱，数据里 dip→done 的比例越高，模型就越
> 悲观；越悲观的想象越教不出恢复技能——**循环锁定**。PPO 不受影响：
> 真实 rollout 里 14.6% 的幸运恢复产生**真实的正 advantage**，直接
> 教会恢复动作。

**边界地图修正**：单步精度（h1 NRMSE 0.088 已达标）**不是**充分条件。
真正的约束是 **off-manifold 价值真实性**——当策略在某区域不胜任时，
那里的想象动力学继承了采集策略的失败统计，想象梯度无法传授当前策略
没有的技能。SuperMimic 不撞这堵墙是因为：稠密 tracking 奖励使策略
快速变强 → 想象始终在流形内 → done 极少点火。

## T8 检验中的假设：teacher 数据注入

若上述机理成立，则**把胜任策略的数据喂给模型**应解锁 BPTT：
模型学会"tilt+修正动作→恢复"的转移后，想象中恢复成为可能，
BPTT 即可经解析梯度教出恢复技能（actor 仍纯 BPTT 训练，teacher
只进 replay buffer）。

实验（`sm_balance_v10`）：每 25 update 用 PPO 对照组最新导出策略在
标准 dual env 收集 512 条轨迹（obs 切前 57 维对齐 self-only 管线），
注入 replay。判据：len 突破 ~90 即机理坐实；若仍卡 80，则说明
off-manifold 问题比"数据覆盖"更深（可能是动作条件化错误——模型
只在 teacher 动作分布下正确，对 learner 的动作插值失真）。

## T9 判定性分解：开环完美 / 闭环经 obs 通道腐蚀（u300+ 结论）

T8 teacher 注入 + BC 热启（PPO 蒸馏策略真实 len=1477）后，BPTT 在
**3 个 update 内把胜任策略毁回 len~60**（v11/v13/v14 三连复现，
含 analytic done + 真实数据接地 critic 的 v14 同样崩）——这排除了
「终止头悲观」和「critic 被想象误差污染」两个假设。分解实验定位
真病灶：

| 环节 | 实测 |
|---|---|
| 模型开环跟踪（真实 teacher 动作序列，H=32） | φ MAE **0.003**，零 dive ——动力学本身近完美 |
| obs 组合误差（feet 力头） | MAE 0.371（通道 std 0.57）≈无信息 |
| obs 组合误差（其余 55 维） | MAE 0.083 |
| obs 误差诱导的动作偏差 | 随想象步长增长至 ~0.10 ≈ actor σ(0.12) |
| 闭环想象（actor 自己的动作） | **41% 轨迹 32 步内 φ<0.5**（真实≈0） |
| feet 通道置零对照 | J 无变化——feet 不是主犯 |

**机理终版**：误差不在动力学而在 **obs→action 反馈环**——想象每步
喂给 actor 的组合 obs 带 ~0.08 系统性偏差，动作被逐步推离其训练
分布，想象状态随即离流形。BPTT 的 ∂J/∂θ 穿过这个失真回路，把策略
拽向"模型以为对的方向"=回归数据分布，与真实收益反相关。

**官方早已内置对此的防御**（读源码发现）：`compute_actor_sensor(
noisy=True)` 给想象每一步的 actor 输入加逐维均匀抖动
（projected_gravity ±0.05、ang_vel ±0.2、joint_pos ±0.01、
joint_vel ±0.5），历史窗 warmstart 同样加噪。策略永远学不到依赖
精确 obs 值 → obs 组合误差无法腐蚀反馈环。我们此前所有 run 的
想象 obs 都是确定性的——actor 可以过拟合模型的精确 obs。

**当前验证**：v15（官方规格 + obs-noise=1.0）/ v16（再叠 analytic
done）从零开跑，判据仍是 len 突破 ~90。

## T10 社区复现横向对比（成立条件细化）

| 实现 | 环境/物理 | 结果 | 关键设计 |
|---|---|---|---|
| 官方 supermimic | MuJoCo-Warp 4096 env | 成功（paper） | rep_state/proprio obs + 均匀抖动；解析 tracking 终止；ref-anchored 稠密奖励 |
| CuboiLeo/SuperMimic-Reproduced | mjlab/MuJoCo-Warp | **0.904** @LAFAN1-40（anchor-relative obs）；paper-faithful obs 平台 0.56 | 全程训练日志+bug 清单；诊断出纯本体 obs 下 root 漂移 1.7°/s 是平台主因 |
| zita-ch/supermimic_vibed | IsaacLab/MJWarp | eval 0.829 @it14000（1.5h） | v2 加速版（只预测加速度）0.842/0.95h |
| Breezewrf/supermimic_mjlab | mjlab | 短跑不升（100 轮 imag-return 健康但 real-return 0.0017） | **明确警告：勿以想象回报判学习**——与我们 J 虚高现象互证 |

**提炼的成立条件交集**（三家成功复现共有、我们缺的）：
1. **参考系锚定**：跟踪任务中目标位姿每步确定性推进，actor 永远
   知道"该在哪"——想象离流形时有显式信号拉回；我们的平衡任务无
   参考，离流形只能靠自己爬回来
2. **obs 抖动**：官方+成功复现都保留；我们直到 v15 才加上
3. **终止解析化**：全部是 tracking-误差规则，无一用学习式终止
4. **想象回报≠真实回报**：mjlab 复现的 100 轮教训与我们 J=1.2
   虚高/真实 0.76 完全同构
