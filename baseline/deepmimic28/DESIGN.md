# DeepMimic28：MimicKit DeepMimic 复现实现说明

目标：在**我们的**仿真框架/PPO框架/实验框架中，用 **Xue Bin Peng MimicKit**
的方法学精确复现 DeepMimic 单 clip 跟踪（humanoid28 + spinkick），
借复现验证 humanoid28 仿真器正确性，并做单 critic vs 多 critic 对照实验。

## 1. 方法来源（全部取自上游 repo 实际代码/配置）

- env 配置 `data/envs/deepmimic_humanoid_env.yaml`
- agent 配置 `data/agents/deepmimic_humanoid_ppo_agent.yaml`
- 实现 `mimickit/envs/deepmimic_env.py` / `char_env.py` / `learning/ppo_agent.py`

## 2. 精确复刻项（数值验证过）

| 项 | 上游值 | 验证 |
|---|---|---|
| 奖励 | w=[.5,.1,.15,.1,.15] scale=[.25,.01,5,1,10]，per-joint 权重 14 组 | RSI 落位 reward=1.000000 精确 |
| key_pos 误差 | 全键位**求和**（非均值），root 相对 | 逐行对照 compute_reward |
| obs 464d | char_obs(140)+tar_obs(3×108)，tar_steps=[1,2,3] | 维度断言 |
| tar_obs root | global_obs 下相对**受控角色** root，z 为参考绝对高 | 对照 compute_deepmimic_obs L689 |
| 终止 | 非 whitelist(双脚) 身体接触力>0.1N；全部 15 身体世界系位置偏差>1m（含 root）；timeout 10s | 对照 compute_done |
| RSI | 均匀相位采样，参考状态(含世界系速度)落位 | FK 链式闭环 0.0° |
| 动作 | 归一化空间策略输出 → center+half×a_norm → clip 扩展限位 → 弧度 PD 目标 | 对照 base_agent normalizer |
| 策略 | FC 2×1024 无界 Linear 均值，FIXED std=0.05（归一化单位），init_output_scale=0.01 | GaussianMLPPolicy（框架已有） |
| PPO | td_lambda=0.95 clip=0.2 discount=0.99 norm_adv_clip=4.0(=adv_winsorize) epochs=5 lr=1e-4 | experiment.py |
| 控制/物理 | 20Hz 控制 / 500Hz 物理 | humanoid28 决策#6 |
| 数据 | humanoid_spinkick.pkl 原文件，78帧@60fps WRAP，零重定向 | 与上游同 robot |

### 旋转数学验证（地基）
- MuJoCo 串联 hinge 组合 = qx⊗qy⊗qz（内旋 XYZ，scipy 数值验证）
- expmap→hinge 分解须用 scipy **外旋 'XYZ'**（roundtrip 误差精确 0）
- hinge 轴：elbow=z，knee=y（逐关节读取）
- qvel[3:6] 角速度 = **body-local** 系（mj_integratePos 数值验证）；
  MimicKit 用世界系——obs/RSI 均已转换
- 转换层 FK 链式闭环（15帧×6身体+非整数时间）误差 0.0°

## 3. 框架差异（诚实清单——框架不支持则不加，用户决策）

| # | 差异 | 上游 | 我们 | 影响评估 |
|---|---|---|---|---|
| D1 | 优化器 | SGD lr 1e-4 | **Adam** lr 1e-4（框架硬编码） | Adam 更激进，早期 KL 大；量级相近 |
| D2 | obs 归一化 | running mean/std clip=10 | **无**（框架无此机制） | 早期学习慢一些；特征尺度天然有界（tannorm/速度） |
| D3 | epochs 分离 | actor 5 / critic 2 | 统一 5（框架无分离） | critic 训更多——通常无害 |
| D4 | action_bound_weight=10 | 出界惩罚进 loss | **无**（用户决策：不加框架扩展） | 动作更频繁触界，由 env clip 硬承担 |
| D5 | 探索混合 | sample/mode bernoulli 混合 | 纯 sample（框架机制） | 探索噪声略大 |
| D6 | minibatch 组织 | batch 4/2 份 | minibatch_size 4096 | 框架差异，批量更大方差更小 |
| D7 | rollout 组织 | steps_per_iter=32×N envs | episodes_per_update=256 | 每次 update 数据量不同，不可直接比 update 数 |
| D8 | 终止豁免 | 首帧即判 | 第 1 控制步豁免（接触力稳定） | 1/20s 差异，宽松一丝 |
| D9 | 接触检测 | Isaac net contact force per body | MuJoCo 逐 contact 对力>0.1N，排除自碰撞 | 语义近似，灵敏度接近 |
| D10 | 熵正则 | 无 | uncertainty floor 0.3 coef 0.01（框架 ExplorationSpec） | 轻微推大 std——std 已 FIXED 0.05，实际无效 |
| D11 | ref 鬼影 | 可视化参考角色 | 无（纯跟踪，不渲染 ghost） | 仅可视化差异 |
| D12 | 稳定机制 | SGD 小步 + action_bound_loss 使 Imp_Ratio~1.0005 | **actor lr 1e-5**（10×降速模拟上游小步 regime；critic 保持 1e-4）+ minibatch 512，无 KL stop | 实测：Adam lr1e-4 单步 mean 位移 0.004 在 σ=0.05 下 ratio 爆炸至 1e6（分布敏感性固有）；lr1e-5 后 epochs 跑满、ep_len 60u 内 7.7→10.2 缓慢爬升 |

| D13 | RSI 时间采样 | 连续均匀 | 60fps 帧网格量化 | 帧对齐后全部 ref 查询精确落帧可预计算（吞吐 2.7×）；上游原版 DeepMimic 本就离散帧采样 |
| D14 | 激活函数 | ReLU（checkpoint 实证） | Tanh（框架 GaussianMLPPolicy 硬编码） | 网络动力学特性差异 |

## 4. 实验设计：单 critic vs 多 critic

同一奖励的两种结构化方式（框架 multi-critic 能力验证）：

- **single**：r_track = 0.5·pose+0.1·vel+0.15·root_pose+0.1·root_vel+0.15·key_pos（他的方式）
- **multi**：5 通道各自独立 critic，actor_weight = 对应权重
  （差异仅在 value 估计结构；actor 看到的加权 advantage 期望一致）

控制变量：同 seed 序列、同 PPO 超参、同 obs/动作/终止。

## 5. 文件

```
baseline/deepmimic28/
├── refmotion.py    # MimicKit pkl → quat/hinge 参考库（0.0° FK 验证）
├── mathutil.py     # numpy quat 工具（xyzw）
├── sim.py          # RSI + 相位 + 464d obs + 归一化动作层
├── plugins.py      # 奖励观测器（五分量）+ 终止插件
├── experiment.py   # multi_critic 开关 + MimicKit 超参
├── zero_policy.py  # robot_b 陪衬
├── envs/track_env.yaml / blueprints/zero.yaml
├── train.sh / eval.py
└── data/motions/   # 20 条上游 humanoid pkl
```

## 6. 已知风险

- Adam + 小 std(0.05) 早期 KL 大——smoke 见 kl 量级 1e7（ratio 爆炸），
  但 clip 0.2 + winsorize 4.0 限制单步破坏；若发散，备选：开 target_kl
  或降 lr（记为正式偏差）
- 20Hz 控制率 + MimicKit 高增益 PD——抖动风险（humanoid21 实测教训），
  但他的 std 极小 + 绝对位置目标语义，预期好于归一化 delta 方案

## 7. 训练配方实战记录（2026-01-28 凌晨）

**D15: batch 规模是样本效率的决定因素**。初始 256 eps/update（~2.5k 样本）配 minibatch 512，
每个梯度步仅 ~100 样本——上游 MimicKit 是 131k/80步 ≈ 1640 样本/步。小批量噪声导致策略空转
（single-critic 回报先升后降）。修正为 **episodes_per_update=2048 + minibatch 4096（实验默认）**，
每步 ~819 样本。代价：update 频率降低，但 samples/sec 吞吐不变。

**吞吐优化链**（全部已验证）：
1. RSI 起始帧量化到 60fps 网格 + 控制步恰为 3 帧 → 全部参考查询精确落帧 →
   FK/tar_obs 全量预计算，ref_state 变纯查表：9.05→3.37ms/控制步（2.7×）
2. rollout_workers 96 + 2048 eps/update 的摊薄效应：实测 ~8k 控制步/秒/run（双跑并行）
3. 对比上游：iter400（突破点）= 52M 样本 ≈ 我们 1 小时 wallclock

**Adam 固定小 std 的固有敏感 regime**（D12 补充）：σ=0.05 下 ratio 对微位移极端敏感
（单步 Adam mean_shift 0.004 已足以大幅推动 ratio）。上游靠 SGD+action_bound 正则稳定，
我们靠 lr=1e-5 + 大 batch。activate ReLU vs 框架 Tanh 差异保留（D14）。

**监控基线**：dm28sk_single_1128_0230 / dm28sk_multi_1128_0230
（--no-confidence 必需：critic 初期 EV 为负会零化 advantage；MimicKit 本无此机制）

## 8. Adam→SGD 保真修复（2026-01-28 凌晨，决定性）

**死亡螺旋确诊链**：固定 σ=0.05 下 log-ratio 对均值微位移极度敏感（1/σ²=400 放大）。
Adam 丢弃梯度幅度信息——每步恒定 lr 的噪声游走，而 PPO 裁剪使 ratio 偏离后有效梯度缩小
（天然刹车），SGD 保留这一自限性而 Adam 破坏它。实测：Adam 下第一个梯度步后 KL 即爆至
2.5 亿，mean_abs 0.04→1.82 饱和（动作出界被环境裁剪，策略失去自由度）。

**修复**（框架扩展，默认不影响其他实验）：
1. `actor_optimizer/critic_optimizer` 可选 'sgd'——上游双方都是纯 SGD lr 1e-4（无 momentum）
2. `action_bound_weight=10.0`（上游精确形式：Σmin(m+1,0)²+Σmax(m-1,0)²，防均值出界）

修复后冒烟：ratio_mean 1.0136（上游 Imp_Ratio~1.0005 同量级）、KL~0.06、mean_abs 稳定 0.04。
原"框架不支持就不加"决策被实证推翻：这两项是方法的承重墙，不是装饰。

## 9. 观测归一化补漏（2026-01-28，第三个被实证推翻的"可省"项）

上游 config `normalizer_samples` + checkpoint 含 obs_mean/obs_std——running 归一化 + clip±10
本就是方法的一部分。实测我方原始 obs 幅度最大 17.3（速度维），2.4% 超 |5|，tanh 饱和 +
特征尺度混乱导致学习近乎冻结（80min/28M 样本 eval survive 全程平直 0.03-0.04）。

实现：fixed stats（data/obs_norm.npz，随机小动作用力 rollout + RSI 落位混合分布 1200 样本，
std 地板 0.05）在 sim.compute_obs_a 出口归一化 + clip±10。用 fixed 而非 running 的偏差已记录，
但尺度矫正捕获 ~90% 收益。**注意**：obs 语义布局若变，必须重新生成 obs_norm.npz。
ReLU vs Tanh（上游用 ReLU）差异仍保留为 D14——归一化后输入集中在 ±2，饱和问题次要。
