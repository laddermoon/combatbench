# Standup：真实 PPO + 误差门控短模型 rollout 实验方案

日期：2026-11-25。状态：方案，尚未实现或启动本方案中的训练。

## 1. 目标与边界

验证一个问题：在真实仿真 PPO 持续学习的前提下，当前策略附近拟合的世界模型，能否通过短 rollout 减少学会 Standup 所需的真实交互，并最终缩短总训练时间？

用户约束：

- 不使用外部 teacher、预训练策略、历史成功轨迹或蒸馏。actor、critic、世界模型均随机初始化。
- 真实仿真采集贯穿训练，负责真实 PPO 更新、模型训练和校准。
- 第一版只关注自己。对手信息不进入策略和模型，物理交互也必须隔离。
- 所有新增或修改文件限定在 `baseline/framework/dynabptt/`；不改公共 PPO、rollout、模拟器或现有实验定义。需要的适配器和插件在本目录实现。
- 本次先写方案。实现、实验运行和扩展预算按下面的阶段执行，不预先启动大规模调参。

第一条验证路线为 **真实 PPO + 模型辅助 PPO**。BPTT 是后续独立分支，不与第一轮混用。模型采用神经网络只是实现选择，模型辅助 PPO 不依赖模型可微。

快速验证允许得出“不通过”“模型暂时不能使用”或“证据不足”。不能用训练损失下降、提议接受率高、最高势能上升代替学会站立。

## 2. 当前代码核查与必须处理的问题

参考文件均相对于项目根目录：

| 文件 | 本方案参考内容 |
|---|---|
| `baseline/experiments_ppo/exp_standup.py` | 任务、奖励、PPO 参数、评估口径 |
| `baseline/framework/ppo/trainer.py` | PPO buffer、概率比、clipping、advantage 标准化 |
| `baseline/framework/ppo/algos/advantages.py` | GAE、timeout bootstrap |
| `baseline/framework/ppo/policies/truncated_normal_mlp.py` | 策略采样、evaluate_actions、sampling context、导出 |
| `baseline/framework/rollout/episode_recorder.py`、`episode.py` | 轨迹数据契约 |
| `envs/framework/env_runtime.py`、`recorder.py` | 动作前 observation 与动作后 observer 的时序 |
| `baseline/framework/dynabptt/corestate.py`、`buffer.py`、`world_model.py` | 可借用的状态布局和模型结构 |
| `baseline/humanoid21/rewards/standing_balance_4stage.py` | 接触依赖、势能和阶段计算 |
| `envs/humanoid21/simulator.py` | 真实 96 维 observation 的构造 |

### 2.1 时序对齐是首要正确性门槛

当前 runtime 明确在执行动作前捕获 observation；CoreState observer 在动作后记录状态。因此 frame[t] 的语义为：

```text
observation[t] = o_t
action[t]      = a_t
corestate[t]   = s_(t+1)
potential[t]  = phi_(t+1)
```

旧 dynabptt buffer 将 observation 注释为 post-step，并在 seed 中把 `cs[f]` 与 `obs[f]` 配对；模型窗口也将同帧 observation 当作下一状态目标。这与当前 runtime 契约冲突。它对历史运行的影响需要结合当时的代码快照确认；之前的平台和梯度结论不应直接继承。

新模块明确构造：

```text
t >= 1:
state        = corestate[t-1]
observation  = observations[t]
action       = actions[t]
next_state   = corestate[t]
next_obs     = observations[t+1]，最后一步使用 final_observation
reward       = 0.01 * potential[t]
```

真实 PPO 保留 t=0 的有效 transition；模型数据在没有可靠 s_0 时显式跳过 t=0，单独统计跳过数量。不能通过无说明的裁剪掩盖长度不一致。

用真实 runtime 连续执行至少两步，对照每次 step 前后的 state、observation、action、reward，逐项验证。不要只用随机数组的 shape 测试。

### 2.2 Self-only 必须同时约束观察和物理

当前 96 维 observation 包含自身 57 维和对手 39 维；旧模型却尝试由自己的 55 维状态预测全部 96 维，这会引入无法由自身状态确定的信息。

新实验统一使用自身 57 维输入，21 维动作。采集、训练、模型头、评估及导出策略均使用同一个本地适配器。索引以当前 simulator 实现为准，并用真实观察对照测试确认；不依赖可能过时的说明文本。

环境从 Standup 蓝图在内存中派生，由本目录的 simulator 子类/适配器关闭两机器人之间的碰撞，保留各自与地面、场地的交互及同一倒地初始化。明确记录派生蓝图和碰撞配置哈希。不能仅靠“初始距离较远”宣称没有交互。

在相同自身初态、相同自身动作序列下改变对手动作，验证自身轨迹与奖励在数值容差内一致。若无法在本目录完成可靠隔离，先停止 self-only 模型实验，报告阻塞。

这会与原始 96 维 PPO 有区别。因此主要对照必须是**同样 57 维、同样隔离环境的本地 PPO**；用户所述约 40 分钟作为外部参考，不作为已复现结果。

### 2.3 状态、观察、奖励不能混为一个 loss

沿用 55 维 CoreState 作为初始模型输入：root position、wxyz quaternion、局部线/角速度、归一化关节角和关节速度。检查执行器是否还有未包含的内部状态；不能预设这 55 维严格满足 Markov 性。

第一版：

- `F(s,a) -> s_next`：残差 MLP，初始尝试 3×512，独立训练集统计量做分组归一化。
- `G(s_next) -> o_self_next`：57 维观察头；优先以真实状态监督，另检查预测状态下的误差。
- `R(s,a,s_next) -> phi_next, stage_next`：转移奖励特征头，phi 回归、stage 分类。奖励严格为 `0.01*phi_next`。
- quaternion 使用单位化、符号等价的姿态损失；评估用旋转夹角，不用原始四元数 MSE。
- 不启用 learned done：此任务允许倒下后继续起身，只有已知的 200 步 timeout。

Standup 奖励含手/脚接触力与负载转移，不能声称只凭位姿已经精确重建奖励。首版允许学习奖励头，但奖励误差必须独立通过准入验证。

模型训练和原始误差统计不使用输出 clamp 隐藏问题；消费端可限制 phi 到 [0,1]，同时记录越界率。持续越界应关闭模型贡献。

## 3. 最小架构与每轮顺序

建议新增 `hybrid_ppo/` 子目录，避免继续在旧 `train.py` 上堆条件分支：

```text
baseline/framework/dynabptt/
  PPO_WORLD_MODEL_EXPERIMENT_PLAN.md
  hybrid_ppo/
    train.py             # 本地入口、每轮调度和计时
    ppo.py               # 本地 PPO/GAE 更新与一致性测试入口
    standup.py           # self-only 环境/策略适配、奖励和评估
    transitions.py       # 显式时序、真实/模型数据分离
    model.py             # 动力学、观察和奖励头
    validation.py        # 留出集验证、门槛和配对诊断
    rollout.py           # 短模型分支、old-policy/log-prob
    configs/             # 固定的实验配置
    tests/               # 关键契约与回归测试
    runs/                # 本地忽略的运行产物
```

目录按实现需要创建，不为占位而创建空文件。ignore 配置也仅放在本目录。共享 EnvRuntime、ParallelRollouter、policy 等可只读 import；若必须改变其行为，在本目录包装或复制，标明来源 commit 和本地差异。

PPO 优先复制现有 GAE、surrogate 和采样上下文相关逻辑；策略分布复用原实现，不能临时换成普通 Gaussian 再自行 tanh。先建立本地 real-only PPO 与原更新逻辑的数值一致性，再加模型。

每轮 u：

1. 冻结旧策略 `pi_old`、旧价值函数 `V_old` 和观察转换规则。
2. `pi_old` 在真实仿真采集。所有真实训练轨迹都用于本轮 PPO。
3. 按 episode ID 划分模型训练/留出数据，同一 episode 的两个机器人必须同组，禁止重叠窗口泄漏。
4. 拟合模型；固定本轮候选模型，使用未参与拟合的最新数据验证。
5. 从本轮真实训练轨迹的状态分布选 seed，按已验证的 horizon 生成短分支。
6. 分别构造真实 PPO advantage 和模型辅助 advantage。真实 critic 只吃真实回报。
7. 在固定 PPO epochs/minibatches 下更新策略与 critic；整个 PPO 更新期不改变模型、old-policy 或 old-value。
8. 更新模型可用性统计，定期在独立真实评估集测效果。

首版顺序执行，先避免训练中途读取变化中的模型。这里的“同步训练”是同轮持续更新，不要求并发线程。异步提速留到证实收益后。

## 4. 模型数据与准入门槛

### 4.1 数据分布

- 真实 PPO 只用本轮 on-policy 数据；历史 replay 仅用于模型拟合。
- 模型 replay 上限先设 1M agent transitions；训练采样 80% 来自最近两轮、20% 来自剩余历史。历史不足时用已有训练池，记录实际组成。
- 独立模型 RNG，保证旁路训练不会改变 PPO shuffle/探索 RNG。
- 当前采集分布包含策略实际随机动作，explore_factor=0；sigma 按 PPO 学习，不沿用旧 BPTT 固定 sigma=0.4。
- 用 episode 哈希固定 80%/20% 模型训练/留出划分；留出轨迹始终不进模型拟合。可用于真实 PPO，但不作为合成 seed。
- 每轮拟合预算初值 60 minibatches，batch=256，autoregressive k=3；误差不通过时不自动增加网络/训练量。
- 验证 horizon 候选 `{1,2,3}`。k=8 或更长 rollout 不进入首轮快速验证。

### 4.2 留出集测量

每轮最多抽取 4096 个窗口，来自至少 64 个独立留出 episode。按真实初始阶段分层报告，缺样本的阶段不开放模型分支。

必须同时报告：

1. 沿真实记录动作的开环 h-step 误差（所有中间步都检查）。
2. 由预测 observation 驱动 pi_old 的闭环误差。
3. 相同状态/相同动作下真实与模型的短期奖励、bootstrap value 和 advantage 误差。

闭环验证从可重现的真实 seed 分支：首选精确保存/恢复完整 MuJoCo integration state 与插件状态；仅设置 55 维 CoreState 不算完整恢复。若现有接口不足，首版用 episode seed + 实际动作前缀重放恢复，并核对分支起点一致。所有验证仿真步都计入真实交互与墙钟预算。

每轮闭环校准最多64个真实分支，每个最长3步，动作前缀重放的成本另计；分支不足以覆盖某阶段时保持关闭。若校准成本超出总预算，不通过减少验证后继续使用模型来规避，而是关闭模型贡献并报告成本问题。

模型和真实分支可使用相同随机数驱动策略采样，但随着观察不同，动作允许不同。另做相同记录动作的开环对照，区分动力学误差和闭环误差。

### 4.3 首版数值门槛（待旁路校准的预注册初值）

以下门槛用于 h≤3 的每个中间步；以阶段分组计算，不能靠大量静止样本稀释失败区域：

| 指标 | 首版要求 |
|---|---|
| root 高度误差 | MAE ≤ 0.02m，P95 ≤ 0.05m |
| root 旋转误差 | 平均 ≤ 5°，P95 ≤ 10° |
| 关节角/速度及 root 速度分组 NRMSE | 各组 ≤ 0.20；尺度只由训练数据估计并设置非零下限 |
| 自身 observation NRMSE | ≤ 0.15；足底两维单独 ≤ 0.25 |
| phi 预测误差 | MAE ≤ 0.03，P95 ≤ 0.08 |
| stage 分类 | 总准确率 ≥ 90%；真实阶段变化窗口 ≥ 80% |
| phi 原始输出越界率 | ≤ 1% |
| 闭环确定性动作均值误差 | RMS ≤ 0.05（动作范围 [-1,1]） |
| 同长度 bootstrap advantage 相对 RMSE | ≤ 0.5，以真实短分支 advantage 的标准差归一化 |
| 有信号样本 advantage 符号一致率 | ≥ 75%；“有信号”固定为绝对值大于真实短分支标准差的 0.25 倍 |

若真实 advantage 标准差过小（<1e-4）、阶段变化样本少于 32 或阶段窗口少于 256，该组记为“证据不足”，不将其判为通过。所有门槛同时通过才开放对应阶段/horizon；无可用阶段则退回普通 PPO。

同长度 advantage 对照用相同 `V_old`，验证的是模型额外引入的偏差；它不证明 V_old 本身准确。另报告 V_old 对真实回报的 explained variance 和预测误差。

旁路阶段允许一次根据真实单位和测量噪声修订门槛，必须先写配置与理由、再开始干预实验。不能看到混合策略效果差后放宽门槛直到“通过”。

### 4.4 开关规则与选择偏差

- 某阶段连续两轮验证通过才开启，任一轮失败立即关闭；h 为连续通过的最大值，初始仍从 h=1 开始。
- 门控只根据真实 seed 的阶段、历史校准结果和预先固定的规则；不按模型预测的高奖励挑选 seed。
- 不因某个随机动作看起来“好”才保留对应样本，否则改变了采样分布，old log-prob 不能纠正这种筛选。
- 若生成时出现 NaN/无效姿态，关闭该轮模型贡献并记录原因；不只丢弃不利动作。未知区域按 seed 级别拒绝。
- 下一轮必须重新验证新 pi_old 下的模型，不能沿用很久以前的“准确”结论。

## 5. 第一条策略路线：模型辅助 PPO

### 5.1 首版只更新真实 seed 上的第一动作

每个 seed 从本轮 pi_old 的真实访问状态抽样，从准确的真实 `s_t,o_t` 起步：

```text
真实 seed → a_0 ~ pi_old(o_0)
          → F → o_hat_1 → a_1 ~ pi_old(o_hat_1) → … → h 步
```

整个 rollout 在 no_grad 下生成。用预测奖励、`V_old` 和 gamma=0.99、lambda=0.95 计算短分支 GAE，仅将第一项 `A_model,0` 和真实 seed 的 `(o_0,a_0,old_log_prob_0)` 加入辅助 PPO loss。

这样避免第一版同时引入“在模型生成的观察分布上更新策略”的额外误差。后续动作只用于估计第一动作的后果。h=1 时相当于模型估计的 TD advantage。

模型切断属于 truncation，末端用 `V_old(o_hat_h)` bootstrap；不能将模型 horizon 当成真实 terminal。符合此任务语义的环境 timeout 同样 bootstrap。不能把失败的模型预测视为奖励为零的 terminal。

### 5.2 Loss 和尺度

```text
rho = exp(log pi(a|o,ctx) - log pi_old(a|o,ctx))
S(rho,A) = min(rho*A, clip(rho,1-epsilon,1+epsilon)*A)

L_actor = -mean_real S(rho,A_real_norm)
          -alpha * mean_model S(rho,A_model_norm)
          + 与基线相同的策略正则项
L_value = 真实轨迹上的 value loss
```

- alpha 初值 0.1；本轮至多生成 `0.25 * N_real` 个 seed。报告实际合成步数（seed_count × h）和用于 actor 的样本数，不能混淆两者。
- `A_real` 按原 PPO 标准化。`A_model` 使用本轮真实 advantage 的尺度归一化并在模型 batch 内中心化；不除以模型自身极小的标准差，避免放大噪声。尺度有明确下限。
- 第一个快速实验固定 alpha，不搜索多个权重。模型项不会稀释真实项的 batch mean。
- 真实 minibatch 数量、epochs、critic 更新数与对照保持一致；每个真实 minibatch 附带一个模型 minibatch，共用一次 actor optimizer step。
- sampling context 与实际采样完全相同；采集时记录或由冻结 pi_old 重算 old log-prob，并逐项检查概率比初始接近 1。
- 真实/模型 KL、clip fraction、梯度范数和两类梯度夹角分别记录。KL 超过统一预算则提前停止策略更新。
- 模型不接受 PPO 梯度；critic 初版不吃模型数据。通过后才考虑全模型轨迹 PPO 或合成 critic target，二者各自单独消融。
- 固定评估集不用于逐 update 策略选择，不启用旧 eval ratchet。保存 best checkpoint 仅用于报告和回放，不反馈优化。

## 6. PPO 对照配置与评估

本地基线沿用 Standup 参数：200 steps/episode、512 episodes/update、gamma=0.99、lambda=0.95、actor/critic LR=3e-4、4 epochs、minibatch=4096、target KL=0.05；clip、梯度裁剪、策略正则和 confidence 机制以源配置为准，完整展开写入 config，所有组一致。

actor/critic 均从相同初始参数开始，策略输入统一 57 维。若复制时简化了原 PPO 的 confidence 等逻辑，必须作为配置差异说明，并先在相同输入数据上验证核心 PPO 数值。

workers 初始设为 24、每次一个活跃训练组、一张固定 GPU、固定 CPU 线程配置。硬件资源有其他负载时记录并避免做精确墙钟速度结论。不能一个组用 96 workers、另一个 24 workers 后比较训练时间。

每 5 updates 在独立的 64 个 episode seeds 上做确定性真实评估：

- `reach_success`：episode 内 max phi ≥ 0.9。
- `final_success`：episode 最后一帧 phi ≥ 0.9。
- `hold_success`：末尾 20 个动作步都满足 phi ≥ 0.9。
- max/final/末尾20步平均 potential、首次达到0.9的步数、各阶段访问占比。
- 对两个 agent 的结果分别记录，并以 episode 为 bootstrap/置信区间聚类单位；不能当作完全独立的 128 次实验。

“学会”预定义为 final_success ≥ 0.9 且 hold_success ≥ 0.8，连续三次评估满足；最终在新的 256 个 episode seeds 上复核并录视频。reach_success 仅作辅助，不能把单次峰值当站稳。

完整记录：训练真实步数、模型校准额外真实步数、评估真实步数、合成步数、梯度步数、样本复用率以及端到端时间。数据效率以训练+校准真实步数为主，评估成本单列并纳入总时间。

## 7. 分阶段快速实验与停止规则

预算均为上限，达到明确结论即停止。达到时间上限但 update 不足，结论为预算内证据不足，不自动延长。

| 阶段 | 实验 | 预算 | 通过条件 / 不通过后的动作 |
|---|---|---|---|
| P0 正确性 | 时序、self-only、PPO 数值与导出检查 | 测试 + 最多2 updates，约10分钟运行预算 | 必须通过关键契约；失败先修实现，不训练长跑 |
| P1 旁路模型 | A=真实 PPO；B=同一 PPO+仅拟合验证模型，alpha=0 | 每组最多20 updates或20分钟，先到为止 | PPO学习与A一致；模型报告至少h=1可用区域；统计旁路成本 |
| P2 干预试验 | 从本次P1的同一 checkpoint 分叉 B继续、C开启门控模型 PPO | 每组最多20 updates或20分钟 | C不明显退化且出现有效模型覆盖；否则停止混合路线并定位 |
| P3 复现筛选 | 新训练 seed，B/C从同一随机初态重新跑，均含旁路warmup | 每组最多40 updates或40分钟 | 两个seed方向一致才称“有前景”；不称最终加速结论 |
| P4 达标比较 | A/C各3个训练seed，从随机初态到统一目标 | 仅P2/P3正向后追加；每组先设60分钟上限 | 报告达到目标的时间/真实步数；未达标报告截尾，禁止外推 |

P1 中 B 的世界模型是旁路，训练 RNG 与 PPO 分离。固定输入数据重放时，A/B 的 PPO 参数更新应数值相同；在线运行允许环境非确定性，但不能把明显差异归为“模型帮助”。

首轮训练 seed=42，P3复现用43，P4使用42/43/44。训练、模型校准、在线评估与最终盲测使用独立的 seed 命名空间；相同对照组共享对应 episode seed 序列。模型数据生成不得消耗真实采集的 RNG。

P2 的 checkpoint 来自本次训练，不是外部 teacher。恢复 actor、critic、优化器、RNG、模型、数据划分与 replay；只改变是否加入模型损失。分叉先做一轮 no-update 配对评估确认起点一致。

P1 若模型通过率为零，停止 P2；输出是哪类误差不过关。若只是个别阶段通过且当前策略几乎不访问，报告覆盖不足，不能用极少样本宣布成功。

P2 筛选规则：

- 模型覆盖至少 20% 的候选真实 seed，连续两轮；否则记为“不能有效干预”。
- C 相比 B 连续两次评估 mean final potential 低 ≥0.05，或 hold_success 低 ≥0.10：停止该干预组。
- 若 C 的 final potential 学习曲线面积在相同累计真实步数下提高 ≥10%，且最终成功指标不下降，视为初步正向；接近零的基线不使用相对比值，报告绝对增益。
- 同时比较墙钟：若新增训练/校准开销超过真实 PPO 时间的30%，即使样本效率正向，也只能称样本效率候选，先分析性能瓶颈。
- 所有阈值是筛选规则，不是统计显著性证明；最终以多seed达标时间和置信区间判断。

P2 负向时只允许一个针对性诊断：在少量可重现 seed 上，把模型后果替换为相同起点/动作的真实短分支，比较辅助 advantage 和一次候选更新。如果真实短分支有效、模型无效，优先定位模型误差；两者都无效，检查短期 bootstrap/辅助 loss，而不是立刻扩大网络或同时扫多个超参。

所有诊断分支成本计入预算。不会每看到平台就再开一组长跑。

## 8. 第二条路线：PPO + 小权重 BPTT（后置）

只有预测和模型辅助 PPO 的基本管道验证后才考虑；也可在第一路线因短期 advantage 信噪比不足时单独立项，不将它当默认补丁。

额外门槛：对真实 seed 的动作施加多个随机小扰动，使用真实重放分支与模型比较同一个短程 bootstrap return 的变化。扰动尺度以实际 PPO 更新产生的动作 RMS 标定。

- 排除真实变化接近数值噪声的样本，记录有效样本比例。
- 至少两个独立 seed 集上，预测变化与真实变化相关系数 >0.5、符号一致率 >70% 才进入小批量试验。
- 同一 actor checkpoint 下比较真实 PPO 候选与 PPO+BPTT 候选，限制相同的真实状态 KL；用独立真实评估比较。
- rollout H=1，最多3；模型冻结，actor 通过模型反传；主要 actor 梯度仍来自真实 PPO。
- critic 仍只用真实数据训练；额外 BPTT 梯度范数起始不超过真实 PPO 梯度的10%，并记录实际比例。
- 与模型 PPO 组分开，不同时开启两项。任何结论限定到所测模型、horizon、种子与预算。

低 state/reward MSE 不能替代这项导数检查，也不能保证BPTT成功。

## 9. 测试、产物与复现

必须覆盖的测试：

1. runtime 两步真实记录对齐，包括最后 next_obs、首步缺失 state 和 timeout。
2. self-only 观察/导出一致性及对手动作不影响自身物理的验证。
3. 采样 log-prob 与 evaluate_actions 一致；初始 ratio≈1；sampling context 往返无损。
4. GAE 的手算短序列、真正 terminal 与模型 truncation 的区别。
5. 相同固定真实 batch 下本地 PPO 与参考 surrogate/GAE/loss/梯度一致。
6. 模型留出 episode 不进入拟合；同 episode 两机器人不跨集合。
7. 模型输出故意扰坏时，门控关闭、真实 PPO 仍执行且更新与旁路组一致。
8. 模型 PPO 的动力学参数梯度为空，critic 不消费合成 target；alpha=0恢复旁路行为。
9. checkpoint 恢复后 fixed-batch 更新可重现，包含优化器、随机状态、replay和门控状态。

每个 run 保存：

- git commit、复制来源 commit、展开后的配置、蓝图哈希、seed、精确命令和真实 PID。
- `metrics.jsonl`：PPO/模型分源统计、校准门槛、覆盖率、样本计数、完整阶段计时。
- checkpoint：actor、critic、model、全部 optimizer、normalizer、RNG、数据划分、replay及门控历史。
- 固定的配对诊断窗口和误差明细；最终评估原始结果和视频。

实现后在本目录补 `HYBRID_PPO_DETAIL.md`（逐次操作、命令、结果）与 `HYBRID_PPO_SUMMARY.md`（有效运行路径、结果和局限）。本方案不冒充已执行日志。

## 10. 预期交付与结论边界

首轮交付是：可独立运行的 self-only PPO 基线、模型旁路训练及校准、通过门槛才开启的短模型 PPO，以及有明确预算的 A/B 结果。

只有 C 比本地匹配基线更少真实步数达到同一真实目标，才能称样本效率提高；只有总墙钟也下降，才能称训练加速。历史 v33、简化 Q-ascent 试验和不同环境的 PPO 曲线不作为主对照。

若模型门槛长期不过，说明当前表示/模型在预算内不能承担这些 rollout；若门槛过了但策略无增益，说明这些门槛或辅助目标不足以保证有效更新。两者都不能泛化成“所有模型方法不可行”。
