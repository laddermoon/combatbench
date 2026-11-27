# SuperMimic 官方实现在 S1..S7 框架下的完整拆解

> 依据：`/tmp/supermimic_mjlab`（官方 mjlab 实现，task=motion）。
> 每个 S 阶段列出：官方对象、输入输出、关键细节、与我们 dynabptt
> 实现的差异。目标：一行不丢地把官方训练循环钉死。

## 概念映射总表

| 我们的概念 | 官方对象 | 位置 |
|---|---|---|
| StartSet | **不存在独立实体**，由两个机制联合承担：<br>① `ResetSampler`（episode 重置点=动作库帧）<br>② `ReplayBuffer`（想象起点=均匀采样真实状态） | `mdp/reset_sampler.py`<br>`data/replay_buffer.py` |
| SubStartSet | `replay.sample_starts(64, history=10, horizon=32)` | `replay_buffer.py` |
| ReplayBuffer | `ReplayBuffer(num_envs=64, capacity=1000)` 环形，带 episode 边界 mask | 同上 |
| WM | `WorldModel`（3×2048 MLP，state_delta） | `models/world_model.py` |
| NStep（rollout/想象） | rollout 教师窗=8，想象 horizon=32 | `rl_cfg.py` |
| Loss | 想象目标 = 软存活折扣奖励 + γ^H·存活·V(s_H) | `rl/imagination.py` |
| Critic | `MotionCritic`（256×128 MLP，GAE 目标） | `models/policy.py` |

默认配置（`OnlineConfig`，全是关键数）：

```
num_envs=64  capacity=1000(步) → replay 64k transitions
warmup=32  collect: 每2次update采5步×64env (=320 transitions)
wm: 10 batches × 256条 × 8步窗  (教师强制)
policy: batch=64起点 × horizon=32
actor: encoder(ref→32)→decoder(32+10×93)→action, (2048,1024,512,256), 确定性
critic: 256×128, 吃物理状态+goal+context
γ=0.99  λ_gae=0.95  action_noise=0.1  episode=10s(500步)
lr=2e-4 cosine→3%   AMP bf16   grad clip 1.0 (actor 另有 element clip 0.1)
```

---

## Init StartSet

官方没有"初始状态集"概念。冷启动是 `bootstrap()`：
1. `collect(warmup_steps=32)`——64 个 env 从动作库帧 reset 跑 32 步，
   政策是未训练的 actor + σ=0.1 噪声，纯粹灌数据
2. 要求 replay 里至少有一条完整 8 步窗
3. `world.fit_normalization()`——用 warmup 数据统计输入 mean/std
   和输出 delta 的 `correction_scale`（floor 保护，位置通道 1e-3、
   速度通道 1e-2）

**与我们差异**：我们的 StartSet 是显式状态集（s₁ 准入）；
官方的"起点分布"完全由 **reset 采样器把 env 重置到动作库任意帧**
产生——起点不是"收集来的状态"，是"参考动作本身的状态"。

## S1: 抽取 SubStartSet

`update_policy()` → `replay.sample_starts(batch=64, history_length=10, horizon=32)`：

- **均匀**拒绝采样 replay 中的所有 (step, env) 位置——不是按 φ
  加权、不是按新鲜度，纯均匀
- 约束条件只有两个：
  - 该点有**同一 episode 内**的 10 帧连续观测历史（不够长则
    左边界 pad 到 episode 起点也算合法，见 `_history_indices`）
  - `frame_id + 32 < clip_length`——**约束的是参考动作库帧界，
    不是真实轨迹界**：想象允许跑出真实 episode 结束之后
- 返回字段：state、context、motion_id、frame_id、history（10×93
  带噪 obs）、clean_history

**与我们差异**：我们的 SubStartSet 从 StartSet 按 φ⁶ 加权采；
官方从 replay 均匀采，且起点必须带 10 帧观测历史（actor 输入
需要）——actor 看的是历史栈不是单帧。

## S2: Rollout（真实环境采集）

`collect(steps=5)`（`online_training.py:194-265`）：

```
inputs = env.policy_input()                # history栈 + future_reference
action = actor(history, future_reference)  # 确定性均值，无采样
noise  = randn × action_noise(0.1)
env.step(action + noise)                   # 执行带噪动作
transition = extras["transition"]          # 记录的是带噪后的 action? 
```

**注意**：`transition["action"]` 在 `env.step()` 内部 clone 的是
**传入的带噪动作**（env.py:310 `action.clone()`）——和我们一样，
记录即执行值。

- 每 transition 记 `policy_version = actor_updates`（off-policy 程度可追溯）
- 每条记录的字段（`REQUIRED_FIELDS`，18 个）：state, next_state,
  action, previous_action, observation, next_observation,
  clean_observation, next_clean_observation, context, reward,
  motion_id, frame_id, episode_id, episode_step, terminated,
  time_out, motion_end
- **带噪 obs 和干净 obs 都存**——WM 学状态不学观测；actor 历史
  用带噪版本（训练和部署输入分布一致）
- episode 结束（终止/超时/clip完）立即走 S7 的 reset 采样器换新帧

**真实终止条件**（`terminations.py`）：height_error>0.25 或
tilt_error>0.8（相对参考帧的根高度/倾角差）+ 10s 超时 + motion_end。

**采集节奏**：`updates_per_collect=2`——每 2 个 update 才采 5 步。
一次 update 周期内：采集 320 transitions → WM 10×256×8 → 想象
64×32。**数据产量:WM训练量 ≈ 1:64**（每批新数据被反复用）。

## S3: Update WM

`update_world()`：`wm_batches=10` 次，每次 `sample_windows(256, 8)`
从 replay 均匀抽 8 步窗。

**状态表示**（`encode_state`，航向不变）：
```
[z(1), g_b(3)重力体坐标, qj(N), v_b(3)体坐标线速度, w_b(3), qd(N)]
输入 = encode_state + action + context → 4N+14 维
```

**context**（域随机化参数，episode 内常量，reset 时写入）：
`[足底摩擦(1), 编码器bias(N), 躯干CoM偏移(3)]`——WM 和 critic
都显式吃这些物理参数。

**预测模式**：`state_delta`（默认）——MLP 输出完整一步增量
（体坐标系下的 Δpos/Δrot/Δq/Δv/Δw/Δqd），`apply_state_delta`
解析应用；末层零初始化（初始预测=恒等+零delta）。
备选 `velocity_residual`：恒速积分+残差校正（`integrate`）。

**Loss**（`world_model_loss`，Eq.1）：教师强制 8 步展开，
`mask` 在 episode 终止后切断：
```
total = normalized_state_error + BCE(termination_logit, terminated)
        # normalized_state_error: 6组分量各自尺度归一
        #   pos/0.3, ori/0.4, q/0.3, v/1.0, w/3.14, qd/5.0
# 最后一批额外算 analytic head 指标（obs/reward）只作诊断，
# 训练 loss 里 analytic_losses=False → 不进梯度
```

**拟合判据**（`evaluate_world_model` 的 qualification gate）：
- horizon 1 & 8：normalized_state_mse **打败恒速基线**
  （`integrate(state)` 零校正）——是相对基线不是绝对百分比！
- horizon 32：四元数范数漂移 < 1e-4（数值稳定性）
- 三个 horizon 的窗覆盖全部 clip
- 全过 → `candidate_for_bptt=True`

**与我们差异**：判据是"beat恒速基线+数值稳定+覆盖"三件套，
不是相对误差百分比——我们的 Bryson 判据在精神上同源
（按物理尺度归一），但官方更松。

## S4: Imagine

`imagine()`（`imagination.py:71-196`），horizon=32：

```python
for t in range(32):
    action      = actor(history, goal_t)            # 确定性 MLP，无噪声无采样
    next_state  = world(state, action, context)     # 干净状态链
    observation = kinematics.observation(next_state, action, context, ref_quat_t+1)
    if sensor_noise:                                 # 默认 True
        observation = add_sensor_noise(observation, ε)  # 均匀噪声，幅度见下
    history = cat(history[:,1:], observation)        # 带噪 obs 进下一步历史
    state = next_state
```

**add_sensor_noise 幅度**（`state.py:126`，均匀 ±）：
`qj 0.01, qd 0.5, root_angvel 0.2, gravity 0.05, 其他 0.05, prev_action 0`
——噪声**只污染 actor 的输入历史**，状态链、奖励、critic 全部
吃干净值。ε 的 requires_grad=False，反传时当常数（∂o_noisy/∂o=I）。

**goal**：actor 的 reference 输入 = 10 帧未来参考的特征窗，
`library.get_future(motion, frame, H+10)` 预先批量取好——
想象轨迹的每一步都看着"接下来10帧该做什么"。

**与我们的差异**：我们 nstep=8 无历史栈无参考轨迹；
官方 horizon=32、actor 吃 10 帧带噪历史 + 10 帧未来参考。

## S5: 计算 Loss

想象完后**整个轨迹批量一次算**（`imagine` 末段）：

```
rewards, failure = task_heads(states, actions, context, ref, prev_action)
  # task_heads = FK(kinematics) → tracking_reward  【解析，非学习头】
  #   + termination_head（唯一学习部件：2维输入→64×64→1 logit）
  # termination_mode="learned": logit(sigmoid)；"analytic": soft_termination

continuation = 1 - failure                        # 软存活概率
alive(t)  = Π_{k≤t} continuation_k                # cumprod
before(t) = Π_{k<t} continuation_k                # 错位一步
reward_objective    = Σ_t γᵗ · before(t) · r_t    # 软存活加权的折扣奖励
bootstrap_objective = γ³² · alive(32) · V(s_32)   # critic 尾部，带梯度
objective = reward + bootstrap
```

**tracking_reward**（`rewards.py`，全解析）：
- BeyondMimic 式：相对参考的 heading-aligned 全身 body_pos/body_quat/
  linvel/angvel 误差 → 各 `exp(-err/scale²)` 高斯基元
- + root_pos/root_ori 绝对误差项（权重0.5）
- + 关节量程越界 cost
- 六路高斯项相加 = 每步 r∈[0,~5.5]，`MAX_MOTION_STEP_REWARD`

**关键结构**：奖励对**整条想象轨迹**在 (B,H) 维批量计算，
只进一次图；V(s_32) 用 critic 但**保留梯度路径**（s_32 的物理
状态可微 → bootstrap 梯度回流整条链）；values（GAE 目标用）
在 `no_grad` 下批量算 + 封顶 `max_value = MAX_STEP_REWARD/(1-γ)`。

## S6: 更新 Policy + Critic

`update_policy()`（`online_training.py:310-419`）：

```python
with frozen_parameters(world, critic):      # 冻参数，保留对state/action的Jacobian
    trajectory = imagine(...)               # 建一次32步图
    checked_step(-objective, actor, opt_a, element_clip=0.1)
    # grad: clip_norm 1.0 + 逐元素 clamp±0.1 + FP64溢出回退

# 同一个 rollout 之后接着训 critic（先用完actor的图再释放）
targets = gae_targets(rewards, continuation, values, γ=0.99, λ=0.95)  # detached
for epoch in 5:                              # critic_epochs
    for minibatch in order.tensor_split(4):  # 整轨迹shuffle后切4批
        loss = masked MSE(V(features), targets)   # 时间维sum，轨迹维mean
        checked_step(loss, critic, opt_c)
```

**critic 的数据来源** = 想象轨迹本身（imagined rewards + imagined
states 的 features）——不是真实 replay 的 reward！这很反直觉但
自洽：critic 学的是"在想象分布下的 V"，而它唯一被用到的地方
也是想象的尾部。

## S7: 更新 "StartSet" —— 官方版是 ResetSampler

`ResetSampler`（`reset_sampler.py`）——**这是官方回答"覆盖问题"
的机制，和我们的 StartSet 哲学完全不同**：

- **bin 划分**：每条 clip 每 50 帧（≈1s）一个 bin，覆盖整个动作库
- **打分**：episode 结束时算 `discounted_return / max_return`，
  对该 episode 起点 bin 做 EMA(0.1) 更新
- **采样概率**：`0.5×uniform + 0.4×softmax(-25·ret/max) + 0.1×historical`
  ——**40% 权重故意采表现差的 bin**（adversarial 课程）+
  10% 均匀历史分布保底
- **softmax_cap**：单个 bin 的概率上限，防止极端集中

**reset 时**（`env.reset`）：sampler 出 (motion, frame) → 取该帧
参考 root/关节状态 → `RESET_NOISE` 扰动：
```
位置 ±(0.05, 0.05, 0.01)m   旋转 ±(0.1,0.1,0.2)rad
线速度 ±(0.5,0.5,0.2)m/s    角速度 ±(0.52,0.52,0.78)rad/s
关节角 ±0.1rad
```
——扰动幅度很大，天然把机器人扔进"非理想"状态带。

**与我们差异（这是整张图里最重要的一个格）**：
| | 我们 StartSet | 官方 ResetSampler |
|---|---|---|
| 起点来源 | 收集轨迹的 s₁（可达域内） | **动作库任意帧**（先验覆盖） |
| 覆盖机制 | φ⁶ 加权 + 严格准入 → 冻结 | 0.4 adversarial + reset噪声扰动 |
| 走廊数据 | 无（分支永不死） | 重置直接到困难帧 + 大扰动 |
| 语义 | 学出来的可达域 | 参考动作定义的全域 |

## 附：一个 update 的完整时序

```
train_iteration():
  [每2轮] collect(5步)                       → replay +320 transitions
  lr = 2e-4 × cosine(fraction)→3%            → 三个optimizer同步调
  update_world():   10×(256窗×8步教师强制)     → WM梯度
  if iter >= wm_only_iters:                   → wm_only 预热期可跳过
    update_policy(): 64起点×32步想象
      → actor一次反传(-objective)
      → critic 5epoch×4minibatch (GAE detached targets)
```

数值纪律（`checked_step`）：loss/grad 非有限 → 整个 step 抛
`FloatingPointError` 拒绝更新；grad norm clip 1.0；actor 额外
逐元素 ±0.1 clip；FP32 平方溢出时 critic_loss 内部用 FP64 算标量。

## 总结：官方相对我们的三个不可忽略的部件

1. **动作库 = 先验覆盖**——StartSet 的覆盖问题在官方框架里
   根本不存在，因为"起始状态"是 motion capture 给的物理合法
   状态全集，加上 reset 噪声扰动带 + adversarial 采样
2. **参考轨迹 = 时序先验**——actor/critic 都吃未来10帧参考，
   想象永远知道"该去哪"；我们的 φ/margin 是无参考的目标，
   方向感全部要 reward 自己给
3. **软存活建模**——学习 termination head + cumprod 软门控，
   让"快要倒"在梯度里是可微信号而不是硬切断
