# CombatBench SB3 Baseline 训练说明

本文档说明这套 baseline 的设计目标、环境修正点、训练分阶段流程，以及为什么这样做。

## 1. 目标

用户需求可以拆成四件事：

1. 在 `combatbench` 目录内部新增正式 baseline
2. baseline 使用 SB3
3. 能支持两个机器人站立对打
4. 如果环境实现、观测或奖励不适合训练，要直接修正

因此这次实现没有只写训练脚本，而是同时补了：

- 环境状态接口
- 训练封装
- 奖励函数
- 观测归一化
- 对战评估入口
- 详细文档

## 2. 这次修了哪些环境问题

### 2.1 `initial_distance` 原来没有真正作用到 reset

之前环境里虽然有 `initial_distance` 参数，但 reset 并没有把这个参数真正写回 MuJoCo 的 root free joint。

这次修复后：

- `root_red`
- `root_blue`

都会在 reset 时按照 `initial_distance` 重新摆位。

### 2.2 训练侧需要结构化状态，原环境只给了原始观测向量

对于 baseline 的奖励 shaping，只依赖硬编码观测索引会很脆弱。

这次在 `info` 里补充了：

- `robot_states`
- `relative_metrics`
- `torso_positions`
- `observation_slices`

这样训练代码可以直接从结构化状态计算奖励，而不是把逻辑全绑死在下标上。

### 2.3 `torso state` 的语义修正

之前 `HumanoidRobot.get_torso_state()` 实际取的是 `pelvis`，名称和语义不一致。

这次改成真正使用 `torso` body，避免：

- 站立高度定义偏差
- upright 判定偏差
- 奖励函数目标高度错位

### 2.4 补了观测维度校验

环境在 `_get_obs()` 里会检查观测是否仍是 127 维。这样如果后续有人改机器人观测结构但忘了同步 baseline，就会尽早报错。

## 3. 为什么 baseline 采用“共享策略 + 对称自博弈”

完整自博弈通常需要：

- 对手池
- checkpoint sampling
- 冻结历史策略
- Elo / 课程学习机制

这些当然可以继续做，但它们会显著增加系统复杂度。

这次 baseline 的目标是：

- 先让系统稳定可训
- 先让机器人学会站立、接近和出手
- 先把训练入口和评估闭环打通

所以这里采用的是更朴素、但更稳的方案：

- 底层仍是双机器人环境
- SB3 看到的是一个单智能体封装环境
- 训练时同一个动作同时施加给红蓝双方
- 总回报取双边 shaped reward 的平均值

这个设计的好处是：

- 实现简单
- 不需要对手管理系统
- 能快速验证 reward shaping 是否有效
- 更适合作为目录内的第一个正式 baseline

它的限制也很明确：

- 这不是严格零和博弈训练
- 共享动作训练可能带来协同性偏差
- 最终策略不一定最强，但更容易先得到“能站立对打”的结果

## 4. 观测归一化是怎么做的

`normalization.py` 把 127 维观测按物理含义分块缩放：

- 关节角 → 除以 `pi`
- 关节角速度 → 除以经验上限
- 高度 → 围绕目标高度中心化
- 线速度 / 角速度 → 除以速度尺度
- 外力 → `tanh` 压缩
- 相对位置 / 关键点位置 → 除以场景尺度

这样做的目的不是“数学上完美”，而是让 PPO 更容易收敛：

- 避免一个量纲过大主导网络
- 避免受力等尖峰量把值函数打坏
- 保证模型看到的输入尺度更均匀

## 5. 奖励函数设计

奖励定义在 `rewards.py`。

### 5.1 `stand` 阶段

主要目标是让机器人：

- 保持 torso 高度
- 保持身体竖直
- 双脚尽量接地
- 不剧烈乱晃
- 不轻易倒地

因此 `stand` 奖励里：

- `height_weight` 更高
- `upright_weight` 更高
- `feet_contact_weight` 更高
- `damage_reward_weight` 很低

### 5.2 `fight` 阶段

主要目标是让机器人：

- 继续保持基本站立稳定
- 面向对手
- 进入合理打击距离
- 造成伤害，同时减少自身掉血

因此 `fight` 奖励里：

- 站立相关项还保留，但权重下降
- `facing_weight` 和 `distance_weight` 提高
- `damage_reward_weight`、`damage_penalty_weight` 提高

### 5.3 为什么还要动作惩罚

如果不加动作惩罚，策略很容易学出：

- 高频大幅震荡
- 用极端抽搐来刷局部奖励
- 短时间爆发后迅速倒地

所以这里加入了：

- `action_penalty_weight`
- `action_change_penalty_weight`

它们会压制不必要的大动作和动作跳变。

## 6. 两阶段训练流程

## 阶段一：站立预训练

目标：先学会“不要立刻倒”。

推荐命令：

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 1000000 \
  --run-name stand_v1
```

建议观察：

- episode 长度是否变长
- 平均 reward 是否稳定提高
- `evaluate.py --mode selfplay` 时是否已经明显更久不倒

## 阶段二：对打微调

目标：在会站的基础上，进一步学会接近、面对和攻击。

推荐命令：

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase fight \
  --timesteps 2000000 \
  --run-name fight_v1 \
  --init-model combatbench/baseline/sb3/runs/stand_v1/model_final.zip
```

为什么要用第一阶段权重初始化：

- 从随机参数直接学对打太难
- 机器人连站都不会时，combat reward 非常稀疏
- 先学平衡，再学对抗，样本效率高很多

## 7. 评估方式

### 7.1 封装环境内自博弈评估

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode selfplay \
  --model combatbench/baseline/sb3/runs/fight_v1/model_final.zip \
  --phase fight \
  --episodes 5
```

这个模式用于看训练时同分布下的表现。

### 7.2 双边独立策略对战

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode match \
  --model combatbench/baseline/sb3/runs/fight_v1/model_final.zip \
  --duration 15 \
  --video combatbench/baseline/sb3/runs/fight_v1/demo_match.mp4
```

这个模式会通过 `SB3CombatPolicy` 适配器，把 SB3 模型接到 `MatchRunner` 上。

好处是：

- 能直接复用项目已有对战工具
- 能输出视频
- 更接近你最终想看的“两个机器人站立对打”效果

### 7.3 稳定导出视频的独立脚本入口

如果你遇到模块路径下的 headless 渲染问题，优先使用：

```bash
python3 combatbench/run_policy_video.py \
  --mode shared_env \
  --model combatbench/baseline/sb3/runs/stand_v1/best_model/best_model.zip \
  --phase stand \
  --duration 10 \
  --control-frequency 20 \
  --initial-distance 2.0 \
  --video combatbench/baseline/sb3/runs/stand_v1/stand_success_10s.mp4 \
  --device cpu
```

### 7.4 对战机制验证

在真正依赖 fight 训练结果之前，建议先验证环境里的敌方观测和战损逻辑：

```bash
python3 -m combatbench.baseline.sb3.validate_fight_mechanics \
  --duration 8 \
  --control-frequency 20 \
  --initial-distance 0.6 \
  --seed 0
```

这个脚本会验证：

- `opponent_*` 观测切片是否和权威状态一致
- `hit_records` 与 HP 变化是否一致
- 一整局里是否出现至少一方掉血

## 8. 推荐参数

如果你只是先跑通：

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 200000 \
  --run-name smoke_stand
```

如果你要认真训练：

- `stand`：`1M - 3M` timesteps
- `fight`：`2M - 5M` timesteps
- `control_frequency`：先用 `20Hz`
- `match_duration`：
  - `stand` 可先 `10s`
  - `fight` 可先 `15s`

如果你只关心站立训练本身，优先直接看：

- `STAND_TRAINING_GUIDE_zh.md`

## 9. 如果训练效果不好，优先调哪几个地方

按优先级建议如下：

1. `rewards.py` 里的 `height_weight / upright_weight / damage_reward_weight`
2. `fight` 阶段的 `target_distance`
3. `fall_penalty`
4. `ent_coef`
5. `n_steps / batch_size`

如果模型只会站，不愿意出手：

- 提高 `damage_reward_weight`
- 稍微提高 `distance_weight`
- 适当降低 `survival_reward`

如果模型会乱打但很快倒：

- 提高 `height_weight`
- 提高 `upright_weight`
- 提高 `fall_penalty`
- 稍微提高 `action_change_penalty_weight`

## 10. 当前 baseline 的边界

这套实现已经能作为正式的 SB3 baseline 起点，但还不是终点。

后续可以继续演进的方向：

- checkpoint opponent pool
- 历史策略采样
- 独立双边动作训练
- Elo / league training
- 更细的命中事件奖励
- 视觉观测版本
- curriculum（站立 → 接近 → 轻击 → 完整对打）

## 11. 最后的建议

如果你的目标是尽快看到“两个机器人能站着打一会儿”：

- 先只跑 `stand`
- 再小步微调 `fight`
- 每次训练后都录一段 `match` 视频看实际行为

在这种环境里，**视频回放比纯 reward 曲线更重要**。因为 reward 上升不一定等于动作真的像“站立对打”。
