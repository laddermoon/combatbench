# 两端点打击训练实现审计

日期：2026-10-20。审计对象：`strike_2phase`。
不以此审计结果推断“端点学习不可行”，不修改 5% 门、MSE/方差或热身分布。

## 1. 数据来源

- 原 run：`train_strike_2phase_ppo_20261020_154404`，u6686–7189
  共 504 条 update 统计；日志明确记录在 u7190 因实验早停结束。
- 使用框架 `baseline/framework/ppo/debug.py` 的 `summary`、`metrics`、
  `inspect`、`query ... merge`，并直接读取 dump 中的 ndarray 复算。
- 诊断 run：`audit_strike_2phase_endpoints`，从原 run u7185 恢复，
  仅跑 u7186，32 episodes、4 workers、1 epoch，关闭 confidence；
  `--dump-at 7186` 生成 `dumps/u07186`。这是缩小批量的诊断截面，
  **不是原训练 u7186 的逐位复现**。
- 修正验证 run：`audit_strike_2phase_terminal_fix`，从用户指定的
  `train_skill_full_mbs_ppo_20261020_104315/checkpoints/checkpoint_u06685.pt`
  恢复，`--reset-update --smoke --no-confidence --dump-at 2`。

示例读取：

```bash
PYTHONPATH=. python3 baseline/framework/ppo/debug.py summary train_strike_2phase_ppo_20261020_154404
PYTHONPATH=. python3 baseline/framework/ppo/debug.py inspect audit_strike_2phase_endpoints:u7186
PYTHONPATH=. python3 baseline/framework/ppo/debug.py query audit_strike_2phase_endpoints:u7186 merge
```

## 2. 已确认的训练语义缺陷：失败轨迹错误 bootstrap

此前 `_piece` 只有成功时才终止打击通道。未过准备门的整条 30 步
轨迹，或者过门但未到落地姿态的 p2，都保持 `is_terminated=False`。
于是失败任务以 `V(final_observation)` 续算，而非以预算耗尽结束。

同时 `CommandSimulator._strike_target_ego()` 仅在
`seg.a <= action_step < seg.b` 时返回回放目标。第 30 步后的 final
observation 落在窗口外，手靶回退为 schedule 的静态目标，训练中间帧
却仍使用动态回放。这给失败轨迹的尾值增加了未受当前轨迹直接训练的输入。

在诊断 dump（64 条全失败 p1 轨迹）中：

| 项 | p1 MSE critic | p1 variance critic |
|---|---:|---:|
| 尾状态 V 均值 | +20.901 | +10.769 |
| 尾状态 V 最大值 | +63.736 | +32.698 |
| 中间帧 V 均值 | +18.068 | +8.690 |
| GAE return 均值 | +17.472 | +8.915 |

MSE 通道的真实任务回报上界为 +1（平时非正，只允许成功尾帧一次
额外 +1）；variance 通道始终非正。上述正值不符合任务定义。
原 run 末期 EV 约 0.992/0.976 只说明 critic 拟合了自己的训练 target，
**不能证明它估计了正确的任务回报**。

末步之前到 final observation，手靶跳变均值分别为 0.439m/0.395m
（两机器人）。固定 final 的本体观测、只把命令换回上一帧命令，
MSE 尾 V 平均下降 0.330，最大下降 13.026；variance 平均下降 1.285。
这证实命令跳变影响尾值，但不足以把全部 critic 漂移归因于这一个输入。

修正：

- 打击通道在成功或 30 步失败时都终止，尾值置零。
- `r_potential` 只在内部 p1→p2 边界继续 bootstrap；任务结束时终止。
- 无成功时不添加 +1；成功帧与后续丢弃规则不变。
- 不继续使用本轮已漂移的 p1 critic；修正验证从原 u6685 出发。

修正后的两次 PPO 更新 smoke 已通过。u2 dump 的 16 条失败轨迹
全部终止，`bootstrap_indices=[]`；p1 MSE 的 GAE return 均值
为 −0.791（范围 −1.670 至 −0.026），variance 为 −0.076
（范围 −0.279 至 −0.001）。这是机制验证，不是收敛结论。
6 个回归测试覆盖成功/失败、末帧过门、尾帧奖励、截断及指标字段。

## 3. 已排除的底层错误

- 参考与 simulator 的 21 个关节顺序一致。
- 关节 limit 最大差约 `8.1e-8` rad；归一化均为 `(q-mid)/half`。
- 以 dump 的下一帧观测（末步用 final_obs）复算去踝 MSE/variance，
  与 observer 输出的最大差分别为 `8.3e-17` / `5.6e-17`。
  未发现奖励符号、关节切片或 action→后状态时序错位。
- 诊断 dump 中每条轨迹只含 p1 与护栏；p2 没有偷偷参与。
- 两个 p1 归一化 advantage 的相关系数约 +0.521，不支持“全局
  优势完全互相抵消”的说法。局部目标竞争仍可能存在。

## 4. 已修正的指标错误

- `on_eval` 原来读取不存在的 `foot` 字段；observer 实际发射
  `foot_disp_m`，导致后脚位移指标被漏掉。
- 过门帧发射的 `gated` 仍为 0，原 eval 会把恰好最后一帧成功的
  episode 记为未过门；改为同时查看当帧已 latch 的 `gate_t`。
- 原 `gate_qmax > 0` 过滤会丢弃真正的零误差；改按实际阶段过滤。

这些指标错误不是当前零过门的解释，但必须修正。

## 5. 仍需区分的设计问题与物理难点

### 指令回放与当前端点目标不同步

阶段一一直优化 clip 0；命令手靶却无论过门与否都按 episode 时钟
回放完整 clip。Heavy 在第 6 步就经过 landing，13 帧后停在恢复末帧；
Jab 对应为第 11 步、22 帧。actor 没有显式的 p1/p2 状态或目标关节
姿态输入。它需要从不断变化的手靶间接识别固定的全身目标。

这是实际存在的条件输入不一致，尚未在本次修复中更换命令契约。
后续可在仍然只学两个端点的前提下，让当前目标明确可观测：
过门前固定准备目标，过门后固定落地目标，或者加入阶段与目标姿态。

### 不只是一个关节卡住

诊断样本中，每条 episode 选取 qmax 最小的那一帧后，平均仍有
约 9–13 个去踝关节超过 0.10。不能根据平均 MSE 与 qmax 的差异
直接断言“只有一两个关节”或“梯度已经为零”。

### 原始参考帧不等于可直接稳定站住的姿态

在 MuJoCo 中设置参考 root/关节姿态和零速度，然后用参考 q_norm
直接作为 PD action 保持 30 步：

- Jab 准备帧有约 6.6cm 脚部穿地与约 0.6cm 上臂/腰部穿插；
  首个控制步后 qmax 已约 0.186，随后失衡。
- Heavy 准备帧有约 1.5cm 脚部穿地；首步 qmax 约 0.014，
  但直接固定 PD target 仍不能持续保持。

该实验说明不能把“参考关节角作为 action”当作已验证的稳定教师。
**它不能证明 5% 目标不可达**：任务放开踝关节，root 姿态和高度
也可调整；直接 PD 缺少负载补偿与闭环平衡。

### 原报告中需要纠正的推断

- pre-gate 的 `hand_err_m` 来自 clip 0 的匹配，而不是证明手已经
  打到 landing；小手误差不能证明“机器人绕过准备姿态直接出拳”。
- `phi` 是包含接触、承重与高度的多阶段站立势能，并非纯高度。
  reward 乘 0.01 也不等于 actor 影响自动缩小百倍：框架对每个
  通道分别标准化 advantage，再按 AW 合并。

## 6. 下一步顺序

先验证终止修正、冷启动 critic 后不再出现不可能的尾值，并看真实
姿态误差走势，再判断是否需要改变学习目标。若端点仍不收敛，优先
做单技能、明确当前目标、近目标起手的可达性实验；通过后再扩展起手
分布。无需在这一步引入中间帧追踪或放宽验收门槛。
