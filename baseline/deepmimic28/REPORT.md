# DeepMimic 复现验证报告（humanoid28 / spinkick）

> 状态时间：2026-01-28 05:50。训练仍在后台继续（single u2500+/multi u2400+，max_updates=10000）。

## 1. 任务与结论摘要

目标：用 MimicKit（彭学斌）的 DeepMimic 方法 + 我方仿真/PPO/实验框架，在 humanoid28 上复现
spinkick 跟踪，借复现验证 humanoid28 仿真器正确性，并对比单 critic vs 多 critic。

**结论（诚实版）**：
1. **方法语义链已完整验证**：观测 464 维布局、动作归一化、奖励五项、终止条件、RSI、
   数据转换（0.0° 误差闭环）全部与上游逐行核对一致；上游预训练策略迁移测试证明
   动作语义正确（step-1 reward 0.708、root_pose 分量 0.98）。
2. **训练动态三个真实 bug 已修复**（每个都被实证为承重墙，非装饰）：
   Adam→SGD（ratio 死亡螺旋）、action_bound 正则（均值出界）、观测归一化（tanh 饱和）。
   修复后 ratio/KL/EV 全部进入上游同款健康 regime。
3. **但学习速度仍显著慢于上游**：38M 样本（≈上游 iter290）时确定性策略从相位 0 仅
   存活 8-11 步，上游同期 test 已存活近百步。训练侧 ep_len/ret 有缓慢正向趋势
   （7.5→8.3、ret 0.9→1.13）但未出现上游 iter400-500 的爆发。
4. **单 vs 多 critic 对比当前无区分度**：两者都未训出，eval 生存 8 vs 11 步，
   训练曲线上 single 持续小幅领先 multi（ep_len 8.3 vs 7.8）。

## 2. 判决实验结果：控制率确证为主因（2026-01-28 05:55）

**实验**：上游预训练策略直接迁移（raw obs + checkpoint 自带归一化），同策略同仿真器，
仅改控制率。结果：

| 控制率 | 存活 | root_dev@t20 | 结论 |
|---|---|---|---|
| 20Hz | 23 步（1.15s） | 0.76m | 一圈未完成 |
| 30Hz | **44 步（1.47s）** | 0.58m | **超过一个完整踢腿循环（1.28s）** |

**含义**：
1. humanoid28 物理上能够完成 spinkick——**仿真器本身不是瓶颈**（用户核心问题的第一答案）
2. 20Hz 是主要制约——注意本测试混淆两个因素（控制带宽 + tar_obs 前瞻语义匹配），
   但两者同向指向 30Hz
3. 即使 30Hz 下上游策略最终仍偏出 1m——Isaac→MuJoCo 跨引擎接触动力学差异客观存在，
   复现精度上限可能略低于上游报告值，但"能否完成动作"不受影响

**待你决策**：是否将跟踪管线迁到 30Hz 控制（物理 480Hz/16 子步可对齐 60fps 网格，
CONTROL_DT 单常量参数化，obs_norm.npz 需重新生成）。20Hz 训练仍在后台继续
（即使慢，趋势仍正向）。

上游控制 30Hz/物理 120Hz，我们 20Hz/物理 500Hz。spinkick 是高速动态动作，
反馈延迟 50ms vs 33ms 对动态平衡的影响是物理性的、非调参可补。
这是**方法栈外唯一剩余的重大差异**（其余：ReLU vs Tanh、arch 1024-512 vs 1024-1024、
fixed vs running 归一化、MuJoCo vs Isaac 接触数值，均次要或不可控）。


## 3. 验证过的关键事实（均可复查）

| 项 | 证据 |
|---|---|
| 数据转换 0.0° | refmotion quat-primary + hinge 查询时分解，闭环逐帧验证 |
| obs 464 维布局 | 与上游 checkpoint obs_norm 维度精确一致 |
| 动作归一化 | 与上游 a_norm 差 1e-5 |
| 奖励精确性 | RSI 落位后五分量全 1.0；key_pos_err 为求和、pose 终止用全部 15 身体（两处对上游源码修正） |
| tar_obs 语义 | global_obs 下相对受控角色 root（非参考鬼影），z 为绝对高度 |
| MuJoCo qvel[3:6] | body-local 系（数值验证），sim 已转世界系 |
| 上游 golden 曲线 | iter100/200/300/500 train_eplen = 5.6/10.9/19.3/52.0（Train 指标才是 RSI 可比项） |

## 4. 训练配方最终形态（DM28v3）

```
optimizer: SGD/SGD lr 1e-4（无 momentum，上游精确）
action_bound_weight: 10.0（上游精确形式）
obs: fixed 归一化 + clip±10（data/obs_norm.npz）
episodes_per_update 2048 / minibatch 8192 / workers 96
固定 std 0.05（register_buffer 冻结）、--no-confidence 必需
吞吐 ~7-9k 控制步/秒/run（RSI 帧对齐预计算 2.7× 优化后）
```

## 5. 运行 artifacts

- 训练：baseline/runs/dm28sk_single_cont / dm28sk_multi_cont（后台运行中）
- 当前 eval 视频：两 run 目录下 eval_mid.mp4（红色 robot 尝试动作但 8-11 步摔倒）
- 上游迁移验证脚本：eval_pretrained.py（注意：我方归一化上线后需加 raw-obs 开关才能正确跑）
- 完整差异清单与诊断历程：DESIGN.md §7-9

## 6. 若继续推进的建议顺序

1. （判决性）30Hz 迁移测试 → 若证实，决策控制率改造（物理 480Hz/16 子步可对齐 60fps 网格）
2. （并行）让当前训练跑满 100M+ 样本——上游曲线显示爆发前平台很长，慢不等于死
3. （若 1 证实仍失败）开环 PD 阶跃响应测试标定执行器数值，排查 humanoid28 接触参数
