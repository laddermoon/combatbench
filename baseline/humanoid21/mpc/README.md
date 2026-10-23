# Humanoid21 Observation-only MPC Baseline

> 当前范围：单机标准站立保持。对手不参与动作规划，在分支仿真中固定为
> canonical standing PD target。

## 目标

验证 CombatBench 的第一条 MPC baseline 路径：

```text
policy-visible 96-D Observation
  -> 解码自身 CoreState（固定全局 yaw gauge）
  -> 注入独立 branch Humanoid21Simulator
  -> rollout 候选低层反馈控制参数
  -> 执行评分最高候选，并在短周期内复用
```

这是一个**有权限边界的 MPC**：策略不读取真实 `MjData`，不调用
`get_core_state()`，也不读取对手内部状态。branch simulator 只从
Observation 可解码出的自身状态开始推演。

## 当前实现

### Observation 解码

`observation_codec.py` 提供 `decode_self_observation()`：

- `obs[0:21]`：关节位置；
- `obs[21:42]`：反解 `sign*sqrt` 关节速度；
- `obs[42:45]`：projected gravity；
- `obs[45]`：root height；
- `obs[46:49]`：机体系 root 线速度；
- `obs[49:52]`：反解 `sign*sqrt` root 角速度；
- `obs[54:57]`：场地中心在机体系中的位置。

Observation 不包含绝对 yaw。当前圆形场地和重力方向使全局绕 Z 旋转成为
对称自由度，因此解码器选择一个 canonical yaw gauge，重建一个与真实状态
**动态等价但不保证绝对 yaw 相同**的 root pose。

### 候选控制律

`standing_mpc.py` 中的 `FeedbackGains` 定义低层反馈参数：

```python
pitch = pitch_angle * projected_gravity[0] + pitch_rate * angular_vel_y
action[ankle_y_left/right] = standing_action + pitch
```

默认候选是一组已经验证过可在 2 秒 branch rollout 中保持站立的增益。
没有加入“零增益 / 弱增益”作为默认可选项：短 horizon 下它们因动作偏差小
而占优，但会导致缓慢的 forward-pitch 失败。

### 分支仿真

每次 replan：

1. 重置一个私有 `Humanoid21Simulator`；
2. 将 Observation 解码出的自身 CoreState 注入受控机器人；
3. 将对手放到远处并保持 standing target；
4. 用 `mj_copyData` 在该近似初始状态上评估每个候选；
5. 每个候选在 branch 中以闭环方式运行：每个控制步重新读取
   Observation-equivalent 状态，再用候选 gains 生成动作。

这避免了从真实 simulator 偷取 solver/contact hidden state；剩余误差来自
Observation 不可见的 solver/context，而不是权限越界。

### 计划复用

MPC 是有状态的：

- 每 `replan_interval` 个控制周期重新评估候选 gains；
- 中间周期继续执行当前 gains 对当前 Observation 的反馈输出；
- 不是盲放一段固定动作，而是复用“已验证的闭环控制参数”。

## 当前默认参数

- 物理步：`0.002 s`
- 控制步：`0.05 s`（25 个物理步）
- MPC horizon：40 个控制步（2 秒）
- replan interval：10 个控制步（0.5 秒）
- 默认控制对象：`robot_a`
- 对手：固定 standing target

## 验证结果

运行：

```bash
PYTHONPATH=. python3 baseline/humanoid21/mpc/run_standing.py --steps 200
```

当前本机结果（10 秒真实 rollout）：

```text
min_height          1.281266
final_height        1.281415
min_uprightness     0.998766
final_uprightness   0.999378
plan calls          20
mean plan latency   0.734 s
p95 plan latency    0.742 s
```

`robot_b` 也可用同一接口运行：

```bash
PYTHONPATH=. python3 baseline/humanoid21/mpc/run_standing.py \
  --steps 40 --robot-id robot_b --horizon 20 --replan-interval 5
```

## 测试

```bash
PYTHONPATH=. pytest -q baseline/humanoid21/mpc/test_standing_mpc.py
```

当前结果：

```text
4 passed
```

## 已知边界

1. **不是 bit-exact simulator clone**：Observation 不包含 contact solver
   warm-start、插件状态、完整对手关节状态。当前场景把对手排除在规划外，
   所以这些缺失不会破坏单机站立任务。
2. **绝对 yaw 不可恢复**：重建的是全局 yaw 等价状态。对圆形 arena、
   单机动力学足够；若以后需要 wall-directed 或绝对目标位置，必须明确
   gauge 或开放 richer privileged state。
3. **当前只做 pitch recovery**：候选 gains 主要修正 forward/backward
   pitch；侧向大步、外部推力恢复、迈步还没有纳入。
4. **规划延迟不满足 50 ms 实时控制**：默认一次 replan 约 0.73 s。当前
   先验证正确性；后续可通过更少候选、分层 horizon、多进程候选评估或
   学习化候选提案降低延迟。
5. **对手被固定**：这是有意限定。未来可将对手动作也作为观测到的不确定
   输入，或在若干对手扰动场景下做 robust rollout。

## 文件

```text
observation_codec.py   # 96-D Observation -> 自身 CoreState（canonical yaw）
standing_mpc.py        # StandingMPCPolicy + FeedbackGains + branch rollout
run_standing.py        # CLI 验证脚本
test_standing_mpc.py   # 解码、动作合法性和 2 秒站立集成测试
```
