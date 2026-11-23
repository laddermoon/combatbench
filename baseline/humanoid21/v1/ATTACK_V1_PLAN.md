# ATTACK V1：可部署攻击策略（Harness 线）

> 类型：方案（已定稿，决策点见 §7）

## 0 目标

把"分层控制器"做成**可部署 Policy**：脚本状态机（Harness 侧）+ 已训基座
（`v2_steparm_r1` 系），输入原始 obs96，输出 action21。v1 只攻不守，
攻击通道只用**拳（臂令序列）**——踢腿留给训练线。

## 1 部署形态

```
obs96 ─→ ego57 = obs[:57]                        ─┐
         opp_rel  = obs[57:60]  （方位/距离，ego系）├→ AttackPlanner(状态机) → cmd22
         opp_head = obs[66:69]  （击头目标，ego系） ─┘        │
obs79 = cat(ego57, cmd22) → steparm.act(obs79) → action21 ─→ env
```

- `AttackPolicy` 是 Policy ABC 的普通实现（脚本 wrapper 合法），
  打包为 blueprint 目录：`policy.py`（**自包含，禁 repo import**——
  导出模板 P0-6 教训）+ `model.pt` + `attack_params.json`。
- 规划全程在 **ego 系**闭环——opp_rel/opp_head 本身就是机体系量，
  不需要世界坐标或里程计。
- `cmd22` 布局即训练协议：`flag_r,tgt_r,flag_l,tgt_l | waist | step_cur | step_nxt`；
  step_next 由 planner 队列**诚实**预览（cur←nxt 契约）。

## 2 状态机（脚本）

| 状态 | 进入条件 | 输出 | 退出 |
|---|---|---|---|
| APPROACH | dist > strike_range | 步令朝对手：sag→缩短距离、dyaw→对bearing（P 纠偏） | dist ≤ strike_range |
| STRIKE | 进入打击圈 | 脚=hold 或微调；臂=拳 tgt 序列（每帧更新，无门控） | 序列播完 / 超时 |
| RECOVER | 序列完成 | 收拳回警戒 tgt；脚=hold | 臂回带 / 超时 → APPROACH |
| GUARD | 自身 phi < 阈值 | 一切令→hold，让底层站稳 | phi 恢复 → APPROACH |

- 步令重规划走 STEPACE 事件门控（land/age 兜底）；臂 tgt 每帧可更新
  （绝对量、非残差，无门控必要）。
- 拳原语 = 一或数帧臂 tgt 序列，离线扫出（§4）后固化进 params。

## 3 验收阶梯（三档）

| 档 | 对手 | 环境 |
|---|---|---|
| L1 冻结假人 | FrozenRobotPlugin(robot_b) | 战斗 env（96 维，scoring observer 开） |
| L2 站立 policy | balance/standing 基座 | 同上，robot_b 跑 policy |
| L3 移动 policy | stepcmd 行走基座 | 同上 |

指标：命中数、命中力、time-to-first-hit、击倒率/KO、
自身 min_phi、被推开距离、**近身 OOD 事件**（摔倒/被挡/贴脸停顿全记录）。

## 4 拳原语发现（CmdProbe 扩展）

CmdProbe 增加手部关键点指标（挥击通道的完成度度量）：

- `hand_pos_traj` / `hand_vel_traj`（ego 系，腕/拳头关键点）
- `reach_max`：拳峰最大前伸（ego +x）
- `v_peak`：手部峰值速度
- `settle_err`：序列结束后回带误差

扫描：tgt 网格（shoulder1/2×elbow）× 序列长度 → 选 `v_peak/reach`
最优、且 `min_phi` 不崩的组合。产出固化进 `attack_params.json`。

## 5 已知边界（先测后补）

- **近身 OOD**：底层从未见过对手碰撞——贴脸/被挡/被推都是分布外。
  v1 接受：OOD 事件计数、录视频，数据交给训练线决定是否开
  近身课程（S5）。
- `foot=0+dyaw` 不可表达（cmd_target 吞保持令转角）——STRIKE 期间
  需要转体对准时用"脚锚定"的移动令（目标=当前脚位）。
- waist 预留维忽略。

## 6 文件规划

| 件 | 说明 |
|---|---|
| `attack_planner.py` | 状态机 + ego 系纠偏（开发版，可 import repo） |
| `attack_policy.py` | 打包用自包含 wrapper（planner 逻辑内联） |
| `attack_accept.py` | 三阶梯验收 harness + 指标 |
| `v2_attack_env.yaml` | 战斗 env：scoring observer + 可配对手档 |
| `punch_sweep.py` | CmdProbe 拳原语扫描 |

## 7 决策记录（用户已拍板）

- D1 攻击通道：**拳**（臂令序列）；不含冲撞、不含踢
- D2 高层形态：**脚本状态机**（非学习式）
- D3 验收：**三阶梯全测**（冻结→站立→移动）
- D4 近身 OOD：**先测后补**，摔倒/被挡记为部署缺口
- D5（自定）：臂 tgt 每帧可更新；步令事件门控
- D8（自定）：指标集如上
- 未决：训练线 S5 内容（近身课程/对手感知/踢腿）——不阻塞本线
