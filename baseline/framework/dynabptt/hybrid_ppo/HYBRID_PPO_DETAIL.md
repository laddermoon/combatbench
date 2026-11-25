# Hybrid PPO + World Model — 实验细节日志

方案文档: `../PPO_WORLD_MODEL_EXPERIMENT_PLAN.md`（commit 183d0ce4）
实现: `hybrid_ppo/`（commit 4a642b54, 57d68ec2）

## [P0] 正确性门槛 — ALL PASS

- `tests/test_p0.py` 6 项全过：对齐抽取、时序（模型分支回放逐帧复现真实 cs/obs）、
  self-only 物理隔离（位掩码关闭 A↔B 碰撞，B 不同动作下 A 逐位一致）、
  PPO 语义（GAE/ratio≈1）、导出一致、垃圾模型被 gate 拒绝。
- 关键坑：插件 `set_episode_seed` 只由 EpisodeRunner 调用；手动 `rt.reset(seed=)`
  不触发 → 测试/验证中需手动给插件播种（见 test_p0.py `_seed_plugins`）。
- 时序确认：`obs[t]`=动作前 obs，`cs[t]`/`phi[t]`=动作后 s_{t+1}；模型窗口从
  `cs[t-1]` 出发，obs 头目标为 `obs[t+1]`/`final_observation`。

## [P1] 双臂对照（seed=42, 512eps/update, 20 updates 或 22min）

- Arm A `runs/hp_p1_ppo_s42/`（--arm ppo）pid 2487079, ~20-26s/update
- Arm B `runs/hp_p1_sidecar_s42/`（--arm sidecar）pid 2492244, ~31-42s/update
- 启动事故：B 首次启动因缺 logs 目录失败，重建后正常。
- RNG 分流生效：A/B 同 update 的 kl/ev 一致到 1e-3，sidecar 不扰动采集流。
- PPO 学习正常：u1 kl≈0.023，EV 0.41→0.96（cold start 后 critic 接管）。
- sidecar 开销 ≈ +10-15s/update（模型拟合60批+留出集验证+GPU分支评估）。

### 模型校准趋势（B 臂, metrics.jsonl 每 update）

| update | nrmse(linvel) | obs_nrmse | phi_mae | act_dev | adv_corr | gate |
|---|---|---|---|---|---|---|
| u5 | 0.65 | 0.52 | 0.017 | 0.35 | 0.21 | 0 |

阈值参考: nrmse_max 0.20, obs_nrmse 0.15, phi_mae 0.03(已过), act_dev 0.05,
adv_rel_rmse 0.50, adv_sign_frac 0.75。

### P1 收官（20 updates 双臂）

- A eval_max_pot: 0.171→0.189；B: 0.158→0.181 —— 曲线在噪声内重合，
  **sidecar 对 PPO 学习零干扰**，代价 ~+50% 墙钟（23s→35s/update）。
- B 模型校准 20 update 无收敛趋势：nrmse_lv ~0.60 平台，obs_nrmse
  0.90→0.46 缓降，adv_corr 0.1-0.3 噪声。gate 全程关闭（正确行为）。

## [标定] 离线密集训练实验 —— 结构上限判定

`offline_calib.py`：取 B 臂 ckpt_u00019 的 replay tail（4050 fit trajs /
806k trans + 800 hold），把模型单独密集训练，周期性跑留出集验证。

| 配置 | 训练量 | s1 linvel NRMSE h1/h3 | s1 obs h3 | act_dev h3 |
|---|---|---|---|---|
| 512³ 续训 | +1200 批 | 0.61→0.60 / 0.66 | 0.44 | (见下) |
| 1024³ 从头 | 1500 批 | **0.50** / 0.56 | 0.45 | 0.095 |

- 训练损失持续下降（wm_s 0.13→0.12）但留出集指标平台化 → **不是欠训练**
- 4× 容量只换 ~18% 误差降幅 → 收益递减，距阈值（0.20）仍 2.5×
- phi_mae 0.02 唯一达标；act_dev（修复后）h3=0.095，接近但未达 0.05

## [Bug] act_dev_rms 指标不可达（已修复）

原实现拿模型驱动的确定性动作比对**录制的随机动作**——探索噪声 σ≈0.3
使指标地板就是 ~0.35，阈值 0.05 永不可达。改为对比 `det(ô_{t+i})` vs
`det(o_{t+i})`（i=0 用 obs[t]，i≥1 用 next_obs）。修复后 h3 act_dev=0.095。

## [根因] 接触冲击主导误差 —— 实测

400 个留出集单步预测，按真实 |Δv|（30ms 内速度变化）分层：

|Δv| 区间 | 占比 | 平均 linvel_err | 平方误差份额 |
|---|---|---|---|
| <0.1 m/s | 1.8% | — | 0.7% |
| 0.1-0.3 | 31% | 0.167 | 14.6% |
| >0.3 | **67%** | 0.268 | **84.7%** |

corr(|Δv|, err)=0.60。该环境几乎所有转移都含接触冲击（脚触地/摔倒），
平滑 MLP 只能拟合冲击响应的条件均值 → NRMSE 地板 ~0.5。
**结论：学习型点估计世界模型在此系统的精度上限，不足以支撑
advantage 估计（配对 adv_corr≈0.2）。gate 按设计正确拒入。**

## 终局判定

Hybrid PPO+WM 路线在 standup 上**按方案停止规则终止**：

- ✅ P0 全过（时序/隔离/PPO语义/兜底）
- ✅ P1：sidecar 安全无干扰，PPO 正常学习（EV 0.96，eval 缓升）
- ❌ 模型校准：结构上限，gate 永不开启 → P2 无意义（assist=sidecar）
- 工具链保留价值：对齐抽取/隔离环境/门控验证框架均可复用

代码: `hybrid_ppo/`（standup/transitions/model/validation/ppo/rollout/train/
offline_calib + tests），commit 至本文件提交点。
