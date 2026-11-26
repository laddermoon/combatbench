# DeepMimic 复现 — 实现说明

> 类型：设计文档（先于实现）
> 目标：在 `baseline/deepmimic/` 内自包含地复现 DeepMimic
> （Peng et al., SIGGRAPH 2018）单 clip 运动跟踪训练管线。
> 参照但不 import `mocap/humanoid21/tracking/`、`exp_track_walk.py`、
> `track_eval.py`（那些是 M2 临时产物，后续会清理）。

## 1. 范围

- **做**：单 clip → 单策略 PPO 模仿训练；训练后确定性整段回放 + 视频 + 指标
- **不做**：多 clip 技能策略 / motion prior / 循环步态（见 §6 局限）
- 数据：`mocap/humanoid21/data/core/*.npz`（172 条 CoreClip，
  `keypoint_pos` 重导出后）

## 2. 文件布局

```
baseline/deepmimic/
├── DESIGN.md            # 本文档
├── README.md            # 快速使用
├── __init__.py
├── ref.py               # CoreRefView：CoreClip 的跟踪侧视图
├── sim.py               # DeepMimicTrackSimulator(Humanoid21Simulator)
├── plugins.py           # DeepMimicGuardPlugin（终止判定）
├── reward.py            # DeepMimicRewarder（模仿奖励 observer）
├── clip_presets.py      # 按 clip meta/label 推导环境参数
├── envs/
│   └── track_env.yaml   # ParameterizedEnvBlueprint
├── experiment.py        # DeepMimicTrack(CombatExperimentPPOBase)
├── train.sh             # 启动脚本：./train.sh <clip_id> [opts]
└── eval.py              # 确定性整段回放评估 + mp4

baseline/experiments_ppo/
└── exp_deepmimic.py     # 注册壳（registry 唯一识别入口，~5 行）
```

`experiments_ppo` 的 registry 只扫描该目录的 `exp_*.py`，故实验类本体放
deepmimic、壳文件做一行转接：`from baseline.deepmimic.experiment import
DeepMimicTrack; EXPERIMENT_CLASS = DeepMimicTrack`。

## 3. 数据路径：CoreClip → 参考帧

**不需要把 CoreClip 伪装成旧 RefClip。** `clip.state(k)` 返回的六字段字典
（`root_pos/root_rot/root_vel_local/root_angular_vel_local/joint_pos_norm/
joint_vel_norm`）与 `sim.get_core_state()`/`set_core_state()` 的原生格式
一一对应。`ref.py` 只做语义映射：

```python
class CoreRefView:
    """CoreClip 的跟踪查询视图。"""
    def __init__(self, clip_id_or_path, clips_dir=DATA_CORE)

    n_ctrl: int            # 控制帧数 = clip.T
    duration_s: float
    meta: dict             # labels / split / phc_quality_level / issues
    rsi_frames: ndarray    # range(T) - meta['rsi_bad_frames']
    keypoint_names: list   # meta['keypoint_names']（重导出后存在）

    def at(k) -> dict:
        frame, phase=k/(T-1),
        q_norm, qv_norm,                    # 直接读 states
        q_rad  = q_norm * half + mid,       # CONTROLLED_JOINTS limits
        qv_rad = qv_norm * half,
        root_pos, root_quat,                # root_rot 直通
        root_linvel_world = R(root_rot) @ root_vel_local,
        root_angvel_world = R(root_rot) @ root_angular_vel_local,
        root_vel_local, root_angvel_local,  # obs 用（见 §4.1）
        ref_action,                         # clip.ref_action[k]（PD 目标）
        keypoints = clip.keypoint_pos[k],
```

`clip_stats()`：`net_dxy`（首末控制帧根 xy 位移）、`net_dyaw_deg`、`T`、
`labels` —— 供 `clip_presets.py` 推导参数。

## 4. 组件设计

### 4.1 `sim.py: DeepMimicTrackSimulator(Humanoid21Simulator)`

- `__init__(clip_id, clips_dir, track_robot='robot_a', rsi='random', **kw)`
- **reset**：`k0 = rng.choice(ref.rsi_frames)`（`options['start_frame']`
  可固定，`rsi='first'` 取 `rsi_frames[0]`）；然后
  `set_core_state({track_robot: ref.at(k0)-CoreState})` 一次落位，
  `_target_pos_norm[track_robot] = ref_action[k0]`（PD 目标对齐避免
  首步冲击）；`_frame = k0`
- **set_action**：`super()` 后 `_frame += 1`（相位 = episode 时间轴）
- **get_observation**：`ref[k+1]`（next-frame 目标，M2 v3 教训）扩展 54 维
  追加到 track_robot 的 96 维基础观测 → obs 150：

  | 段 | dim | 来源 |
  |---|---|---|
  | phase | 1 | `k/(T-1)` |
  | ref_q_norm | 21 | states |
  | ref_qv_norm ×0.2 | 21 | states |
  | ref_root_linvel_refframe | 3 | **= `root_vel_local`**（CoreClip 原生
     就是"参考朝向系投影"，v5 手动投影的语义免费获得） |
  | root_disp_ego ×0.5 | 3 | `R_self⁻¹(ref_pos − self_pos)` |
  | dyaw sin/cos | 2 | `wrap(yaw_ref − yaw_self)` |
  | ref_root_angvel_refframe | 3 | = `root_angular_vel_local` |

  根级误差用 ego 系而非绝对量——基础 96 维刻意不含自身绝对 xy/yaw，
  发绝对目标 = 不可观测（M2 v4 教训）。
- **get_derived_state** 增加 `motion_track` 键：`ref_at(k)` 全集 +
  `self_keypoints`（track_robot 的 KEYPOINT_BODIES 世界系 xpos，body id
  由 KEYPOINT_BODIES+suffix 解析）+ `frame/phase/ended`。
- `track_ended` = `_frame >= n_ctrl - 1`

### 4.2 `plugins.py: DeepMimicGuardPlugin(BasePlugin)`

`on_post_action_step` 按序判定（先判 clip_end）：

1. `ended` → `track_clip_end`
2. **fall 判定**，三种模式（蓝图参数 `fall_mode`，见 §5-D6）：
   - `height`（默认）：`root_z < fall_height` 或 `head_z < pelvis_z`
   - `contact`：`get_derived_state('contacts')` SoA 中 track_robot 的
     **非足 geoms**（foot1/foot2_{l,r} 以外）接触地面/墙 → 更接近论文
     "undesired contact"；需 geom→body 反查表
   - `none`：关闭（getup/lie/crawl 类 clip 的合法低姿态）
3. **drift 判定**（阈值非 null 时）：`dxy > drift_dxy` 或
   `|dyaw| > drift_dyaw` → `track_drift`（把根跟踪从软奖励升级为生存
   条件，M2 v6 关键修复）

### 4.3 `reward.py: DeepMimicRewarder(BaseObserverPlugin)`

```
reward = w_p·r_pose + w_v·r_vel + w_kp·r_kp + w_h·r_height + w_root·r_root
r_pose   = exp(-2   · mean((q_norm  − ref_q_norm)²))
r_vel    = exp(-0.02· mean((qv_norm − ref_qv_norm)²))
r_kp     = exp(-40  · mean(pelvis系关键点误差²))   # torso/head/双手/双足
r_height = exp(-20  · (root_z − ref_root_z)²)
r_root   = 0.5·exp(-3·dxy²) + 0.5·exp(-1·dyaw²)
```

每步输出 dict：`reward` + 分项 + `kp_err_m/kp_err_rel_m/root_dxy_m/
root_dyaw_deg/q_rmse_deg/frame/phase`。权重五元组蓝图可配。

### 4.4 `envs/track_env.yaml`

ParameterizedEnvBlueprint，参数：`clip_id, clips_dir, rsi, max_steps,
fall_mode, fall_height, drift_dxy, drift_dyaw_deg, initial_distance,
w_pose/w_vel/w_kp/w_height/w_root`。runtime `phy_steps_per_action=25`。

### 4.5 `experiment.py: DeepMimicTrack(CombatExperimentPPOBase)`

- `name="deepmimic"`，`obs_dim=150`，`action_dim=21`
- `clip_id: str = ""`（必须 `--set clip_id=X`）；`clips_dir` 默认
  `mocap/humanoid21/data/core`
- `_env_pb()` 覆写：从 `deepmimic/envs/track_env.yaml` 加载
- `build_jobs`：`materialize(clip_id, rsi, max_steps, fall_mode,
  drift_*, 奖励权重)`；`clip_presets.suggest()` 自动补
  `max_steps=n_ctrl` 和 label→fall_mode/drift 默认值
- `build_trajectories`：单通道 `r_track`，`track_fall/track_drift/
  track_clip_end` → 真终止，`timeout` → bootstrap
- `on_eval`：`survive/complete_rate/fall_rate/kp_err/q_rmse/dxy/dyaw`，
  best = `survive − 0.5·fall_rate`
- PPO 超参沿用 TrackWalk 已验证值（lr 3e-4, kl 0.05, γ .99, λ .95,
  epochs 4, eps/update 256）

### 4.6 脚本

```bash
# 训练（auto max_steps=n_ctrl、run-name=dm_<clip>_<ts>）
./baseline/deepmimic/train.sh cmu_13_17            # 后台
./baseline/deepmimic/train.sh cmu_13_17 --smoke    # 冒烟
./baseline/deepmimic/train.sh cmu_13_17 --fg --set drift_dxy=0.5

# 评估视频（确定性 rsi=first 整段，M2 验收同款）
python3 baseline/deepmimic/eval.py \
    --policy baseline/runs/dm_cmu_13_17_*/policy_exports/u01900 \
    --clip cmu_13_17 --video /tmp/eval.mp4 --json /tmp/metrics.json
```

`eval.py` 指标 = M2 验收口径：completed / phase_reached /
reward_mean / kp_err_m / kp_err_rel_m / dxy / dyaw / q_rmse（mean+max）。

训练中视频由框架自动产 `runs/<run>/videos/uNNNNN.mp4`
（`video_eval_interval` 控制）。

## 5. 与 DeepMimic 论文的差异清单

论文配方（对照基准）：`r = .65·r_p + .10·r_v + .15·r_ee + .10·r_com`；
`r_p=exp(−2·Σw_j·Δq_j²)`（per-joint 权重）；`r_v=exp(−0.1·Σw_j·Δq̇_j²)`；
`r_ee=exp(−40·Σ·世界系误差²)`（手脚）；`r_com=exp(−10·Δcom²)`；
早停=非足 body 触地；RSI 均匀采样；标量 phase obs；PD 目标动作；PPO。

| # | 组件 | 论文 | 本实现 | 性质 |
|---|---|---|---|---|
| D1 | r_pose | 四元数差 + per-joint 权重 + 含 root | 21 hinge 归一化角 MSE、均匀权重、root 不在此项 | **近似等价**：hinge 关节四元数差≡角度差；root 姿态由 r_root(dyaw) + kp + height 分担；per-joint 权重可实现（蓝图 dict），默认不开 |
| D2 | r_vel | 同 pose 权重结构 | norm 均方、均匀 | 近似等价（均匀权重） |
| D3 | r_ee | 世界系手脚位置 | 骨盆系相对位置（torso/head/双手/双足） | **有意偏差**：M2 v2 实测世界系 kp 对平移滞后过罚导致退化；代价=ee 不约束全局轨迹 → r_root + drift 终止补偿 |
| D4 | r_com | 真 COM 项 | r_height(root z) + r_root(ego dxy/dyaw) | **偏差**：无显式 COM；r_root 是论文没有的新增项——修复"原地打拳"退化的关键（骨盆系 kp + 关节 pose 对根 yaw/xy 完全不变） |
| D5 | 权重 | .65/.10/.15/.10 | .35/.10/.25/.10/.20 | 偏差（M2 六轮调参结果，蓝图可改回论文值） |
| D6 | 早停 | 非足 body 触地 | 高度代理 + 可选 contact 模式 + drift | **偏差**：contact 模式可实现（contacts SoA 已可用）；head<pelvis 在 getup/lie/crawl 类 clip 误伤 → per-clip fall_mode |
| D7 | 循环相位 | 步态 clip phase 循环 | clip_end 终止 | **不支持**：wrap 处状态不连续（边缘帧本就在 rsi_bad_frames）；想循环需源数据本身循环对齐 |
| D8 | obs | 本体感知 + 标量 φ | 96 战斗 obs + 54 参考 ext（含 next-frame ref pose/vel） | **偏差（更简单）**：论文只给 φ；`ref_obs_mode=phase_only` 可退化到论文设定做消融 |
| D9 | 场景 | 单角色空场 | robot_b standing@6m 场景机 | **偏差（已选）**：B 不参与评分；A 的 obs 含对手维（近常数，无害浪费） |
| D10 | 控制率 | ~30-60Hz 控制 / ~1.2kHz 物理 | 20Hz 控制 / 500Hz 物理（CONTROLSPEC 契约） | **限制**：快速段带宽不足，M2 已知 foot jitter/mild tremor；不可改 |
| D11 | PD | 位置 PD 目标 | 同 | 等价；均无目标速度前馈项 |
| D12 | RSI | 均匀采样 | 同 + rsi_bad_frames 排除 | **改进**：避免非物理速度 seed（差分放大的跳变帧 16-20 rad/s） |
| D13 | 参考速度 | retarget 原生 | 20Hz 网格中心差分重导（v·dt=Δq 严格成立） | 偏差（自洽性更好，轻度低通） |
| D14 | 策略类 | Gaussian 系 | TruncatedNormalPolicy（有界探索） | 偏差（蓝图可换 Gaussian 系 init_policy） |
| D15 | 数据可执行性 | 假设 clip 物理可达 | kinematic retarget，**未验证**；172 条中 91 条 L1_veto | **数据限制**：phc_issues 白名单选用（foot_penetration 踝映射残差可吸收；frame_jump 用 rsi_bad_frames 屏蔽） |
| D16 | 步态基元 | — | 缺动作平滑项 → M2 观察到手/脚高频颤动 | 已知缺陷（reward 无 action-rate 惩罚），可在奖励侧补 `r_smooth` 留作后续 |

## 6. 验证协议

1. `--smoke` 跑通（2 updates / 8 eps）
2. **回归基准 `cmu_13_17`**（core 内有，M2 v6b 旧管线已验收：
   complete=1.0, dxy 0.28m(max .66), dyaw 10.8°(max 40°), kp_err
   0.37/0.16m, q_rmse 21°(max 34°)）——新管线同指标对齐
3. accad punch + kick 各一条 + 一条 walk（验证净位移 clip 的 drift 自适应）
4. 验收以 `eval.py` 确定性回放为准（训练 eval 指标会掩盖原地退化，
   M2 v3 教训）

## 7. 待决策/开放问题

1. **PHC bundle 可恢复** → `phc2core --bundle core` 重导出全部 keypoints；
   不可恢复 → 写 patch 脚本对现有 npz 原地 `fk_keypoints` 补字段（等价，
   FK 源就是 CoreClip states 本身）
2. **label→fall_mode 默认映射**：getup/fall/recovery + general_misc 里的
   lie/crawl → `contact` or `none`？（建议 contact，论文语义）
3. **`ref_obs_mode`**：默认 full（54d ext）；phase_only 消融留参数位
4. val split 目前只 1 条——若要多 clip 评估可再标
