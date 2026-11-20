# V1 训练日志（追加式）

> 每次训练结束后追加一条记录。字段：实验名 / Run 目录 / 启动命令 /
> 目的 / 结果。不删改历史条目。

---

## 2026-11-17 `v2_s1_standup_step` — 79 维从零起身（S1）✅ 收敛

- **实验名**：`v2_s1_standup_step`（`experiments_ppo/exp_v2_s1_standup_step.py`）
- **Run 目录**：`baseline/runs/v2_s1_standup_step`
- **启动命令**：
  ```bash
  cd /data1/mono/things/combatbench
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s1_standup_step --algo ppo \
    --run-name v2_s1_standup_step --background
  ```
- **目的**：新输入维度模型第一段（D8 布局 79 维 = 57 本体 + 22
  指令段全零，flag=0 策略自控双臂）。参照 `hier_s1_s42i_scratch`
  配方：倒地→起身→s42i 标准踱步单阶段，奖励不变
  （r_potential + r_std_clip φ² 门控，γ0.99/λ0.95，MBS K=3）。
- **结果**：跑满 max_updates=1500（~10.2s/update，2026-11-17
  18:08 起 ~4.2h）。**收敛达标**：
  - u250 即 success=1.0（起身收敛速度与 66 维参照持平）
  - 终局 eval（u1485–1500）：**success≈1.0，clip=0.748，
    cad=0.951，h≈1.20，cycles≈17.1，max_stage≈4.0，
    final_pot≈0.94**
  - 对比参照 `hier_s1_s42i_scratch`（clip=0.607）：**79 维零指令段
    不损害收敛，clip 反而更高**——指令段零初始化输入层无干扰
  - 最佳导出：`baseline/runs/v2_s1_standup_step/policy/`
  - 代码快照：`exp/v2_s1_standup_step_20261117_180835`
    （commit `d5762ded`）；冒烟残留
    `runs/train_v2_s1_standup_step_ppo_20261117_180805`（u2）可忽略

---

## 2026-11-18 `v2_s2_balance` — 79 维推力平衡课程（S2）✅ 封顶手动停

- **实验名**：`v2_s2_balance`（`experiments_ppo/exp_v2_s2_balance.py`）
- **Run 目录**：`baseline/runs/v2_s2_balance`
- **启动命令**：
  ```bash
  cd /data1/mono/things/combatbench
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s2_balance --algo ppo \
    --resume-from baseline/runs/v2_s1_standup_step/checkpoints/checkpoint_u01500.pt \
    --reset-update --run-name v2_s2_balance --background
  ```
- **目的**：S1 收敛模型续训 12 级推力课程（40N→100N，recovery≥0.8
  ×2 晋级），保持/恢复 s42i 标准步态。79 维指令段全零 flag=0——
  策略自控双臂、臂可用于抗推平衡。通道：r_potential·3 +
  r_fall·2 + r_std_clip·2φ²·非受推（与 hier_s2_balance 相同）。
- **结果**：u1690 手动停（2026-11-18 00:50 起 ~9.5h，
  ~20s/update）。首个 eval 即 success=1.0（warm 权重立即生效）。
  - 课程进度：u5→265 level 0→5；u840 level 7；**u1255 level 11
    封顶**（100N 最高级），后 76 eval 平台期
  - **最佳 u1680（new_best）**：success=1.0，recovery=0.901，
    push_falls=75/760（9.9%），clip=0.355，cad=0.404，
    cycles=23.0，h 均值≈1.17
  - 对照参照 `hier_s2_balance_s42i_b` 终局（recovery=0.948、
    push_falls=40/764≈5.2%、clip=0.388、cad=0.61）：封顶同级但
    **抗推恢复质量与步态保真度略逊**——100N 下步态节奏变形
    更大（cad 0.40 vs 0.61），recovery 差 ~5pt
  - 停训依据：level 11 后 recovery 在 0.86–0.90 振荡 ~400
    update 无趋势，与参照同为手动停
  - 续训起点：`checkpoints/checkpoint_u01680.pt`（供 S3
    warm-start）；最佳导出 `policy/`；快照
    `exp/v2_s2_balance_20261118_005049`（commit `068865aa`）

---

## 2026-11-18 `v2_s3_balance_arm` — 随机臂脚本接管推力平衡（S3）✅ 已收

- **实验名**：`v2_s3_balance_arm`（`experiments_ppo/exp_v2_s3_balance_arm.py`，
  继承 V2S2Balance）
- **Run 目录**：`baseline/runs/v2_s3_balance_arm`（原始）→
  `v2_s3_balance_arm_r2`（u1450 停后续训）→
  `v2_s3_balance_arm_r3`（重启掉线续训）
- **启动命令**（warm-start S2 最强 ckpt u1680，`--reset-update`）：
  ```bash
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s3_balance_arm --algo ppo \
    --resume-from baseline/runs/v2_s2_balance/checkpoints/checkpoint_u01680.pt \
    --reset-update --run-name v2_s3_balance_arm --background
  ```
- **目的**：S2 收敛权重上重训同一 12 级推力课程，加随机臂脚本接管：
  站稳后 flag=1，env 覆写 act[15:21] 为随机脚本目标——抗推同时撤掉
  双臂，逼出纯腿部平衡策略（为 S4 步令打基础）。
- **u1450 误停**（11-18 ~10:23 起 ~11h）：实验级早停
  `Step._no_improvement_limit=200` 触发——level 7 长推平台期
  quality 长期不涨被误判收敛失败。停时指标仍在线：
  success=1.000、max_pot=0.983、recovery=0.639、falls=1.13、
  level=7.0。
- **修复**：`V2S3BalanceArm` 覆盖 `no_improvement_limit=1e9` 关闭
  早停（仅本实验；best 追踪/保存不变），commit `9dc89b78`。
- **续训**（11-18 晚）：
  ```bash
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s3_balance_arm --algo ppo \
    --resume-from baseline/runs/v2_s3_balance_arm/checkpoints/checkpoint_u01450.pt \
    --run-name v2_s3_balance_arm_r2 --background
  ```
  无 `--reset-update` → u1451 续编，实验状态（level=7、
  best 追踪）完整恢复。跑到 u2290 时**机器重启**被杀
  （22:32 关机、23:22 二次重启、23:33 起——日志无报错，
  纯外部 kill）。11-19 00:3x 以 u2290 ckpt 起 r3：
  ```bash
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s3_balance_arm --algo ppo \
    --resume-from baseline/runs/v2_s3_balance_arm_r2/checkpoints/checkpoint_u02290.pt \
    --run-name v2_s3_balance_arm_r3 --background
  ```
  u2291 续编恢复。注意：level 卡在 7.0 已达 ~800 update，
  r2 全程 eval success=1.0/recovery≈0.64 平台——待观察。
- **r3 结局**（11-19 ~10:00 手动收）：跑到 u4234 停。
  **level 7 卡死全程 ~2770 updates**（晋级阈值
  `PROMOTE_RECOVERY_RATE=0.8` 不可及——臂被脚本接管后
  100N×16-20 帧推的 recovery 峰值只有 0.673）。平台期非但不涨
  还在退化：recovery 0.67→0.52、push_falls 290→380、
  mixture_weight_entropy≈0.03 单模态塌缩。判定边际收益为零，
  停训保留算力进 S4。
- **★ S4 warm-start 用最优 ckpt**：
  `baseline/runs/v2_s3_balance_arm_r3/checkpoints/checkpoint_u02315.pt`
  ——best-of-run（quality=70+0.673，全训练史最后一次
  [new_best]，`policy/model.pt` 即该 update 导出）。
  备选同档：`checkpoint_u02300.pt`（rec 0.642）。
  避用 u4000+ 尾部 ckpt（退化段）。
- **S3 最终能力画像**（u2315 best）：100N×16-20 帧推力课程
  level 7，eval success=1.0（撑满回合）、recovery=0.673、
  push_falls=283/866、臂接管下自主站立+碎步平衡。
- **备注**：重启时发现全部 8 卡被外来 `wildrig-multi` 矿工
  （pearlhash → 155.133.196.229，cwd `/var/lib/prl/gpu-*`，
  ~974MiB/卡）占用——自 11-18 17:34 起在跑，待排查处置。

## S4 r1 — v2_s4_step_r1（步令跟随首训）

- **启动**（11-19 ~11:1x，GPU0 独占）：
  ```bash
  PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
    python3 -B baseline/framework/train.py \
    --experiment v2_s4_step --algo ppo --background \
    --resume-from baseline/runs/v2_s3_balance_arm_r3/checkpoints/checkpoint_u02315.pt \
    --reset-update --run-name v2_s4_step_r1
  ```
  PID 3060348；~19s/update；512 eps/1024 trajs/update。
- **任务**：S4_PLAN.md —— V2StepSimulator 双机步令流
  （相位机边界 WAIT_LIFT→IN_SWING、超时 age 6.0、墙 collision_fn），
  指令段 obs[67:78]=step_cur(5+age)/step_next。
- **通道**：r_potential·3 + r_fall·2 + r_step_foot·1.5
  + r_step_mv/st/yaw·2（边界帧稀疏；去 r_std_clip、无推力）。
- **warm-start**：S3 r3 best u2315（actor 续 + 4 新 critic 冷启）。
- **u1-u2 初值**：r_step_foot mean 0.078（有迈对落地）、
  r_step_mv mean -0.089、r_step_yaw mean -0.017、r_step_st≈0
  （钉脚站姿 stance 无滑移——符合预期）；r_fall -0.0004
  （平衡未劣化）；ep_len 全部 400 timeout 存活。
- **eval 指标**：step_events/step_foot_rate/step_timeout_rate/
  step_d_mv/step_d_st/step_d_yaw + quality
  （= foot_rate − 0.5·d_mv/0.5 − 0.5·d_st/0.3 − 0.3·d_yaw/1.0，
  各项 clip[0,1]）。

### r1→r2 插曲：on_eval 契约修复

- r1（u1-u88）`on_eval` 误把指标平铺返回——框架契约是
  `{is_new_best, stop_training, info}`，eval_info 全程为空、
  best 追踪不工作（训练本身无影响，u85 ckpt 完整）。
- 修复后 r2 续跑：`--resume-from r1/checkpoints/checkpoint_u00085.pt`
  （不 reset-update，u86 续编）。PID 3390678。
- **u90 首个有效 eval**：step_ev=7064（边界事件密集）、
  foot_rate=0.521（≈随机——策略还没理会指令脚）、
  timeout=0.002（脚频抬落，相位机几乎不走超时路径）、
  d_mv=0.596m、d_yaw=0.347rad、d_st≈0、quality=-0.083。
  结论：机制端到端正常，订单在真实落地中结算；策略当前
  把指令当噪声，待观察 foot_rate 爬升。

## S4 noarm r1 — v2_s4_step_noarm_r1（消融：无臂接管 + S2 底座）

- **目的**：对照 v2_s4_step_r2，隔离两个变量——① 臂脚本接管
  是否干扰步令学习；② S2（无臂扰动史）vs S3（臂接管史）底座差异。
- **差异**：env=v2_step_env_noarm.yaml（仅 `arm_script: false`，
  flag/arm_tgt 恒 0 与 S2 观测语义一致）；其余全继承。
- **启动**（GPU1）：`--resume-from v2_s2_balance/checkpoints/
  checkpoint_u01680.pt --reset-update`（S2 best：level11+
  recovery0.901，r_std_clip critic 不在通道集自动丢弃）。
  PID 3554454；~34s/update。
- **u1 初值**：r_step_foot 0.075（≈arm 组 0.078，同随机水位）、
  r_fall≈0（S2 站姿完好）。
- **对比看板**：foot_rate / d_mv / d_yaw 爬坡速度两组对比；
  若 noarm 显著快 → 臂接管是干扰项，后续阶段再叠加。

## S4 D16 语义切换 — 时限预算到期结算（取代事件驱动）

- **动因**（dump 实测 u155/u26）：相位机机制全对但语义根伤——
  订单中位 4 帧结算、指令脚净行程中位 9cm，令在给碎步盖戳；
  踝高重建示稳态抬幅 p90 仅 3~7cm、18% 帧贴地零受力（蹭步），
  "落地事件"在该行为流形上失去物理意义。
- **改法**：令=(目标构型, 时限 H)，H=0.15~0.50s 按动脚行程线性
  插值（s_max=0.53m），到期帧测 d_mv/d_st/d_yaw；foot_ok=
  d_mv<0.10m。step_boundary.py 删除，age→age/H 归一化。
- **重启**（旧相位机 run 作废；obs dim5 后改为"剩余预算"
  故二次重启）：
  - `v2_s4_step_d16b_r1` GPU0 PID 773255，warm u2315
  - `v2_s4_step_noarm_d16b_r2` GPU1 PID 809292，warm S2 u1680
  - 注：`v2_s4_step_d16_r1`/`_noarm_d16_r1`/`noarm_d16b_r1`
    为废弃目录（旧 obs 语义/yaml 漏改崩溃），已删
- **smoke 验证**：到期结算/轮转/归一化 age/通道装配全通；
  u3 实测 step 通道 adv_std 健康（mv 0.28/foot 0.17）。
- **观察点**：foot_rate(到位率) 爬坡、d_mv 破 0.4、budget 均值
  分布（~8f 初值说明订单多在 ~40cm 行程档）。

## S3 臂令跟随分支 — v2_s3_armcmd_r1（D17，第三轨）

- **机制**：臂目标只进观测（seg[1:7]），act[15:21] 不覆写；
  r_arm_r/l = −‖臂输出−目标‖ 逐帧稠密（w=1, aw=0.5）。
  **S3 血统**（平衡+臂令，无 S4 步令任务），5 通道。
- **对照**：vs S3 接管史（v2_s3_balance_arm）——同 S2 u1680
  底座、同推力插件，唯一变量=臂令载体（奖励 vs 覆写）。
- **启动**（GPU2）：PID 2868065，warm u1680 + reset-update。
  （v2_s4_armcmd_r1 为误建的 S4 步令版，两次启动均已废弃删除）
- **扰动**：固定 40N × 31-40 帧躯干推（40N 档最高级，单级
  无课程）；训练/eval 同强度——eval 指标即推力下兼顾度。
- **看板**：arm_err_r/l 应 ~1.0→<0.3；recovery/falls 对比
  S3 同期水位——抗推能力不退 + 臂令跟随学会 = 方案成立；
  臂通道 grad 占比过高（首测 clip_frac~0.85）则调 w_arm。
- **r1 中断与恢复**：宿主机 2026-11-20 00:35/03:00 两次重启
  杀光三 run（日志无报错戛然而止）。r1 死于 u2230——
  死前已学 550u：arm_err_r 1.33→0.86、arm_err_l 1.21→0.59
  （左臂快于右臂）、recovery 全程 ~0.994（~770 推/eval
  推落 ~0.6%）、falls ~0.02-0.05——40N 下"平衡+臂令"兼顾
  初步成立且曲线未收敛。
- **重启**：`v2_s3_armcmd_r2` GPU2 PID 360882，
  resume r1 ckpt_u02230，计数续 u2231（无 reset-update）。

## 恢复 — v2_s4_step_noarm_d16b_r3（GPU1）

- r2 死于宿主机重启（u813，同批事故）。死前 810u 单调爬坡
  未收敛：foot_rate 0.027→0.333、d_mv 0.52→0.23、d_yaw
  →0.12、quality 刚转正（+0.071）、falls ~0.02。
- **重启**：`v2_s4_step_noarm_d16b_r3` GPU1 PID 518815，
  resume r2 ckpt_u00810，计数续 u811（无 reset-update）。
- S4 硬劫持组（v2_s4_step_d16b_r1 @u921）暂未恢复——同期
  foot_rate 钉死基线，对照结论已出（接管干扰学习），
  恢复与否待定。
- **D19 释放机制**：r2 死于 yaml 热改冲突（运行中
  build_jobs 重读新 yaml 带 arm_release_prob，旧代码工人
  TypeError）——教训：运行中的 env yaml 不能改，改配置=
  必须重启。`v2_s3_armcmd_r3` GPU2 resume u2425，携带
  release_prob=0.25 上线；eval 新增 arm_rel 释放率诊断。

## V1 步令实验（D20 残差编码）— v2_stepcmd_r1（GPU0）

- **任务**：0.5s/10f 固定周期连续指令流，obs[67:79] 逐帧
  残差（动脚踝→世界锚目标，本体系）；无站立门控帧0施令；
  无推力、无臂脚本、r_std_clip 移除。5 通道：r_potential
  (h_cap=1.10→φ≤0.889) + r_fall + r_step_mv/st/yaw 逐帧积分
  （积分=ASAP 语义）。
- **启动**（GPU0）：PID 3124988，warm S2 u1680 + reset-update，
  3 步令 critic 冷启。
- **u5 基线**：success=1.000（thresh 0.85 适配封顶 φ）、
  falls=0.04、hold=0.097、arr_rate=0.002、d_mv=0.453、
  d_yaw=0.283、skate=0.649（未迈步时读数虚高为基线）。
- **看板**：arr_rate/d_mv 应 ~0→升/降；falls 应保持 <0.1
  （失衡忽略令学会）；skate 在迈步出现后需看是否回落
  （真滑行 vs 未起步）。
