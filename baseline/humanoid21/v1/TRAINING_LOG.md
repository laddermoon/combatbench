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
- **u1565 进展**（+6h 值守）：arr 0.002→0.645、d_mv 0.453→0.134、
  d_yaw→0.211、quality −0.58→+0.36（持续 new_best）、falls≤0.04、
  hold 0.097→0.15 缓升。健康爬坡，无实现级异常。
- **完结（u5000，达 max_updates 自然终止）**：arr_rate 0.905、
  d_mv 0.088（低于 0.10 首达阈值——令终态基本落在目标上）、
  d_yaw 0.140、quality 0.72、falls ≤0.03。**结论：D20 残差
  编码+积分奖励教会了"读令迈步"，核心问题已答完**；后继：
  臂令叠加=v2_steparm_r1，随机时长=v2_steptvar_r1。

## STEPARM V1 联合实验（步令×双臂独立软劫持）— v2_steparm_r1（GPU1）

- **任务**（STEPARM_V1_PLAN.md）：stepcmd 任务 ∪ ARMDUAL 逐臂
  劫持——obs[57:65]=[flag_r,tgt_r,flag_l,tgt_l]（flag=0 臂全零）
  与 obs[67:79] 步令残差共存，两套指令流独立计时。7 通道：
  r_potential(h_cap)·3 + r_fall·2 + r_step_mv/st/yaw(1/0.5/0.5)
  + r_arm_r/l·0.7（臂总 1.4 = 步总 2.0 的 70%）。
- **启动**（GPU1）：PID 2913997，warm `v2_stepcmd_r1` ckpt
  u03020 + reset-update，r_arm_r/l 两 critic 冷启（臂令段
  死维→活跃 = 观测分布迁移，步令指标预期瞬态回落）。
- **实现级验证**（先 smoke 再起，防"白训"）：双指令流共存
  ✓、自由臂 tgt 全零+err 0 ✓、occ 贴名义 ✓、aw 3/2/1/0.5/0.5/
  0.7/0.7 ✓。smoke 暴露两处 eval 设计缺陷已修：站稳门
  φ≥0.9 被 h_cap 封顶锁死 → 降 0.85；arm-free 桶恒空（恒≥1臂
  被劫）→ 交叉指标改 arr_1arm/arr_2arm（被劫臂数分桶）。
- **u5 基线**：arr 0.664（热启动步令跟随即存活）、
  arm_err_r/l 1.232/1.125（≈随机水位，待降）、
  err_stand 1.277 < err_down 1.760（结构在）、
  occ 0.101/0.452/0.447 ≈ 名义、falls 0.01、quality 0.380
  （vs stepcmd 成熟值 0.585——迁移瞬态进行中）。
- **看板**：arm_err 应向 armdual 水位（~0.13-0.19）收敛；
  arr 瞬态回落后应回升至 ≥0.6；arr_1arm vs arr_2arm 裂口
  >0.15 → 双臂同劫拖累迈步（调权重/降 dual_prob 决策点）；
  falls 应 ≤0.05。

## STEPTVAR V1 随机时长步令 — v2_steptvar_r1（GPU2）

- **任务**（STEPTVAR_V1_PLAN.md）：stepcmd 指令周期改逐令随机
  U(1,20) 帧（0.05s–1.0s），**时长不写 obs**——唯一鲁棒策略=
  ASAP；短令制造半空换令、长令检验驻留。5 通道原样。
- **启动**（GPU2）：PID 3526737，warm `v2_stepcmd_r1` ckpt
  u05000 + reset-update。
- **实现级验证**（先 smoke）：dur 1-20 全范围轮转 ✓、1 帧令
  存在 ✓、诊断字段 dur/res0/st_z 上报 ✓、分桶 eval 工作 ✓。
- **u5 基线**：arr 0.692（vs 固定期 0.90——短令结构性下移，
  预期）、分桶 arr_s/m/l=0.580/0.721/0.750、
  hold_err 0.063（长令驻留已可用）、air_frac 0.008（成熟策略
  下换令时脚多已落地）、res0 0.162、falls 0.01。
- **看板**：res0 应降（preview 预摆位学习中）；air_frac 若升
  =策略不再能准时落地（正常代价）若降=学会踩点落地（更强
  结论）；arr_s 是 ASAP 压力测试主指标；hold_err 应 →~0.05；
  falls ≤0.05。

## 已终止实验收尾

### v2_armsoft_r1（ARMSOFT V1，臂软劫持捆绑版）— ✅ 成功，终于 u3025

- **任务**：r3 最小化变体——无推力、flag 恒 1（令恒激活）、
  r_std_clip 移除（std_pose 含臂关节与令冲突）。4 通道：
  r_potential(无cap)·3 + r_fall·2 + r_arm_r/l·0.5。
- **启动**：PID 3537440，warm S2 u1680 + reset-update，
  r_arm_r/l 两 critic 冷启。
- **终态**：err_stand 0.085 / err_down 1.088、arm_err_r/l
  0.132/0.121、falls 0.02、success 1.0。
- **核心结论——"学会忽略"成立**：err_stand 0.94→0.085（站稳
  精确跟随）vs err_down 全程钉 ~1.1（失衡帧=随机水位），裂口
  ~1.0——策略明确学会了"站稳服从、失衡忽略"。
- **行为形态**：跟随=瞬时吸附（≤1 帧到位）+ ~0.11 残差地板
  （关节响应+U(−1,1)³ 全域目标的可达边界）。cycles→~0.5：
  无节律引导下步态退化为近似原地站立——对本实验无碍，
  但揭示"不动是最优"的退化方向（后续若需步态需保留节律项）。
- 说明：右臂慢于左臂的固有不对称再现（0.132 vs 0.121 接近
  收敛差）。

### v2_armdual_r1（ARMDUAL V1，双臂独立软劫持）— ✅ 成功，终于 u2035

- **任务**：逐臂独立指令——obs[57:65]=[flag_r,tgt_r,flag_l,tgt_l]
  （flag=0 臂 tgt 全零=无目标），腰段剩 2 维。同步周期调度：
  P(0.1) 双臂 / P(0.9) 单臂(50/50)，dur U(2,20)，恒 ≥1 臂在令。
- **启动**：PID 2012515，warm S2 u1680 + reset-update。
- **终态**：err_stand 0.134 / err_down 1.643、arm_err_r/l
  0.187/0.176、occ 0.099/0.453/0.448 ≈ 名义、falls 0.02。
- **核心结论**：逐臂版"学会忽略"成立且更强——err_down 全程
  钉 ~1.7（从 u5 起就没学过失衡服从，比 armsoft 的"先跟后放"
  更纯粹）；err_stand 0.134 仅落后捆绑版 0.03 却只用 ~55%
  计分帧预算，且白赚一个自由臂。
- **自由臂**：实测输出活动量 = 被劫臂的 62%——结构上活跃，
  具备充当平衡器的物理前提（定性视频确认留待需要时）。
- **代价**：单臂跟令未推高 falls（≤0.03）——设计假设
  "劫持一臂不损平衡"证实。
- 实现验证：occ 全程贴名义值（调度器✓）、奖励掩码精确
  （自由臂帧 0 分✓）、err 两臂对称收敛。
- **遗留**：`policy/` best 导出停在 ~u30（best 追踪看电势系
  质量，封顶后不更新）——迁移请用 `policy_exports/` 最新版；
  armsoft 同病（停在 u~300）。教训：best 追踪应纳入任务指标。

### v2_s4_step_noarm_d16b 线（D13/D16b 类别编码）— 终于 u1770

### v2_s4_step_noarm_d16b 线（D13/D16b 类别编码）— 终于 u1770

- 接力：r1(u810 死于重启) → r2(u810 resume) → r3(u1770 人工停)。
- 终态：foot_rate 0.408、d_mv 0.185、d_yaw 0.090、quality 0.197、
  falls 0.01——单调爬坡后进入缓坡 plateau。
- **结论**：类别编码的开环天花板即此水位——发令时刻 SE(2)
  锚定 + 单帧 obs 无绝对朝向 → 残差地板=执行方差。该 plateau
  正是 D20 残差编码（v2_stepcmd_r1）的动机证据。
- 对比锚：v2_s4_step（臂接管版）同期 foot_rate 钉死基线——
  接管干扰学习的对照结论已出。

### v2_s3_armcmd 线（D17 奖励制臂令）— 终于 u3235

- 接力：r1(u2230 死于重启) → r2(u2425 死于 yaml 热改冲突)
  → r3(u3235 人工停，携 release_prob=0.25)。
- 终态：arm_err_r 0.663 / arm_err_l 0.412（累计 −50%/−66%）、
  arm_rel 0.249≈0.25、recovery 0.992（40N×772 推仅倒 6）、
  falls 0.02、clip 0.45/cad 0.71。
- **结论**：奖励制"可忽略指令"在 40N 推力下"平衡+跟随"
  兼顾成立；右臂残差未收敛但速率平缓。最小化验证版
  （无推+令恒激活+去 clip 冲突）= v2_armsoft_r1。
- 教训（记 D19）：运行中的 env yaml 不可改——build_jobs
  重读会与新参数冲突杀工人，改配置=必须重启。

### v2_steparm_r1 线（步令+臂令双通道）— 终于 u4924，人工停

- 终态：arr_rate 0.90、d_mv 0.086、d_st 0、d_yaw 0.17、skate 0.60、
  arm_err_r/l 0.21、falls 0.01、quality 0.71——收敛平台。
- **用途**：Attack Harness v1 的基座（导出于 u4735）。攻击验收
  暴露其短板=步令动态性/衔接——正是 stepbudget 的动机。

### v2_steptvar_r1 线（随机时长 U(1,20)）— 终于 u2206，人工停

- 终态：arr 0.75（结构性下移）、d_mv 0.115、hold_err 0.056、
  air_frac 0.012、res0 0.156、quality 0.53。
- **结论**：不可见时长下 ASAP 语义学出，短令 d_mv 与长令相当；
  半空发令仅 1.2%（预览承诺契约下，半空晋升被策略规避）。
- **迁移到 stepbudget 的教训**：预算可见化后节奏由指令给，
  需速度奖励防止"按预算磨洋工"；预览承诺必须维持（cur←nxt）。

### v2_stepbudget_r1 线（方向+时间预算，STEPBUDGET_V1_PLAN）— 启动

- 语义变化：步长=执行结果（预算×脚速，L2W 包络封顶）；朝向=
  独立绝对目标（ψ* 累积、±45° 夹）；Clip 组合 80% + 随机 20%。
- 起训：S2 balance u1680，--reset-update（指令输入语义全变）。
- 已知监测项（smoke/对照实测）：
  1. **站宽超包络**：S2 自然站宽 0.80–0.89m > L2W 0.625m，
     未训态 96% 步终态违规——前期信号被站宽惩罚主导，
     若 viol 不降则放宽至 ~0.85m。
  2. ψ* 累计夹 ±45°：朝向目标会追身（执行差时目标不退）。
  3. 包络违规只判结算帧终态；中途过冲由 s_max 封顶。
- **u818 巡查（~5h）**：无实现级异常，优化器健康（kl≈0.011、
  eff_std≈0.35 稳定、falls 0.00–0.03）。
  - 站宽冲突**已自行消解**：viol 0.963→0.06（u105 即降到 0.1），
    包络 0.625m 维持不放宽。
  - yaw_e 0.996→0.18（u650 后平台）；dir_err 1.19→0.42。
  - stride 0.02(u455)→0.109(u815)，线性爬坡未饱和；quality 0.30→0.48。
  - 离线诊断 u820（12ep/470 步）：前向 s_max 中位 ~0.5 但 stride
    b3-6/7-10/11-14/15-20 = 0.06/0.13/0.15/0.22，触顶率 ≤12%——
    **步幅小是策略保守，不是包络封顶**。预算→步长已单调。
  - 代价项：land 0.94→0.87、pv_air 0.06→0.12（迈大步后半空发令
    增多）；c_fast 0.92→0.71、c_shuffle 0.87→0.78（短预算着地难）。
  - 决定：继续跑；若 stride 增速停滞再考虑提 w_prog。
- **r1→r2 预算接力（u4785）**：r1 max_updates=5000 将耗尽，停机
  resume 至 `v2_stepbudget_r2`（GPU0, pid 73930），
  `--param max_updates=15000` 生效（框架 per-update 参数补丁，
  `param_overrides` 是进程内代码不可热改——不停机加预算做不到）。
  u4786 起连续计数，优化器/critic/rng 全量恢复。
- **u4770 状态**：stride 0.228 / vf 0.66m/s@b3-6 / land 0.90 /
  yaw_e 0.08 / dir_err 0.30 / viol 0.045 / quality 0.83——u2400
  平台后持续爬升，Lf3 步幅 0.35m。
- 攻击走位 Harness（u02617 基座）：frozen/stand/walk 三档
  reach_rate 1.0、t_reach≈90f、band_frac 0.80、绕行 240–275°、
  零摔倒——旧残差协议的三大短板（逼近慢/近身失稳/绕不了圈）
  全部解决。

### v2_stepbudget_r2 线（续训）— 终于 u6140，人工停（收敛）

- 终态：stride 0.233 / cpl 0.556 / land 0.93 / dir_err 0.31 /
  yaw_e 0.077 / viol 0.039 / vf 0.74m/s@b3-6 / quality 0.83。
- 标定曲线（给规划层）：前向 b3-6/7-10/11-14/15-20 =
  0.165/0.282/0.369/0.353m；最速 b3-6，最大步幅 b11-14。
- **结论**：方向+预算协议验证成功——比 stepcmd 残差协议
  最速 0.37→0.74 m/s（+100%）；Harness 走位三档满分。
- stride 收敛于包络 ~70%（0.37/0.53m）：奖励均衡点，
  再迈大需调权重（r2 的事）。

### v2_stepbudarm_r1 线（步budget × ARMDUAL）— 启动

- = stepbudget + 双臂独立软劫持（P0.1 双臂 / 单臂 50/50，
  dur U(2,20)f，tgt U(-1,1)³——ARMDUAL 参数照抄）。
- 通道 9：stepbudget 原 7 + r_arm_r/l·0.7（被劫帧 −err）。
- 起训：stepbudget_r2 u6140 ckpt --reset-update（臂令段死维
  →活跃，r_arm critic 冷启，步指标预期瞬态回落）。
- max_updates=15000（启动即加，避免 r1 式预算耗尽接力）。
- 交叉指标改造：land_1arm/land_2arm + stride_1arm/stride_2arm
  （按令实际生存窗分桶，替代 stepcmd 的 arr/dmv 桶）。
