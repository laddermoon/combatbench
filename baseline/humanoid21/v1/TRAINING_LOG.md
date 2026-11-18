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
