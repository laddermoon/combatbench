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
