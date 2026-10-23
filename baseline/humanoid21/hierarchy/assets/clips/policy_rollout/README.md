# policy_rollout clips — 生成记录

本目录存放**训练策略在仿真器中实际执行**的 CoreClip（格式契约见
`mocap/humanoid21/clip/CORECLIP_SPEC.md`）。与人捕 retarget clip
（`clips/{full,extend,retract}/`）分开管理。

## 当前内容

| 文件 | 技能 | 时长 | 说明 |
|---|---|---|---|
| `eval_jab_l_u09260_robot_a.npz` | jab_l | 24 帧 / 1.2s | 左站架抱架：左脚在前、右手护脸 |
| `eval_heavy_l_u09260_robot_b.npz` | heavy_l | 24 帧 / 1.2s | 宽站架抱架：大步距前后脚、双手交叉护脸 |

两条均 `check` basic PASS + `fix`（脚部穿地 root-lift ≤5mm）+
`ref_action` PASS（记录动作序列物理回放 0 误差 = 可执行 clip）。
对应视频在 `assets/videos/policy_rollout/`（`clip play` 运动学回放）。

## 来源

- **Run**：`baseline/runs/train_strike_kp_2phase_ppo_20261021_001547`
  （实验 `strike_kp_2phase`，CommandSimulator 66-dim obs / 21-dim act）
- **策略**：`policy_exports/u09260`（最后导出的 checkpoint；`policy/` 是
  best-of-run u08760，两者不同，注意别混）
- **指令**：与该 run eval 视频完全一致的 `videos/video_options.json`——
  robot_a=`jab_l`、robot_b=`heavy_l`、warmup `turn_right×2`/`step×4`、
  `initial_distance=2.0`、`max_steps=30`、seed `997777`

## 复现步骤

```bash
cd /data1/mono/things/combatbench
POLICY=baseline/runs/train_strike_kp_2phase_ppo_20261021_001547/policy_exports/u09260

# 1. policy blueprint（file: 引用导出目录）
cat > /tmp/policy_u09260.yaml <<EOF
version: 1
cls: file:${PWD}/${POLICY}/policy.py:ExportedStateMixtureBoundedStdTruncNormPolicy
config: {}
EOF

# 2. options：直接取 run 的 video_options.json，仅把
#    warmup_policy_cls 的 file: 路径改到 u09260（warmup 步由同 policy 执行）

# 3. 录制：round_runner + CoreClipPostRecorder（PostActionRecorder
#    适配器，因果配对 (s_t, a_t)，episode 末每 agent 落一个 CoreClip）
MUJOCO_GL=egl PYTHONPATH=. python3 -m envs.framework.round_runner \
    --env-blueprint baseline/runs/train_strike_kp_2phase_ppo_20261021_001547/videos/video_env_blueprint.yaml \
    --policy-a-blueprint /tmp/policy_u09260.yaml \
    --policy-b-blueprint /tmp/policy_u09260.yaml \
    --options-json <options.json> \
    --seed 997777 --video <ep.mp4> \
    --recorder "mocap.humanoid21.clip.recorder:CoreClipPostRecorder?output_dir=<dir>&clip_id=<id>"

# 4. 切片去掉 warmup（步骤 0–5）取 strike 指令段 6–29：
#    CoreClip.load(npz).slice(6, 30, clip_id=...).save(...)

# 5. 校验 + 修复 + 渲染
PYTHONPATH=. python3 -m mocap.humanoid21.clip check --clip <x.npz> --only basic,ref_action
PYTHONPATH=. python3 -m mocap.humanoid21.clip fix   --clip <x.npz> --out <x_fix.npz> --mode lowest
MUJOCO_GL=egl PYTHONPATH=. python3 -m mocap.humanoid21.clip play  --clip <x.npz> --out <x.mp4> --robot <robot_a|robot_b>
```

## 注意事项

- `ref_action` 语义为 `pd_target_norm`（策略实际发出的归一化 PD 目标），
  `check --only ref_action` 回放即该 clip 可执行性的唯一真验证。
- lineage：`record → slice(6,30) → fix_ground`，见 npz `meta_json`。
- 完整 60 步录制（含 warmup 与更长保持段）在
  `/data1/dev/policy_capture/strike_kp_2phase_u09260*/core/`（不入库）。
- 非标准观测的策略（如本实验 66-dim）不能用
  `policy_capture/collect.py`，必须走本页的 `round_runner` 路径并自带
  该实验的 env/options。
