Train:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 baseline/humanoid21/curriculum/train_curriculum.py &> train.log & 

Resume:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 -u baseline/humanoid21/curriculum/train_curriculum.py \
    --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260531_172059/checkpoints/checkpoint_u03895.pt &> train_resume10.log & 

Watch Video:
python3 -m http.server 8999 --bind 0.0.0.0   --directory /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260531_172059/videos


Gen Video:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260528_175538/policy_exports/u00539/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260528_175538/policy_exports/u00539/policy_blueprint.yaml \
  --video out1.mp4

每隔一段时间，按照Rollout一样的配置，生成一个视频。



DEBUG:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/curriculum_env.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260529_112738/policy_exports/u01925/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260529_112738/policy_exports/u01925/policy_blueprint.yaml \
  --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=_debug/run03

python3 -m envs.framework.recorder_viewer --no-browser _debug/run03


things/combatbench/baseline/humanoid21/runs/curriculum_20260527_175448/policy_exports/u00837/policy_blueprint.yaml
总体的处理流程：
Rollout得到原始数据。

进行奖励计算。 计算4种奖励。 使用原始奖励。



1. 