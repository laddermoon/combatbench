Train:
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_curriculum.py

Resume:
cd /data1/mono/things/combatbench
PYTHONPATH=. nohup python3 -u baseline/humanoid21/curriculum/train_curriculum.py \
    --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_20260526_165233/checkpoints/checkpoint_u01110.pt &> train.log & 

Gen Video:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260527_012459/policy/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260526_115427/policy/policy_blueprint.yaml \
  --video out.mp4


DEBUG:
cd /data1/mono/things/combatbench
python3 -m envs.framework.round_runner \
  --env-blueprint baseline/humanoid21/blueprints/curriculum_env.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_20260526_115427/policy/policy_blueprint.yaml \
  --policy-b-blueprint baseline/humanoid21/runs/curriculum_20260526_115427/policy/policy_blueprint.yaml \
  --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=_debug/run01

python3 -m envs.framework.recorder_viewer --no-browser _debug/run01



总体的处理流程：
Rollout得到原始数据。

进行奖励计算。 计算4种奖励。 使用原始奖励。


