debug env :

python3 -m envs.framework.round_runner \
    --blueprint baseline/humanoid21/blueprints/curriculum.yaml \
    --policy-a policy.random.policy:RandomCombatPolicy \
    --policy-b policy.random.policy:RandomCombatPolicy \
    --recorder envs.framework.recorder:BaseFrameRecorder?output_dir=baseline/humanoid21/blueprints/out


start viewer:

python3 -m envs.framework.recorder_viewer --no-browser baseline/humanoid21/blueprints/out

在浏览中看到8765