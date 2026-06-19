

PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment basic_balance_v2 &> basic_balance_v2.log & 

python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py basic_balance_v2.log  --watch


/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_v2_20260617_111918/checkpoints/checkpoint_u00685.pt
用时大概7小时。


PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_basic_balance_v2_20260617_111918/checkpoints/checkpoint_u00685.pt &> balance_recover_v2.log & 



python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_v2.log  --watch


/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_v2_20260617_195446/checkpoints/checkpoint_u03480.pt
训练大概12小时。  一个可能的改进是把MaxStep调小。


PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --experiment balance_recover_plus_v2 --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_20260618_131501/checkpoints/checkpoint_u04875.pt &> balance_recover_plus_v2.log & 


python3 /data1/mono/things/combatbench/baseline/humanoid21/curriculum/analyze_logs.py balance_recover_plus_v2.log  --watch


撑地后站立
http://180.76.152.227:8999/curriculum_balance_recover_plus_v2_20260618_225956/videos/u07245.mp4