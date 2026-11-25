import sys, os, pathlib, subprocess
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/data1/mono/things/combatbench/mocap/gmr')
import numpy as np, mujoco, cv2, pickle
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

SMPLX_FOLDER = pathlib.Path('/data1/mono/things/combatbench/mocap/gmr/assets/body_models')
AMASS = '/data1/mono/things/combatbench/mocap/amass/smplx_gender_specific'
OUT = '/data1/mono/things/combatbench/mocap/retarget_out'

CLIPS = [
    ('SSM/20160330_03333/punching_stageii.npz',            'ssm_punching'),
    ('SSM/20160330_03333/side_hopping_stageii.npz',        'ssm_side_hopping'),
    ('SSM/20160330_03333/chicken_wings_stageii.npz',       'ssm_chicken_wings'),
    ('SSM/20160330_03333/shake_hips_stageii.npz',          'ssm_shake_hips'),
    ('SSM/20160930_50032/jumping_jacks_sync_stageii.npz',  'ssm_jumping_jacks'),
    ('SSM/20160930_50032/running_on_the_spot_sync_stageii.npz', 'ssm_running_spot'),
    ('SSM/20160930_50032/ATUSquat_sync_stageii.npz',       'ssm_squat'),
    ('SSM/20160930_50032/special_move_sync_stageii.npz',   'ssm_special_move'),
    ('SSM/20161014_50033/dance_sync_stageii.npz',          'ssm_dance'),
    ('SSM/20161014_50033/dribble_kick_sync_stageii.npz',   'ssm_dribble_kick'),
    ('SSM/20161014_50033/hop_on_one_leg_sync_stageii.npz', 'ssm_hop_one_leg'),
]

mm = mujoco.MjModel.from_xml_path(
    '/data1/mono/things/combatbench/mocap/gmr/assets/humanoid21/humanoid21_mocap.xml')
dd = mujoco.MjData(mm)
r = mujoco.Renderer(mm, height=480, width=640)
cam = mujoco.MjvCamera(); cam.distance = 2.8; cam.elevation = -12; cam.azimuth = 100
bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, 'torso_a')

for rel, tag in CLIPS:
    path = os.path.join(AMASS, rel)
    try:
        smplx_data, body_model, smplx_output, h = load_smplx_file(path, SMPLX_FOLDER)
        frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
        rt = GMR(actual_human_height=h, src_human='smplx', tgt_robot='humanoid21')
        qpos = np.array([rt.retarget(f) for f in frames])
        with open(f'{OUT}/{tag}.pkl', 'wb') as fo:
            pickle.dump({'fps': fps, 'qpos': qpos, 'src': rel}, fo)
        raw = f'/tmp/{tag}.mp4'
        w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for q in qpos:
            dd.qpos[:] = q; mujoco.mj_forward(mm, dd)
            cam.lookat[:] = dd.xpos[bid]
            r.update_scene(dd, camera=cam)
            w.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        w.release()
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                        f'{OUT}/{tag}.mp4'], check=True)
        os.remove(raw)
        print(f'[OK] {tag}: {len(qpos)} frames')
    except Exception as e:
        print(f'[FAIL] {tag}: {e}')
