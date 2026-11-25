"""Render PHC bundle clips to h264 mp4 for review.

Usage:
  mocap/gmr/.venv/bin/python mocap/humanoid21/gmr/render_phc.py \
      --bundle dev [--clip_id X | --all] [--outdir mocap/retarget_out/phc]
"""
import argparse, os, pathlib, pickle, subprocess, sys

os.environ.setdefault('MUJOCO_GL', 'egl')
HERE = pathlib.Path(__file__).resolve().parent
MOCAP = HERE.parents[1]
sys.path.insert(0, str(MOCAP / 'gmr'))

import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R

PHC_DIR = pathlib.Path('/data1/dev/combatbench_mocap/phc')
DEFAULT_OUT = MOCAP / 'retarget_out' / 'phc'


def render_clip(clip, out_path):
    mm = mujoco.MjModel.from_xml_path(str(HERE / 'humanoid21_mocap.xml'))
    dd = mujoco.MjData(mm)
    r = mujoco.Renderer(mm, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.distance = 3.3; cam.elevation = -8
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, 'torso_a')
    # PHC xyzw -> mujoco wxyz, reassemble qpos
    qpos = np.concatenate([clip['root_trans_offset'],
                           clip['root_rot'][:, [3, 0, 1, 2]],
                           clip['dof']], axis=1)
    # camera tracks root yaw -> always a front 3/4 view (robot faces +X at
    # identity).  GOTCHA: mujoco azimuth is the LOOK direction — the camera
    # body sits at azimuth+180deg, so front view needs az=yaw+180ish.
    CAM_YAW = 150.0
    fps = int(clip['fps'])
    raw = f'/tmp/_phc_render.mp4'
    w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
    for q in qpos:
        dd.qpos[:] = q
        mujoco.mj_forward(mm, dd)
        fwd = R.from_quat(q[3:7], scalar_first=True).apply([1, 0, 0])
        cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])) + CAM_YAW
        cam.lookat[:] = dd.xpos[bid]
        cam.lookat[2] = max(0.45, dd.xpos[bid][2] - 0.45)
        r.update_scene(dd, camera=cam)
        w.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    w.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    out_path], check=True)
    os.remove(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', default='dev')
    p.add_argument('--clip_id', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--outdir', default=str(DEFAULT_OUT))
    args = p.parse_args()

    bundle = pickle.load(open(PHC_DIR / f'combatbench_{args.bundle}_v1.pkl', 'rb'))
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ids = sorted(bundle) if args.all else [args.clip_id]
    for cid in ids:
        render_clip(bundle[cid], str(outdir / f'{cid}.mp4'))
        iss = bundle[cid]['meta'].get('issues', [])
        print(f'[OK] {cid}.mp4  issues={iss}')


if __name__ == '__main__':
    main()
