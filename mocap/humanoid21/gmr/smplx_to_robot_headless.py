"""Headless variant of smplx_to_robot.py: no GLFW viewer, EGL offscreen render."""
import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
GMR_ROOT = (HERE / ".." / ".." / "gmr").resolve()  # mocap/gmr (upstream clone)
sys.path.insert(0, str(GMR_ROOT))
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smplx_file, get_smplx_data_offline_fast)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smplx_file", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--render_video", default=None)
    args = p.parse_args()

    SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER)
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    retarget = GMR(actual_human_height=actual_human_height,
                   src_human="smplx", tgt_robot=args.robot)
    qpos_list = [retarget.retarget(frames[i]) for i in range(len(frames))]
    qpos_list = np.asarray(qpos_list)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7][:, [1, 2, 3, 0]]  # wxyz->xyzw
    dof_pos = qpos_list[:, 7:]
    motion_data = {"fps": aligned_fps, "root_pos": root_pos, "root_rot": root_rot,
                   "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None,
                   "qpos": qpos_list}
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved {len(qpos_list)} frames to {args.save_path}")

    if args.render_video:
        import mujoco
        from general_motion_retargeting.params import ROBOT_XML_DICT
        mm = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        dd = mujoco.MjData(mm)
        r = mujoco.Renderer(mm, height=480, width=640)
        import cv2
        vw = cv2.VideoWriter(args.render_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             aligned_fps, (640, 480))
        for q in qpos_list[::2]:
            dd.qpos[:] = q
            mujoco.mj_forward(mm, dd)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = dd.xpos[1]
            cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 90
            r.update_scene(dd, camera=cam)
            vw.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"video -> {args.render_video}")


if __name__ == "__main__":
    main()
