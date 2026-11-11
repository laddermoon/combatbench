#!/usr/bin/env python3
"""e2e：连续应用 action_for_balance 的站桩闭环演示视频。

用法（仓库根目录）：

    PYTHONPATH=. MUJOCO_GL=osmesa python3 \
        point5/humanoid21/e2e_balance_video.py [out.mp4]

流程：sim reset 站姿 → 每控制周期
    a = st.action_for_balance() → st = st.take_action(a) → render。
t_push 时刻注入一次前冲扰动（root_linvel += push_v）观察踝策略
恢复过程。帧左上叠 t / margin / |a−a_hold| 诊断。

产物：mp4（ffmpeg libx264，缺失时 cv2 mp4v 兜底）。
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from point5.humanoid21.robot_state import RobotState
from point5.humanoid21.mechanics import rad_to_norm


def make_standing_state() -> RobotState:
    """sim reset 态 -> RobotState（与 test_robot_state 同构造）。"""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from envs.humanoid21.meta import Humanoid21Meta
    sim = Humanoid21Simulator()
    sim.reset()
    r = Humanoid21Meta.build_runtime_tables(
        sim.model)["robots"]["robot_a"]
    d = sim.data
    adr, vadr = int(r["root_qpos_adr"]), int(r["root_qvel_adr"])
    quat = np.asarray(d.qpos[adr + 3:adr + 7])
    Rm = np.empty(9)
    mujoco.mju_quat2Mat(Rm, quat)
    Rm = Rm.reshape(3, 3)
    zyx = np.array([
        np.arctan2(-Rm[2, 0], np.hypot(Rm[0, 0], Rm[1, 0])),
        np.arctan2(Rm[2, 1], Rm[2, 2]),
        np.arctan2(Rm[1, 0], Rm[0, 0])])
    mujoco.mj_subtreeVel(sim.model, d)
    return RobotState(
        qpos=d.qpos[np.asarray(r["qpos_indices"])].copy(),
        root_position=d.qpos[adr:adr + 3].copy(),
        root_orientation=zyx,
        qvel=d.qvel[np.asarray(r["qvel_indices"])].copy(),
        root_linvel=d.qvel[vadr:vadr + 3].copy(),
        root_angvel=Rm @ d.qvel[vadr + 3:vadr + 6],
        ang_mom=np.asarray(
            d.subtree_angmom[r["root_body_id"]]).copy())


class _Writer:
    """ffmpeg 管道写 mp4；ffmpeg 缺失退化 cv2。"""

    def __init__(self, path: Path, fps: int, size):
        self.path, self.fps, self.size = path, fps, size
        w, h = size
        try:
            subprocess.run(["ffmpeg", "-version"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                 "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "faststart", "-crf", "20", str(path)],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.cv2 = None
        except (FileNotFoundError, subprocess.CalledProcessError):
            import cv2
            self.proc = None
            self.cv2 = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h))

    def write(self, rgb: np.ndarray):
        if self.proc is not None:
            self.proc.stdin.write(rgb.tobytes())
        else:
            self.cv2.write(rgb[..., ::-1])          # RGB->BGR

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv2.release()


def annotate(img, lines):
    """PIL 图上叠诊断文本（黑底黄字）。"""
    from PIL import ImageDraw, ImageFont
    img = img.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = "\n".join(lines)
    x, y = 14, 12
    bbox = d.multiline_textbbox((x, y), text, font=font, spacing=4)
    d.rectangle((bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6),
                fill=(0, 0, 0, 180))
    d.multiline_text((x, y), text, font=font, spacing=4,
                     fill=(255, 230, 80))
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else "/data1/dev/combatbench_executor/e2e/balance_e2e.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    dt, steps = 0.05, 90                  # 4.5s @ 20Hz
    t_push, push_v = 1.0, np.array([0.35, 0.0, 0.0])
    W, H = 1280, 720

    st = make_standing_state()
    writer = _Writer(out, fps=int(1 / dt), size=(W, H))
    log = []
    t0 = time.time()
    for k in range(steps):
        t = k * dt
        if abs(t - t_push) < 1e-9:        # 注入前冲扰动
            st = dataclasses.replace(
                st, root_linvel=st.root_linvel + push_v)
        a_hold = np.clip(rad_to_norm(st.qpos), -1, 1)
        ss0 = st.to_support_state()
        ts = time.time()
        a = st.action_for_balance(dt=dt, max_evals=200,
                                  eps=0.08, horizon=3)
        ms = (time.time() - ts) * 1e3
        st = st.take_action(a, dt=dt)
        ss = st.to_support_state()
        m = ss.balance_margin()
        da = float(np.abs(a - a_hold).max())
        tag = "PUSH" if abs(t - t_push) < 1e-9 else ""
        print(f"[{k:3d}] t={t:4.2f} m0={ss0.balance_margin():+.3f} "
              f"m={m:+.3f} |da|={da:.3f} solve={ms:5.0f}ms "
              f"com_z={st.root_position[2]:.3f} {tag}", flush=True)
        log.append(m)
        frame = annotate(st.render(width=W, height=H), [
            f"t = {t:4.2f}s   step {k}/{steps}",
            f"balance_margin = {m:+.3f} m",
            f"|a - a_hold|max = {da:.3f}",
            f"root_z = {st.root_position[2]:.3f} m  {tag}"])
        writer.write(np.asarray(frame))
        if st.root_position[2] < 0.35:    # 已倒，提前收尾
            print(f"[{k:3d}] root_z={st.root_position[2]:.3f} — 倒地终止")
            for _ in range(10):           # 倒地画面停留半秒
                writer.write(np.asarray(frame))
            break
    writer.close()
    n_inf = sum(1 for m in log if not np.isfinite(m))
    print(f"\n{out}  ({time.time()-t0:.0f}s, {len(log)} 步, "
          f"margin>0 步数={sum(1 for m in log if m > 0)}/{len(log)}, "
          f"inf={n_inf})")


if __name__ == "__main__":
    main()
