#!/usr/bin/env python3
"""e2e：ready 起步的"骨盆横移+回摆承接"第一步演示视频。

用法（仓库根目录）：

    PYTHONPATH=. python3 point5/humanoid21/e2e_walk_video.py [out.mp4]

机制（对应人横向移重心的真实做法——骨盆滑动，膝几乎不弯）：

    READY      双支撑（ready 前倾位）
    SHIFT      髋横移 hxL=−0.15/hxR=+0.15 + 躯干回直(abdomen_y −0.20)
               → 骨盆滑向右脚，com 到脚内沿即停（约 −0.09）
    RETURN     等 com 钟摆回正（vy>0 且 com_y>−0.10）
               —— 单支撑里 com 自然向摆动侧倒
    SWING      左腿前摆快落（hyL=−0.55/kl=−0.70 浅抬快下，
               hxL=−0.30 加宽落点接横向漂）
    LANDED     落地后冻结当前姿态

实测：左脚落在 (+0.28,+0.13) 前方偏左，站稳 ~0.7s。
后续步需要 com 横向动量管理（每步落地接住漂移），开环姿态
序列到第一步为止是稳定边界。

帧左上叠相位 / 支撑类 / com / 脚高诊断。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 渲染需在 mujoco 导入前选定 GL 后端（本机只有 osmesa 可用；
# simulator.py 会把 PYOPENGL_PLATFORM 钉成 egl，render() 内部再协调）。
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from point5.humanoid21.robot_state import RobotState
from point5.humanoid21.mechanics import _MID, _HALF, JOINT_NAMES


def _norm_target(name: str, rad: float) -> float:
    i = JOINT_NAMES.index(name)
    return float(np.clip((rad - _MID[i]) / _HALF[i], -1.0, 1.0))


def _action(base: np.ndarray, **kwargs) -> np.ndarray:
    a = base.copy()
    for n, v in kwargs.items():
        a[JOINT_NAMES.index(n)] = _norm_target(n, v)
    return a


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


def annotate(img: np.ndarray, lines) -> np.ndarray:
    """PIL 图上叠诊断文本（黑底黄字）。"""
    im = Image.fromarray(img).copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = "\n".join(lines)
    x, y = 14, 12
    bbox = d.multiline_textbbox((x, y), text, font=font, spacing=4)
    d.rectangle((bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6),
                fill=(0, 0, 0))
    d.multiline_text((x, y), text, font=font, spacing=4,
                     fill=(255, 230, 80))
    return np.asarray(im)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else "/data1/dev/combatbench_executor/e2e/walk_step1.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    dt, max_steps = 0.05, 90           # 4.5s @ 20Hz
    W, H = 1280, 720

    st = RobotState.ready()
    base = st.corresponding_action()

    # ---- 相位动作 ----
    # SHIFT: 骨盆横移到右脚 + 躯干回直（abdomen_y −0.20 比 ready −0.36 直些）
    act_shift = _action(base,
                        hip_x_left=-0.15, hip_x_right=+0.15,
                        abdomen_y=-0.20)
    # SWING: 左腿前送+落点加宽、支撑膝保持屈度
    act_swing = _action(base,
                        hip_x_left=-0.30, hip_x_right=0.0,
                        abdomen_y=-0.20,
                        hip_y_left=-0.75, knee_left=-0.50,
                        ankle_y_left=0.0, knee_right=-0.30)

    wr = _Writer(out, fps=20, size=(W, H))
    phase = "SHIFT"
    lifted = fallen = False
    t0 = td = 0.0
    history = []

    for k in range(max_steps):
        g = st.to_support_state()
        t = k * dt

        if phase == "SHIFT":
            act = act_shift
            # 到位：com 过右脚内沿且开始回摆（vy 回正）
            if g.com_vel[1] > 0.02 and g.com[1] > -0.10:
                phase, t0 = "SWING", t
                history.append(f"{t:.2f} 回摆→SWING com=({g.com[0]:+.2f},{g.com[1]:+.2f}) vy={g.com_vel[1]:+.2f}")
            if t > 1.4:
                phase, t0 = "SWING", t
                history.append(f"{t:.2f} 超时→SWING com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
        elif phase == "SWING":
            act = act_swing
            if g.left_foot_height > 0.02:
                lifted = True
            if lifted and g.left_foot_height < 0.012 and t - t0 > 0.15:
                phase, td = "LANDED", t
                history.append(f"{t:.2f} 落地 com=({g.com[0]:+.2f},{g.com[1]:+.2f}) "
                               f"fL=({g.left_foot[0]:+.2f},{g.left_foot[1]:+.2f})")
            if t - t0 > 1.3:
                history.append(f"{t:.2f} 摆超时 fhL={g.left_foot_height:.3f} z={st.root_position[2]:.2f}")
                break
        else:                                       # LANDED
            act = st.corresponding_action()
            if st.root_position[2] < 0.9 and not fallen:
                fallen = True
                history.append(f"{t:.2f} 失稳(站了{t-td:.2f}s) com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
            if t - td > 1.6 or st.root_position[2] < 0.5:
                history.append(f"{t:.2f} 落地后续 {t-td:.2f}s com=({g.com[0]:+.2f},{g.com[1]:+.2f}) z={st.root_position[2]:.2f}")
                break

        st = st.take_action(act, dt=dt)

        img = np.asarray(st.render(width=W, height=H))
        img = annotate(img, [
            f"t={t:4.2f}s  phase={phase}",
            f"support={g.support_class().name}",
            f"com=({g.com[0]:+.2f},{g.com[1]:+.2f}) v=({g.com_vel[0]:+.2f},{g.com_vel[1]:+.2f})",
            f"fhL={g.left_foot_height:.3f} fhR={g.right_foot_height:.3f}",
            f"z={st.root_position[2]:.2f}",
        ])
        wr.write(img)

    wr.close()
    print("相位历史:")
    for h in history:
        print(" ", h)
    print("saved:", out)


if __name__ == "__main__":
    main()
