#!/usr/bin/env python3
"""e2e：ready 起步的简化步态 FSM 演示视频（前向走 1~2 步）。

用法（仓库根目录）：

    PYTHONPATH=. python3 point5/humanoid21/e2e_walk_video.py [out.mp4]

相位机（用户的"让载-接住-摆动-落地"人类机制）：

    DOUBLE     ready 双支撑
    YIELD      右膝加深屈曲(让载) + 左膝伸展(推) → com 向右自由漂移
    CATCH      右膝回弹近伸直 → 接住 com（门控: 距右脚<6cm 且 |vy|<0.2）
    SWING_L    给目标迈步姿态 → 左腿前摆、右腿后拖、右臂前左臂后
    SETTLE     落地后冻结当前姿态 → 等 com 漂过左脚 → 镜像第二步
    SWING_R    镜像迈步姿态（第二步）
    DONE       落地后冻结，观察能否稳住

落地检测：先见摆脚抬升 >4cm 再回落 <2cm（防起点贴地短路）。
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


def _set_joints(qpos: np.ndarray, **kwargs) -> np.ndarray:
    for n, v in kwargs.items():
        qpos[JOINT_NAMES.index(n)] = v
    return qpos


def _pose_action(st: RobotState, **kwargs) -> np.ndarray:
    q = st.qpos.copy()
    _set_joints(q, **kwargs)
    return np.clip((q - _MID) / _HALF, -1.0, 1.0)


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
               else "/data1/dev/combatbench_executor/e2e/walk_fsm.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    dt, max_steps = 0.05, 170          # 8.5s @ 20Hz
    W, H = 1280, 720

    st = RobotState.ready()
    base = st.corresponding_action()

    # ---- 相位动作（烘焙自 ready 的关节修改） ----
    # 让载+推: 右膝加深、左膝伸展 → com 右漂
    act_yield = base.copy()
    act_yield[JOINT_NAMES.index("knee_right")] = _norm_target("knee_right", -0.80)
    act_yield[JOINT_NAMES.index("knee_left")] = _norm_target("knee_left", -0.30)
    # 接住: 右膝回弹至近伸直 + 踝回位
    act_catch = base.copy()
    act_catch[JOINT_NAMES.index("knee_right")] = _norm_target("knee_right", -0.20)
    # 第一步姿态: 左腿前摆、右腿后拖、右臂前左臂后、左髋外摆加宽落点
    act_step_l = _pose_action(
        st,
        hip_x_left=-0.25, hip_y_left=-0.90, knee_left=-0.60,
        ankle_y_left=-0.10,
        hip_y_right=+0.10, knee_right=-0.35, ankle_y_right=-0.28,
        shoulder2_right=+0.55, shoulder2_left=-0.85,
        shoulder1_right=+0.15, shoulder1_left=+0.20)
    # 第二步镜像
    act_step_r = _pose_action(
        st,
        hip_x_right=-0.25, hip_y_right=-0.90, knee_right=-0.60,
        ankle_y_right=-0.10,
        hip_y_left=+0.10, knee_left=-0.35, ankle_y_left=-0.28,
        shoulder2_left=+0.55, shoulder2_right=-0.85,
        shoulder1_left=+0.15, shoulder1_right=+0.20)

    wr = _Writer(out, fps=20, size=(W, H))
    phase = "YIELD"
    lifted_l = lifted_r = False
    t0 = td = tc = 0.0
    hold = None
    history = []

    for k in range(max_steps):
        g = st.to_support_state()
        t = k * dt

        if phase == "YIELD":
            act = act_yield
            if g.com[1] < -0.04 or t > 0.6:
                phase, tc = "CATCH", t
        elif phase == "CATCH":
            act = act_catch
            d = float(np.linalg.norm(g.com[:2] - g.right_foot))
            if (d < 0.06 and abs(g.com_vel[1]) < 0.2) or t > tc + 0.9:
                phase, t0 = "SWING_L", t
                history.append(f"{t:.2f} SWING_L com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
        elif phase == "SWING_L":
            act = act_step_l
            if g.left_foot_height > 0.04:
                lifted_l = True
            if lifted_l and g.left_foot_height < 0.02:
                phase, td = "SETTLE", t
                history.append(f"{t:.2f} 左脚落地 com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
            if t > t0 + 1.6:
                history.append(f"{t:.2f} 摆动超时"); break
        elif phase == "SETTLE":
            act = st.corresponding_action()          # 冻结当前姿态
            # 躯干回直立：落地瞬间是前弓姿态，冻结会锁住前倾 → 持续前扑。
            # abdomen_y 回 0 明显延长二步（落地 3.05→3.80s、行程 ~1.5m）。
            act[JOINT_NAMES.index("abdomen_y")] = _norm_target("abdomen_y", 0.0)
            if (g.com[1] > 0.05 and g.com_vel[1] > 0) or t > td + 1.2:
                phase, t0 = "SWING_R", t
                history.append(f"{t:.2f} SWING_R com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
        elif phase == "SWING_R":
            act = act_step_r
            act[JOINT_NAMES.index("abdomen_y")] = _norm_target("abdomen_y", 0.0)
            if g.right_foot_height > 0.04:
                lifted_r = True
            if lifted_r and g.right_foot_height < 0.02:
                phase, td = "DONE", t
                history.append(f"{t:.2f} 右脚落地 com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
            if t > t0 + 1.6:
                history.append(f"{t:.2f} 第二步超时"); break
        else:                                       # DONE
            if hold is None:
                hold = st.corresponding_action()
                hold[JOINT_NAMES.index("abdomen_y")] = _norm_target("abdomen_y", 0.0)
            act = hold
            if st.root_position[2] < 0.9:           # 已倒：记一次后收尾
                history.append(f"{t:.2f} 失稳 com=({g.com[0]:+.2f},{g.com[1]:+.2f})")
                break
            if t > td + 1.6:
                history.append(f"{t:.2f} 两步后稳住 1.6s"); break

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
