"""矢状面抡臂验收：θ 整圆扫掠（前→上→后→下→前），固定 φ/ext。

θ 是 B 约定下的扫掠角，本脚本验证：(1) 整圆连续性与分支稳定
（帧间传 prev）；(2) 不可达弧段（正后/近下）的最近点投影形态；
(3) dir_err/pos_err 随 θ 的分布。帧 → PNG + ffmpeg 合成 mp4。

    MUJOCO_GL=egl PYTHONPATH=. python3 -m \
        baseline.humanoid21.v1.sweep_arm_validate \
        --pol 90 --ext 0.95 --step 10 --out <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw

from baseline.humanoid21.v1.arm_ik import ArmIK, ArmPose
from baseline.humanoid21.v1.render_arm_poses import _load, _set_arms, _render
from baseline.humanoid21.v1.sample_arm_validate import _font


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", type=float, default=90.0,
                    help="外展角：90=纯矢状面圆环 <90=外张锥面")
    ap.add_argument("--ext", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=10.0, help="θ 步长(度)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="baseline/humanoid21/v1/sweep_renders")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ik = ArmIK()
    m, d = _load()
    font = _font(26)
    prev, rows = None, []

    # θ: 0(前) → +90(上) → 180(后) → -90(下) → 360(回前)
    azs = np.arange(0.0, 360.0 + args.step * 0.5, args.step)
    azs[azs > 180.0] -= 360.0          # wrap 到 (−180,180] 表达
    for i, az in enumerate(azs):
        sol = ik.solve("right", ArmPose.deg(float(az), args.pol, args.ext),
                       prev=prev)
        prev = sol.angles
        _set_arms(m, d, sol.angles, np.zeros(3))
        img = np.concatenate([_render(m, d, 0), _render(m, d, 90)], axis=1)
        im = Image.fromarray(img)
        tag = f"#{i:02d}  az={az:+.0f}  pol={args.pol:.0f}  ext={args.ext:.2f}"
        if sol.clipped:
            tag += f"  clip:{sol.clipped}"
        if sol.pos_err > 0.01:
            tag += f"  pos_err:{sol.pos_err:.2f}m"
        ImageDraw.Draw(im).text(
            (10, 8), tag, fill=(255, 255, 0), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0))
        im.save(os.path.join(args.out, f"sweep_{i:02d}.png"))
        rows.append((az, sol.dir_err_deg, sol.pos_err,
                     sol.clipped or ""))

    # contact sheet
    thumbs = [Image.open(os.path.join(args.out, f"sweep_{i:02d}.png"))
              .resize((256, 144)) for i in range(len(azs))]
    cols = 6
    n_rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (256 * cols, 144 * n_rows), (30, 30, 30))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 144))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))

    # mp4
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps),
         "-i", os.path.join(args.out, "sweep_%02d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         os.path.join(args.out, "sweep.mp4")],
        check=True, capture_output=True)

    # 手位轨迹摘要（θ→方向/残差，看哪段不可达）
    print(f"{len(azs)} frames → {args.out} (+sweep.mp4)")
    for az, de, pe, cl in rows:
        bar = "#" * int(pe * 50)
        print(f"  az={az:+6.0f}  dir_err={de:5.1f}deg  "
              f"pos_err={pe:.3f}m  {cl:24s} {bar}")


if __name__ == "__main__":
    main()
