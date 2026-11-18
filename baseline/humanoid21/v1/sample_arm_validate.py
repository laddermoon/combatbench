"""描述符空间随机采样验收：N 张渲染图，标注 (θ,φ,e) + clip/err。

采样域：θ~U(−180°,180°)、φ~U(0°,180°)、e~U(0.02,1.0)——全空间
含不可达区，验收投影行为。右臂反解、左臂零位中性；front(看φ)
+side(看θ) 双视角拼接，角标参数。附 10×10 contact sheet。

    MUJOCO_GL=egl PYTHONPATH=. python3 -m \
        baseline.humanoid21.v1.sample_arm_validate \
        --n 100 --out <dir>
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from baseline.humanoid21.v1.arm_ik import ArmIK, ArmPose
from baseline.humanoid21.v1.render_arm_poses import _load, _set_arms, _render


def _font(sz: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="baseline/humanoid21/v1/sample_renders")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ik = ArmIK()
    m, d = _load()
    rng = np.random.default_rng(args.seed)
    font = _font(26)
    thumbs, errs, n_clipped = [], [], 0

    for i in range(args.n):
        az = float(rng.uniform(-180, 180))
        pol = float(rng.uniform(0, 180))
        ext = float(rng.uniform(0.02, 1.0))
        sol = ik.solve("right", ArmPose.deg(az, pol, ext))
        _set_arms(m, d, sol.angles, np.zeros(3))

        img = np.concatenate([_render(m, d, 0), _render(m, d, 90)], axis=1)
        im = Image.fromarray(img)
        tag = f"#{i:03d}  az={az:+.0f}  pol={pol:.0f}  ext={ext:.2f}"
        if sol.clipped:
            tag += f"  clip:{sol.clipped}"
        if sol.pos_err > 0.01:
            tag += f"  pos_err:{sol.pos_err:.2f}m"
        elif sol.dir_err_deg > 1:
            tag += f"  err:{sol.dir_err_deg:.0f}deg"
        ImageDraw.Draw(im).text(
            (10, 8), tag, fill=(255, 255, 0), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0))
        im.save(os.path.join(args.out, f"sample_{i:03d}.png"))
        thumbs.append(im.resize((256, 144)))
        n_clipped += bool(sol.clipped)
        errs.append(sol.dir_err_deg)

    cols = 10
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (256 * cols, 144 * rows), (30, 30, 30))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 144))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))

    print(f"{args.n} samples → {args.out}")
    print(f"clipped={n_clipped}/{args.n}  "
          f"dir_err mean={np.mean(errs):.1f}° max={np.max(errs):.0f}°")


if __name__ == "__main__":
    main()
