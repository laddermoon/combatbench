"""P5 studio: web-based Point5 viewer/editor.

    PYTHONPATH=. python3 mocap/point5/studio/server.py --dir <workdir> [--port 8901]

JSON API under /api; the SPA lives in static/.  Dragging a point talks to
the two-stage mechanics solver: head/foot drags hit check_legal live and
the stage-1 solve on release; hand drags run the analytic stage-2 check
in real time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mocap.point5.humanoid21.mechanics import (  # noqa: E402
    LANDMARK_ORDER, Skeleton, _stage2_core, check_legal,
    norm_to_rad, solve_equilibrium)

sk = Skeleton()
app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"),
            static_url_path="/static")
WORK = Path.cwd()

def _extract_geoms():
    """Real Humanoid21 geoms from the MuJoCo model, per mechanics body.

    Returns {body: [{"type","size","pos","quat","rgba"}]} — spheres,
    capsules, cylinders, boxes only.
    """
    import mujoco
    xml = ROOT / "envs/humanoid21/battle_circular_v2.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    tmap = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid",
            5: "cylinder", 6: "box"}
    out = {}
    for gi in range(m.ngeom):
        bid = int(m.geom_bodyid[gi])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname.endswith("_a"):
            continue
        g = {"type": tmap.get(int(m.geom_type[gi])),
             "size": m.geom_size[gi].tolist(),
             "pos": m.geom_pos[gi].tolist(),
             "quat": m.geom_quat[gi].tolist(),
             "rgba": m.geom_rgba[gi].tolist()}
        if g["type"] is None:
            continue
        out.setdefault(bname[:-2], []).append(g)
    return out


VIS = _extract_geoms()
POINT_COLOR = {"head": "#e0483e", "hand_right": "#3ba53b",
               "hand_left": "#3b7fd4", "foot_right": "#e0a02a",
               "foot_left": "#b56ae0"}

# ---- state ------------------------------------------------------------
S = {"p5": None, "yaw": None, "res": None, "frozen": None,
     "clip": [], "sel": -1}


def _initial_p5():
    npz = ROOT / "baseline/humanoid21/hierarchy/assets/initial_states" \
        "/strike_start_v1/strike_start_p5.npz"
    if npz.exists():
        d = np.load(npz)
        return d["p5"][0].astype(np.float64), float(d["torso_yaw"][0])
    q = np.zeros(21)
    lm = sk.landmarks(np.array([0.0, 0.0, 0.79]), np.array([1., 0, 0, 0]),
                      norm_to_rad(q))
    return lm, None


def _body_payload(res):
    xf = sk.body_transforms(res["root_pos"], res["root_quat_wxyz"],
                            res["q_rad"])
    from scipy.spatial.transform import Rotation as _R
    return {n: {"p": list(map(float, p)),
                "q": list(map(float, _R.from_matrix(m).as_quat()))}
            for n, (p, m) in xf.items()}


def _res_payload(res):
    return {"satisfied": res["satisfied"],
            "failed_stage": res["failed_stage"],
            "point_errors": res["point_errors"].tolist(),
            "energy": res["energy"],
            "bodies": _body_payload(res),
            "q_norm": res["q_norm"].tolist()}


def _commit_solve():
    S["res"] = solve_equilibrium(S["p5"], torso_yaw=S["yaw"], skeleton=sk)
    if S["res"]["satisfied"]:
        q = S["res"]["q_norm"].copy()
        q[[i for i, n in enumerate(
            __import__('mocap.point5.humanoid21.mechanics',
                       fromlist=['JOINT_NAMES']).JOINT_NAMES)
            if 'shoulder' in n or 'elbow' in n]] = 0.0
        S["frozen"] = {"root_pos": S["res"]["root_pos"],
                       "root_quat": S["res"]["root_quat_wxyz"], "q": q}
    return _res_payload(S["res"])


def _state_payload():
    out = {"p5": S["p5"].tolist(), "yaw": S["yaw"],
           "clip": S["clip"], "sel": S["sel"],
           "point_names": list(LANDMARK_ORDER),
           "point_colors": POINT_COLOR}
    if S["res"] is not None:
        out["res"] = _res_payload(S["res"])
    return out


# ---- api ----------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/init")
def api_init():
    return jsonify({"workdir": str(WORK), "vis": VIS,
                    **_state_payload()})


@app.post("/api/drag")
def api_drag():
    d = request.get_json(force=True)
    name, pos, phase = d["point"], np.asarray(d["pos"], float), d["phase"]
    idx = LANDMARK_ORDER.index(name)
    cand = S["p5"].copy()
    cand[idx] = pos

    if "hand" in name:                      # stage-2 check (real time)
        if S["frozen"] is None:
            return jsonify({"ok": None, "note": "no stage-1 pose yet"})
        t0 = time.time()
        r = _stage2_core(sk, cand, S["frozen"]["root_pos"],
                         S["frozen"]["root_quat"], S["frozen"]["q"],
                         0.01, None)
        ok = r["satisfied"]
        out = {"ok": ok, "ms": round((time.time() - t0) * 1000, 1)}
        if ok:
            out["res"] = _res_payload(r)
            if phase == "end":
                S["p5"] = cand
                S["res"] = r
        elif phase == "end":
            out["res"] = _res_payload(S["res"])
        return jsonify(out)

    # head/foot: fast envelope on move, full solve on end
    if phase == "move":
        c = check_legal(cand, S["yaw"])
        return jsonify({"ok": c["legal"], "violations": c["violations"]})
    t0 = time.time()
    old_p5, old_res = S["p5"], S["res"]
    S["p5"] = cand
    try:
        pay = _commit_solve()
    except Exception as e:                  # solver blew up
        S["p5"], S["res"] = old_p5, old_res
        return jsonify({"ok": False, "error": str(e),
                        "res": _res_payload(old_res)})
    ok = pay["satisfied"]
    if not ok:
        S["p5"], S["res"] = old_p5, old_res
        pay["res"] = _res_payload(old_res)
    pay.update({"ok": ok, "ms": round((time.time() - t0) * 1000, 1)})
    return jsonify(pay)


@app.post("/api/set_yaw")
def api_yaw():
    S["yaw"] = request.get_json(force=True).get("yaw")
    return jsonify({"ok": True, "res": _commit_solve()})


@app.get("/api/files")
def api_files():
    return jsonify([{"name": p.name, "size": p.stat().st_size}
                    for p in sorted(WORK.glob("*.json"))])


@app.post("/api/save")
def api_save():
    d = request.get_json(force=True)
    name = Path(d["name"]).name
    data = {"type": "p5clip",
            "point_names": list(LANDMARK_ORDER),
            "keyframes": S["clip"],
            "current": {"pos": S["p5"].tolist(), "torso_yaw": S["yaw"]}}
    (WORK / name).write_text(json.dumps(data, indent=1))
    return jsonify({"ok": True, "name": name})


@app.post("/api/load")
def api_load():
    d = request.get_json(force=True)
    data = json.loads((WORK / Path(d["name"]).name).read_text())
    if data.get("type") == "p5clip":
        S["clip"] = data["keyframes"]
        cur = data.get("current") or {}
        if cur.get("pos") is not None:
            S["p5"] = np.asarray(cur["pos"], float)
            S["yaw"] = cur.get("torso_yaw")
    elif "pos" in data:
        S["p5"] = np.asarray(data["pos"], float)
        S["yaw"] = data.get("torso_yaw")
    return jsonify({"ok": True, "res": _commit_solve(),
                    **_state_payload()})


@app.post("/api/clip/add")
def clip_add():
    t = (S["clip"][-1]["t"] + 1.0) if S["clip"] else 0.0
    S["clip"].append({"t": t, "pos": S["p5"].tolist(),
                      "torso_yaw": S["yaw"]})
    S["sel"] = len(S["clip"]) - 1
    return jsonify({"clip": S["clip"], "sel": S["sel"]})


@app.post("/api/clip/commit")
def clip_commit():
    """Write the editor's current p5 into the selected keyframe."""
    if 0 <= S["sel"] < len(S["clip"]):
        S["clip"][S["sel"]]["pos"] = S["p5"].tolist()
        S["clip"][S["sel"]]["torso_yaw"] = S["yaw"]
    return jsonify({"clip": S["clip"], "sel": S["sel"]})


@app.post("/api/clip/sel")
def clip_sel():
    i = int(request.get_json(force=True)["i"])
    if 0 <= i < len(S["clip"]):
        S["sel"] = i
        kf = S["clip"][i]
        S["p5"] = np.asarray(kf["pos"], float)
        S["yaw"] = kf.get("torso_yaw")
        return jsonify({"res": _commit_solve(), "sel": i})
    return jsonify({"error": "bad index"}), 400


@app.post("/api/clip/del")
def clip_del():
    i = int(request.get_json(force=True)["i"])
    if 0 <= i < len(S["clip"]):
        S["clip"].pop(i)
        S["sel"] = min(S["sel"], len(S["clip"]) - 1)
    return jsonify({"clip": S["clip"], "sel": S["sel"]})


@app.post("/api/clip/time")
def clip_time():
    d = request.get_json(force=True)
    S["clip"][int(d["i"])]["t"] = float(d["t"])
    return jsonify({"clip": S["clip"]})


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--port", type=int, default=8901)
    a = ap.parse_args()
    WORK = Path(a.dir).resolve()
    S["p5"], S["yaw"] = _initial_p5()
    _commit_solve()
    app.run(host="0.0.0.0", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
