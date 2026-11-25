"""Offline model-calibration probe: is the gate unreachable because the
model is undertrained (60 batches/update) or because of a structural
ceiling?

Loads a sidecar checkpoint's replay tail + weights, refits the model
hard (many batches), and validates periodically — open-loop metrics and
closed-loop action divergence only (no paired env check).

Usage:
    PYTHONPATH=. python3 -m baseline.framework.dynabptt.hybrid_ppo.offline_calib \
        --ckpt baseline/framework/dynabptt/hybrid_ppo/runs/hp_p1_sidecar_s42/checkpoints/ckpt_u00019.pt \
        --batches 1200 --val-every 300
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)

from .model import SelfWorldModel, fit_model
from .standup import ACTION_DIM, SELF_OBS_DIM, STATE_DIM
from .train import ModelConfig
from .transitions import ModelReplay
from .validation import validate_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replay", default=None,
                    help="override replay pkl (e.g. collected at another dt)")
    ap.add_argument("--batches", type=int, default=1200)
    ap.add_argument("--val-every", type=int, default=300)
    ap.add_argument("--val-windows", type=int, default=1024)
    ap.add_argument("--fresh", action="store_true",
                    help="start model from scratch instead of ckpt weights")
    ap.add_argument("--hidden", type=int, nargs="+", default=None,
                    help="override dynamics hidden dims (e.g. 1024 1024 1024)")
    ap.add_argument("--dt", type=float, default=0.05,
                    help="control step seconds (0.05 = 25x2ms, 0.02 = 10x2ms)")
    ap.add_argument("--save-out", default=None,
                    help="save trained model state_dict to this path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(777)
    torch.manual_seed(777)

    ckpt = torch.load(args.ckpt, map_location=device)
    rp = Path(args.replay) if args.replay else Path(
        str(args.ckpt).replace(".pt", ".replay.pkl"))
    data = pickle.load(open(rp, "rb"))

    replay = ModelReplay()
    replay.fit = list(data["fit"])
    replay.hold = list(data["hold"])
    replay._fit_trans = data["fit_trans"]
    replay._hold_seen = set(data["hold_seen"])
    cur_u = max(t.update for t in replay.fit)
    print(f"replay: {len(replay.fit)} fit trajs "
          f"({replay._fit_trans} trans), {len(replay.hold)} hold trajs, "
          f"latest update={cur_u}", flush=True)

    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=str(device))
    actor.load_state_dict(ckpt["actor"])
    critic = CriticMLP(obs_dim=SELF_OBS_DIM, hidden_dim=256).to(device)
    critic.load_state_dict(ckpt["critic"])
    hidden = tuple(args.hidden) if args.hidden else (512, 512, 512)
    dec = round(args.dt / 0.002)
    model = SelfWorldModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, obs_dim=SELF_OBS_DIM,
        hidden=hidden, dt=args.dt,
        substep_coeff=(dec + 1) / (2 * dec)).to(device)
    if not args.fresh and args.hidden is None:
        try:
            model.load_state_dict(ckpt["model"])
            print("loaded model weights from ckpt")
        except (RuntimeError, KeyError) as e:
            print(f"model weights incompatible with current arch "
                  f"({e}) — starting fresh")
    opt_m = torch.optim.Adam(model.parameters(), lr=3e-4)

    mcfg = ModelConfig()
    nw = replay.sample_windows(1024, mcfg.model_k, rng, cur_update=cur_u)
    model.fit_normalization(nw)

    def val(tag: str) -> None:
        rep = validate_model(
            model, actor, critic, replay, rng, device,
            n_windows=args.val_windows, env_bp=None, paired_seeds=0)
        row = {"tag": tag}
        for stg, hm in rep.metrics.items():
            h1, h3 = hm[1], hm[3]
            row[f"s{stg}_lv"] = h1.get("nrmse_linvel", float("nan"))
            row[f"s{stg}_lv3"] = h3.get("nrmse_linvel", float("nan"))
            row[f"s{stg}_obs"] = h3.get("obs_nrmse", float("nan"))
            row[f"s{stg}_phi"] = h3.get("phi_mae", float("nan"))
            row[f"s{stg}_adev3"] = h3.get("act_dev_rms", float("nan"))
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float)
                         else f"{k}={v}" for k, v in row.items()),
              flush=True)

    val("init")
    done = 0
    t0 = time.time()
    while done < args.batches:
        step = min(args.val_every, args.batches - done)
        mcfg.model_fit_batches = step
        stats = fit_model(model, opt_m, replay, mcfg, rng, cur_u, device)
        done += step
        print(f"[{done} batches, {time.time()-t0:.0f}s] "
              f"wm_s={stats['s']:.4f} wm_obs={stats['obs']:.4f} "
              f"wm_phi={stats['phi']:.4f} wm_stage={stats['stage']:.4f}",
              flush=True)
        val(f"b{done}")

    if args.save_out:
        torch.save({"model": model.state_dict(),
                    "hidden": list(hidden), "dt": args.dt},
                   args.save_out)
        print(f"saved model -> {args.save_out}", flush=True)


if __name__ == "__main__":
    main()
