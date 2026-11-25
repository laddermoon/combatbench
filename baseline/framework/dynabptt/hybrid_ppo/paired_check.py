"""Paired real-vs-model advantage check on a calibrated model.

Loads a model saved by offline_calib --save-out plus actor/critic from a
training checkpoint, builds the branch-restoration env, and runs the full
validate_model gate including the paired advantage check.

    PYTHONPATH=. python3 -m baseline.framework.dynabptt.hybrid_ppo.paired_check \
        --ckpt .../ckpt_u00019.pt --model /tmp/model_dt20.pt \
        --replay /tmp/replay_dt20_big.pkl --phy-steps 10
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)

from .model import RepCritic, SelfWorldModel, fit_rep_critic
from .standup import ACTION_DIM, SELF_OBS_DIM, STATE_DIM, make_standup_env_bp
from .transitions import ModelReplay
from .validation import validate_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--phy-steps", type=int, default=10)
    ap.add_argument("--paired-seeds", type=int, default=64)
    ap.add_argument("--paired-h", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(555)

    ckpt = torch.load(args.ckpt, map_location=device)
    blob = torch.load(args.model, map_location=device)
    data = pickle.load(open(Path(args.replay), "rb"))

    replay = ModelReplay()
    replay.fit = list(data["fit"])
    replay.hold = list(data["hold"])
    replay._fit_trans = data["fit_trans"]
    replay._hold_seen = set(data["hold_seen"])

    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=str(device))
    actor.load_state_dict(ckpt["actor"])
    critic = CriticMLP(obs_dim=SELF_OBS_DIM, hidden_dim=256).to(device)
    critic.load_state_dict(ckpt["critic"])

    hidden = tuple(blob["hidden"])
    dt = float(blob["dt"])
    dec = round(dt / 0.002)
    model = SelfWorldModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, obs_dim=SELF_OBS_DIM,
        hidden=hidden, dt=dt,
        substep_coeff=(dec + 1) / (2 * dec)).to(device)
    model.load_state_dict(blob["model"])
    print(f"model loaded: hidden={hidden} dt={dt}", flush=True)

    vrep = RepCritic().to(device)
    opt_v = torch.optim.Adam(vrep.parameters(), lr=3e-4)
    vl = fit_rep_critic(vrep, opt_v, replay, model, rng, device,
                        batches=80)
    print(f"vrep trained (loss {vl:.4f})", flush=True)

    env_bp = make_standup_env_bp(
        max_steps=500, phy_steps_per_action=args.phy_steps)
    for tag, v in (("obsC", None), ("repC", vrep)):
        report = validate_model(
            model, actor, critic, replay, rng, device,
            n_windows=1024, env_bp=env_bp,
            paired_seeds=args.paired_seeds, paired_h=args.paired_h,
            vrep=v)
        print(f"[{tag}] paired:",
              {k: round(x, 4) for k, x in report.paired.items()},
              flush=True)
        print(f"[{tag}] gate: horizon={report.horizon} "
              f"stages={report.admitted_stages} "
              f"pass={report.all_passed}", flush=True)
    for stg, hm in report.metrics.items():
        for h in (1, 2, 3):
            m = hm[h]
            print(f"stage{stg} h{h}: lv={m.get('nrmse_linvel', float('nan')):.3f} "
                  f"obs={m.get('obs_nrmse', float('nan')):.3f} "
                  f"phi={m.get('phi_mae', float('nan')):.3f} "
                  f"adev={m.get('act_dev_rms', float('nan')):.3f} "
                  f"rotp95={m.get('rot_deg_p95', float('nan')):.1f}",
                  flush=True)
    print("gate notes:", report.notes, flush=True)


if __name__ == "__main__":
    main()
