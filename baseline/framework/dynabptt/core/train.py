"""DynaBPTT core orchestrator — main-experiment spec (MAIN_EXP_DECISIONS.md).

Per update:
  S1  SubStartSet: |SS|<2048 -> every element once deterministically,
      then phi^6-weighted resamples with sigma noise to fill 2048;
      |SS|>=2048 -> phi^6-weighted draw of 2048, all deterministic.
  S2  Rollout NStep=8 in the real env -> branches into replay.
      Strict s1 admission into StartSet (no dedup, no cap).
  S3  Fit the WM (rep-space transition regression) on
      all-new windows + recency-weighted history until the 1-step
      normalized RMSE on THIS update's new windows < threshold.
      Not converged -> collect+fit only, no policy update.
  S4  Imagine H=8 deterministically through the WM from the same seeds.
  S5  Loss = -J, J = sum_t phi(rep_t) (gamma=1, no critic, no shaping).
  S6  4 BPTT gradient steps on the actor.

WMState = (aux, rep, feet): rep/aux from cs analytically, feet from
obs[52:54].  Obs for the policy = compose_obs(rep, aux, feet).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.dynabptt.core.collect import BranchCollector
from baseline.framework.dynabptt.core.model import (
    RepWorldModel, phi_of_rep, state_loss, wmstate_breakdown,
    wmstate_nrmse)
from baseline.framework.dynabptt.core.startset import StartSetManager
from baseline.framework.dynabptt.hybrid_ppo import gencoord as gc
from baseline.framework.dynabptt.smbptt.task import (
    STANDING_HEIGHT, analytic_phi, make_balance_env_bp)
from baseline.framework.dynabptt.hybrid_ppo.standup import (
    ACTION_DIM, SELF_OBS_DIM)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
FEET = slice(52, 54)


def phi_np(cs: np.ndarray) -> np.ndarray:
    """phi from raw core state(s), numpy. cs (...,55) -> (...)"""
    q = cs[..., 3:7]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ux, uy, uz = -x, -y, -z
    vx, vy, vz = 0.0, 0.0, -1.0
    tx = 2 * (uy * vz - uz * vy)
    ty = 2 * (uz * vx - ux * vz)
    tz = 2 * (ux * vy - uy * vx)
    gx = vx + w * tx + (uy * tz - uz * ty)
    gy = vy + w * ty + (uz * tx - ux * tz)
    gz = vz + w * tz + (ux * ty - uy * tx)
    return (-gz) * cs[..., 2] / STANDING_HEIGHT


def actor_sd_numpy(actor) -> dict:
    return {k: v.detach().cpu().numpy()
            for k, v in actor.state_dict().items()}


def _state_windows(sel, device, k_max=8):
    """Stack branches into WMState-unroll tensors.

    k = min(k_max, shortest branch len).  rep/aux come from cs,
    feet from obs[52:54].  Returns None-tuple if all branches empty.
    """
    k = min([k_max] + [len(b["act"]) for b in sel])
    if k < 1:
        return (None,) * 7
    cs0 = torch.as_tensor(np.stack([b["cs"][0] for b in sel]),
                          dtype=torch.float32, device=device)
    acts = torch.as_tensor(np.stack([b["act"][:k] for b in sel]),
                           dtype=torch.float32, device=device)
    cs_t = torch.as_tensor(np.stack([b["cs"][1:k + 1] for b in sel]),
                           dtype=torch.float32, device=device)
    feet0 = torch.as_tensor(np.stack(
        [b["obs"][0][FEET] for b in sel]),
        dtype=torch.float32, device=device)
    feet_t = torch.as_tensor(np.stack(
        [b["obs"][1:k + 1, FEET] for b in sel]),
        dtype=torch.float32, device=device)
    rep0, aux0 = gc.represent(cs0), gc.extract_aux(cs0)
    rep_t, aux_t = gc.represent(cs_t), gc.extract_aux(cs_t)
    return rep0, aux0, feet0, acts, rep_t, aux_t, feet_t


def _fit_batch(model, opt_m, win, feet_std):
    rep0, aux0, feet0, acts, rep_t, aux_t, feet_t = win
    rep, aux, feet = rep0, aux0, feet0
    L = 0.0
    for m in range(acts.shape[1]):
        aux, rep, feet = model.next_state(rep, aux, feet, acts[:, m])
        L = L + state_loss(rep, aux, rep_t[:, m], aux_t[:, m],
                           feet, feet_t[:, m], feet_std)
    L = L / acts.shape[1]
    opt_m.zero_grad(set_to_none=True)
    L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt_m.step()
    return float(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phy-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--episodes", type=int, default=2048,
                   help="rollout episodes (branches) per update")
    p.add_argument("--nstep", type=int, default=8)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--sigma", type=float, default=0.15,
                   help="action noise for the fill-up phase only")
    p.add_argument("--kappa", type=float, default=6.0)
    p.add_argument("--model-hidden", type=int, default=1024)
    p.add_argument("--model-batch", type=int, default=256)
    p.add_argument("--fit-threshold", type=float, default=0.01,
                   help="1-step normalized RMSE on new windows")
    p.add_argument("--fit-max-batches", type=int, default=2000,
                   help="safety cap on WM fit batches per update")
    p.add_argument("--hist-tau", type=float, default=100.0,
                   help="recency decay (updates) for replay weighting")
    p.add_argument("--imag-epochs", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--model-lr", type=float, default=2e-4)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-nstep", type=int, default=100)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env_bp = make_balance_env_bp(
        max_steps=args.max_steps, phy_steps_per_action=args.phy_steps)
    dt = args.phy_steps * 0.002
    sub_coeff = (args.phy_steps + 1) / (2 * args.phy_steps)

    actor = TruncatedNormalPolicy(
        obs_dim=SELF_OBS_DIM, action_dim=ACTION_DIM,
        hidden_dim=256, device=str(device))
    actor.log_std.requires_grad_(False)
    model = RepWorldModel(
        action_dim=ACTION_DIM, hidden=(args.model_hidden,) * 3,
        dt=dt, substep_coeff=sub_coeff).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.actor_lr,
                             betas=(0.9, 0.95))
    opt_m = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    collector = BranchCollector(env_bp, num_workers=args.workers)
    atexit.register(collector.close)
    def _sigterm(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    ss = StartSetManager(
        cap=2 ** 31, dedup_eps=0.0, evict="phi", rng=rng,
        kappa=args.kappa, weighted_frac=1.0)
    replay: list[dict] = []          # append-only; each carries "update"

    start_u = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device,
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        model.load_state_dict(ck["model"])
        ss.load_state_dict(ck["startset"])
        start_u = int(ck["update"]) + 1
        rp = Path(str(args.resume_from).replace(".pt", ".replay.pkl"))
        if rp.exists():
            replay = pickle.load(open(rp, "rb"))["replay"]

    if len(ss) == 0:
        boot_env = env_bp.build()
        for pl in boot_env.plugins:
            pl.set_episode_seed(args.seed)
        boot_env.reset(seed=args.seed, base_seed=args.seed)
        from baseline.framework.dynabptt.corestate import (
            flatten_core_state)
        cs0 = flatten_core_state(
            boot_env.simulator.get_core_state(), "robot_a")
        rep0 = gc.represent(torch.as_tensor(
            cs0[None], dtype=torch.float32))[0].numpy()
        ss.add_initial(cs0, float(phi_np(cs0)), rep0)
        del boot_env
        print(f"[init] StartSet bootstrapped phi={float(phi_np(cs0)):.3f}",
              flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    def jlog(rec: dict) -> None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    jlog({"type": "header", "run": args.run_name, "seed": args.seed,
          "episodes": args.episodes, "nstep": args.nstep,
          "fit_threshold": args.fit_threshold, "hist_tau": args.hist_tau,
          "git": os.popen("git rev-parse HEAD").read().strip()})

    t_start = time.time()
    for u in range(start_u, args.max_updates):
        t0 = time.time()
        rec = {"type": "update", "u": u}

        # ---- S1: SubStartSet -------------------------------------------
        n_all = len(ss)
        sigmas = np.zeros(args.episodes, dtype=np.float32)
        if n_all < args.episodes:
            det_idx = np.arange(n_all)
            n_fill = args.episodes - n_all
            fill_idx = ss.sample(n_fill, rng)
            sigmas[n_all:] = args.sigma
            idx = np.concatenate([det_idx, fill_idx])
        else:
            idx = ss.sample(args.episodes, rng)
        seeds_cs = ss.states(idx)                       # (N,55)

        # ---- S2: rollout -----------------------------------------------
        sd_np = actor_sd_numpy(actor)
        branches = collector.collect(
            sd_np, seeds_cs, nstep=args.nstep, sigma=sigmas,
            base_seed=args.seed + 1000 + u * 13)

        # strict wavefront: only s1 of each branch enters StartSet
        succ_cs, succ_phi, succ_rep = [], [], []
        for b in branches:
            if len(b["cs"]) >= 2 and not b["term"][0]:
                cs1 = b["cs"][1]
                succ_cs.append(cs1)
                succ_phi.append(float(phi_np(cs1)))
                succ_rep.append(gc.represent(torch.as_tensor(
                    cs1[None], dtype=torch.float32))[0].numpy())
        n_admit = ss.admit_successors(succ_cs, succ_phi, succ_rep) \
            if succ_cs else 0

        for b in branches:
            b["update"] = u
            replay.append(b)

        surv = float(np.mean([
            not b["term"][-1] if len(b["term"]) else True
            for b in branches])) if branches else 0.0
        rec.update({
            "n_sub": len(idx), "n_det": int((sigmas == 0).sum()),
            "admit": n_admit, "ss_size": len(ss),
            "branch_surv": surv,
            "branch_len": float(np.mean([len(b["act"]) for b in branches])),
            **{f"ss_{k}": v for k, v in ss.stats().items()},
        })

        # ---- S3: WM fit until criterion on NEW windows -----------------
        new_branches = [b for b in replay if b["update"] == u]
        hist_branches = [b for b in replay if b["update"] < u]
        fit_ok = False
        nrmse = float("nan")
        n_batches = 0
        wm_loss = 0.0

        def _pool_states(pool):
            cs = torch.as_tensor(np.concatenate(
                [b["cs"] for b in pool]), dtype=torch.float32)
            ob = torch.as_tensor(np.concatenate(
                [b["obs"] for b in pool])[:, FEET],
                dtype=torch.float32)
            return torch.cat(
                [gc.represent(cs), ob], dim=-1)   # (N, 54)

        # feet std for the fit loss
        win0 = _state_windows(
            new_branches[:min(512, len(new_branches))], device)
        feet_std = win0[6].std(0).clamp_min(1e-3) if win0[0] is not None \
            else torch.ones(2, device=device)

        while not fit_ok:
            model.fit_normalization(_pool_states(
                new_branches + hist_branches[:min(
                    len(hist_branches), 4 * len(new_branches))]))

            order = rng.permutation(len(new_branches))
            for i in range(0, len(order), args.model_batch):
                sel = [new_branches[j] for j in order[i:i + args.model_batch]]
                win = _state_windows(sel, device)
                if win[0] is not None:
                    wm_loss = _fit_batch(model, opt_m, win, feet_std)
                    n_batches += 1
            if hist_branches:
                age = np.asarray([u - b["update"] for b in hist_branches],
                                 dtype=np.float32)
                w = np.exp(-age / args.hist_tau)
                w = w / w.sum()
                hi = rng.choice(len(hist_branches),
                                size=min(len(new_branches),
                                         len(hist_branches)),
                                replace=True, p=w)
                hsel = [hist_branches[j] for j in hi]
                horder = rng.permutation(len(hsel))
                for i in range(0, len(horder), args.model_batch):
                    sel = [hsel[j] for j in horder[i:i + args.model_batch]]
                    win = _state_windows(sel, device)
                    if win[0] is not None:
                        wm_loss = _fit_batch(model, opt_m, win, feet_std)
                        n_batches += 1

            # criterion: teacher-forced 1-step WMState nRMSE on NEW windows
            win = _state_windows(new_branches, device, k_max=1)
            if win[0] is None:
                break
            rep0, aux0, feet0, acts, rep_t, aux_t, feet_t = win
            with torch.no_grad():
                aux_p, rep_p, feet_p = model.next_state(
                    rep0, aux0, feet0, acts[:, 0])
            nrmse = wmstate_nrmse(
                rep_p, aux_p, feet_p,
                rep_t[:, 0], aux_t[:, 0], feet_t[:, 0])
            fit_ok = nrmse < args.fit_threshold
            if n_batches >= args.fit_max_batches:
                rec["fit_top_dims"] = [
                    {"dim": d, "share": round(s, 3),
                     "rmse": round(r, 4), "std": round(sd, 4)}
                    for d, s, r, sd in wmstate_breakdown(
                        rep_p, aux_p, feet_p,
                        rep_t[:, 0], aux_t[:, 0], feet_t[:, 0])[:5]]
                break

        rec.update({"wm_loss": wm_loss, "fit_nrmse": nrmse,
                    "fit_batches": n_batches, "fit_ok": bool(fit_ok)})

        # diagnostic: paired closed-loop phi-MAE on 64 of this update's
        # branches (replay their real actions through the model)
        paired_mae = float("nan")
        if len(new_branches) >= 8:
            didx = rng.choice(len(new_branches),
                              min(64, len(new_branches)), replace=False)
            dbr = [new_branches[i] for i in didx]
            win = _state_windows(dbr, device, k_max=args.nstep)
            if win[0] is not None:
                rep0, aux0, feet0, acts, rep_t, aux_t, feet_t = win
                real_phi = analytic_phi(rep_t)
                with torch.no_grad():
                    rep, aux, feet = rep0, aux0, feet0
                    preds = []
                    for m in range(acts.shape[1]):
                        aux, rep, feet = model.next_state(
                            rep, aux, feet, acts[:, m])
                        preds.append(phi_of_rep(rep))
                    pred_phi = torch.stack(preds, 1)
                paired_mae = float((pred_phi - real_phi).abs().mean())
        rec["paired_phi_mae"] = paired_mae

        # ---- S4-S6: imagine + BPTT (only when fit converged) -----------
        if fit_ok:
            win = _state_windows(branches, device, k_max=1)
            rep_i, aux_i, feet_i = win[0], win[1], win[2]
            for pp in model.parameters():
                pp.requires_grad_(False)
            actor.train()
            J_tot = gn = 0.0
            for epoch in range(args.imag_epochs):
                rep, aux, feet = rep_i, aux_i, feet_i
                o = model.obs_of(rep, aux, feet)
                J = 0.0
                for t in range(args.nstep):
                    a = actor.deterministic_action(o).clamp(-1.0, 1.0)
                    aux, rep, feet = model.next_state(rep, aux, feet, a)
                    J = J + phi_of_rep(rep)
                    o = model.obs_of(rep, aux, feet)
                loss = -J.mean()
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
                gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()
                J_tot += float(J.mean())
            for pp in model.parameters():
                pp.requires_grad_(True)
            rec["J"] = J_tot / args.imag_epochs
            rec["gnorm"] = float(gn)

        # ---- eval: deterministic long branches -------------------------
        if (u + 1) % args.eval_every == 0 and len(ss) > 0:
            eidx = ss.sample(min(128, len(ss)), rng)
            ev = collector.collect(
                actor_sd_numpy(actor), ss.states(eidx),
                nstep=args.eval_nstep, sigma=0.0,
                base_seed=args.seed + 555 + u)
            rec["eval_surv"] = float(np.mean([
                len(b["act"]) / args.eval_nstep for b in ev]))
            rec["eval_full"] = float(np.mean([
                not (b["term"][-1] if len(b["term"]) else False)
                for b in ev]))
            for b in ev:
                b["update"] = u
                replay.append(b)

        rec["t"] = time.time() - t0
        jlog(rec)
        print(f"[u{u:04d}] ss={len(ss)} det={rec['n_det']} "
              f"surv={rec['branch_surv']:.2f} "
              f"fit={'OK' if fit_ok else 'X'}({nrmse:.4f},{n_batches}b) "
              f"pair={paired_mae:.3f} "
              f"J={rec.get('J', float('nan')):.2f} "
              f"t={rec['t']:.1f}s", flush=True)

        if (u + 1) % 25 == 0:
            torch.save({
                "actor": actor.state_dict(), "model": model.state_dict(),
                "update": u,
                "opt_a": opt_a.state_dict(), "opt_m": opt_m.state_dict(),
                "args": vars(args),
                "startset": ss.state_dict(),
            }, run_dir / f"ckpt_u{u:05d}.pt")
            pickle.dump({"replay": replay[-2000:]},
                        open(run_dir / f"ckpt_u{u:05d}.replay.pkl", "wb"))

    collector.close()


if __name__ == "__main__":
    main()
