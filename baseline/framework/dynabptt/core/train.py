"""DynaBPTT core orchestrator — Balance implementation.

Implements the S0-S6 loop of DYNABPTT_FRAMEWORK.md:
  S0 update StartSet (wavefront admission of first-step successors)
  S1 sample SubStartSet (1024, phi^6-weighted + uniform floor)
  S2 branch rollout in the real env (deterministic policy + sigma noise)
  S3 fit the world model on replay windows + gate evaluation
  S4 imagine H=NStep in the frozen model from the SAME SubStartSet
  S5 loss = -J + real-branch critic fit
  S6 BPTT policy update (gated)
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.dynabptt.core.collect import BranchCollector
from baseline.framework.dynabptt.core.startset import StartSetManager
from baseline.framework.dynabptt.hybrid_ppo import gencoord as gc
from baseline.framework.dynabptt.hybrid_ppo.model import (
    RepCritic, SelfWorldModel, bryson_state_loss)
from baseline.framework.dynabptt.smbptt.task import (
    OBS_NOISE_HALF, STANDING_HEIGHT, analytic_phi, compose_obs,
    make_balance_env_bp)
from baseline.framework.dynabptt.hybrid_ppo.standup import (
    ACTION_DIM, SELF_OBS_DIM, STATE_DIM)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy)

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def band_frames(b, rng, max_take=1):
    """Sparse mid-branch frames in the recoverable band (phi>=0.7) that the
    branch survives >=4 further steps — proven-viable hard states, not dead
    zones under the current policy."""
    L = len(b["act"])
    valid = []
    for k in range(4, max(4, L - 4)):
        ph = float(phi_np(b["cs"][k]))
        if ph >= 0.7 and not b["term"][k:k + 4].any():
            valid.append((k, ph))
    if not valid:
        return []
    take = min(max_take, len(valid))
    idx = rng.choice(len(valid), size=take, replace=False)
    return [(b["cs"][valid[i][0]], valid[i][1]) for i in idx]


def phi_np(cs: np.ndarray) -> np.ndarray:
    """phi from raw core state(s), numpy. cs (...,55) -> (...)"""
    q = cs[..., 3:7]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # conj-rotate [0,0,-1] by q: gravity direction in body frame
    ux, uy, uz = -x, -y, -z
    vx, vy, vz = 0.0, 0.0, -1.0
    # t = 2*cross(u,v)
    tx = 2 * (uy * vz - uz * vy)
    ty = 2 * (uz * vx - ux * vz)
    tz = 2 * (ux * vy - uy * vx)
    # v' = v + w*t + cross(u,t)
    gx = vx + w * tx + (uy * tz - uz * ty)
    gy = vy + w * ty + (uz * tx - ux * tz)
    gz = vz + w * tz + (ux * ty - uy * tx)
    upright = -gz
    return upright * cs[..., 2] / STANDING_HEIGHT


def actor_sd_numpy(actor) -> dict:
    return {k: v.detach().cpu().numpy()
            for k, v in actor.state_dict().items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phy-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--subset", type=int, default=1024)
    p.add_argument("--nstep", type=int, default=32)
    p.add_argument("--fit-unroll", type=int, default=16)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--admit-intermediate", action="store_true")
    p.add_argument("--evict", choices=["phi", "random"], default="phi")
    p.add_argument("--cap", type=int, default=4096)
    p.add_argument("--kappa", type=float, default=6.0)
    p.add_argument("--weighted-frac", type=float, default=0.85)
    p.add_argument("--obs-noise-scale", type=float, default=1.0)
    p.add_argument("--model-hidden", type=int, default=1024)
    p.add_argument("--fit-phi-weight", type=float, default=0.0,
                   help="extra WM loss weight on degraded states: "
                        "w = 1 + k*max(0, 0.95-phi_target)")
    p.add_argument("--anchor-slots", type=int, default=32,
                   help="reserved SubStartSet slots for the natural "
                        "initial state (index 0)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--model-batches", type=int, default=40)
    p.add_argument("--model-batch", type=int, default=256)
    p.add_argument("--imag-epochs", type=int, default=4)
    p.add_argument("--critic-batches", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=2e-4)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--time-budget-min", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gate-phi-mae", type=float, default=0.08)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-nstep", type=int, default=100)
    p.add_argument("--resume-from", type=str, default=None)
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
    # imagination uses the deterministic mean: freeze log_std out of BPTT
    actor.log_std.requires_grad_(False)
    model = SelfWorldModel(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, obs_dim=SELF_OBS_DIM,
        hidden=(args.model_hidden,) * 3, dt=dt,
        substep_coeff=sub_coeff).to(device)
    vrep = RepCritic().to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.actor_lr,
                             betas=(0.9, 0.95))
    opt_m = torch.optim.Adam(model.parameters(), lr=2e-4)
    opt_v = torch.optim.Adam(vrep.parameters(), lr=2e-4)

    noise_half = torch.as_tensor(
        OBS_NOISE_HALF, dtype=torch.float32, device=device) \
        * args.obs_noise_scale

    collector = BranchCollector(env_bp, num_workers=args.workers)
    atexit.register(collector.close)
    def _sigterm(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    # --- initial standup state: one in-process probe via the collector ---
    # bootstrap: use a fresh env state captured by a zero-action rollout
    ss = StartSetManager(cap=args.cap, evict=args.evict, rng=rng,
                         kappa=args.kappa, weighted_frac=args.weighted_frac)
    replay: list[dict] = []       # branches as fit windows (len=nstep)
    hold: list[dict] = []         # small holdout for gate eval
    critic_pool: list[dict] = []  # ALL branches incl. terminated + eval
    REPLAY_CAP = 4000
    CRITIC_CAP = 6000

    start_u = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device,
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        model.load_state_dict(ck["model"])
        vrep.load_state_dict(ck["vrep"])
        ss.load_state_dict(ck["startset"])
        start_u = int(ck["update"]) + 1
        rp = Path(str(args.resume_from).replace(".pt", ".replay.pkl"))
        if rp.exists():
            data = pickle.load(open(rp, "rb"))
            replay, hold = data["replay"], data["hold"]

    if len(ss) == 0:
        # bootstrap StartSet with the natural standing initial state:
        # reset a throwaway in-process env (NO injection) and read cs0.
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
          "nstep": args.nstep, "subset": args.subset,
          "git": os.popen("git rev-parse HEAD").read().strip()})

    v_max = 1.0 / (1.0 - args.gamma)  # r=phi in [0,1]
    t_start = time.time()
    for u in range(start_u, args.max_updates):
        t0 = time.time()
        rec = {"type": "update", "u": u}

        # ---- S1: sample SubStartSet -----------------------------------
        if len(ss) == 0:
            raise RuntimeError("StartSet must be bootstrapped before loop")
        n_sub = min(args.subset, len(ss))
        idx = ss.sample(n_sub, rng)
        n_anchor = min(args.anchor_slots, n_sub) if len(ss) else 0
        if n_anchor:
            idx[:n_anchor] = 0  # deployment anchor: natural init state
        seeds_cs = ss.states(idx)                       # (N,55)

        # ---- S2: branch rollout ---------------------------------------
        sd_np = actor_sd_numpy(actor)
        branches = collector.collect(
            sd_np, seeds_cs, nstep=args.nstep, sigma=args.sigma,
            base_seed=args.seed + 1000 + u * 13)

        # survivors produce first-step successors for the wavefront
        succ_cs, succ_phi, succ_rep = [], [], []
        for b in branches:
            if len(b["cs"]) >= 2 and not b["term"][0]:
                cs1 = b["cs"][1]
                succ_cs.append(cs1)
                succ_phi.append(float(phi_np(cs1)))
                rep1 = gc.represent(torch.as_tensor(
                    cs1[None], dtype=torch.float32))[0].numpy()
                succ_rep.append(rep1)
            if args.admit_intermediate:
                f = band_frames(b, rng, max_take=1)
                for csk, ph in f:
                    succ_cs.append(csk); succ_phi.append(ph)
                    succ_rep.append(gc.represent(torch.as_tensor(
                        csk[None], dtype=torch.float32))[0].numpy())
        n_admit = ss.admit_successors(succ_cs, succ_phi, succ_rep) \
            if succ_cs else 0

        # replay/holdout split for WM windows (need full nstep length);
        # critic pool keeps EVERYTHING — early-terminated branches carry
        # the only true terminal zeros that ground the value function.
        for b in branches:
            if len(b["act"]) >= args.fit_unroll:
                (hold if rng.random() < 0.05 else replay).append(b)
            critic_pool.append(b)
        if len(replay) > REPLAY_CAP:
            del replay[:len(replay) - REPLAY_CAP]
        if len(critic_pool) > CRITIC_CAP:
            del critic_pool[:len(critic_pool) - CRITIC_CAP]
        n_term = sum(1 for b in critic_pool
                     if len(b["term"]) and b["term"][-1])
        rec["pool_term_frac"] = n_term / max(len(critic_pool), 1)

        surv = float(np.mean([
            not b["term"][-1] if len(b["term"]) else True
            for b in branches])) if branches else 0.0
        mean_len = float(np.mean([len(b["act"]) for b in branches]))
        rec.update({
            "n_sub": n_sub, "admit": n_admit, "ss_size": len(ss),
            "branch_surv8": surv, "branch_len": mean_len,
            **{f"ss_{k}": v for k, v in ss.stats().items()},
        })

        # ---- S3: WM fit -----------------------------------------------
        wm_stats = {"loss": 0.0, "n": 0}
        if len(replay) >= args.model_batch:
            for _ in range(args.model_batches):
                bidx = rng.integers(0, len(replay), args.model_batch)
                fu = args.fit_unroll
                sel = [replay[i] for i in bidx]
                ofs = [int(rng.integers(0, len(b["act"]) - fu + 1))
                       for b in sel]
                cs0 = torch.as_tensor(np.stack(
                    [b["cs"][o] for b, o in zip(sel, ofs)]),
                    dtype=torch.float32, device=device)
                acts = torch.as_tensor(np.stack(
                    [b["act"][o:o + fu] for b, o in zip(sel, ofs)]),
                    dtype=torch.float32, device=device)
                cs_t = torch.as_tensor(np.stack(
                    [b["cs"][o + 1:o + fu + 1] for b, o in zip(sel, ofs)]),
                    dtype=torch.float32, device=device)
                obs_t = torch.as_tensor(np.stack(
                    [b["obs"][o + 1:o + fu + 1] for b, o in zip(sel, ofs)]),
                    dtype=torch.float32, device=device)
                rep_t = gc.represent(cs_t)
                aux_t = gc.extract_aux(cs_t)
                rep = gc.represent(cs0)
                aux = gc.extract_aux(cs0)
                L_s = L_obs = L_phi = 0.0
                for m in range(fu):
                    aux, rep = model.next_state(rep, aux, acts[:, m])
                    h = model.heads(rep, aux)
                    if args.fit_phi_weight > 0:
                        # per-sample weighted bryson loss: degraded target
                        # states are scarce, weight them up
                        pb = gc.loss_blocks(rep, aux)
                        tb = gc.loss_blocks(rep_t[:, m], aux_t[:, m])
                        num = den = 0.0
                        for name, blk in pb.items():
                            bw = model._loss_weights[name]
                            se = ((blk - tb[name]) ** 2).mean(-1)
                            num = num + bw * blk.shape[-1] * se
                            den = den + blk.shape[-1]
                        ls = num / den  # (B,)
                        w = 1.0 + args.fit_phi_weight * (
                            0.95 - analytic_phi(rep_t[:, m])).clamp(min=0)
                        ls = (ls * w).mean()
                    else:
                        ls = bryson_state_loss(
                            model, rep, aux, rep_t[:, m], aux_t[:, m])
                    L_s = L_s + ls
                    L_obs = L_obs + F.mse_loss(
                        h["obs_norm"], model.norm_o(obs_t[:, m]))
                    L_phi = L_phi + F.mse_loss(
                        h["phi"], analytic_phi(rep))
                loss = (L_s + 1.0 * L_obs + 0.5 * L_phi) / fu
                opt_m.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt_m.step()
                wm_stats["loss"] += float(loss)
                wm_stats["n"] += 1
        rec["wm_loss"] = (wm_stats["loss"] / max(wm_stats["n"], 1))

        # ---- S3b: gate — open-loop phi MAE on holdout ------------------
        gate_ok = False
        phi_mae = float("nan")
        if len(hold) >= 8:
            hn = min(64, len(hold))
            hidx = rng.choice(len(hold), hn, replace=False)
            cs0h = torch.as_tensor(np.stack(
                [hold[i]["cs"][0] for i in hidx]),
                dtype=torch.float32, device=device)
            gh = min(args.nstep, min(len(hold[i]["act"]) for i in hidx))
            actsh = torch.as_tensor(np.stack(
                [hold[i]["act"][:gh] for i in hidx]),
                dtype=torch.float32, device=device)
            real_phi = np.stack([
                phi_np(hold[i]["cs"][1:gh + 1]) for i in hidx])
            with torch.no_grad():
                rep, aux = model.rep_aux_from_cs(cs0h)
                preds = []
                for m in range(gh):
                    aux, rep = model.next_state(rep, aux, actsh[:, m])
                    preds.append(analytic_phi(rep))
                pred_phi = torch.stack(preds, 1).cpu().numpy()
            phi_mae = float(np.abs(pred_phi - real_phi).mean())
            gate_ok = phi_mae < args.gate_phi_mae
        rec["gate_phi_mae"] = phi_mae
        rec["gate_ok"] = bool(gate_ok)

        # ---- S5a: critic fit on REAL branches (GAE) --------------------
        if len(critic_pool) >= 64:
            for _ in range(args.critic_batches):
                bidx = rng.integers(0, len(critic_pool), 64)
                targets_all, reps_all, auxs_all = [], [], []
                # oversample terminal branches: half the batch if possible
                term_idx = [i for i in range(len(critic_pool))
                            if len(critic_pool[i]["term"])
                            and critic_pool[i]["term"][-1]]
                n_tb = min(32, len(term_idx))
                if n_tb:
                    tb = rng.choice(term_idx, n_tb, replace=len(term_idx) < n_tb)
                    bidx = np.concatenate([np.asarray(tb), bidx[:64 - n_tb]])
                for i in bidx:
                    b = critic_pool[int(i)]
                    L = len(b["act"])
                    csb = torch.as_tensor(
                        b["cs"][:L + 1], dtype=torch.float32,
                        device=device)  # incl. successor state s_L
                    rr = gc.represent(csb)
                    aa = gc.extract_aux(csb)
                    with torch.no_grad():
                        v_all = vrep(rr, aa).clamp(max=v_max)
                    r = torch.as_tensor(
                        phi_np(b["cs"][1:L + 1]), dtype=torch.float32,
                        device=device)
                    d = torch.as_tensor(
                        b["term"][:L].astype(np.float32),
                        device=device)
                    v = v_all[:L]
                    terminal = bool(d[-1])
                    adv = torch.zeros((), device=device)
                    tgt = torch.zeros_like(v)
                    if terminal:
                        # pure Monte-Carlo return-to-go, no bootstrap:
                        # fully grounded by the real terminal zero
                        ret = torch.zeros((), device=device)
                        for t in reversed(range(L)):
                            ret = r[t] + args.gamma * ret * (1.0 - d[t])
                            tgt[t] = ret
                    else:
                        nv = v_all[L]
                        for t in reversed(range(L)):
                            nd = 1.0 - d[t]
                            delta = r[t] + nd * args.gamma * nv - v[t]
                            adv = delta + nd * args.gamma * args.lam * adv
                            tgt[t] = adv + v[t]
                            nv = v[t]
                    targets_all.append(tgt)
                    reps_all.append(rr[:L])
                    auxs_all.append(aa[:L])
                rb = torch.cat(reps_all)
                ab = torch.cat(auxs_all)
                tgt = torch.cat(targets_all).clamp(max=v_max)
                vpred = vrep(rb, ab)
                vloss = F.mse_loss(vpred, tgt)
                opt_v.zero_grad(set_to_none=True)
                vloss.backward()
                opt_v.step()
                rec["vloss"] = float(vloss)

        # ---- S4+S6: imagine + BPTT (gated) -----------------------------
        if gate_ok:
            cs0i = torch.as_tensor(seeds_cs, dtype=torch.float32,
                                   device=device)
            o0i = torch.as_tensor(np.stack([
                b["obs"][0] for b in branches]), dtype=torch.float32,
                device=device)
            for p in list(model.parameters()) + list(vrep.parameters()):
                p.requires_grad_(False)
            actor.train()
            J_tot = 0.0
            for epoch in range(args.imag_epochs):
                rep, aux = model.rep_aux_from_cs(cs0i)
                o = o0i
                surv = torch.ones(len(cs0i), device=device)
                J = 0.0
                for t in range(args.nstep):
                    # deterministic mean + external Gaussian noise — SAME
                    # semantics as collection (framework spec): imagined
                    # (s,a) stays inside the noise-cloud coverage the WM
                    # was trained on, blocking coordinated-drift exploits
                    a = actor.deterministic_action(o)
                    a = (a + torch.randn_like(a) * args.sigma) \
                        .clamp(-1.0, 1.0)
                    aux, rep = model.next_state(rep, aux, a)
                    hd = model.heads(rep, aux)
                    phi = analytic_phi(rep)
                    d = torch.sigmoid((0.45 - phi) / 0.05)
                    J = J + (args.gamma ** t) * surv * phi
                    surv = surv * (1.0 - d)
                    feet = torch.zeros(
                        rep.shape[0], 2, device=device)
                    o = compose_obs(rep, aux, feet)
                    o = o + (torch.rand_like(o) * 2 - 1) * noise_half
                if not args.no_critic:
                    with torch.no_grad():
                        v_last = vrep(rep, aux).clamp(max=v_max)
                    J = J + (args.gamma ** args.nstep) * v_last * surv
                loss = -J.mean()
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
                gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()
                J_tot += float(J.mean())
            for p in list(model.parameters()) + list(vrep.parameters()):
                p.requires_grad_(True)
            rec["J"] = J_tot / args.imag_epochs
            rec["gnorm"] = float(gn)

        # ---- eval: long branches every eval-every ----------------------
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
            ev_cs, ev_phi = [], []
            for b in ev:
                critic_pool.append(b)
                if len(b["act"]) >= args.fit_unroll:
                    replay.append(b)
                if args.admit_intermediate:
                    # long deploy branches traverse the deep sink corridor
                    # (t=40..90) that no 32-step seed branch ever reaches —
                    # seeding it closes the last coverage gap
                    for csk, ph in band_frames(b, rng, max_take=4):
                        ev_cs.append(csk); ev_phi.append(ph)
            if ev_cs:
                ev_rep = gc.represent(torch.as_tensor(
                    np.asarray(ev_cs), dtype=torch.float32)).numpy()
                ss.admit_successors(ev_cs, ev_phi, list(ev_rep))
            if len(replay) > REPLAY_CAP:
                del replay[:len(replay) - REPLAY_CAP]
            if len(critic_pool) > CRITIC_CAP:
                del critic_pool[:len(critic_pool) - CRITIC_CAP]

        with torch.no_grad():
            cs_probe = torch.as_tensor(
                seeds_cs[:256], dtype=torch.float32, device=device)
            rp_, ax_ = gc.represent(cs_probe), gc.extract_aux(cs_probe)
            rec["v_mean"] = float(vrep(rp_, ax_).mean())
        rec["t"] = time.time() - t0
        jlog(rec)
        print(f"[u{u:04d}] ss={len(ss)} surv8={rec['branch_surv8']:.2f} "
              f"wm={rec.get('wm_loss', 0):.3f} "
              f"gate={'OK' if gate_ok else 'X'}({phi_mae:.3f}) "
              f"J={rec.get('J', float('nan')):.2f} "
              f"t={rec['t']:.1f}s", flush=True)

        if (u + 1) % 25 == 0:
            torch.save({
                "actor": actor.state_dict(), "model": model.state_dict(),
                "vrep": vrep.state_dict(), "update": u,
                "opt_a": opt_a.state_dict(), "opt_m": opt_m.state_dict(),
                "opt_v": opt_v.state_dict(),
                "startset": ss.state_dict(),
            }, run_dir / f"ckpt_u{u:05d}.pt")
            pickle.dump({"replay": replay[-500:], "hold": hold},
                        open(run_dir / f"ckpt_u{u:05d}.replay.pkl", "wb"))

        if args.time_budget_min > 0 and \
                (time.time() - t_start) / 60 > args.time_budget_min:
            print("[budget] time budget reached", flush=True)
            break

    collector.close()


if __name__ == "__main__":
    main()
