"""Held-out validation + admission gate for model rollouts.

Three complementary checks on never-fitted recent episodes:

1. Open-loop: roll the model along recorded actions h = 1..H_MAX,
   compare predicted states / obs / phi / stage to ground truth.
2. Closed-loop: feed predicted obs through pi_old; measure action
   divergence vs the actions pi_old takes on the real obs sequence.
3. Paired real-vs-model branch: restore the env to a held-out seed
   state via set_core_state, apply the model branch's own action
   sequence in the real sim, and compare short-horizon bootstrap
   advantages under the same V_old.

Admission is per start-stage (stage of s_t): a stage must pass all
thresholds at horizon h to admit h-step branches seeded there.
``consecutive_pass`` requires two clean validations in a row before
the gate opens; any failure closes it immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import SelfWorldModel, grouped_state_metrics
from .standup import (REWARD_COEF, REWARDER_NAMES,
                      unflatten_core_state)
from .transitions import AlignedTraj

H_MAX = 3

# First-version thresholds (see plan §4.3) — applied per start-stage and
# per horizon step h <= H_MAX.  Stage gates: a stage is evaluated only
# when it owns >= MIN_STAGE_WINDOWS of the sampled windows.
GATE = {
    "root_z_mae": 0.02, "root_z_p95": 0.05,
    "rot_deg_mean": 5.0, "rot_deg_p95": 10.0,
    "nrmse_max": 0.20,            # max over vel/joint groups
    "obs_nrmse": 0.15, "obs_feet_nrmse": 0.25,
    "phi_mae": 0.03, "phi_p95": 0.08,
    "stage_acc": 0.90, "stage_trans_acc": 0.80,
    "phi_oob_frac": 0.01,
    "act_dev_rms": 0.05,
    "adv_rel_rmse": 0.50,
    "adv_sign_frac": 0.75,
}
# Second-tier admission for the PPO-assist path (first-action advantage
# from real seeds at h<=3).  The strict tier certifies state accuracy for
# long/BPTT-style rollouts; the assist tier gates the quantity actually
# consumed by the aux update — reward/stage heads, action fidelity, and
# the paired advantage itself — at looser state-metric bounds.
ASSIST_GATE = {
    "root_z_mae": 0.04, "rot_deg_mean": 8.0,
    "phi_mae": 0.05, "act_dev_rms": 0.15,
    "adv_rel_rmse": 0.50, "adv_sign_frac": 0.75, "adv_corr": 0.60,
}
MIN_STAGE_WINDOWS = 256
MIN_STAGE_TRANSITIONS = 32
ADV_SIGNAL_FLOOR_FRAC = 0.25     # "significant" |A| > 0.25 * std_real


@dataclass
class GateReport:
    """Outcome of one validation round."""
    horizon: int = 0
    admitted_stages: Tuple[int, ...] = ()
    consecutive_pass: int = 0
    assist_horizon: int = 0
    assist_stages: Tuple[int, ...] = ()
    assist_passed: bool = False
    # stage -> horizon -> metrics
    metrics: Dict[int, Dict[int, Dict[str, float]]] = field(
        default_factory=dict)
    paired: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    all_passed: bool = False

    def flat(self) -> Dict[str, float]:
        """Flatten for metrics.jsonl."""
        out = {
            "gate_passed": float(self.all_passed),
            "gate_horizon": float(self.horizon),
            "gate_consec": float(self.consecutive_pass),
            "gate_n_stages": float(len(self.admitted_stages)),
            "assist_passed": float(self.assist_passed),
            "assist_horizon": float(self.assist_horizon),
            "assist_n_stages": float(len(self.assist_stages)),
        }
        for stg, per_h in self.metrics.items():
            for h, m in per_h.items():
                for k, v in m.items():
                    out[f"val_s{stg}_h{h}_{k}"] = v
        for k, v in self.paired.items():
            out[f"val_pair_{k}"] = v
        return out


class GateState:
    """Persistent admission controller across updates."""

    def __init__(self) -> None:
        self.consecutive_pass = 0
        self.horizon = 0
        self.admitted_stages: Tuple[int, ...] = ()
        self.assist_consec = 0
        self.assist_horizon = 0
        self.assist_stages: Tuple[int, ...] = ()

    @property
    def admitted(self) -> bool:
        return self.horizon > 0 and bool(self.admitted_stages)

    @property
    def assist_admitted(self) -> bool:
        return self.assist_horizon > 0 and bool(self.assist_stages)

    def update(self, report: "GateReport") -> None:
        if report.all_passed:
            self.consecutive_pass += 1
        else:
            self.consecutive_pass = 0
            self.horizon = 0
            self.admitted_stages = ()
        if self.consecutive_pass >= 2:
            self.horizon = report.horizon
            self.admitted_stages = report.admitted_stages
        if report.assist_passed:
            self.assist_consec += 1
        else:
            self.assist_consec = 0
            self.assist_horizon = 0
            self.assist_stages = ()
        if self.assist_consec >= 2:
            self.assist_horizon = report.assist_horizon
            self.assist_stages = report.assist_stages

    def state_dict(self) -> Dict:
        return {"consecutive_pass": self.consecutive_pass,
                "horizon": self.horizon,
                "admitted_stages": list(self.admitted_stages),
                "assist_consec": self.assist_consec,
                "assist_horizon": self.assist_horizon,
                "assist_stages": list(self.assist_stages)}

    def load_state_dict(self, d: Dict) -> None:
        self.consecutive_pass = int(d["consecutive_pass"])
        self.horizon = int(d["horizon"])
        self.admitted_stages = tuple(d["admitted_stages"])
        self.assist_consec = int(d.get("assist_consec", 0))
        self.assist_horizon = int(d.get("assist_horizon", 0))
        self.assist_stages = tuple(d.get("assist_stages", ()))


# --- internal helpers ----------------------------------------------------

def _rot_deg(qp: np.ndarray, qt: np.ndarray) -> np.ndarray:
    qp = qp / np.linalg.norm(qp, axis=-1, keepdims=True)
    qt = qt / np.linalg.norm(qt, axis=-1, keepdims=True)
    cosang = np.clip(np.abs((qp * qt).sum(-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosang))


def _nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).mean())
                 / max(float(true.std()), 1e-6))


@torch.no_grad()
def _model_rollout(model: SelfWorldModel, s0: torch.Tensor,
                   acts: torch.Tensor, horizon: int):
    """Open-loop rollout along recorded actions.

    acts: (B, H>=horizon, act_dim) — uses the FIRST `horizon` actions.
    s0 is a 55-dim cs; the model rolls in (rep, aux) space and each
    step's cs is reconstructed for metric compatibility.
    Returns stacked preds: cs (B,H,55), obs (B,H,57), phi (B,H),
    stage_logits (B,H,4).
    """
    rep = model.rep_aux_from_cs(s0)[0]
    aux = model.rep_aux_from_cs(s0)[1]
    cs, obs, phis, logits = [], [], [], []
    for m in range(horizon):
        aux, rep = model.next_state(rep, aux, acts[:, m])
        h = model.heads(rep, aux)
        cs.append(model.cs_from_rep_aux(rep, aux))
        obs.append(h["obs"])
        phis.append(h["phi"].clamp(0.0, 1.0))
        logits.append(h["stage_logits"])
    return (torch.stack(cs, 1), torch.stack(obs, 1),
            torch.stack(phis, 1), torch.stack(logits, 1))


@torch.no_grad()
def _closed_loop_rollout(model: SelfWorldModel, actor,
                         s0: torch.Tensor, o0: torch.Tensor,
                         horizon: int, device):
    """Roll the model feeding predicted obs through pi_old
    (deterministic actions — measures the policy's sensitivity to
    model-obs error, not noise)."""
    rep, aux = model.rep_aux_from_cs(s0)
    o = o0
    acts, cs = [], []
    for _ in range(horizon):
        a = actor.deterministic_action(o)
        acts.append(a)
        aux, rep = model.next_state(rep, aux, a)
        cs.append(model.cs_from_rep_aux(rep, aux))
        o = model.heads(rep, aux)["obs"]
    return torch.stack(acts, 1), torch.stack(cs, 1)


def _stage_of(traj: AlignedTraj, t: int) -> int:
    return int(traj.stage[t - 1])


def validate_model(model: SelfWorldModel, actor, critic, replay,
                   rng: np.random.Generator, device: torch.device,
                   n_windows: int = 4096,
                   env_bp=None, paired_seeds: int = 64,
                   paired_h: int = 3, vrep=None) -> GateReport:
    """Run all validation checks; returns a GateReport.

    ``env_bp``: blueprint used to build the branch-restoration env for
    the paired real-vs-model advantage check.  ``paired_seeds <= 0``
    skips that check (metrics still computed; paired fields stay NaN).
    """
    report = GateReport()
    windows = replay.sample_holdout_windows(n_windows, H_MAX, rng)
    if len(windows) < 64:
        report.notes.append(
            f"insufficient holdout windows ({len(windows)}) — gate closed")
        report.all_passed = False
        return report

    # --- batch the windows ---
    by_stage: Dict[int, List[Tuple[AlignedTraj, int]]] = {}
    for tr, t in windows:
        by_stage.setdefault(_stage_of(tr, t), []).append((tr, t))

    for stg, items in sorted(by_stage.items()):
        if len(items) < MIN_STAGE_WINDOWS:
            report.notes.append(
                f"stage {stg}: only {len(items)} windows "
                f"(< {MIN_STAGE_WINDOWS}) — insufficient evidence")
            continue
        s0 = torch.as_tensor(
            np.stack([tr.cs[t - 1] for tr, t in items]), device=device)
        o0 = torch.as_tensor(
            np.stack([tr.obs[t] for tr, t in items]), device=device)
        acts_all = np.stack([tr.act[t:t + H_MAX] for tr, t in items])
        true_cs = np.stack([tr.cs[t:t + H_MAX] for tr, t in items])
        true_phi = np.stack([tr.phi[t:t + H_MAX] for tr, t in items])
        true_stage = np.stack(
            [tr.stage[t:t + H_MAX] for tr, t in items]).astype(int)
        true_obs = np.stack(
            [[tr.next_obs(t + m) for m in range(H_MAX)]
             for tr, t in items])
        # reference actions: the DETERMINISTIC policy applied to the REAL
        # obs sequence — comparing against recorded stochastic actions
        # would floor this metric at the exploration sigma (~0.3) and
        # make the act_dev gate unreachable by construction.
        # cl_acts[:,i] = det(o_hat_{t+i}); the matching real reference is
        # det(o_{t+i}): i=0 -> obs[t] (o0), i>=1 -> true_obs[:, i-1].
        ref_obs = np.concatenate(
            [o0.cpu().numpy()[:, None, :], true_obs[:, :H_MAX - 1]],
            axis=1)
        ref_obs_t = torch.as_tensor(
            ref_obs.reshape(-1, ref_obs.shape[-1]), device=device)
        with torch.no_grad():
            real_pi_acts = actor.deterministic_action(
                ref_obs_t).cpu().numpy().reshape(
                    ref_obs.shape[:-1] + (acts_all.shape[-1],))

        s0_t = s0
        o0_t = o0
        acts_t = torch.as_tensor(acts_all, device=device)
        cs_pred, obs_pred, phi_pred, logits_pred = _model_rollout(
            model, s0_t, acts_t, H_MAX)
        cl_acts, cl_cs = _closed_loop_rollout(
            model, actor, s0_t, o0_t, H_MAX, device)

        cs_pred_np = cs_pred.cpu().numpy()
        obs_pred_np = obs_pred.cpu().numpy()
        phi_pred_np = phi_pred.cpu().numpy()
        stage_pred = logits_pred.argmax(-1).cpu().numpy() + 1
        cl_acts_np = cl_acts.cpu().numpy()

        stage_metrics: Dict[int, Dict[str, float]] = {}
        for h in range(1, H_MAX + 1):
            i = h - 1
            m: Dict[str, float] = {}
            g = grouped_state_metrics(
                cs_pred[:, :h].reshape(-1, 55),
                torch.as_tensor(true_cs[:, :h]).reshape(-1, 55))
            m.update(g)
            m["rot_deg_p95"] = float(np.percentile(
                _rot_deg(cs_pred_np[:, :h, 3:7].reshape(-1, 4),
                         true_cs[:, :h, 3:7].reshape(-1, 4)), 95))
            m["obs_nrmse"] = _nrmse(
                obs_pred_np[:, :h], true_obs[:, :h])
            m["obs_feet_nrmse"] = _nrmse(
                obs_pred_np[:, :h, 54:56], true_obs[:, :h, 54:56])
            perr = np.abs(phi_pred_np[:, :h] - true_phi[:, :h])
            m["phi_mae"] = float(perr.mean())
            m["phi_p95"] = float(np.percentile(perr, 95))
            m["phi_oob_frac"] = float(
                ((phi_pred_np[:, :h] < 0.0)
                 | (phi_pred_np[:, :h] > 1.0)).mean())
            m["stage_acc"] = float(
                (stage_pred[:, :h] == true_stage[:, :h]).mean())
            # transition windows: frames where the true stage changes
            # within the horizon prefix — measure accuracy there.
            tr_mask = np.zeros_like(true_stage[:, :h], dtype=bool)
            tr_mask[:, 0] = true_stage[:, 0] != np.asarray(
                [_stage_of(tr, t) for tr, t in items])
            if h > 1:
                tr_mask[:, 1:] = (
                    true_stage[:, 1:h] != true_stage[:, 0:h - 1])
            n_tr = int(tr_mask.sum())
            m["stage_trans_n"] = float(n_tr)
            if n_tr >= MIN_STAGE_TRANSITIONS:
                m["stage_trans_acc"] = float(
                    (stage_pred[:, :h][tr_mask]
                     == true_stage[:, :h][tr_mask]).mean())
            else:
                m["stage_trans_acc"] = float("nan")
            # closed-loop action divergence at step h: compare model-
            # driven det action to the RECORDED action at that step.
            dev = cl_acts_np[:, i] - real_pi_acts[:, i]
            m["act_dev_rms"] = float(np.sqrt((dev ** 2).mean()))
            stage_metrics[h] = m
        report.metrics[stg] = stage_metrics

    # --- paired real-vs-model branch advantage -------------------------
    if env_bp is not None and paired_seeds > 0:
        report.paired = _paired_branch_adv(
            model, actor, critic, replay, env_bp, rng, device,
            paired_seeds, paired_h, report, vrep=vrep)

    # --- admission decision ---------------------------------------------
    horizon, stages, notes = _admission_decision(report)
    report.horizon = horizon
    report.admitted_stages = stages
    report.notes.extend(notes)
    report.all_passed = horizon > 0 and bool(stages)
    ah, ast, anotes = _assist_decision(report)
    report.assist_horizon = ah
    report.assist_stages = ast
    report.assist_passed = ah > 0 and bool(ast)
    report.notes.extend(anotes)
    return report


def _admission_decision(report: GateReport):
    """Apply thresholds per stage per horizon; longest passing h wins."""
    notes: List[str] = []
    stage_h: Dict[int, int] = {}
    for stg, per_h in report.metrics.items():
        best = 0
        for h in range(1, H_MAX + 1):
            m = per_h.get(h)
            if m is None:
                break
            fails = []
            if m["root_z_mae"] > GATE["root_z_mae"]:
                fails.append("root_z")
            if m["rot_deg_mean"] > GATE["rot_deg_mean"]:
                fails.append("rot")
            if max(m["nrmse_linvel"], m["nrmse_angvel"],
                   m["nrmse_jpos"], m["nrmse_jvel"]) > GATE["nrmse_max"]:
                fails.append("nrmse")
            if m["obs_nrmse"] > GATE["obs_nrmse"]:
                fails.append("obs")
            if m["obs_feet_nrmse"] > GATE["obs_feet_nrmse"]:
                fails.append("feet")
            if m["phi_mae"] > GATE["phi_mae"]:
                fails.append("phi")
            if m["phi_oob_frac"] > GATE["phi_oob_frac"]:
                fails.append("oob")
            if m["stage_acc"] < GATE["stage_acc"]:
                fails.append("stage")
            if not np.isnan(m["stage_trans_acc"]) and (
                    m["stage_trans_acc"] < GATE["stage_trans_acc"]):
                fails.append("stage_trans")
            if m["act_dev_rms"] > GATE["act_dev_rms"]:
                fails.append("act_dev")
            if fails:
                notes.append(
                    f"stage {stg} h={h} failed: {','.join(fails)}")
                break
            best = h
        if best > 0:
            stage_h[stg] = best
    if not stage_h:
        return 0, (), notes
    horizon = min(stage_h.values())
    stages = tuple(s for s, hh in stage_h.items() if hh >= horizon)
    if not stages:
        return 0, (), notes
    # paired-branch check applies globally
    pr = report.paired
    if pr and not np.isnan(pr.get("adv_rel_rmse", float("nan"))):
        if pr["adv_rel_rmse"] > GATE["adv_rel_rmse"]:
            notes.append(
                f"paired adv_rel_rmse {pr['adv_rel_rmse']:.3f} "
                f"> {GATE['adv_rel_rmse']} — gate closed")
            return 0, (), notes
        if pr["adv_sign_frac"] < GATE["adv_sign_frac"]:
            notes.append(
                f"paired adv_sign_frac {pr['adv_sign_frac']:.3f} "
                f"< {GATE['adv_sign_frac']} — gate closed")
            return 0, (), notes
    return horizon, stages, notes


def _assist_decision(report: GateReport):
    """Assist-tier admission: gate the quantities the aux PPO update
    actually consumes (reward/stage heads, action fidelity, paired
    advantage) rather than raw state NRMSE."""
    notes: List[str] = []
    stage_h: Dict[int, int] = {}
    for stg, per_h in report.metrics.items():
        best = 0
        for h in range(1, H_MAX + 1):
            m = per_h.get(h)
            if m is None:
                break
            fails = []
            if m["root_z_mae"] > ASSIST_GATE["root_z_mae"]:
                fails.append("root_z")
            if m["rot_deg_mean"] > ASSIST_GATE["rot_deg_mean"]:
                fails.append("rot")
            if m["phi_mae"] > ASSIST_GATE["phi_mae"]:
                fails.append("phi")
            if m["act_dev_rms"] > ASSIST_GATE["act_dev_rms"]:
                fails.append("act_dev")
            if fails:
                notes.append(
                    f"assist stage {stg} h={h} failed: {','.join(fails)}")
                break
            best = h
        if best > 0:
            stage_h[stg] = best
    if not stage_h:
        return 0, (), notes
    horizon = min(stage_h.values())
    stages = tuple(s for s, hh in stage_h.items() if hh >= horizon)
    pr = report.paired
    if not pr or np.isnan(pr.get("adv_rel_rmse", float("nan"))):
        notes.append("assist: no paired evidence — closed")
        return 0, (), notes
    if pr["adv_rel_rmse"] > ASSIST_GATE["adv_rel_rmse"]:
        notes.append(f"assist adv_rel_rmse {pr['adv_rel_rmse']:.3f} — closed")
        return 0, (), notes
    if pr["adv_sign_frac"] < ASSIST_GATE["adv_sign_frac"]:
        notes.append(f"assist adv_sign_frac {pr['adv_sign_frac']:.3f} — closed")
        return 0, (), notes
    if pr.get("adv_corr", 0.0) < ASSIST_GATE["adv_corr"]:
        notes.append(f"assist adv_corr {pr.get('adv_corr', 0.0):.3f} — closed")
        return 0, (), notes
    return horizon, stages, notes


def _paired_branch_adv(model, actor, critic, replay, env_bp,
                       rng, device, n_seeds: int, h: int,
                       report: GateReport, gamma: float = 0.99,
                       lam: float = 0.95, vrep=None) -> Dict[str, float]:
    """Model-branch vs real-branch short-horizon advantage.

    For each held-out seed:
      * model branch: actions sampled from pi_old on predicted obs.
      * real branch: env restored to the seed's focal-agent state via
        ``set_core_state`` (opponent is physically decoupled, so its
        state/action cannot affect the focal robot — it is stepped with
        zero actions), stepped with the SAME action sequence; rewards
        and observations read from the real env after each step.

    Both advantages use the same V_old (frozen critic) and GAE params,
    so the comparison isolates model-induced bias in reward + state +
    obs + value propagation.
    """
    out: Dict[str, float] = {
        "adv_rel_rmse": float("nan"), "adv_sign_frac": float("nan"),
        "adv_corr": float("nan"), "n_seeds": 0.0,
        "adv_std_real": float("nan"), "n_significant": 0.0,
    }
    seeds = replay.sample_seeds(n_seeds, rng, pool="hold")
    if len(seeds) < 16:
        report.notes.append(
            f"paired check: only {len(seeds)} holdout seeds — skipped")
        return out

    # --- model branches (batched, no_grad) ---
    s0 = torch.as_tensor(
        np.stack([tr.cs[t - 1] for tr, t in seeds]), device=device)
    o0 = torch.as_tensor(
        np.stack([tr.obs[t] for tr, t in seeds]), device=device)
    B = s0.shape[0]
    with torch.no_grad():
        rep, aux = model.rep_aux_from_cs(s0)
        o = o0
        acts: List[torch.Tensor] = []
        phis: List[torch.Tensor] = []
        obs_seq: List[torch.Tensor] = []
        rep_seq: List[Tuple[torch.Tensor, torch.Tensor]] = [(rep, aux)]
        for _ in range(h):
            a, _ = actor.sample_action(o)
            acts.append(a)
            aux, rep = model.next_state(rep, aux, a)
            hd = model.heads(rep, aux)
            o = hd["obs"]
            obs_seq.append(o)
            rep_seq.append((rep, aux))
            phis.append(hd["phi"].clamp(0.0, 1.0))
        act_seq = torch.stack(acts, 1).cpu().numpy()      # (B,h,21)
        phi_model = torch.stack(phis, 1)                  # (B,h)
        if vrep is not None:
            v_all = torch.stack(
                [vrep(r, a) for r, a in rep_seq], dim=1)  # (B,h+1)
        else:
            obs_seq_t = torch.stack(obs_seq, 1)           # (B,h,57)
            v_all = critic(torch.cat(
                [o0.unsqueeze(1), obs_seq_t], dim=1
            ).reshape(-1, o0.shape[-1])).reshape(B, h + 1)
        adv_model = _gae(REWARD_COEF * phi_model, v_all[:, :-1],
                         v_all[:, -1], gamma, lam).cpu().numpy()

    # --- real branches (in-process env restoration) ---
    from baseline.framework.dynabptt.corestate import flatten_core_state
    runtime = env_bp.build()
    phi_real = np.zeros((B, h), dtype=np.float32)
    obs_real = np.zeros((B, h + 1, o0.shape[-1]), dtype=np.float32)
    cs_real = (np.zeros((B, h + 1, s0.shape[-1]), dtype=np.float32)
               if vrep is not None else None)
    for i, (tr, t) in enumerate(seeds):
        aid = tr.agent_id
        # plugin RNGs must be seeded explicitly — EpisodeRunner does this
        # for rollout jobs; direct EnvRuntime use is our responsibility.
        bseed = int(rng.integers(2**31))
        for pl in runtime.plugins:
            pl.set_episode_seed(bseed)
        runtime.reset(seed=bseed, base_seed=bseed)
        runtime.simulator.set_core_state(
            {aid: unflatten_core_state(tr.cs[t - 1])})
        o_restored = runtime.get_observation()
        obs_real[i, 0] = np.asarray(
            o_restored[0 if aid == "robot_a" else 1], dtype=np.float32)
        if cs_real is not None:
            cs_real[i, 0] = tr.cs[t - 1]
        for m in range(h):
            a_self = act_seq[i, m]
            zero = np.zeros(21, dtype=np.float32)
            runtime.step(
                a_self if aid == "robot_a" else zero,
                zero if aid == "robot_a" else a_self,
            )
            obs_now = runtime.get_observation()
            obs_real[i, m + 1] = np.asarray(
                obs_now[0 if aid == "robot_a" else 1], dtype=np.float32)
            phi_real[i, m] = float(runtime.get_observer_output(
                REWARDER_NAMES[aid])["potential"])
            if cs_real is not None:
                cs_real[i, m + 1] = flatten_core_state(
                    runtime.simulator.get_core_state(), aid)

    with torch.no_grad():
        if vrep is not None:
            cs_t = torch.as_tensor(cs_real, device=device)
            rep_r, aux_r = model.rep_aux_from_cs(
                cs_t.reshape(-1, cs_t.shape[-1]))
            v_real = vrep(rep_r, aux_r).reshape(B, h + 1)
        else:
            v_real = critic(torch.as_tensor(
                obs_real, device=device).reshape(-1, obs_real.shape[-1])
            ).reshape(B, h + 1)
        adv_real = _gae(REWARD_COEF * torch.as_tensor(
            phi_real, device=device), v_real[:, :-1],
            v_real[:, -1], gamma, lam).cpu().numpy()

    std_r = float(adv_real.std())
    out["n_seeds"] = float(B)
    out["adv_std_real"] = std_r
    out["adv_mean_real"] = float(adv_real.mean())
    out["adv_mean_model"] = float(adv_model.mean())
    out["adv_bias"] = float((adv_model - adv_real).mean())
    if std_r < 1e-4:
        report.notes.append(
            f"paired check: real adv std {std_r:.2e} < 1e-4 — "
            "insufficient signal, gate closed")
        return out
    out["adv_rel_rmse"] = float(
        np.sqrt(((adv_model - adv_real) ** 2).mean()) / std_r)
    out["adv_corr"] = float(np.corrcoef(adv_model, adv_real)[0, 1])
    sig = np.abs(adv_real) > ADV_SIGNAL_FLOOR_FRAC * std_r
    out["n_significant"] = float(sig.sum())
    if sig.sum() >= 16:
        out["adv_sign_frac"] = float(
            (np.sign(adv_model[sig]) == np.sign(adv_real[sig])).mean())
    return out


def _gae(r: torch.Tensor, v: torch.Tensor, v_last: torch.Tensor,
         gamma: float, lam: float) -> torch.Tensor:
    """First-step GAE advantage on a short branch.

    r (B,h): rewards r_t for transitions t..t+h-1.
    v (B,h): V at the states where actions were taken (o_t..o_{t+h-1}).
    v_last (B,): bootstrap V at the post-branch observation.
    Returns adv_0 (B,) — the GAE advantage of the first action.
    """
    B, h = r.shape
    last = torch.zeros(B, device=r.device)
    adv0 = torch.zeros(B, device=r.device)
    for t in reversed(range(h)):
        v_next = v_last if t == h - 1 else v[:, t + 1]
        last = r[:, t] + gamma * v_next - v[:, t] + gamma * lam * last
        if t == 0:
            adv0 = last
    return adv0
