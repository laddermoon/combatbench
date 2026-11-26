"""StartSetManager — strict wavefront expansion seed set.

Semantics (DYNABPTT_FRAMEWORK.md):
  * init with the standup state (phi=1)
  * each update: only the FIRST-step successor s_1 of each SubStartSet
    seed's rollout enters the set; the remaining 7 steps feed WM fitting
    and paired validation only
  * sampling: phi^6 weighted (85%) + uniform floor (15%)
  * dedup on admission by rep-space nearest-neighbor distance
  * cap enforced by phi-ascending eviction (dead zone first out)
"""
import numpy as np
import torch

KAPPA = 6.0
WEIGHTED_FRAC = 0.85
DEDUP_EPS = 0.10
CAP = 4096


class StartSetManager:
    def __init__(self, cap: int = CAP, dedup_eps: float = DEDUP_EPS):
        self.cs = []          # list of (D,) float32 arrays — core states
        self.phi = []         # parallel list of float
        self.reps = None      # cached (N, R) float32 reps for dedup/dist
        self.cap = int(cap)
        self.dedup_eps = float(dedup_eps)
        self.n_admitted = 0
        self.n_dedup_skip = 0
        self.n_evicted = 0

    def __len__(self):
        return len(self.cs)

    def add_initial(self, cs: np.ndarray, phi: float, rep: np.ndarray):
        self._admit(cs, phi, rep, skip_dedup=True)

    def _admit(self, cs, phi, rep, skip_dedup=False):
        if not skip_dedup and self.reps is not None and len(self.reps):
            d = np.linalg.norm(self.reps - rep[None, :], axis=1).min()
            if d < self.dedup_eps:
                self.n_dedup_skip += 1
                return False
        self.cs.append(np.asarray(cs, dtype=np.float32).copy())
        self.phi.append(float(phi))
        r = np.asarray(rep, dtype=np.float32).reshape(1, -1)
        self.reps = r if self.reps is None else np.concatenate(
            [self.reps, r], axis=0)
        self.n_admitted += 1
        return True

    def admit_successors(self, cs_list, phi_list, rep_list):
        """Admit first-step successors from this update's rollouts."""
        n = 0
        for cs, p, rp in zip(cs_list, phi_list, rep_list):
            n += int(self._admit(cs, p, rp))
        if len(self.cs) > self.cap:
            order = np.argsort(np.asarray(self.phi))  # lowest phi first
            kill = order[:len(self.cs) - self.cap]
            keep = np.ones(len(self.cs), dtype=bool)
            keep[kill] = False
            idx = np.nonzero(keep)[0]
            self.cs = [self.cs[i] for i in idx]
            self.phi = [self.phi[i] for i in idx]
            self.reps = self.reps[idx]
            self.n_evicted += len(kill)
        return n

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """phi^6-weighted (85%) + uniform floor (15%); returns indices."""
        N = len(self.cs)
        if N == 0:
            raise RuntimeError("StartSet empty")
        n = min(int(n), N)
        n_u = min(N, int(round(n * (1.0 - WEIGHTED_FRAC))))
        n_w = n - n_u
        out = []
        if n_w > 0:
            w = np.power(np.maximum(np.asarray(self.phi), 1e-3), KAPPA)
            w = w / w.sum()
            out.append(rng.choice(N, size=n_w, replace=N < n_w, p=w))
        if n_u > 0:
            out.append(rng.choice(N, size=n_u, replace=N < n_u))
        return np.concatenate(out)

    def states(self, idx) -> np.ndarray:
        return np.stack([self.cs[i] for i in idx])

    def stats(self) -> dict:
        p = np.asarray(self.phi) if self.phi else np.zeros(1)
        return {
            "size": len(self.cs), "phi_mean": float(p.mean()),
            "phi_p10": float(np.percentile(p, 10)),
            "phi_p50": float(np.percentile(p, 50)),
            "phi_p90": float(np.percentile(p, 90)),
            "admitted": self.n_admitted, "dedup_skip": self.n_dedup_skip,
            "evicted": self.n_evicted,
        }

    def state_dict(self):
        return {"cs": self.cs, "phi": self.phi, "reps": self.reps,
                "n_admitted": self.n_admitted,
                "n_dedup_skip": self.n_dedup_skip,
                "n_evicted": self.n_evicted}

    def load_state_dict(self, sd):
        self.cs, self.phi, self.reps = sd["cs"], sd["phi"], sd["reps"]
        self.n_admitted = sd["n_admitted"]
        self.n_dedup_skip = sd["n_dedup_skip"]
        self.n_evicted = sd["n_evicted"]
