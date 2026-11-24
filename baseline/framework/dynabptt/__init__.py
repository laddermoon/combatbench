"""DynaBPTT — Dyna-style world-model + BPTT training (SuperMimic recipe).

Self-contained v1: a learned differentiable world model f(s,a)->s' is fit on
logged core-state transitions; the policy is improved by backpropagating
imagined returns through the frozen model (Dyna-style seeding from a replay
buffer), with a critic bootstrapping beyond the imagination horizon.

v1 scope is SELF-ONLY: the world model ignores the opponent entirely
(f(ŝ_self, a_self) -> ŝ'_self). Valid for tasks where the two robots never
make contact (balance / standup / in-air skills).
"""
