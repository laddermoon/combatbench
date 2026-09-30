"""Audit probe (regularization Phase 1) — NOT a fix.

Question under audit: ``Humanoid21Simulator._data_cache`` is cleared on every
``physical_step`` / ``reset`` / ``set_action``, but ``_cached_contacts_vec``
(the SoA contacts array that ``_get_feet_forces`` reads first) is a SEPARATE
attribute only cleared in ``reset`` — never in ``physical_step``. If a caller
populates it (via ``get_derived_state(['contacts'])``) and then steps physics,
the next ``get_observation()`` / ``_get_feet_forces`` may consume the PREVIOUS
step's contacts — i.e. feet_forces dims 44-45 of the 96-dim observation lag
one action step behind.

Probe strategy: populate the contacts cache, apply a strong external force to
change the contact pattern, step physics once, then compare
``_get_feet_forces`` (possibly stale path) against forces recomputed from a
fresh ``_extract_contacts`` at the same instant.
"""
from __future__ import annotations

import numpy as np

from envs.humanoid21 import Humanoid21Simulator


def _forces_from_contacts(sim: Humanoid21Simulator, robot_id: str, contacts) -> np.ndarray:
    """Recompute feet forces from a *fresh* contacts SoA — mirrors
    ``_get_feet_forces`` but never touches ``_cached_contacts_vec``."""
    cache = sim._robot(robot_id)
    kp = cache['keypoint_body_ids']
    ground_geom_id = sim._ground_geom_id
    ncon = contacts['ncon']
    if ncon == 0:
        return np.zeros(2, dtype=np.float32)
    geom1, geom2 = contacts['geom1'], contacts['geom2']
    body1, body2 = contacts['body1'], contacts['body2']
    force_mag = contacts['force_mag']
    g1_ground = geom1 == ground_geom_id
    g2_ground = geom2 == ground_geom_id
    ground_mask = g1_ground | g2_ground
    if not np.any(ground_mask):
        return np.zeros(2, dtype=np.float32)
    other_body = np.where(g1_ground, body2, body1)
    forces = force_mag[ground_mask]
    bodies = other_body[ground_mask]
    return np.array(
        [float(np.sum(forces[bodies == kp['foot_right']])),
         float(np.sum(forces[bodies == kp['foot_left']]))],
        dtype=np.float32,
    ) / cache['body_weight']


def test_cached_contacts_vec_survives_physical_step():
    """Invariant check: a physics step must invalidate the contacts cache."""
    sim = Humanoid21Simulator()
    sim.reset()
    sim.get_derived_state(['contacts'])          # populate _cached_contacts_vec
    cached_before = sim._cached_contacts_vec
    assert cached_before is not None
    sim.physical_step()
    # After stepping, the cached contacts describe a past state. The cache
    # is *not* invalidated — this documents the current (likely buggy)
    # behaviour. If this assertion ever fails, the bug is already fixed.
    assert sim._cached_contacts_vec is cached_before


def test_feet_forces_can_go_stale_after_push():
    """Behavioural check: change contacts via external push, then compare the
    cached-path feet forces vs a fresh extraction at the same instant."""
    sim = Humanoid21Simulator()
    sim.reset()

    # settle a few steps and populate the contacts cache
    for _ in range(50):
        sim.physical_step()
    sim.get_derived_state(['contacts'])          # cv now = contacts@t0

    # Push robot_a's torso hard upward → feet leave ground → contacts change.
    sim.apply_external_force('torso', np.array([0.0, 0.0, 4000.0]), robot_id='robot_a')
    sim.physical_step()                          # contacts@t1 != contacts@t0

    stale = sim._get_feet_forces('robot_a')                     # uses stale cv
    fresh = _forces_from_contacts(sim, 'robot_a', sim._extract_contacts())

    # With a strong upward push the ground reaction must drop. If the two
    # paths disagree, the observation's feet_forces dims are stale.
    assert not np.allclose(stale, fresh, rtol=0.05, atol=0.02), (
        f"expected stale-vs-fresh mismatch to demonstrate the bug; "
        f"stale={stale}, fresh={fresh}"
    )
