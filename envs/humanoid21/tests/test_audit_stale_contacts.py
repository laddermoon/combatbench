"""Audit regression — stale contacts cache (was: audit probe, now fixed).

Historical defect: ``Humanoid21Simulator._data_cache`` was cleared on every
``physical_step`` / ``reset``, but ``_cached_contacts_vec`` (the SoA contacts
array that ``_get_feet_forces`` reads first) was a SEPARATE attribute only
cleared in ``reset`` — never in ``physical_step``. A caller populating it
(via ``get_derived_state(['contacts'])``) then stepping physics left the
next ``get_observation()`` / ``_get_feet_forces`` consuming the PREVIOUS
step's contacts — feet_forces dims 44-45 of the 96-dim observation lagging
behind.

Fix: contacts now live solely under ``_data_cache['_derived_contacts']``
and share its epoch (cleared by ``physical_step`` / ``reset``). These tests
assert the invalidation invariant and that the feet-forces path can never
read cross-step contacts again.
"""
from __future__ import annotations

import numpy as np

from envs.humanoid21 import Humanoid21Simulator


def _forces_from_contacts(sim: Humanoid21Simulator, robot_id: str, contacts) -> np.ndarray:
    """Recompute feet forces from a *fresh* contacts SoA — mirrors
    ``_get_feet_forces`` but never touches the data cache."""
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


def test_contacts_cache_invalidated_by_physical_step():
    """Invariant: a physics step must invalidate the contacts cache."""
    sim = Humanoid21Simulator()
    sim.reset()
    sim.get_derived_state(['contacts'])  # populate _data_cache['_derived_contacts']
    assert sim._data_cache.get('_derived_contacts') is not None
    sim.physical_step()
    assert sim._data_cache.get('_derived_contacts') is None


def test_feet_forces_fresh_after_push():
    """Behavioural: after a push changes contacts, the feet-forces path must
    agree with a fresh extraction at the same instant (no stale read)."""
    sim = Humanoid21Simulator()
    sim.reset()

    # settle a few steps and populate the contacts cache
    for _ in range(50):
        sim.physical_step()
    sim.get_derived_state(['contacts'])  # cv now = contacts@t0

    # Push robot_a's torso hard upward → feet leave ground → contacts change.
    sim.apply_external_force('torso', np.array([0.0, 0.0, 4000.0]), robot_id='robot_a')
    sim.physical_step()  # contacts@t1 != contacts@t0; cache invalidated

    observed = sim._get_feet_forces('robot_a')
    fresh = _forces_from_contacts(sim, 'robot_a', sim._extract_contacts())

    assert np.allclose(observed, fresh, rtol=0.05, atol=0.02), (
        f"feet_forces read stale contacts after physics step; "
        f"observed={observed}, fresh={fresh}"
    )
