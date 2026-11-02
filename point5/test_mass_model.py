"""Tests for point5.mass_model — the model-free COM/inertia proxy."""
import numpy as np
import pytest

from point5.contracts import POINT_INDEX, TaskValidationError
from point5.mass_model import (
    DEFAULT_MASSES, MassModelConfig, mass_proxy, virtual_body)

# a plausible standing local pose (ankles planted, hands forward-ish)
STAND = np.array([
    [0.00, 0.00, 1.47],    # head
    [0.36, -0.17, 1.34],   # hand_right
    [0.36, 0.17, 1.34],    # hand_left
    [0.00, -0.09, 0.087],  # foot_right
    [0.00, 0.09, 0.087],   # foot_left
])


def test_shapes_and_types():
    mp = mass_proxy(STAND)
    assert mp.com.shape == (3,)
    assert mp.inertia_per_mass.shape == (3, 3)
    assert mp.segments.shape == (14, 3)
    assert np.isfinite(mp.com).all()
    assert np.isfinite(mp.inertia_per_mass).all()


def test_com_plausibility():
    """Standing COM: between ankles and head, near the support centre."""
    c = mass_proxy(STAND).com
    assert 0.5 < c[2] < STAND[0, 2]
    assert abs(c[0]) < 0.3 and abs(c[1]) < 0.1


def test_translation_equivariance():
    mp1 = mass_proxy(STAND)
    shift = np.array([0.3, -0.2, 0.1])
    mp2 = mass_proxy(STAND + shift)
    np.testing.assert_allclose(mp2.com - mp1.com, shift, atol=1e-9)
    np.testing.assert_allclose(mp2.inertia_per_mass,
                               mp1.inertia_per_mass, atol=1e-9)


def test_mirror_symmetry():
    m = np.diag([1.0, -1.0, 1.0])
    mirror = np.empty_like(STAND)
    mirror[0] = STAND[0] @ m.T
    mirror[1], mirror[2] = STAND[2] @ m.T, STAND[1] @ m.T
    mirror[3], mirror[4] = STAND[4] @ m.T, STAND[3] @ m.T
    mp = mass_proxy(mirror)
    base = mass_proxy(STAND)
    np.testing.assert_allclose(mp.com, base.com @ m.T, atol=1e-9)


def test_reaching_hand_pulls_com():
    reach = STAND.copy()
    reach[POINT_INDEX["hand_right"]] += np.array([0.25, 0.0, 0.0])
    d = mass_proxy(reach).com - mass_proxy(STAND).com
    assert d[0] > 0.005          # com moves forward, not backward
    assert abs(d[1]) < 0.02 and abs(d[2]) < 0.02


def test_crouch_lowers_com():
    crouch = STAND.copy()
    crouch[0, 2] -= 0.25         # head down
    assert mass_proxy(crouch).com[2] < mass_proxy(STAND).com[2] - 0.05


def test_inertia_properties():
    J = mass_proxy(STAND).inertia_per_mass
    np.testing.assert_allclose(J, J.T, atol=1e-12)
    eig = np.linalg.eigvalsh(J)
    assert (eig >= -1e-9).all()
    assert 0.01 < mass_proxy(STAND).gyration_z < 1.0


def test_invalid_input():
    with pytest.raises(TaskValidationError):
        mass_proxy(np.zeros((4, 3)))
    with pytest.raises(TaskValidationError):
        mass_proxy(np.full((5, 3), np.nan))


def test_custom_masses():
    # half mass on pelvis anchor + half on head (both on the y=0 axis)
    cfg = MassModelConfig(
        masses=(0.5, 0.0, 0.0, 0.5) + (0.0,) * 10)
    c = mass_proxy(STAND, cfg).com
    assert abs(c[1]) < 1e-6
    assert c[2] > STAND[0, 2] - 0.58 - 1e-6
