"""Quaternion helpers shared by clip tools (wxyz convention)."""
import numpy as np


def _rot(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz -> 3x3."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_log_wxyz(q: np.ndarray) -> np.ndarray:
    """Log map of a unit wxyz quaternion -> rotation vector."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    v = q[1:4]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return 2.0 * v
    ang = 2.0 * np.arctan2(s, np.clip(q[0], -1.0, 1.0))
    return v / s * ang


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])
