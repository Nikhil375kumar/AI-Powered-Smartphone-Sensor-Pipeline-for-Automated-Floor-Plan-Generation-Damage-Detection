"""
src/utils/geometry.py
~~~~~~~~~~~~~~~~~~~~~
Pure 3-D / 2-D geometry helpers.

All functions are stateless, numpy-based utilities — no config dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ─── Quaternion ───────────────────────────────────────────────────────────────

def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> NDArray:
    """
    Convert a unit quaternion (x, y, z, w) to a 3×3 rotation matrix.

    Args:
        qx, qy, qz, qw: Quaternion components (unit quaternion expected).

    Returns:
        R: (3, 3) numpy rotation matrix.
    """
    # Normalise to guard against floating-point drift
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    return R


def build_transform_matrix(x: float, y: float, z: float,
                            qx: float, qy: float, qz: float, qw: float) -> NDArray:
    """
    Build a 4×4 homogeneous transform matrix from translation + quaternion.

    Args:
        x, y, z:           Translation vector (world coords).
        qx, qy, qz, qw:    Orientation quaternion.

    Returns:
        T: (4, 4) transform matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


# ─── Point Cloud Projection ───────────────────────────────────────────────────

def depth_to_pointcloud(
    depth_map: NDArray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    confidence_map: NDArray | None = None,
    min_confidence: int = 1,
) -> NDArray:
    """
    Back-project a depth image into a (N, 3) point cloud in the camera frame.

    Args:
        depth_map:       (H, W) uint16/float depth image.
        fx, fy:          Focal lengths in pixels.
        cx, cy:          Principal point in pixels.
        depth_scale:     Multiply depth values by this to get metres.
        depth_min:       Discard points closer than this (metres).
        depth_max:       Discard points farther than this (metres).
        confidence_map:  Optional (H, W) per-pixel confidence (0/1/2).
        min_confidence:  Minimum confidence level to keep a point.

    Returns:
        points: (N, 3) float64 array of XYZ points in camera frame.
    """
    depth_m = depth_map.astype(np.float64) * depth_scale

    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # pixel grid

    # Valid mask: depth in range
    valid = (depth_m > depth_min) & (depth_m < depth_max)

    # Apply confidence mask if provided
    if confidence_map is not None:
        valid &= (confidence_map >= min_confidence)

    # Backproject: X = (u - cx) * Z / fx
    Z = depth_m[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy

    return np.stack([X, Y, Z], axis=-1)


def transform_points(points: NDArray, transform: NDArray) -> NDArray:
    """
    Apply a 4×4 homogeneous transform to an (N, 3) point array.

    Args:
        points:    (N, 3) array in source frame.
        transform: (4, 4) homogeneous transform to target frame.

    Returns:
        (N, 3) array in target frame.
    """
    N = points.shape[0]
    pts_h = np.hstack([points, np.ones((N, 1), dtype=np.float64)])  # (N, 4)
    transformed = (transform @ pts_h.T).T                            # (N, 4)
    return transformed[:, :3]


# ─── 2-D Geometry ─────────────────────────────────────────────────────────────

def polygon_area(vertices: NDArray) -> float:
    """
    Compute the area of a 2-D polygon using the shoelace formula.

    Args:
        vertices: (N, 2) array of polygon vertices in order.

    Returns:
        Absolute area in the same units squared.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def segment_length(p1: NDArray, p2: NDArray) -> float:
    """Euclidean length of a line segment between two 2-D or 3-D points."""
    return float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))


def point_to_line_distance(point: NDArray, line_start: NDArray, line_end: NDArray) -> float:
    """
    Perpendicular distance from *point* to the infinite line through
    *line_start* and *line_end*.
    """
    d = line_end - line_start
    n = d / (np.linalg.norm(d) + 1e-12)
    v = point - line_start
    return float(np.linalg.norm(v - np.dot(v, n) * n))
