"""
src/reconstruction/plane_detection.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detects dominant planes (floor, ceiling, walls) in a 3-D point cloud
using iterative RANSAC plane fitting.

KEY FIX: iOS ARKit LiDAR data does NOT use Z as world-up.
The actual gravity direction is estimated from:
  1. IMU accelerometer mean (most accurate)
  2. Auto-detection: try X, Y, Z axes — pick the one where two large
     parallel planes exist (floor + ceiling pair)

Returns typed PlaneResult objects — no raw dicts leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─── Data Containers ──────────────────────────────────────────────────────────

@dataclass
class PlaneResult:
    """A detected plane with its equation and inlier indices."""
    plane_id: str
    normal: NDArray          # (3,) unit normal vector
    offset: float            # d in: normal · x = d
    inlier_indices: List[int]
    inlier_count: int
    label: Optional[str] = None   # "floor" | "ceiling" | "wall_N" | None

    @property
    def equation(self) -> Tuple[float, float, float, float]:
        """Returns (a, b, c, d) of plane equation ax+by+cz+d=0."""
        a, b, c = self.normal
        return float(a), float(b), float(c), float(-self.offset)


# ─── Detector ─────────────────────────────────────────────────────────────────

class PlaneDetector:
    """
    Iteratively fits planes to a point cloud using Open3D's RANSAC.

    Args:
        cfg:           ``settings.plane_detection`` config dict.
        gravity_down:  Optional (3,) unit vector pointing DOWN in world frame.
                       When None, auto-detected from point cloud structure or IMU.
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        gravity_down: Optional[NDArray] = None,
    ) -> None:
        self.cfg = cfg or settings.plane_detection
        # gravity_down: unit vector pointing DOWN in world frame
        # world UP = -gravity_down
        self._gravity_down: Optional[NDArray] = gravity_down

    # ── Public ────────────────────────────────────────────────────────────────

    def set_gravity_from_imu(self, imu_readings) -> None:
        """
        Estimate gravity direction from IMU accelerometer readings.

        The mean of all accelerometer readings approximates the gravity vector
        (when the device is not undergoing rapid linear acceleration).

        Args:
            imu_readings: List of IMUReading objects with accel_x/y/z fields.
        """
        if not imu_readings:
            return

        accel = np.array(
            [[r.accel_x, r.accel_y, r.accel_z] for r in imu_readings],
            dtype=np.float64,
        )
        mean_accel = accel.mean(axis=0)
        norm = np.linalg.norm(mean_accel)
        if norm > 0.1:
            self._gravity_down = mean_accel / norm
            log.info(
                "Gravity estimated from IMU: [{:.3f}, {:.3f}, {:.3f}]",
                *self._gravity_down,
            )
        else:
            log.warning("IMU gravity vector is too small — will auto-detect UP axis.")

    def detect_all_planes(
        self,
        pcd: o3d.geometry.PointCloud,
        max_planes: Optional[int] = None,
    ) -> List[PlaneResult]:
        """
        Iteratively segment planes until fewer than ``min_inlier_ratio`` of
        remaining points form a plane.

        Args:
            pcd:        Input point cloud (each iteration consumes inlier points).
            max_planes: Hard cap on planes to extract. Defaults to config value.

        Returns:
            List of :class:`PlaneResult` sorted by inlier count (descending).
        """
        distance_thresh: float = self.cfg.get("distance_threshold", 0.02)
        ransac_n: int          = self.cfg.get("ransac_n", 3)
        num_iter: int          = self.cfg.get("num_iterations", 1000)
        min_ratio: float       = self.cfg.get("min_inlier_ratio", 0.01)
        if max_planes is None:
            max_planes = self.cfg.get("max_planes", 15)

        remaining = pcd
        planes: List[PlaneResult] = []
        total_pts = len(np.asarray(pcd.points))

        for i in range(max_planes):
            if len(np.asarray(remaining.points)) < ransac_n:
                break

            eq, inliers = remaining.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=ransac_n,
                num_iterations=num_iter,
            )

            inlier_ratio = len(inliers) / total_pts
            if inlier_ratio < min_ratio:
                log.debug("Plane {} has only {:.2%} inliers — stopping", i, inlier_ratio)
                break

            a, b, c, d = eq
            normal = np.array([a, b, c], dtype=np.float64)
            normal_len = np.linalg.norm(normal)
            if normal_len > 0:
                normal /= normal_len

            plane = PlaneResult(
                plane_id=f"plane_{i:02d}",
                normal=normal,
                offset=-d / normal_len,
                inlier_indices=inliers,
                inlier_count=len(inliers),
            )
            planes.append(plane)
            log.debug(
                "Detected {} inliers={} ({:.1%}) normal=[{:.2f},{:.2f},{:.2f}]",
                plane.plane_id, plane.inlier_count, inlier_ratio, *normal,
            )

            # Remove inliers — continue with remaining outliers
            remaining = remaining.select_by_index(inliers, invert=True)

        # Auto-detect UP axis if gravity not set from IMU
        if self._gravity_down is None:
            self._gravity_down = self._auto_detect_down_axis(planes)

        # Label each plane as floor / ceiling / wall
        planes = self._label_planes(planes)
        planes.sort(key=lambda p: p.inlier_count, reverse=True)

        up = self._get_up()
        log.info(
            "Detected {} planes | UP axis: [{:.2f},{:.2f},{:.2f}]",
            len(planes), *up,
        )
        return planes

    def extract_floor_plane(
        self, planes: List[PlaneResult], points: NDArray
    ) -> Optional[PlaneResult]:
        """
        Identify the floor plane — the horizontal plane with the lowest
        average centroid along the gravity/UP axis.

        Args:
            planes: List of detected planes.
            points: (N, 3) array of all point cloud points.

        Returns:
            The floor :class:`PlaneResult` or None if not identifiable.
        """
        up = self._get_up()

        # Horizontal = normal within 60° of UP
        horizontal = [
            p for p in planes
            if abs(np.dot(p.normal, up)) > 0.5
        ]

        if not horizontal:
            log.warning("No horizontal planes found — returning largest plane as floor fallback.")
            return planes[0] if planes else None

        # Floor = horizontal plane with the lowest mean position along UP
        best = min(
            horizontal,
            key=lambda p: float(np.dot(
                np.mean(points[p.inlier_indices], axis=0), up
            )),
        )
        log.info(
            "Floor plane: {} | label={} | inliers={} | normal=[{:.2f},{:.2f},{:.2f}]",
            best.plane_id, best.label, best.inlier_count, *best.normal,
        )
        return best

    def get_up_vector(self) -> NDArray:
        """Return the world UP unit vector."""
        return self._get_up()

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_up(self) -> NDArray:
        """Return world UP as unit vector (opposite of gravity_down)."""
        if self._gravity_down is not None:
            return -self._gravity_down
        return np.array([0.0, 0.0, 1.0])

    def _auto_detect_down_axis(self, planes: List[PlaneResult]) -> NDArray:
        """
        Find the gravity direction by checking which principal axis has the
        most inlier points in horizontal (floor+ceiling) planes.

        Tries ±X, ±Y, ±Z — returns best match.
        """
        candidates = [
            np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
            np.array([0, 0, 1.0]), np.array([0, 0, -1.0]),
        ]

        best_axis = np.array([0.0, 0.0, -1.0])   # default: -Z is down
        best_score = 0

        for axis in candidates:
            aligned = [p for p in planes if abs(np.dot(p.normal, axis)) > 0.7]
            score = sum(p.inlier_count for p in aligned)
            if score > best_score:
                best_score = score
                best_axis = axis.copy()

        log.info(
            "Auto-detected DOWN axis: [{:.0f},{:.0f},{:.0f}] (score={:,})",
            *best_axis, best_score,
        )
        return best_axis

    def _label_planes(self, planes: List[PlaneResult]) -> List[PlaneResult]:
        """Assign semantic labels (floor / ceiling / wall_NN) via UP alignment."""
        up = self._get_up()

        horizontal_planes = sorted(
            [p for p in planes if abs(np.dot(p.normal, up)) > 0.5],
            key=lambda p: p.inlier_count,
            reverse=True,
        )
        vertical_planes = [p for p in planes if abs(np.dot(p.normal, up)) <= 0.5]

        if horizontal_planes:
            horizontal_planes[0].label = "floor"
        if len(horizontal_planes) > 1:
            horizontal_planes[1].label = "ceiling"

        for i, vp in enumerate(vertical_planes):
            vp.label = f"wall_{i:02d}"

        return planes
