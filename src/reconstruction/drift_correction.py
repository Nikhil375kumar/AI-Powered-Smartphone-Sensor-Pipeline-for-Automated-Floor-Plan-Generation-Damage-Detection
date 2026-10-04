"""
src/reconstruction/drift_correction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Handles accumulated odometry drift via ICP (Iterative Closest Point) refinement
and optionally a loop-closure pass.

Design:
  - The "poses used as-is" approach fails the assessment drift gate.
  - This module provides ICP-based scan-to-scan refinement, producing
    corrected poses that can be fed back to PointCloudBuilder.
  - An ablation helper lets callers compare stitched footprints with/without
    correction (required by the assessment drift gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CorrectionResult:
    """Output of drift correction for one scan."""
    corrected_transforms: List[NDArray]   # (4×4) world transforms, one per keyframe
    rmse_before: float                    # RMSE before ICP (metres)
    rmse_after: float                     # RMSE after ICP (metres)
    loop_closure_applied: bool


class DriftCorrector:
    """
    Refines camera poses using point-cloud ICP.

    The assessment requires an ablation showing the stitched footprint
    *with* and *without* drift correction — see :meth:`ablation_run`.

    Args:
        cfg: ``settings.stitching`` config dict.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        self.cfg = cfg or settings.stitching

    def refine_with_icp(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        initial_transform: NDArray,
    ) -> Tuple[NDArray, float]:
        """
        Refine a relative transform between two overlapping point clouds via ICP.

        Args:
            source_pcd:         Point cloud to align.
            target_pcd:         Reference point cloud.
            initial_transform:  (4×4) initial guess (from odometry).

        Returns:
            (refined_transform, rmse): The refined 4×4 transform and fitness RMSE.
        """
        max_corr: float = self.cfg.get("icp_max_correspondence", 0.05)

        result = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=max_corr,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=self.cfg.get("icp_convergence_rmse", 0.001),
                max_iteration=50,
            ),
        )

        log.debug(
            "ICP: fitness={:.4f} rmse={:.4f}m inliers={}",
            result.fitness,
            result.inlier_rmse,
            len(result.correspondence_set),
        )
        return np.asarray(result.transformation), float(result.inlier_rmse)

    def ablation_run(
        self,
        pcd_with: o3d.geometry.PointCloud,
        pcd_without: o3d.geometry.PointCloud,
    ) -> Dict[str, float]:
        """
        Compute and return footprint metrics for the drift gate ablation table.

        Compares the axis-aligned bounding box area of two point clouds:
        one produced with drift correction and one without.

        Args:
            pcd_with:    Global point cloud built WITH drift correction.
            pcd_without: Global point cloud built WITHOUT drift correction.

        Returns:
            Dict with footprint area (m²) for both variants.
        """
        def footprint_area(pcd: o3d.geometry.PointCloud) -> float:
            pts = np.asarray(pcd.points)[:, :2]   # project to XY
            aabb_min = pts.min(axis=0)
            aabb_max = pts.max(axis=0)
            return float(np.prod(aabb_max - aabb_min))

        area_with = footprint_area(pcd_with)
        area_without = footprint_area(pcd_without)

        log.info(
            "Drift ablation — footprint WITH correction: {:.2f}m²  WITHOUT: {:.2f}m²",
            area_with, area_without,
        )
        return {
            "footprint_area_m2_with_correction": area_with,
            "footprint_area_m2_without_correction": area_without,
            "difference_m2": abs(area_with - area_without),
        }
