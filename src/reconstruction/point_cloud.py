"""
src/reconstruction/point_cloud.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Builds a global (world-frame) 3-D point cloud from raw depth frames + poses.

Core logic:
  For each sampled frame:
    1. Back-project depth map → camera-frame 3-D points
    2. Apply camera-to-world pose transform
    3. Accumulate in global buffer
  Finally voxel-downsample for memory/speed efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from tqdm import tqdm

from src.capture.data_loader import CameraPose, DataLoader, ScanData
from src.config import settings
from src.utils.geometry import build_transform_matrix, depth_to_pointcloud, transform_points
from src.utils.logger import get_logger

log = get_logger(__name__)


class PointCloudBuilder:
    """
    Builds a merged, downsampled point cloud for a single scan session.

    Args:
        scan_data: Populated :class:`ScanData` from :class:`DataLoader`.
        cfg:       ``settings.point_cloud`` config dict (injected, not imported inline).
        data_cfg:  ``settings.data`` config dict.
    """

    def __init__(
        self,
        scan_data: ScanData,
        cfg: Optional[Dict] = None,
        data_cfg: Optional[Dict] = None,
    ) -> None:
        self.scan_data = scan_data
        self.cfg = cfg or settings.point_cloud
        self.data_cfg = data_cfg or settings.data

        # Build a fast frame_id → pose lookup
        self._pose_map: Dict[str, CameraPose] = {
            p.frame_id: p for p in scan_data.poses
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self) -> o3d.geometry.PointCloud:
        """
        Process all depth frames and return one merged Open3D PointCloud.

        Returns:
            Voxel-downsampled global point cloud in world coordinates.
        """
        loader = DataLoader(self.scan_data.scan_dir)
        stride: int = self.cfg.get("frame_stride", 5)
        voxel_size: float = self.cfg.get("voxel_size", 0.02)

        depth_scale: float = self.data_cfg.get("depth_scale_factor", 0.001)
        depth_min: float   = self.data_cfg.get("depth_min_meters", 0.1)
        depth_max: float   = self.data_cfg.get("depth_max_meters", 10.0)
        min_conf: int      = self.data_cfg.get("confidence_threshold", 1)

        all_points: List[NDArray] = []

        frame_iter = loader.iter_depth_frames(self.scan_data, stride=stride)
        for fid, depth_map, conf_map in tqdm(frame_iter, desc="Building point cloud"):
            pose = self._pose_map.get(fid)
            if pose is None:
                log.debug("No pose for frame {} — skipping", fid)
                continue

            # Back-project depth → camera-frame points
            cam_pts = depth_to_pointcloud(
                depth_map=depth_map,
                fx=pose.intrinsics.fx,
                fy=pose.intrinsics.fy,
                cx=pose.intrinsics.cx,
                cy=pose.intrinsics.cy,
                depth_scale=depth_scale,
                depth_min=depth_min,
                depth_max=depth_max,
                confidence_map=conf_map,
                min_confidence=min_conf,
            )

            if cam_pts.shape[0] == 0:
                continue

            # Transform camera-frame → world-frame
            T = build_transform_matrix(
                pose.x, pose.y, pose.z,
                pose.qx, pose.qy, pose.qz, pose.qw,
            )
            world_pts = transform_points(cam_pts, T)
            all_points.append(world_pts)

        if not all_points:
            raise RuntimeError("No valid depth frames processed — check data and config thresholds.")

        merged = np.vstack(all_points)
        log.info("Total raw points: {:,}", merged.shape[0])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)

        # Voxel downsample for efficiency
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        log.info("After voxel downsample (size={}m): {:,} points", voxel_size, len(pcd.points))

        # Estimate surface normals (needed for plane detection)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        return pcd
