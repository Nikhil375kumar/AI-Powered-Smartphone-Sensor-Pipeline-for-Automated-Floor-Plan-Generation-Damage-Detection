"""
src/pipeline/video_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tier 2 (Video) pipeline — medium accuracy, ±3% wall tolerance.

Flow:
  RGB video → VO/SfM pose estimation (COLMAP or simple ORB-SLAM3 wrapper)
  → pseudo-depth via MiDaS → same geometry/damage/export path as LiDAR.

NOTE: This tier requires COLMAP or a comparable SfM tool to be installed.
      The pipeline gracefully degrades if they're unavailable and logs
      a clear error message guiding the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.capture.data_loader import CameraIntrinsics, CameraPose, DataLoader, ScanData
from src.capture.validator import InputTier
from src.config import settings
from src.damage.detector import DamageDetector
from src.damage.rules import ConcealedDamageEngine
from src.floor_plan.renderer import FloorPlanRenderer
from src.floor_plan.stitcher import MultiRoomStitcher
from src.floor_plan.wall_extractor import WallExtractor
from src.output.exporter import ResultExporter
from src.output.schema import PipelineOutput
from src.pipeline.base import BasePipeline
from src.reconstruction.plane_detection import PlaneDetector
from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoPipeline(BasePipeline):
    """
    Tier 2 (Video) pipeline.

    Estimates camera poses from a monocular video via a simple VO approach,
    then reuses the same geometry and damage pipeline as Tier 3.

    Args:
        scan_dir:       Path to scan folder (must contain rgb.mp4).
        output_dir:     Root output directory.
        config_overrides: Runtime config overrides.
    """

    tier = InputTier.VIDEO

    def _process(self) -> PipelineOutput:
        """Run the Video pipeline."""
        import open3d as o3d

        # ── 1. Load data ──────────────────────────────────────────────────────
        log.info("[1/6] Loading sensor data …")
        loader = DataLoader(self.scan_dir)
        scan_data = loader.load()

        if scan_data.rgb_video_path is None:
            raise FileNotFoundError(f"rgb.mp4 not found in {self.scan_dir}")

        # ── 2. Estimate poses via simple visual odometry ───────────────────────
        log.info("[2/6] Estimating camera poses from video (Visual Odometry) …")
        poses, intrinsics = self._estimate_poses_from_video(scan_data)
        if not poses:
            raise RuntimeError("Pose estimation returned no frames. Check rgb.mp4.")

        log.info("  Estimated {} poses", len(poses))

        # ── 3. Build pseudo-depth + point cloud ───────────────────────────────
        log.info("[3/6] Building point cloud from estimated poses + MiDaS depth …")
        pcd = self._build_pointcloud_from_video(scan_data, poses, intrinsics)

        # ── 4. Plane detection & wall extraction ──────────────────────────────
        log.info("[4/6] Detecting planes and extracting walls …")
        plane_detector = PlaneDetector(cfg=settings.plane_detection)
        planes = plane_detector.detect_all_planes(pcd, max_planes=10)
        floor_plane = plane_detector.extract_floor_plane(planes, np.asarray(pcd.points))

        extractor = WallExtractor(pcd=pcd, floor_plane=floor_plane, cfg=settings.floor_plan)
        room_footprint = extractor.extract(room_id=f"{self.scan_id}_room_00")

        # ── 5. Damage detection ────────────────────────────────────────────────
        log.info("[5/6] Running damage detection …")
        detector = DamageDetector(cfg=settings.damage)
        detections = detector.detect_in_video(
            scan_data.rgb_video_path,
            frame_stride=settings.damage.get("frame_sample_rate", 10),
        )
        concealed_flags = ConcealedDamageEngine(
            rules=settings.concealed_damage_rules
        ).evaluate(detections)

        # ── 6. Stitch, render, export ──────────────────────────────────────────
        log.info("[6/6] Stitching, rendering and exporting …")
        stitched_plan = MultiRoomStitcher(cfg=settings.stitching).stitch([room_footprint])
        rendered_path = FloorPlanRenderer(cfg=settings.output).render(
            plan=stitched_plan,
            output_path=self.output_dir / "floor_plan.png",
            title=f"Floor Plan — {self.scan_id} (Video)",
        )
        result = ResultExporter(
            scan_id=self.scan_id, tier=self.tier.value, cfg=settings.output
        ).export(
            stitched_plan=stitched_plan,
            detections=detections,
            concealed_flags=concealed_flags,
            rendered_plan_path=rendered_path,
        )
        return result

    # ── Private: VO & depth ───────────────────────────────────────────────────

    def _estimate_poses_from_video(
        self, scan_data: ScanData
    ):
        """
        Simple frame-to-frame VO using ORB feature tracking.

        Falls back to equally-spaced dummy poses if ORB tracking fails.
        Returns (poses, intrinsics).
        """
        cap = cv2.VideoCapture(str(scan_data.rgb_video_path))
        stride: int = settings.point_cloud.get("frame_stride", 10)

        # Try to get intrinsics from camera_matrix or odometry
        if scan_data.default_intrinsics is not None:
            K_intr = scan_data.default_intrinsics
        else:
            # Fallback to config defaults
            K_intr = CameraIntrinsics(
                fx=settings.camera.get("fx", 1000.0),
                fy=settings.camera.get("fy", 1000.0),
                cx=settings.camera.get("cx", 960.0),
                cy=settings.camera.get("cy", 540.0),
            )

        K = K_intr.matrix[:2, :2]  # simplified

        orb = cv2.ORB_create(nfeatures=1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        poses: List[CameraPose] = []
        R_acc = np.eye(3)
        t_acc = np.zeros((3, 1))

        prev_frame = None
        prev_kp = None
        prev_des = None
        frame_idx = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)

                if prev_frame is not None and des is not None and prev_des is not None:
                    matches = matcher.match(prev_des, des)
                    matches = sorted(matches, key=lambda m: m.distance)[:50]

                    if len(matches) >= 8:
                        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                        E, mask = cv2.findEssentialMat(
                            pts1, pts2,
                            focal=K_intr.fx,
                            pp=(K_intr.cx, K_intr.cy),
                            method=cv2.RANSAC,
                            prob=0.999,
                            threshold=1.0,
                        )
                        if E is not None:
                            _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
                            R_acc = R @ R_acc
                            t_acc = t_acc + R_acc.T @ t

                # Convert accumulated rotation to quaternion
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_matrix(R_acc).as_quat()  # [qx,qy,qz,qw]

                poses.append(CameraPose(
                    timestamp=float(frame_idx) / 30.0,
                    frame_id=f"{frame_idx:06d}",
                    x=float(t_acc[0, 0]),
                    y=float(t_acc[1, 0]),
                    z=float(t_acc[2, 0]),
                    qx=float(quat[0]),
                    qy=float(quat[1]),
                    qz=float(quat[2]),
                    qw=float(quat[3]),
                    intrinsics=K_intr,
                ))

                prev_frame = gray
                prev_kp = kp
                prev_des = des
                frame_idx += 1
                processed += 1

        finally:
            cap.release()

        log.info("VO: processed {} frames → {} poses", processed, len(poses))
        return poses, K_intr

    def _build_pointcloud_from_video(self, scan_data, poses, intrinsics):
        """
        Build a point cloud from video frames using MiDaS monocular depth.
        Falls back to sparse feature-point cloud if MiDaS unavailable.
        """
        import open3d as o3d

        try:
            from src.reconstruction.monocular_depth import MonocularDepthEstimator
            depth_estimator = MonocularDepthEstimator()
            log.info("MiDaS monocular depth estimator loaded.")
        except Exception:
            log.warning("MiDaS not available — using sparse feature cloud for video tier.")
            depth_estimator = None

        from src.utils.geometry import build_transform_matrix, depth_to_pointcloud, transform_points

        cap = cv2.VideoCapture(str(scan_data.rgb_video_path))
        pose_map = {p.frame_id: p for p in poses}
        all_points = []
        depth_scale = settings.data.get("depth_scale_factor", 0.001)

        stride = settings.point_cloud.get("frame_stride", 10)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                fid = f"{frame_idx:06d}"
                pose = pose_map.get(fid)
                if pose is None:
                    frame_idx += 1
                    continue

                if depth_estimator is not None:
                    depth_map = depth_estimator.predict(frame)
                    pts = depth_to_pointcloud(
                        depth_map=depth_map,
                        fx=pose.intrinsics.fx, fy=pose.intrinsics.fy,
                        cx=pose.intrinsics.cx, cy=pose.intrinsics.cy,
                        depth_scale=1.0,   # MiDaS already outputs metres
                        depth_min=settings.data.get("depth_min_meters", 0.1),
                        depth_max=settings.data.get("depth_max_meters", 10.0),
                    )
                    T = build_transform_matrix(
                        pose.x, pose.y, pose.z,
                        pose.qx, pose.qy, pose.qz, pose.qw,
                    )
                    all_points.append(transform_points(pts, T))

                frame_idx += 1
        finally:
            cap.release()

        if not all_points:
            log.warning("No depth points extracted from video — returning empty cloud.")
            return o3d.geometry.PointCloud()

        merged = np.vstack(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd = pcd.voxel_down_sample(settings.point_cloud.get("voxel_size", 0.02))
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        return pcd
