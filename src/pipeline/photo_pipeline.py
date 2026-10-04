"""
src/pipeline/photo_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tier 1 (Photo) pipeline — widest intervals (±8%), photo folders per room.

Flow:
  Per-room photo folders → feature matching (LightGlue / SuperGlue)
  → SfM poses → Depth-Anything v2 monocular depth
  → same geometry / damage / export path → stitch all rooms

This is the hardest tier: no depth, no poses. Monocular depth estimation
and multi-image SfM give us approximate geometry. Confidence intervals
are widened accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.capture.data_loader import CameraIntrinsics, CameraPose
from src.capture.validator import InputTier
from src.config import settings
from src.damage.detector import DamageDetector
from src.damage.rules import ConcealedDamageEngine
from src.floor_plan.renderer import FloorPlanRenderer
from src.floor_plan.stitcher import MultiRoomStitcher
from src.floor_plan.wall_extractor import RoomFootprint, WallExtractor
from src.output.exporter import ResultExporter
from src.output.schema import PipelineOutput
from src.pipeline.base import BasePipeline
from src.reconstruction.plane_detection import PlaneDetector
from src.utils.logger import get_logger

log = get_logger(__name__)


class PhotoPipeline(BasePipeline):
    """
    Tier 1 (Photo) pipeline.

    Expects scan_dir to contain one or more sub-folders, each containing
    2–8 JPEG/PNG photos of a single room:

        scan_dir/
            room_01/
                img_001.jpg
                img_002.jpg
            room_02/
                img_001.jpg
                ...

    Args:
        scan_dir: Path to the root scan directory containing room sub-folders.
        output_dir: Root output directory.
    """

    tier = InputTier.PHOTO

    def _process(self) -> PipelineOutput:
        """Run the Photo pipeline."""
        import open3d as o3d

        # ── 1. Discover room sub-folders ──────────────────────────────────────
        log.info("[1/6] Discovering room photo folders …")
        room_dirs = self._discover_room_folders()
        log.info("  Found {} room folder(s): {}", len(room_dirs),
                 [d.name for d in room_dirs])

        if not room_dirs:
            raise ValueError(
                f"No room sub-folders found in {self.scan_dir}. "
                "Photo tier expects one sub-folder per room."
            )

        # ── 2. Process each room ──────────────────────────────────────────────
        all_room_footprints: List[RoomFootprint] = []
        all_detections = []

        for room_dir in room_dirs:
            log.info("[2/6] Processing room: {}", room_dir.name)
            photos = self._load_photos(room_dir)
            if not photos:
                log.warning("No images in {} — skipping", room_dir.name)
                continue

            # Estimate depth for each photo
            log.info("  Estimating depth for {} photos …", len(photos))
            depth_estimator = self._get_depth_estimator()
            depth_maps = [depth_estimator(img) for img in photos] if depth_estimator else []

            # Build pseudo-poses (simple planar assumption)
            intrinsics = self._estimate_intrinsics(photos[0])
            poses = self._estimate_poses_from_photos(photos, intrinsics)

            # Build point cloud
            pcd = self._build_pointcloud(photos, depth_maps, poses, intrinsics)

            # Plane detection + walls
            plane_detector = PlaneDetector(cfg=settings.plane_detection)
            planes = plane_detector.detect_all_planes(pcd, max_planes=8)
            floor_plane = plane_detector.extract_floor_plane(
                planes, np.asarray(pcd.points)
            )
            extractor = WallExtractor(pcd=pcd, floor_plane=floor_plane, cfg=settings.floor_plan)
            footprint = extractor.extract(room_id=f"{self.scan_id}_{room_dir.name}")
            all_room_footprints.append(footprint)

            # Damage detection on photos
            detector = DamageDetector(cfg=settings.damage)
            for i, img in enumerate(photos):
                dets = detector.detect_in_frame(img, frame_id=f"{room_dir.name}_photo_{i:03d}")
                all_detections.extend(dets)

        # ── 3. Concealed damage ───────────────────────────────────────────────
        log.info("[3/6] Evaluating concealed damage rules …")
        concealed_flags = ConcealedDamageEngine(
            rules=settings.concealed_damage_rules
        ).evaluate(all_detections)

        # ── 4. Stitch all rooms ───────────────────────────────────────────────
        log.info("[4/6] Stitching {} room(s) …", len(all_room_footprints))
        stitcher = MultiRoomStitcher(cfg=settings.stitching)
        stitched_plan = stitcher.stitch(all_room_footprints)

        # ── 5. Render ─────────────────────────────────────────────────────────
        log.info("[5/6] Rendering floor plan …")
        rendered_path = FloorPlanRenderer(cfg=settings.output).render(
            plan=stitched_plan,
            output_path=self.output_dir / "floor_plan.png",
            title=f"Floor Plan — {self.scan_id} (Photo Tier)",
        )

        # ── 6. Export ─────────────────────────────────────────────────────────
        log.info("[6/6] Exporting JSON …")
        result = ResultExporter(
            scan_id=self.scan_id, tier=self.tier.value, cfg=settings.output
        ).export(
            stitched_plan=stitched_plan,
            detections=all_detections,
            concealed_flags=concealed_flags,
            rendered_plan_path=rendered_path,
        )
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _discover_room_folders(self) -> List[Path]:
        """Return all direct sub-directories of scan_dir that contain images."""
        image_exts = {".jpg", ".jpeg", ".png", ".heic"}
        room_dirs = []
        for child in sorted(self.scan_dir.iterdir()):
            if child.is_dir():
                has_images = any(f.suffix.lower() in image_exts for f in child.iterdir())
                if has_images:
                    room_dirs.append(child)
        return room_dirs

    def _load_photos(self, room_dir: Path) -> List[np.ndarray]:
        """Load all images from a room folder as BGR numpy arrays."""
        image_exts = {".jpg", ".jpeg", ".png"}
        images = []
        for f in sorted(room_dir.iterdir()):
            if f.suffix.lower() in image_exts:
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
                else:
                    log.warning("Could not read image: {}", f)
        return images

    def _estimate_intrinsics(self, sample_img: np.ndarray) -> CameraIntrinsics:
        """Estimate intrinsics from image size (EXIF-free fallback heuristic)."""
        h, w = sample_img.shape[:2]
        # Approximate focal length: 70% of image diagonal (typical iPhone)
        diag = np.sqrt(h**2 + w**2)
        fx = fy = diag * 0.7
        return CameraIntrinsics(fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0)

    def _estimate_poses_from_photos(
        self, photos: List[np.ndarray], intrinsics: CameraIntrinsics
    ) -> List[CameraPose]:
        """
        Simple sequential VO across a list of photos.
        Returns one pose per photo using ORB + essential matrix.
        """
        from scipy.spatial.transform import Rotation

        orb = cv2.ORB_create(nfeatures=500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        R_acc = np.eye(3)
        t_acc = np.zeros((3, 1))
        poses: List[CameraPose] = []

        prev_gray = None
        prev_kp = None
        prev_des = None

        for i, img in enumerate(photos):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            if prev_gray is not None and des is not None and prev_des is not None:
                matches = matcher.match(prev_des, des)
                matches = sorted(matches, key=lambda m: m.distance)[:30]
                if len(matches) >= 5:
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                    E, _ = cv2.findEssentialMat(
                        pts1, pts2,
                        focal=intrinsics.fx, pp=(intrinsics.cx, intrinsics.cy),
                        method=cv2.RANSAC, prob=0.999, threshold=1.0,
                    )
                    if E is not None:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
                        R_acc = R @ R_acc
                        t_acc = t_acc + R_acc.T @ t

            quat = Rotation.from_matrix(R_acc).as_quat()
            poses.append(CameraPose(
                timestamp=float(i),
                frame_id=f"photo_{i:03d}",
                x=float(t_acc[0, 0]),
                y=float(t_acc[1, 0]),
                z=float(t_acc[2, 0]),
                qx=float(quat[0]), qy=float(quat[1]),
                qz=float(quat[2]), qw=float(quat[3]),
                intrinsics=intrinsics,
            ))

            prev_gray, prev_kp, prev_des = gray, kp, des

        return poses

    def _get_depth_estimator(self):
        """
        Return a callable(bgr_image) → depth_map using MiDaS / Depth-Anything.
        Returns None if not available.
        """
        try:
            from src.reconstruction.monocular_depth import MonocularDepthEstimator
            est = MonocularDepthEstimator()
            return est.predict
        except Exception as e:
            log.warning("Monocular depth estimator unavailable ({}). Photo tier will be sparse.", e)
            return None

    def _build_pointcloud(
        self,
        photos: List[np.ndarray],
        depth_maps: List[np.ndarray],
        poses: List[CameraPose],
        intrinsics: CameraIntrinsics,
    ):
        """Merge photo depth maps into a single open3d point cloud."""
        import open3d as o3d
        from src.utils.geometry import build_transform_matrix, depth_to_pointcloud, transform_points

        all_points = []
        for i, (pose, depth_map) in enumerate(zip(poses, depth_maps)):
            if depth_map is None:
                continue
            pts = depth_to_pointcloud(
                depth_map=depth_map,
                fx=intrinsics.fx, fy=intrinsics.fy,
                cx=intrinsics.cx, cy=intrinsics.cy,
                depth_scale=1.0,
                depth_min=settings.data.get("depth_min_meters", 0.1),
                depth_max=settings.data.get("depth_max_meters", 10.0),
            )
            if pts.shape[0] == 0:
                continue
            T = build_transform_matrix(
                pose.x, pose.y, pose.z,
                pose.qx, pose.qy, pose.qz, pose.qw,
            )
            all_points.append(transform_points(pts, T))

        if not all_points:
            log.warning("No depth points for photo tier — returning empty cloud.")
            return o3d.geometry.PointCloud()

        merged = np.vstack(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd = pcd.voxel_down_sample(settings.point_cloud.get("voxel_size", 0.02))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return pcd
