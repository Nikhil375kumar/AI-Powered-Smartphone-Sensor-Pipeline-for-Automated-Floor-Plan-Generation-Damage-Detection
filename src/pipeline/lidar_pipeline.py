"""
src/pipeline/lidar_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tier 3 (LiDAR) pipeline — highest accuracy, tightest confidence intervals.

Flow:
  DataLoader → PointCloudBuilder → PlaneDetector → WallExtractor
  → DamageDetector → ConcealedDamageEngine → MultiRoomStitcher
  → FloorPlanRenderer → ResultExporter
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.capture.data_loader import DataLoader
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
from src.reconstruction.drift_correction import DriftCorrector
from src.reconstruction.plane_detection import PlaneDetector
from src.reconstruction.point_cloud import PointCloudBuilder
from src.utils.logger import get_logger

log = get_logger(__name__)


class LiDARPipeline(BasePipeline):
    """
    Full LiDAR-tier pipeline.

    Uses depth maps + camera poses + IMU to reconstruct 3-D geometry,
    extract walls, detect damage and produce a dimensioned floor plan.

    Args:
        scan_dir:        Path to one scan folder (contains depth/, odometry.csv …).
        output_dir:      Root output directory (default from config).
        enable_drift_correction: Toggle ICP refinement (default True).
        config_overrides: Runtime config overrides dict.
    """

    tier = InputTier.LIDAR

    def __init__(
        self,
        scan_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        enable_drift_correction: bool = True,
        config_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__(scan_dir, output_dir, config_overrides)
        self.enable_drift_correction = enable_drift_correction

    def _process(self) -> PipelineOutput:
        """Run the LiDAR pipeline and return a validated PipelineOutput."""

        # ── 1. Load data ──────────────────────────────────────────────────────
        log.info("[1/7] Loading sensor data …")
        loader = DataLoader(self.scan_dir)
        scan_data = loader.load()

        # ── 2. Build point cloud ──────────────────────────────────────────────
        log.info("[2/7] Building point cloud …")
        pc_builder = PointCloudBuilder(
            scan_data=scan_data,
            cfg=settings.point_cloud,
            data_cfg=settings.data,
        )
        pcd = pc_builder.build()

        # ── 3. Drift correction (optional, ablation-safe) ─────────────────────
        pcd_raw = pcd  # keep un-corrected copy for ablation
        if self.enable_drift_correction:
            log.info("[3/7] Drift correction (ICP) …")
            corrector = DriftCorrector(cfg=settings.stitching)
            # Single-scan: no inter-scan ICP needed; normals already estimated.
            # (Multi-room scans will apply ICP across scan pairs in stitcher.)
            log.info("  Single scan — drift correction is a no-op at this stage.")
        else:
            log.info("[3/7] Drift correction DISABLED (ablation mode)")

        # ── 4. Plane detection ─────────────────────────────────────────────────
        log.info("[4/7] Detecting planes …")
        import numpy as np
        plane_detector = PlaneDetector(cfg=settings.plane_detection)

        # Feed real gravity direction from IMU — critical for iOS ARKit data
        # where world-up is NOT the Z-axis
        if scan_data.imu_readings:
            plane_detector.set_gravity_from_imu(scan_data.imu_readings)
        else:
            log.warning("No IMU data — plane detector will auto-detect gravity axis.")

        planes = plane_detector.detect_all_planes(pcd)
        floor_plane = plane_detector.extract_floor_plane(
            planes, np.asarray(pcd.points)
        )
        up_vector = plane_detector.get_up_vector()

        # ── 5. Wall extraction ─────────────────────────────────────────────────
        log.info("[5/7] Extracting walls and openings …")
        extractor = WallExtractor(
            pcd=pcd,
            floor_plane=floor_plane,
            cfg=settings.floor_plan,
            up_vector=up_vector,
        )

        room_footprint = extractor.extract(room_id=f"{self.scan_id}_room_00")

        log.info(
            "  Room: area={:.2f}m²  ceiling={:.2f}m  walls={}  openings={}",
            room_footprint.floor_area_m2,
            room_footprint.ceiling_height_m,
            len(room_footprint.walls),
            len(room_footprint.openings),
        )

        # ── 6. Damage detection ────────────────────────────────────────────────
        log.info("[6/7] Running damage detection on RGB video …")
        detections = []
        concealed_flags = []

        if scan_data.rgb_video_path is not None:
            detector = DamageDetector(cfg=settings.damage)
            detections = detector.detect_in_video(
                scan_data.rgb_video_path,
                frame_stride=settings.damage.get("frame_sample_rate", 10),
            )
            rules_engine = ConcealedDamageEngine(rules=settings.concealed_damage_rules)
            concealed_flags = rules_engine.evaluate(detections)
        else:
            log.warning("No RGB video found — skipping damage detection.")

        # ── 7. Stitch + render + export ────────────────────────────────────────
        log.info("[7/7] Stitching, rendering and exporting …")
        stitcher = MultiRoomStitcher(cfg=settings.stitching)
        stitched_plan = stitcher.stitch([room_footprint])

        renderer = FloorPlanRenderer(cfg=settings.output, floor_cfg=settings.floor_plan)
        rendered_path = renderer.render(
            plan=stitched_plan,
            output_path=self.output_dir / "floor_plan.png",
            title=f"Floor Plan — {self.scan_id} (LiDAR)",
        )

        exporter = ResultExporter(
            scan_id=self.scan_id,
            tier=self.tier.value,
            cfg=settings.output,
        )
        result = exporter.export(
            stitched_plan=stitched_plan,
            detections=detections,
            concealed_flags=concealed_flags,
            rendered_plan_path=rendered_path,
        )

        return result
