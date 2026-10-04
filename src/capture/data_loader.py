"""
src/capture/data_loader.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads raw sensor data from a scan directory.

Expected directory layout (LiDAR tier — sample data):
    <scan_dir>/
        imu.csv
        odometry.csv
        camera_matrix.csv
        depth/
            000000.png
            000001.png
            ...
        confidence/
            000000.png    (optional)
            ...
        rgb.mp4           (optional)

All loading logic is here — no path strings live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─── Data Containers ──────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> NDArray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)


@dataclass
class CameraPose:
    """6-DOF camera pose for a single frame."""
    timestamp: float
    frame_id: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    intrinsics: CameraIntrinsics   # per-frame intrinsics from odometry


@dataclass
class IMUReading:
    """Single IMU sample."""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class ScanData:
    """All raw data for one scan session."""
    scan_id: str
    scan_dir: Path
    poses: List[CameraPose] = field(default_factory=list)
    imu_readings: List[IMUReading] = field(default_factory=list)
    default_intrinsics: Optional[CameraIntrinsics] = None
    depth_dir: Optional[Path] = None
    confidence_dir: Optional[Path] = None
    rgb_video_path: Optional[Path] = None
    frame_ids: List[str] = field(default_factory=list)


# ─── Loader ───────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Reads all sensor files from a scan directory and returns a :class:`ScanData`.

    Args:
        scan_dir: Path to the scan folder (e.g. ``Dataset/single_scan_floor_only/1a8384c3f6``).
    """

    def __init__(self, scan_dir: str | Path) -> None:
        self.scan_dir = Path(scan_dir).resolve()
        if not self.scan_dir.is_dir():
            raise FileNotFoundError(f"Scan directory not found: {self.scan_dir}")
        self.scan_id = self.scan_dir.name
        log.info("DataLoader initialised for scan '{}'", self.scan_id)

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> ScanData:
        """Load all available sensor data and return a :class:`ScanData` object."""
        data = ScanData(
            scan_id=self.scan_id,
            scan_dir=self.scan_dir,
        )

        data.default_intrinsics = self._load_camera_matrix()
        data.poses = self._load_odometry(data.default_intrinsics)
        data.imu_readings = self._load_imu()

        depth_dir = self.scan_dir / "depth"
        if depth_dir.is_dir():
            data.depth_dir = depth_dir
            data.frame_ids = sorted([p.stem for p in depth_dir.glob("*.png")])
            log.info("Found {} depth frames", len(data.frame_ids))

        confidence_dir = self.scan_dir / "confidence"
        if confidence_dir.is_dir():
            data.confidence_dir = confidence_dir
            log.info("Confidence maps available")

        rgb_path = self.scan_dir / "rgb.mp4"
        if rgb_path.is_file():
            data.rgb_video_path = rgb_path
            log.info("RGB video found: {}", rgb_path.name)

        return data

    def iter_depth_frames(
        self,
        data: ScanData,
        stride: int = 1,
    ) -> Iterator[Tuple[str, NDArray, Optional[NDArray]]]:
        """
        Yield (frame_id, depth_map, confidence_map) tuples one at a time.

        Args:
            data:   Loaded :class:`ScanData` (must have ``depth_dir`` set).
            stride: Yield every Nth frame to reduce processing time.

        Yields:
            frame_id:        Zero-padded frame number string (e.g. ``"000042"``).
            depth_map:       (H, W) uint16 array — raw depth values.
            confidence_map:  (H, W) uint8 array or None.
        """
        if data.depth_dir is None:
            raise RuntimeError("No depth directory found in ScanData.")

        for i, fid in enumerate(data.frame_ids):
            if i % stride != 0:
                continue

            depth_path = data.depth_dir / f"{fid}.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                log.warning("Could not read depth frame: {}", depth_path)
                continue

            conf = None
            if data.confidence_dir is not None:
                conf_path = data.confidence_dir / f"{fid}.png"
                if conf_path.is_file():
                    conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)

            yield fid, depth, conf

    def iter_rgb_frames(
        self, data: ScanData, stride: int = 1
    ) -> Iterator[Tuple[int, NDArray]]:
        """
        Yield (frame_index, bgr_frame) from the RGB video.

        Args:
            data:   Loaded :class:`ScanData` (must have ``rgb_video_path`` set).
            stride: Sample every Nth video frame.

        Yields:
            frame_index: 0-based integer frame index.
            bgr_frame:   (H, W, 3) uint8 BGR image.
        """
        if data.rgb_video_path is None:
            raise RuntimeError("No RGB video found in ScanData.")

        cap = cv2.VideoCapture(str(data.rgb_video_path))
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    yield idx, frame
                idx += 1
        finally:
            cap.release()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_camera_matrix(self) -> CameraIntrinsics:
        """Load the 3×3 camera intrinsic matrix from camera_matrix.csv."""
        path = self.scan_dir / "camera_matrix.csv"
        if not path.is_file():
            raise FileNotFoundError(f"camera_matrix.csv not found in {self.scan_dir}")

        mat = np.loadtxt(str(path), delimiter=",")
        if mat.shape != (3, 3):
            raise ValueError(f"camera_matrix.csv must be 3×3, got {mat.shape}")

        intrinsics = CameraIntrinsics(
            fx=float(mat[0, 0]),
            fy=float(mat[1, 1]),
            cx=float(mat[0, 2]),
            cy=float(mat[1, 2]),
        )
        log.debug("Loaded camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                  intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        return intrinsics

    def _load_odometry(self, default_intrinsics: CameraIntrinsics) -> List[CameraPose]:
        """
        Load odometry.csv → list of :class:`CameraPose`.

        Columns: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
        """
        path = self.scan_dir / "odometry.csv"
        if not path.is_file():
            raise FileNotFoundError(f"odometry.csv not found in {self.scan_dir}")

        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"odometry.csv missing columns: {missing}")

        poses: List[CameraPose] = []
        for _, row in df.iterrows():
            # Per-frame intrinsics if available, else fall back to camera_matrix.csv
            fx = float(row["fx"]) if "fx" in df.columns and pd.notna(row.get("fx")) else default_intrinsics.fx
            fy = float(row["fy"]) if "fy" in df.columns and pd.notna(row.get("fy")) else default_intrinsics.fy
            cx = float(row["cx"]) if "cx" in df.columns and pd.notna(row.get("cx")) else default_intrinsics.cx
            cy = float(row["cy"]) if "cy" in df.columns and pd.notna(row.get("cy")) else default_intrinsics.cy

            poses.append(CameraPose(
                timestamp=float(row["timestamp"]),
                frame_id=str(int(float(row["frame"]))).zfill(6),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                qx=float(row["qx"]),
                qy=float(row["qy"]),
                qz=float(row["qz"]),
                qw=float(row["qw"]),
                intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
            ))

        log.info("Loaded {} camera poses from odometry.csv", len(poses))
        return poses

    def _load_imu(self) -> List[IMUReading]:
        """Load imu.csv → list of :class:`IMUReading`."""
        path = self.scan_dir / "imu.csv"
        if not path.is_file():
            log.warning("imu.csv not found — IMU data will be unavailable")
            return []

        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]

        readings: List[IMUReading] = []
        for _, row in df.iterrows():
            readings.append(IMUReading(
                timestamp=float(row["timestamp"]),
                accel_x=float(row.get("a_x", 0.0)),
                accel_y=float(row.get("a_y", 0.0)),
                accel_z=float(row.get("a_z", 0.0)),
                gyro_x=float(row.get("alpha_x", 0.0)),
                gyro_y=float(row.get("alpha_y", 0.0)),
                gyro_z=float(row.get("alpha_z", 0.0)),
            ))

        log.info("Loaded {} IMU readings from imu.csv", len(readings))
        return readings
