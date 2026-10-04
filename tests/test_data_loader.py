"""tests/test_data_loader.py — Integration tests for the DataLoader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.capture.data_loader import DataLoader, ScanData


def _make_fake_scan_dir(tmp_path: Path) -> Path:
    """Create a minimal fake scan directory for testing."""
    scan = tmp_path / "fake_scan"
    scan.mkdir()

    # camera_matrix.csv
    (scan / "camera_matrix.csv").write_text(
        "1596.0, 0.0, 955.0\n0.0, 1596.0, 717.0\n0.0, 0.0, 1.0\n"
    )

    # odometry.csv
    header = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy\n"
    rows = "\n".join(
        f"{i * 0.033:.6f}, {i:06d}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1596.0, 1596.0, 955.0, 717.0"
        for i in range(5)
    )
    (scan / "odometry.csv").write_text(header + rows)

    # imu.csv
    imu_header = "timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n"
    imu_rows = "\n".join(
        f"{i * 0.01:.6f}, 0.0, -9.8, 0.0, 0.0, 0.0, 0.0" for i in range(10)
    )
    (scan / "imu.csv").write_text(imu_header + imu_rows)

    # depth/ with one PNG
    import cv2
    depth_dir = scan / "depth"
    depth_dir.mkdir()
    depth_img = np.full((10, 10), 2000, dtype=np.uint16)  # 2 m
    cv2.imwrite(str(depth_dir / "000000.png"), depth_img)

    return scan


class TestDataLoader:
    def test_load_returns_scan_data(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        loader = DataLoader(scan_dir)
        data = loader.load()
        assert isinstance(data, ScanData)

    def test_scan_id_is_folder_name(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        loader = DataLoader(scan_dir)
        data = loader.load()
        assert data.scan_id == "fake_scan"

    def test_camera_intrinsics_loaded(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        data = DataLoader(scan_dir).load()
        assert data.default_intrinsics is not None
        assert abs(data.default_intrinsics.fx - 1596.0) < 1e-3

    def test_poses_loaded(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        data = DataLoader(scan_dir).load()
        assert len(data.poses) == 5
        assert data.poses[0].frame_id == "000000"

    def test_imu_loaded(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        data = DataLoader(scan_dir).load()
        assert len(data.imu_readings) == 10

    def test_depth_dir_found(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        data = DataLoader(scan_dir).load()
        assert data.depth_dir is not None
        assert data.depth_dir.is_dir()

    def test_frame_ids_populated(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        data = DataLoader(scan_dir).load()
        assert "000000" in data.frame_ids

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader("/nonexistent/path")

    def test_iter_depth_frames(self, tmp_path):
        scan_dir = _make_fake_scan_dir(tmp_path)
        loader = DataLoader(scan_dir)
        data = loader.load()
        frames = list(loader.iter_depth_frames(data, stride=1))
        assert len(frames) == 1
        fid, depth, conf = frames[0]
        assert fid == "000000"
        assert depth.shape == (10, 10)
