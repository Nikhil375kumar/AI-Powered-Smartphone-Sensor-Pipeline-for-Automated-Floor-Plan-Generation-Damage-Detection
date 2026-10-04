"""tests/test_geometry.py — Unit tests for geometry utilities."""

import numpy as np
import pytest

from src.utils.geometry import (
    build_transform_matrix,
    depth_to_pointcloud,
    polygon_area,
    quaternion_to_rotation_matrix,
    segment_length,
    transform_points,
)


class TestQuaternionToRotation:
    def test_identity(self):
        R = quaternion_to_rotation_matrix(0, 0, 0, 1)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_z_rotation(self):
        # 90° around Z: quat = (0, 0, sin45°, cos45°)
        s = np.sin(np.pi / 4)
        R = quaternion_to_rotation_matrix(0, 0, s, s)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_orthogonality(self):
        s = np.sin(np.pi / 6)
        c = np.cos(np.pi / 6)
        R = quaternion_to_rotation_matrix(s, 0, 0, c)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        s = np.sin(np.pi / 3)
        c = np.cos(np.pi / 3)
        R = quaternion_to_rotation_matrix(0, s, 0, c)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestTransformPoints:
    def test_identity_transform(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        T = np.eye(4)
        result = transform_points(pts, T)
        np.testing.assert_allclose(result, pts)

    def test_translation_only(self):
        pts = np.array([[0, 0, 0]], dtype=float)
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = transform_points(pts, T)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])

    def test_output_shape(self):
        pts = np.random.rand(100, 3)
        T = np.eye(4)
        result = transform_points(pts, T)
        assert result.shape == (100, 3)


class TestDepthToPointcloud:
    def test_basic_backprojection(self):
        # 2×2 depth map, all at distance 1 m (1000 mm raw)
        depth = np.full((2, 2), 1000, dtype=np.uint16)
        pts = depth_to_pointcloud(depth, fx=1000, fy=1000, cx=0.5, cy=0.5, depth_scale=0.001)
        assert pts.shape[1] == 3
        # All Z values should be 1.0 m
        np.testing.assert_allclose(pts[:, 2], 1.0, atol=1e-6)

    def test_depth_range_filter(self):
        depth = np.array([[500, 5000, 15000]], dtype=np.uint16)  # 0.5, 5, 15 m
        pts = depth_to_pointcloud(
            depth, fx=1000, fy=1000, cx=0, cy=0,
            depth_scale=0.001, depth_min=0.1, depth_max=10.0
        )
        # Only 0.5 m and 5.0 m should be kept; 15 m filtered
        assert pts.shape[0] == 2

    def test_empty_on_all_invalid(self):
        depth = np.zeros((10, 10), dtype=np.uint16)
        pts = depth_to_pointcloud(depth, fx=500, fy=500, cx=5, cy=5, depth_scale=0.001)
        assert pts.shape[0] == 0


class TestPolygonArea:
    def test_unit_square(self):
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        area = polygon_area(verts)
        assert abs(area - 1.0) < 1e-10

    def test_rectangle(self):
        verts = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=float)
        area = polygon_area(verts)
        assert abs(area - 6.0) < 1e-10


class TestSegmentLength:
    def test_zero_length(self):
        assert segment_length(np.array([1, 2]), np.array([1, 2])) == pytest.approx(0.0)

    def test_known_length(self):
        assert segment_length(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)
