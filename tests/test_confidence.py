"""tests/test_confidence.py — Unit tests for confidence interval estimation."""

import pytest

from src.output.confidence import ConfidenceEstimator


class TestConfidenceEstimator:
    @pytest.mark.parametrize("tier", ["lidar", "video", "photo"])
    def test_ci_bounds_ordering(self, tier):
        est = ConfidenceEstimator(tier=tier)
        ci = est.wall_length(3.0)
        assert ci.lower <= ci.value <= ci.upper

    def test_lidar_tighter_than_photo(self):
        lidar = ConfidenceEstimator("lidar").wall_length(3.0)
        photo = ConfidenceEstimator("photo").wall_length(3.0)
        lidar_spread = lidar.upper - lidar.lower
        photo_spread = photo.upper - photo.lower
        assert lidar_spread < photo_spread

    def test_ceiling_height_unit(self):
        est = ConfidenceEstimator("lidar")
        ci = est.ceiling_height(2.5)
        assert ci.unit == "metres"
        assert abs((ci.upper - ci.lower) / 2 - 0.015) < 1e-9  # ±1.5 cm

    def test_summary_returns_correct_tier(self):
        est = ConfidenceEstimator("video")
        summary = est.summary()
        assert summary.tier == "video"
        assert summary.wall_length_tolerance_pct == pytest.approx(3.0)

    def test_floor_area_wider_than_wall(self):
        est = ConfidenceEstimator("lidar")
        wall_ci = est.wall_length(3.0)
        area_ci = est.floor_area(9.0)
        wall_pct = (wall_ci.upper - wall_ci.lower) / wall_ci.value
        area_pct = (area_ci.upper - area_ci.lower) / area_ci.value
        assert area_pct >= wall_pct
