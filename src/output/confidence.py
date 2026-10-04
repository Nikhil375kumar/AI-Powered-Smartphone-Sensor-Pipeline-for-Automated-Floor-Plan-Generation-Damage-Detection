"""
src/output/confidence.py
~~~~~~~~~~~~~~~~~~~~~~~~
Computes confidence intervals for every measurement, based on input tier.

Intervals widen as sensor data thins:
  LiDAR  → tightest (raw depth, known poses)
  Video  → medium   (estimated poses via VO)
  Photo  → widest   (monocular depth, no poses)

All tolerance values come from config — never hardcoded.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.config import settings
from src.output.schema import ConfidenceInterval, ConfidenceSummaryOut


class ConfidenceEstimator:
    """
    Wraps tier-specific tolerance parameters and builds :class:`ConfidenceInterval`
    objects for every measured quantity.

    Args:
        tier: "lidar" | "video" | "photo"
        cfg:  ``settings.confidence`` config dict.
    """

    def __init__(self, tier: str, cfg: Optional[Dict] = None) -> None:
        self.tier = tier.lower()
        self.cfg = cfg or settings.confidence

        # Map tier → wall length % tolerance
        self._wall_pct_map = {
            "lidar": self.cfg.get("lidar_wall_length_pct", 1.0),
            "video": self.cfg.get("video_wall_length_pct", 3.0),
            "photo": self.cfg.get("photo_wall_length_pct", 8.0),
        }
        self.wall_length_pct: float = self._wall_pct_map.get(self.tier, 5.0)
        self.ceiling_cm: float = self.cfg.get("ceiling_height_cm", 1.5)
        self.opening_cm: float = self.cfg.get("opening_width_cm", 2.0)

    # ── Public helpers ─────────────────────────────────────────────────────────

    def wall_length(self, value_m: float) -> ConfidenceInterval:
        """Return a CI for a wall length measurement in metres."""
        delta = value_m * (self.wall_length_pct / 100.0)
        return ConfidenceInterval(
            value=round(value_m, 4),
            lower=round(value_m - delta, 4),
            upper=round(value_m + delta, 4),
            unit="metres",
        )

    def ceiling_height(self, value_m: float) -> ConfidenceInterval:
        """Return a CI for ceiling height in metres."""
        delta = self.ceiling_cm / 100.0
        return ConfidenceInterval(
            value=round(value_m, 4),
            lower=round(value_m - delta, 4),
            upper=round(value_m + delta, 4),
            unit="metres",
        )

    def floor_area(self, value_m2: float) -> ConfidenceInterval:
        """Return a CI for floor area in m² (approximate — derived from wall CIs)."""
        # Area error ≈ 2 * wall_pct for a rectangular room (rough)
        delta_pct = 2.0 * self.wall_length_pct / 100.0
        delta = value_m2 * delta_pct
        return ConfidenceInterval(
            value=round(value_m2, 4),
            lower=round(value_m2 - delta, 4),
            upper=round(value_m2 + delta, 4),
            unit="m²",
        )

    def opening_width(self, value_m: float) -> ConfidenceInterval:
        """Return a CI for door/window width in metres."""
        delta = self.opening_cm / 100.0
        return ConfidenceInterval(
            value=round(value_m, 4),
            lower=round(value_m - delta, 4),
            upper=round(value_m + delta, 4),
            unit="metres",
        )

    def summary(self) -> ConfidenceSummaryOut:
        """Return a summary object for the top-level output."""
        return ConfidenceSummaryOut(
            tier=self.tier,
            wall_length_tolerance_pct=self.wall_length_pct,
            ceiling_height_tolerance_cm=self.ceiling_cm,
            opening_width_tolerance_cm=self.opening_cm,
        )
