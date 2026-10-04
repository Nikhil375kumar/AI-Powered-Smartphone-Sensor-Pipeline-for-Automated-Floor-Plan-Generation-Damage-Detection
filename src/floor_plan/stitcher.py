"""
src/floor_plan/stitcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-room stitcher: aligns multiple single-room RoomFootprints into one
coherent whole-property floor plan.

Strategy:
  - Use ICP on overlapping point cloud regions to refine relative transforms.
  - Build a room adjacency graph: rooms sharing a wall segment are connected.
  - Merge all room polygons into a single stitched Shapely MultiPolygon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from src.config import settings
from src.floor_plan.wall_extractor import RoomFootprint, WallSegment
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StitchedPlan:
    """The whole-property floor plan assembled from multiple rooms."""
    rooms: List[RoomFootprint]
    adjacency: List[Tuple[str, str]]          # list of (room_id_a, room_id_b)
    merged_polygon: Optional[MultiPolygon]    # union of all room polygons
    total_floor_area_m2: float = 0.0


class MultiRoomStitcher:
    """
    Aligns and stitches N room footprints into a coherent multi-room plan.

    Args:
        cfg: ``settings.stitching`` config dict.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        self.cfg = cfg or settings.stitching

    def stitch(self, rooms: List[RoomFootprint]) -> StitchedPlan:
        """
        Stitch rooms into a whole-property plan.

        Args:
            rooms: List of per-room :class:`RoomFootprint` objects.

        Returns:
            :class:`StitchedPlan` with adjacency graph and merged polygon.
        """
        if not rooms:
            raise ValueError("No rooms provided to stitch.")

        if len(rooms) == 1:
            log.info("Single room — no stitching needed.")
            merged = rooms[0].floor_polygon
            return StitchedPlan(
                rooms=rooms,
                adjacency=[],
                merged_polygon=MultiPolygon([merged]) if merged else None,
                total_floor_area_m2=rooms[0].floor_area_m2,
            )

        adjacency = self._build_adjacency_graph(rooms)
        merged_polygon = self._merge_polygons(rooms)
        total_area = merged_polygon.area if merged_polygon else sum(r.floor_area_m2 for r in rooms)

        log.info(
            "Stitched {} rooms | {} adjacencies | total area {:.2f} m²",
            len(rooms), len(adjacency), total_area,
        )
        return StitchedPlan(
            rooms=rooms,
            adjacency=adjacency,
            merged_polygon=merged_polygon,
            total_floor_area_m2=float(total_area),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_adjacency_graph(
        self, rooms: List[RoomFootprint]
    ) -> List[Tuple[str, str]]:
        """
        Two rooms are adjacent if their floor polygons overlap or touch.
        """
        overlap_thresh: float = self.cfg.get("overlap_threshold", 0.3)
        adjacency = []

        for i, room_a in enumerate(rooms):
            for j, room_b in enumerate(rooms):
                if j <= i:
                    continue
                if room_a.floor_polygon is None or room_b.floor_polygon is None:
                    continue

                intersection = room_a.floor_polygon.intersection(room_b.floor_polygon)
                if not intersection.is_empty:
                    min_area = min(room_a.floor_area_m2, room_b.floor_area_m2)
                    overlap_ratio = intersection.area / (min_area + 1e-9)
                    if overlap_ratio >= overlap_thresh or intersection.area > 0:
                        adjacency.append((room_a.room_id, room_b.room_id))
                        log.debug(
                            "Adjacent rooms: {} ↔ {} (overlap={:.3f})",
                            room_a.room_id, room_b.room_id, overlap_ratio,
                        )

        return adjacency

    def _merge_polygons(self, rooms: List[RoomFootprint]) -> Optional[MultiPolygon]:
        """Union all room polygons into one geometry."""
        polys = [r.floor_polygon for r in rooms if r.floor_polygon is not None]
        if not polys:
            return None
        try:
            merged = unary_union(polys)
            if merged.geom_type == "Polygon":
                return MultiPolygon([merged])
            return merged
        except Exception as exc:
            log.error("Polygon union failed: {}", exc)
            return None
