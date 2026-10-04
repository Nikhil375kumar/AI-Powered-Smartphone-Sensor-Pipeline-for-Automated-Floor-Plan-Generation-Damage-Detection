"""
src/floor_plan/wall_extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts 2-D wall segments and room boundaries from a 3-D point cloud.

Algorithm:
  1. Flatten the wall-band slice of the point cloud to a 2-D occupancy grid.
  2. Apply morphological closing to fill gaps.
  3. Extract contours → wall candidate polygons.
  4. Fit line segments to each contour edge.
  5. Identify gaps (openings) in walls — doors / windows.

All thresholds come from config, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union

from src.config import settings
from src.reconstruction.plane_detection import PlaneResult
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─── Data Containers ──────────────────────────────────────────────────────────

@dataclass
class WallSegment:
    """A single detected wall segment."""
    wall_id: str
    start: Tuple[float, float]   # (x, y) in metres (world frame)
    end: Tuple[float, float]
    length_m: float
    surface_id: str = ""

    @property
    def as_linestring(self) -> LineString:
        return LineString([self.start, self.end])


@dataclass
class Opening:
    """A gap in a wall (door / window / archway)."""
    opening_id: str
    wall_id: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    width_m: float
    opening_type: str = "unknown"   # assigned downstream
    position_along_wall_m: float = 0.0


@dataclass
class RoomFootprint:
    """2-D footprint of a single room."""
    room_id: str
    walls: List[WallSegment] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    floor_polygon: Optional[Polygon] = None    # Shapely polygon in world metres
    floor_area_m2: float = 0.0
    ceiling_height_m: float = 0.0


# ─── Extractor ────────────────────────────────────────────────────────────────

class WallExtractor:
    """
    Extracts walls, openings and a room footprint from a global point cloud.

    Args:
        pcd:        World-frame point cloud.
        floor_plane: Detected floor plane (used to orient coordinate system).
        cfg:        ``settings.floor_plan`` config dict.
    """

    def __init__(
        self,
        pcd: o3d.geometry.PointCloud,
        floor_plane: Optional[PlaneResult] = None,
        cfg: Optional[Dict] = None,
        up_vector: Optional[np.ndarray] = None,
    ) -> None:
        self.pcd = pcd
        self.floor_plane = floor_plane
        self.cfg = cfg or settings.floor_plan
        # World UP unit vector — use IMU-derived value or fall back to Z
        self.up = np.array(up_vector, dtype=float) if up_vector is not None else np.array([0.0, 0.0, 1.0])
        self.up /= np.linalg.norm(self.up)
        self._points = np.asarray(pcd.points)

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, room_id: str = "room_00") -> RoomFootprint:
        """
        Run the full wall-extraction pipeline.

        Returns:
            :class:`RoomFootprint` with walls, openings, area and ceiling height.
        """
        floor_z = self._estimate_floor_z()
        ceiling_height = self._estimate_ceiling_height(floor_z)

        # Slice points in the wall-band
        wall_pts = self._slice_wall_band(floor_z)
        if wall_pts.shape[0] < 10:
            log.warning("Too few wall-band points ({}), result may be inaccurate", wall_pts.shape[0])

        # Build occupancy grid and extract contours
        grid, origin, resolution = self._build_occupancy_grid(wall_pts)
        contours = self._extract_contours(grid)

        # Fit wall segments
        walls = self._fit_wall_segments(contours, origin, resolution)
        log.info("Extracted {} wall segments", len(walls))

        # Detect openings (gaps)
        openings = self._detect_openings(walls, wall_pts, origin, resolution)
        log.info("Detected {} openings", len(openings))

        # Build room polygon from wall segments
        floor_polygon = self._build_room_polygon(walls)
        floor_area = floor_polygon.area if floor_polygon else 0.0

        return RoomFootprint(
            room_id=room_id,
            walls=walls,
            openings=openings,
            floor_polygon=floor_polygon,
            floor_area_m2=float(floor_area),
            ceiling_height_m=float(ceiling_height),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _estimate_floor_z(self) -> float:
        """
        Estimate floor position along the UP axis as the 5th percentile
        of all point projections onto the world-UP vector.
        """
        proj = self._points @ self.up   # scalar projection along UP
        floor_z = float(np.percentile(proj, 5))
        log.debug("Estimated floor level (along UP) = {:.3f} m", floor_z)
        return floor_z

    def _estimate_ceiling_height(self, floor_z: float) -> float:
        """Ceiling height = span from floor to Nth percentile along UP."""
        pct: float = self.cfg.get("ceiling_sample_percentile", 95)
        proj = self._points @ self.up
        ceiling_z = float(np.percentile(proj, pct))
        height = ceiling_z - floor_z
        log.debug("Estimated ceiling height = {:.3f} m", height)
        return height

    def _slice_wall_band(self, floor_z: float) -> NDArray:
        """Keep only points in the vertical wall-band slice (projected onto UP)."""
        z_min_offset: float = self.cfg.get("wall_height_band_min", 0.1)
        z_max_offset: float = self.cfg.get("wall_height_band_max", 1.8)
        proj = self._points @ self.up
        mask = (proj >= floor_z + z_min_offset) & (proj <= floor_z + z_max_offset)
        # Project the wall-band points onto the horizontal plane (perpendicular to UP)
        # Build two orthogonal axes in the horizontal plane
        wall_pts_3d = self._points[mask]
        # Gram-Schmidt: find two vectors perpendicular to self.up
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, self.up)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        ax1 = ref - np.dot(ref, self.up) * self.up
        ax1 /= np.linalg.norm(ax1)
        ax2 = np.cross(self.up, ax1)
        ax2 /= np.linalg.norm(ax2)
        # Return 2D projection onto horizontal plane
        pts_2d = np.column_stack([wall_pts_3d @ ax1, wall_pts_3d @ ax2])
        self._ax1 = ax1   # store for later reconstruction if needed
        self._ax2 = ax2
        return pts_2d

    def _build_occupancy_grid(
        self, pts_2d: NDArray
    ) -> Tuple[NDArray, NDArray, float]:
        """
        Rasterise 2-D points into a binary occupancy grid.

        Returns:
            grid:       (H, W) uint8 binary image (255 = occupied).
            origin:     (2,) array — bottom-left corner in world metres.
            resolution: metres per pixel.
        """
        resolution: float = self.cfg.get("occupancy_grid_resolution", 0.02)

        x_min, y_min = pts_2d.min(axis=0) - resolution
        x_max, y_max = pts_2d.max(axis=0) + resolution

        W = int(np.ceil((x_max - x_min) / resolution))
        H = int(np.ceil((y_max - y_min) / resolution))

        grid = np.zeros((H, W), dtype=np.uint8)

        # Quantise points to grid indices
        ix = np.clip(((pts_2d[:, 0] - x_min) / resolution).astype(int), 0, W - 1)
        iy = np.clip(((pts_2d[:, 1] - y_min) / resolution).astype(int), 0, H - 1)
        grid[iy, ix] = 255

        # Morphological closing to fill small gaps in walls
        kernel_size = max(3, int(0.1 / resolution))   # ~10 cm kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        return grid, np.array([x_min, y_min]), resolution

    def _extract_contours(self, grid: NDArray) -> List[NDArray]:
        """Extract external contours from the occupancy grid."""
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        log.debug("Found {} raw contours", len(contours))
        return list(contours)

    def _fit_wall_segments(
        self, contours: List[NDArray], origin: NDArray, resolution: float
    ) -> List[WallSegment]:
        """
        Approximate each contour with a polygon and convert edges to WallSegments.
        """
        min_wall: float = self.cfg.get("min_wall_length", 0.3)
        walls: List[WallSegment] = []
        wall_idx = 0

        for contour in contours:
            # Douglas-Peucker polygon approximation
            epsilon = 0.02 / resolution   # ~2 cm precision in pixel units
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            pts_px = approx[:, 0, :]   # (N, 2) pixel coords

            # Convert to world metres
            pts_world = pts_px * resolution + origin  # broadcast

            for j in range(len(pts_world)):
                p1 = tuple(pts_world[j])
                p2 = tuple(pts_world[(j + 1) % len(pts_world)])
                length = float(np.linalg.norm(np.array(p2) - np.array(p1)))

                if length < min_wall:
                    continue

                walls.append(WallSegment(
                    wall_id=f"wall_{wall_idx:03d}",
                    start=p1,
                    end=p2,
                    length_m=length,
                    surface_id=f"surface_{wall_idx:03d}",
                ))
                wall_idx += 1

        return walls

    def _detect_openings(
        self,
        walls: List[WallSegment],
        pts_2d: NDArray,
        origin: NDArray,
        resolution: float,
    ) -> List[Opening]:
        """
        Detect gaps in wall segments as potential door/window openings.

        Strategy: for each wall segment, sample perpendicular slabs and find
        point-density gaps wider than min_opening_width.
        """
        min_w: float = self.cfg.get("min_opening_width", 0.5)
        max_w: float = self.cfg.get("max_opening_width", 3.0)
        openings: List[Opening] = []
        op_idx = 0

        for wall in walls:
            gaps = self._find_gaps_in_wall(wall, pts_2d, min_w, max_w)
            for gap_start, gap_end, gap_width in gaps:
                openings.append(Opening(
                    opening_id=f"opening_{op_idx:03d}",
                    wall_id=wall.wall_id,
                    start=gap_start,
                    end=gap_end,
                    width_m=gap_width,
                ))
                op_idx += 1

        return openings

    def _find_gaps_in_wall(
        self,
        wall: WallSegment,
        pts_2d: NDArray,
        min_w: float,
        max_w: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        Project nearby points onto a wall's axis and find density gaps.

        Returns list of (gap_start_world, gap_end_world, gap_width_m).
        """
        p1 = np.array(wall.start)
        p2 = np.array(wall.end)
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-6:
            return []
        wall_unit = wall_vec / wall_len
        wall_normal = np.array([-wall_unit[1], wall_unit[0]])

        # Keep only points within a narrow slab around the wall
        rel = pts_2d - p1
        proj_along = rel @ wall_unit    # projection along wall
        proj_perp  = rel @ wall_normal  # distance from wall

        slab_width = 0.15   # 15 cm either side
        mask = (
            (proj_along >= 0) & (proj_along <= wall_len) &
            (np.abs(proj_perp) <= slab_width)
        )
        proj_on_wall = proj_along[mask]

        if proj_on_wall.size < 5:
            return []

        # Histogram along wall axis (5 cm bins)
        bin_size = 0.05
        bins = np.arange(0, wall_len + bin_size, bin_size)
        hist, bin_edges = np.histogram(proj_on_wall, bins=bins)

        # Find consecutive empty bins → gaps
        density_threshold = 1
        is_empty = hist <= density_threshold
        gaps = []
        in_gap = False
        gap_start_pos = 0.0

        for k, empty in enumerate(is_empty):
            pos = bin_edges[k]
            if empty and not in_gap:
                in_gap = True
                gap_start_pos = pos
            elif not empty and in_gap:
                in_gap = False
                gap_width = pos - gap_start_pos
                if min_w <= gap_width <= max_w:
                    gs_world = tuple(p1 + gap_start_pos * wall_unit)
                    ge_world = tuple(p1 + pos * wall_unit)
                    gaps.append((gs_world, ge_world, gap_width))

        if in_gap:
            gap_width = wall_len - gap_start_pos
            if min_w <= gap_width <= max_w:
                gs_world = tuple(p1 + gap_start_pos * wall_unit)
                ge_world = tuple(p2)
                gaps.append((gs_world, ge_world, gap_width))

        return gaps

    def _build_room_polygon(self, walls: List[WallSegment]) -> Optional[Polygon]:
        """Build a Shapely Polygon from the outer wall segments."""
        if not walls:
            return None
        lines = [wall.as_linestring for wall in walls]
        try:
            merged = unary_union(lines)
            if merged.geom_type == "LineString":
                return Polygon(merged.coords)
            if merged.geom_type == "MultiLineString":
                # Attempt to build polygon from convex hull of all points
                all_pts = []
                for geom in merged.geoms:
                    all_pts.extend(geom.coords)
                from shapely.geometry import MultiPoint
                return MultiPoint(all_pts).convex_hull
        except Exception as exc:
            log.warning("Could not build room polygon: {}", exc)
        return None
