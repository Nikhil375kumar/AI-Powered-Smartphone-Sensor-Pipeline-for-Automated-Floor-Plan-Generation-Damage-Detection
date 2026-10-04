"""
src/output/exporter.py
~~~~~~~~~~~~~~~~~~~~~~
Assembles all pipeline results into a :class:`PipelineOutput` and writes:
  - <output_dir>/<scan_id>/output.json   (validated against schema)
  - <output_dir>/<scan_id>/floor_plan.png

Also generates scope line items from damage detections.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings
from src.damage.detector import Detection
from src.damage.rules import ConcealedFlag
from src.floor_plan.stitcher import StitchedPlan
from src.floor_plan.wall_extractor import Opening, RoomFootprint, WallSegment
from src.output.confidence import ConfidenceEstimator
from src.output.schema import (
    ConcealedDamageFlagOut,
    DamageRegionOut,
    OpeningOut,
    PipelineOutput,
    RoomOut,
    ScopeLineItemOut,
    StitchedPlanOut,
    WallOut,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class ResultExporter:
    """
    Converts internal pipeline objects → validated :class:`PipelineOutput`
    and persists JSON + rendered plan to disk.

    Args:
        scan_id: Unique scan identifier (folder name).
        tier:    Input tier ("lidar" | "video" | "photo").
        cfg:     ``settings.output`` config dict.
    """

    def __init__(
        self,
        scan_id: str,
        tier: str,
        cfg: Optional[Dict] = None,
    ) -> None:
        self.scan_id = scan_id
        self.tier = tier
        self.cfg = cfg or settings.output
        self.ci = ConfidenceEstimator(tier=tier)

        # Output directory: <base_dir>/<scan_id>/
        base_dir = Path(self.cfg.get("base_dir", "./outputs"))
        self.out_dir = base_dir / scan_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def export(
        self,
        stitched_plan: StitchedPlan,
        detections: List[Detection],
        concealed_flags: List[ConcealedFlag],
        rendered_plan_path: Optional[Path] = None,
        processing_time: float = 0.0,
    ) -> PipelineOutput:
        """
        Assemble and export the full pipeline output.

        Args:
            stitched_plan:      Stitched floor plan from MultiRoomStitcher.
            detections:         Raw damage detections from DamageDetector.
            concealed_flags:    Fired concealed-damage flags.
            rendered_plan_path: Path to rendered floor plan PNG (if rendered).
            processing_time:    Total pipeline elapsed time in seconds.

        Returns:
            Validated :class:`PipelineOutput` (also written to disk).
        """
        rooms_out = self._build_rooms(
            stitched_plan.rooms, detections, concealed_flags
        )

        # Build polygon coordinates for stitched plan
        polygon_coords: List = []
        if stitched_plan.merged_polygon is not None:
            try:
                geom = stitched_plan.merged_polygon
                if hasattr(geom, "geoms"):
                    geom = max(geom.geoms, key=lambda g: g.area)
                coords = list(geom.exterior.coords)
                polygon_coords = [(round(x, 4), round(y, 4)) for x, y in coords]
            except Exception:
                pass

        stitched_out = StitchedPlanOut(
            room_count=len(stitched_plan.rooms),
            total_floor_area_m2=round(stitched_plan.total_floor_area_m2, 4),
            adjacency_graph=list(stitched_plan.adjacency),
            rendered_plan_path=str(rendered_plan_path) if rendered_plan_path else "",
            floor_plan_polygon_meters=polygon_coords,
        )

        output = PipelineOutput(
            scan_id=self.scan_id,
            tier=self.tier,
            processing_time_seconds=round(processing_time, 2),
            rooms=rooms_out,
            stitched_plan=stitched_out,
            confidence_summary=self.ci.summary(),
        )

        self._write_json(output)
        log.info("Export complete → {}", self.out_dir)
        return output

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_rooms(
        self,
        rooms: List[RoomFootprint],
        detections: List[Detection],
        concealed_flags: List[ConcealedFlag],
    ) -> List[RoomOut]:
        rooms_out: List[RoomOut] = []
        det_map = {d.frame_id: d for d in detections}  # simplified lookup

        for room in rooms:
            walls_out = self._build_walls(room.walls)
            openings_out = self._build_openings(room.openings)
            damage_out = self._build_damage_regions(detections)
            flags_out = self._build_concealed_flags(concealed_flags)
            scope_out = self._build_scope_items(damage_out)

            rooms_out.append(RoomOut(
                room_id=room.room_id,
                floor_area_m2=round(room.floor_area_m2, 4),
                floor_area_confidence=self.ci.floor_area(room.floor_area_m2),
                ceiling_height_m=round(room.ceiling_height_m, 4),
                ceiling_height_confidence=self.ci.ceiling_height(room.ceiling_height_m),
                walls=walls_out,
                openings=openings_out,
                damage_regions=damage_out,
                concealed_damage_flags=flags_out,
                scope_line_items=scope_out,
            ))

        return rooms_out

    def _build_walls(self, walls: List[WallSegment]) -> List[WallOut]:
        return [
            WallOut(
                wall_id=w.wall_id,
                length_m=round(w.length_m, 4),
                length_confidence=self.ci.wall_length(w.length_m),
                start_point=(round(w.start[0], 4), round(w.start[1], 4)),
                end_point=(round(w.end[0], 4), round(w.end[1], 4)),
                surface_id=w.surface_id,
            )
            for w in walls
        ]

    def _build_openings(self, openings: List[Opening]) -> List[OpeningOut]:
        return [
            OpeningOut(
                opening_id=o.opening_id,
                type=o.opening_type,
                width_m=round(o.width_m, 4),
                width_confidence=self.ci.opening_width(o.width_m),
                wall_id=o.wall_id,
                position_along_wall_m=round(o.position_along_wall_m, 4),
            )
            for o in openings
        ]

    def _build_damage_regions(self, detections: List[Detection]) -> List[DamageRegionOut]:
        out: List[DamageRegionOut] = []
        for i, det in enumerate(detections):
            out.append(DamageRegionOut.model_validate({
                "damage_id": f"damage_{i:04d}",
                "class": det.class_name,
                "area_m2": 0.0,   # physical area computed if depth available
                "bounding_box_px": det.bbox_xyxy,
                "confidence_score": round(det.confidence, 4),
                "frame_id": det.frame_id,
            }))
        return out

    def _build_concealed_flags(self, flags: List[ConcealedFlag]) -> List[ConcealedDamageFlagOut]:
        out: List[ConcealedDamageFlagOut] = []
        for flag in flags:
            out.append(ConcealedDamageFlagOut.model_validate({
                "flag_id": flag.flag_id,
                "class": flag.class_name,
                "trigger_damage_id": flag.trigger_detection.frame_id,
                "rule_fired": flag.rule_fired,
                "explanation": flag.explanation,
                "location_description": flag.location_description,
            }))
        return out

    def _build_scope_items(self, damage_regions: List[DamageRegionOut]) -> List[ScopeLineItemOut]:
        """Generate one repair scope line item per damage region."""
        scope_descriptions = {
            "crack": "Crack repair and surface filling",
            "water_stain": "Stain treatment and surface re-coat",
            "mold": "Mold remediation and anti-fungal treatment",
            "peeling_paint": "Surface preparation and repainting",
            "structural_damage": "Structural assessment and repair",
            "efflorescence": "Salt removal and waterproofing membrane application",
        }
        items: List[ScopeLineItemOut] = []
        for i, region in enumerate(damage_regions):
            desc = scope_descriptions.get(
                region.class_name, f"Inspect and repair: {region.class_name}"
            )
            items.append(ScopeLineItemOut(
                item_id=f"scope_{i:04d}",
                description=desc,
                surface_id=region.surface_id,
                damage_id=region.damage_id,
                estimated_area_m2=round(region.area_m2, 4),
            ))
        return items

    def _write_json(self, output: PipelineOutput) -> None:
        json_path = self.out_dir / "output.json"
        json_path.write_text(output.to_json(), encoding="utf-8")
        log.info("JSON output → {}", json_path)
