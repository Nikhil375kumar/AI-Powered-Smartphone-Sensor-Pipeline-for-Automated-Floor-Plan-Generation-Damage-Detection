"""
src/output/schema.py
~~~~~~~~~~~~~~~~~~~~
Pydantic v2 models that mirror config/output_schema.json.

Every field leaving the pipeline is validated here — no raw dicts are
exported. This ensures the JSON always conforms to the published schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ─── Atomic Building Blocks ────────────────────────────────────────────────────

class ConfidenceInterval(BaseModel):
    value: float
    lower: float
    upper: float
    unit: str

    @field_validator("lower", "upper")
    @classmethod
    def _check_bounds(cls, v: float, info: Any) -> float:
        return v  # bounds checked at construction time in confidence.py


class WallOut(BaseModel):
    wall_id: str
    length_m: float
    length_confidence: ConfidenceInterval
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    surface_id: str = ""


class OpeningOut(BaseModel):
    opening_id: str
    type: str = "unknown"
    width_m: float
    width_confidence: ConfidenceInterval
    height_m: Optional[float] = None
    wall_id: str = ""
    position_along_wall_m: float = 0.0


class DamageRegionOut(BaseModel):
    damage_id: str
    class_name: str = Field(alias="class")
    surface_id: str = ""
    area_m2: float = 0.0
    bounding_box_px: List[int] = Field(default_factory=list)
    confidence_score: float = 0.0
    frame_id: str = ""

    model_config = {"populate_by_name": True}


class ConcealedDamageFlagOut(BaseModel):
    flag_id: str
    class_name: str = Field(alias="class")
    trigger_damage_id: str = ""
    rule_fired: str = ""
    explanation: str = ""
    location_description: str = ""

    model_config = {"populate_by_name": True}


class ScopeLineItemOut(BaseModel):
    item_id: str
    description: str
    surface_id: str = ""
    damage_id: str = ""
    estimated_area_m2: float = 0.0


# ─── Room ─────────────────────────────────────────────────────────────────────

class RoomOut(BaseModel):
    room_id: str
    floor_area_m2: float
    floor_area_confidence: ConfidenceInterval
    ceiling_height_m: float
    ceiling_height_confidence: ConfidenceInterval
    walls: List[WallOut] = Field(default_factory=list)
    openings: List[OpeningOut] = Field(default_factory=list)
    damage_regions: List[DamageRegionOut] = Field(default_factory=list)
    concealed_damage_flags: List[ConcealedDamageFlagOut] = Field(default_factory=list)
    scope_line_items: List[ScopeLineItemOut] = Field(default_factory=list)


# ─── Stitched Plan ─────────────────────────────────────────────────────────────

class StitchedPlanOut(BaseModel):
    room_count: int
    total_floor_area_m2: float
    adjacency_graph: List[Tuple[str, str]] = Field(default_factory=list)
    rendered_plan_path: str = ""
    floor_plan_polygon_meters: List[Tuple[float, float]] = Field(default_factory=list)


class ConfidenceSummaryOut(BaseModel):
    tier: str
    wall_length_tolerance_pct: float
    ceiling_height_tolerance_cm: float
    opening_width_tolerance_cm: float


# ─── Top-Level Output ─────────────────────────────────────────────────────────

class PipelineOutput(BaseModel):
    """Complete output contract for one capture — matches output_schema.json v1."""

    scan_id: str
    schema_version: str = "1.0.0"
    tier: str
    capture_route: str = "route2_stock_protocol"
    pipeline_version: str = "0.1.0"
    processing_time_seconds: float = 0.0
    rooms: List[RoomOut] = Field(default_factory=list)
    stitched_plan: StitchedPlanOut
    confidence_summary: ConfidenceSummaryOut

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string using Pydantic's model_dump_json."""
        return self.model_dump_json(indent=indent, by_alias=True)
