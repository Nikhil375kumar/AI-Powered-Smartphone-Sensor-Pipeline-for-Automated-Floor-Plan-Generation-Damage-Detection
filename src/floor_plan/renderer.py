"""
src/floor_plan/renderer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Renders a :class:`StitchedPlan` into a dimensioned floor plan PNG image.

Produces a plan a homeowner would recognise (Magicplan / Poly.cam style):
  - Rooms drawn as filled polygons
  - Walls drawn as thick lines with dimension labels
  - Openings shown as gaps in walls
  - Damage regions highlighted
  - North arrow, scale bar, legend
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/pipeline use

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon

from src.config import settings
from src.floor_plan.stitcher import StitchedPlan
from src.floor_plan.wall_extractor import Opening, RoomFootprint, WallSegment
from src.utils.logger import get_logger

log = get_logger(__name__)


class FloorPlanRenderer:
    """
    Renders a stitched floor plan to a PNG file.

    Args:
        cfg:        ``settings.output`` config dict.
        floor_cfg:  ``settings.floor_plan`` config dict.
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        floor_cfg: Optional[Dict] = None,
    ) -> None:
        self.cfg = cfg or settings.output
        self.floor_cfg = floor_cfg or settings.floor_plan

    def render(
        self,
        plan: StitchedPlan,
        output_path: str | Path,
        title: str = "Floor Plan",
    ) -> Path:
        """
        Render the stitched plan and save to *output_path*.

        Args:
            plan:        Stitched multi-room plan.
            output_path: Where to save the PNG.
            title:       Figure title (scan ID, tier etc.).

        Returns:
            Resolved path to the saved PNG.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dpi: int = self.cfg.get("render_dpi", 150)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)
        ax.set_aspect("equal")
        ax.set_facecolor("#F7F4EF")
        fig.patch.set_facecolor("#F7F4EF")

        for room in plan.rooms:
            self._draw_room(ax, room)

        self._draw_legend(ax)
        self._draw_scale_bar(ax, plan)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        ax.set_xlabel("X (metres)", fontsize=9)
        ax.set_ylabel("Y (metres)", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#aaaaaa")

        plt.tight_layout()
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        log.info("Floor plan rendered → {}", output_path)
        return output_path.resolve()

    # ── Private ───────────────────────────────────────────────────────────────

    def _draw_room(self, ax: plt.Axes, room: RoomFootprint) -> None:
        """Draw a single room: polygon fill, walls, openings, labels."""
        floor_color = [c / 255 for c in self.cfg.get("render_floor_color", [245, 240, 230])]
        wall_color  = [c / 255 for c in self.cfg.get("render_wall_color", [30, 30, 30])]
        open_color  = [c / 255 for c in self.cfg.get("render_opening_color", [100, 180, 255])]

        # Floor polygon
        if room.floor_polygon is not None and room.floor_polygon.is_valid:
            xs, ys = room.floor_polygon.exterior.xy
            ax.fill(xs, ys, color=floor_color, alpha=0.7, zorder=1)
            ax.plot(xs, ys, color=wall_color, linewidth=2.0, zorder=3)

        # Wall segments with dimension labels
        for wall in room.walls:
            x1, y1 = wall.start
            x2, y2 = wall.end
            ax.plot([x1, x2], [y1, y2], color=wall_color, linewidth=3.0, zorder=4)

            # Dimension label at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            label = f"{wall.length_m:.2f}m"
            ax.annotate(
                label,
                (mx, my),
                fontsize=6,
                ha="center",
                va="center",
                color="#333333",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Openings (gaps in walls)
        for opening in room.openings:
            x1, y1 = opening.start
            x2, y2 = opening.end
            ax.plot([x1, x2], [y1, y2], color=open_color, linewidth=4.0, zorder=6, solid_capstyle="butt")
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.annotate(
                f"{opening.width_m:.2f}m",
                (mx, my),
                fontsize=5.5,
                ha="center",
                color="#1155cc",
                zorder=7,
            )

        # Room label (centroid)
        if room.floor_polygon is not None:
            cx, cy = room.floor_polygon.centroid.x, room.floor_polygon.centroid.y
        elif room.walls:
            cx = np.mean([w.start[0] for w in room.walls])
            cy = np.mean([w.start[1] for w in room.walls])
        else:
            return

        area_label = f"{room.room_id}\n{room.floor_area_m2:.1f} m²\nh={room.ceiling_height_m:.2f} m"
        ax.text(
            cx, cy, area_label,
            fontsize=7, ha="center", va="center",
            color="#111111", fontweight="bold", zorder=8,
        )

    def _draw_legend(self, ax: plt.Axes) -> None:
        """Add a colour legend to the plot."""
        floor_color = [c / 255 for c in self.cfg.get("render_floor_color", [245, 240, 230])]
        open_color  = [c / 255 for c in self.cfg.get("render_opening_color", [100, 180, 255])]
        damage_color = [c / 255 for c in self.cfg.get("render_damage_color", [220, 60, 60])]

        patches = [
            mpatches.Patch(color=floor_color, label="Floor area"),
            mpatches.Patch(color=open_color, label="Opening (door/window)"),
            mpatches.Patch(color=damage_color, label="Damage region"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)

    def _draw_scale_bar(self, ax: plt.Axes, plan: StitchedPlan) -> None:
        """Draw a 1-metre scale bar at the bottom-left."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.04
        ax.plot([x0, x0 + 1.0], [y0, y0], "k-", linewidth=3, zorder=10)
        ax.text(x0 + 0.5, y0 + 0.1, "1 m", ha="center", fontsize=7, zorder=10)
