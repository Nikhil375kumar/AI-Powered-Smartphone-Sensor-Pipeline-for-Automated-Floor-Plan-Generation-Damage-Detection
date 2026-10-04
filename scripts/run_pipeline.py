"""
scripts/run_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~
ONE-COMMAND entry point for the indoor mapping pipeline.

Usage examples:
  # LiDAR tier (sample data):
  python scripts/run_pipeline.py --scan-dir Dataset/single_scan_floor_only/1a8384c3f6 --tier lidar

  # Video tier:
  python scripts/run_pipeline.py --scan-dir Dataset/my_scan --tier video

  # Photo tier:
  python scripts/run_pipeline.py --scan-dir Dataset/my_house_photos --tier photo

  # Disable drift correction (ablation):
  python scripts/run_pipeline.py --scan-dir Dataset/single_scan_floor_only/1a8384c3f6 --tier lidar --no-drift-correction

  # Custom output directory:
  python scripts/run_pipeline.py --scan-dir Dataset/scan1 --tier lidar --output-dir ./results/scan1
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make sure src/ is importable when run from repo root ─────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import LiDARPipeline, PhotoPipeline, VideoPipeline
from src.utils.logger import get_logger, setup_logging

console = Console()
app = typer.Typer(
    name="indoor-mapping-pipeline",
    help="Automated indoor mapping: photos/video/LiDAR → floor plan + damage report.",
    add_completion=False,
)
log = get_logger(__name__)


@app.command()
def run(
    scan_dir: Path = typer.Option(
        ...,
        "--scan-dir", "-s",
        help="Path to the scan directory (contains depth/, rgb.mp4, odometry.csv, etc.)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    tier: str = typer.Option(
        ...,
        "--tier", "-t",
        help="Input tier: 'lidar' | 'video' | 'photo'",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Root output directory. Defaults to ./outputs/<scan_id>/",
    ),
    no_drift_correction: bool = typer.Option(
        False,
        "--no-drift-correction",
        help="Disable ICP drift correction (use for ablation comparison).",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG | INFO | WARNING | ERROR",
    ),
) -> None:
    """
    Run the indoor mapping pipeline on a single scan directory.

    Produces:
      <output_dir>/<scan_id>/output.json    — validated schema output
      <output_dir>/<scan_id>/floor_plan.png — rendered floor plan
    """
    # Logging setup (before any pipeline code runs)
    setup_logging(log_level=log_level, log_dir="./logs")

    # Show startup banner
    console.print(Panel.fit(
        f"[bold cyan]Indoor Mapping Pipeline[/bold cyan]\n"
        f"Scan: [yellow]{scan_dir.name}[/yellow]  "
        f"Tier: [green]{tier.upper()}[/green]",
        border_style="cyan",
    ))

    # ── Select and run the correct pipeline ───────────────────────────────────
    tier_lower = tier.strip().lower()

    try:
        if tier_lower == "lidar":
            pipeline = LiDARPipeline(
                scan_dir=scan_dir,
                output_dir=output_dir,
                enable_drift_correction=not no_drift_correction,
            )
        elif tier_lower == "video":
            pipeline = VideoPipeline(
                scan_dir=scan_dir,
                output_dir=output_dir,
            )
        elif tier_lower == "photo":
            pipeline = PhotoPipeline(
                scan_dir=scan_dir,
                output_dir=output_dir,
            )
        else:
            console.print(f"[red]Unknown tier '{tier}'. Choose: lidar | video | photo[/red]")
            raise typer.Exit(code=1)

        result = pipeline.run()

    except Exception as exc:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}")
        log.exception("Pipeline error")
        raise typer.Exit(code=1)

    # ── Print results summary ─────────────────────────────────────────────────
    _print_summary(result)
    console.print("\n[bold green]✓ Pipeline completed successfully.[/bold green]")


def _print_summary(result) -> None:
    """Print a rich table summary of the pipeline output."""
    table = Table(title="Pipeline Output Summary", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Scan ID", result.scan_id)
    table.add_row("Tier", result.tier.upper())
    table.add_row("Processing Time", f"{result.processing_time_seconds:.1f}s")
    table.add_row("Rooms", str(len(result.rooms)))

    if result.rooms:
        for room in result.rooms:
            table.add_row(f"  {room.room_id} — Area", f"{room.floor_area_m2:.2f} m²")
            table.add_row(f"  {room.room_id} — Ceiling", f"{room.ceiling_height_m:.2f} m")
            table.add_row(f"  {room.room_id} — Walls", str(len(room.walls)))
            table.add_row(f"  {room.room_id} — Openings", str(len(room.openings)))
            table.add_row(f"  {room.room_id} — Damage regions", str(len(room.damage_regions)))

    table.add_row("Total Floor Area", f"{result.stitched_plan.total_floor_area_m2:.2f} m²")
    table.add_row("Rendered Plan", result.stitched_plan.rendered_plan_path)

    console.print(table)


# Re-export for `python -m scripts.run_pipeline`
from typing import Optional

if __name__ == "__main__":
    app()
