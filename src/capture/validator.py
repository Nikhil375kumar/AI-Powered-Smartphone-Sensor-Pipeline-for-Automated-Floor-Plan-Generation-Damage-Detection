"""
src/capture/validator.py
~~~~~~~~~~~~~~~~~~~~~~~~
Validates that a scan directory has the minimum required files for a given tier
before any processing begins — fail-fast with informative errors.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)


class InputTier(str, Enum):
    LIDAR = "lidar"
    VIDEO = "video"
    PHOTO = "photo"


class ValidationError(Exception):
    """Raised when a scan directory fails validation."""


_REQUIRED_FILES: dict[InputTier, List[str]] = {
    InputTier.LIDAR: [
        "odometry.csv",
        "camera_matrix.csv",
        "depth",          # directory
    ],
    InputTier.VIDEO: [
        "rgb.mp4",
    ],
    InputTier.PHOTO: [],  # Photos tier: validated per-room — no fixed filenames
}

_RECOMMENDED_FILES: dict[InputTier, List[str]] = {
    InputTier.LIDAR:  ["imu.csv", "confidence", "rgb.mp4"],
    InputTier.VIDEO:  ["odometry.csv", "camera_matrix.csv"],
    InputTier.PHOTO:  [],
}


def validate_scan_dir(
    scan_dir: str | Path,
    tier: InputTier,
) -> Tuple[bool, List[str]]:
    """
    Check that *scan_dir* has the mandatory files for *tier*.

    Args:
        scan_dir: Path to the scan folder.
        tier:     Input tier to validate against.

    Returns:
        (is_valid, warnings):
            is_valid — True iff all required files are present.
            warnings — List of human-readable warning strings.

    Raises:
        ValidationError: If any required file is missing.
    """
    scan_dir = Path(scan_dir).resolve()
    errors: List[str] = []
    warnings: List[str] = []

    if not scan_dir.is_dir():
        raise ValidationError(f"Scan directory does not exist: {scan_dir}")

    for item in _REQUIRED_FILES[tier]:
        candidate = scan_dir / item
        if not candidate.exists():
            errors.append(f"MISSING required {'directory' if '.' not in item else 'file'}: {item}")

    for item in _RECOMMENDED_FILES[tier]:
        candidate = scan_dir / item
        if not candidate.exists():
            warnings.append(f"Optional item not found (pipeline may be less accurate): {item}")

    # LiDAR-specific: check depth folder has at least one PNG
    if tier == InputTier.LIDAR:
        depth_dir = scan_dir / "depth"
        if depth_dir.is_dir():
            depth_frames = list(depth_dir.glob("*.png"))
            if not depth_frames:
                errors.append("depth/ directory exists but contains no PNG files")
            else:
                log.debug("depth/ contains {} frames", len(depth_frames))

    if errors:
        msg = "\n  ".join(errors)
        raise ValidationError(
            f"Scan directory '{scan_dir.name}' failed validation for tier '{tier.value}':\n  {msg}"
        )

    for w in warnings:
        log.warning("Validation warning: {}", w)

    log.info("Scan directory '{}' passed validation for tier '{}'", scan_dir.name, tier.value)
    return True, warnings
