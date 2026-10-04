"""
src/pipeline/base.py
~~~~~~~~~~~~~~~~~~~~
Abstract base class that every tier pipeline (LiDAR / Video / Photo) inherits.

Enforces the contract:
  - Every pipeline implements `run()` and returns a `PipelineOutput`.
  - Shared pre/post logic (validation, timing, export) lives here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.capture.validator import InputTier, validate_scan_dir
from src.config import settings
from src.output.schema import PipelineOutput
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


class BasePipeline(ABC):
    """
    Template-method base for all input-tier pipelines.

    Subclasses implement :meth:`_process` — base handles:
      - Validation
      - Timing
      - Logging setup
      - Output directory creation
    """

    tier: InputTier  # set by each subclass

    def __init__(
        self,
        scan_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        config_overrides: Optional[dict] = None,
    ) -> None:
        self.scan_dir = Path(scan_dir).resolve()
        self.scan_id = self.scan_dir.name

        self.output_dir = Path(
            output_dir or settings.output_dir
        ) / self.scan_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Allow caller to override any config key at runtime
        self._overrides = config_overrides or {}

        # Initialise logging (idempotent — re-calling is safe)
        setup_logging(
            log_level=settings.log_level,
            log_dir=settings.logging.get("log_dir", "./logs"),
        )

    # ── Template Method ───────────────────────────────────────────────────────

    def run(self) -> PipelineOutput:
        """
        Execute the full pipeline:
          1. Validate inputs
          2. Process (implemented by subclass)
          3. Return validated output

        Returns:
            :class:`PipelineOutput` conforming to the published schema.
        """
        log.info("=" * 60)
        log.info("Pipeline: {}  |  Tier: {}  |  Scan: {}",
                 self.__class__.__name__, self.tier.value, self.scan_id)
        log.info("=" * 60)

        validate_scan_dir(self.scan_dir, self.tier)

        t0 = time.perf_counter()
        result = self._process()
        elapsed = time.perf_counter() - t0

        result.processing_time_seconds = round(elapsed, 2)
        log.info("Pipeline complete in {:.1f}s", elapsed)
        return result

    @abstractmethod
    def _process(self) -> PipelineOutput:
        """Subclass implements all tier-specific logic here."""
        ...
