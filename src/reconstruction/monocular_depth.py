"""
src/reconstruction/monocular_depth.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wrapper around MiDaS / Depth-Anything v2 for monocular depth estimation.

Used by the Video and Photo pipelines when LiDAR depth is not available.
Downloads model weights on first run via torch.hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class MonocularDepthEstimator:
    """
    Estimates a dense depth map from a single RGB image using MiDaS.

    The output is a relative depth map in normalised [0, 1] scale.
    For metric depth (metres), a scale-shift alignment against known
    structure is applied downstream.

    Args:
        model_type: MiDaS model variant. Options: "DPT_Large", "DPT_Hybrid",
                    "MiDaS_small". Smaller = faster, less accurate.
    """

    def __init__(self, model_type: str = "MiDaS_small") -> None:
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._model_type = model_type
        self._load()

    def _load(self) -> None:
        """Load MiDaS model from torch hub. Downloads weights on first call."""
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading MiDaS ({}) on {}", self._model_type, self._device)

            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self._model_type,
                trust_repo=True,
            )
            self._model.to(self._device)
            self._model.eval()

            transforms_hub = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            if self._model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = transforms_hub.dpt_transform
            else:
                self._transform = transforms_hub.small_transform

            log.info("MiDaS loaded successfully.")

        except Exception as exc:
            log.error("Failed to load MiDaS: {}. Monocular depth unavailable.", exc)
            self._model = None

    def predict(self, bgr_image: NDArray) -> Optional[NDArray]:
        """
        Predict a depth map for a single BGR image.

        Args:
            bgr_image: (H, W, 3) uint8 BGR image.

        Returns:
            (H, W) float32 depth map in relative scale (0=near, 1=far),
            or None if model is unavailable.
        """
        if self._model is None:
            return None

        import torch

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)

        return depth
