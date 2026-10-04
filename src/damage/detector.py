"""
src/damage/detector.py
~~~~~~~~~~~~~~~~~~~~~~
Damage detection using YOLOv8 on RGB video frames or extracted images.

Design:
  - Wraps the ultralytics YOLO model behind a thin interface.
  - All thresholds/class names come from config — never hardcoded.
  - Falls back gracefully if model weights are not found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import settings
from src.utils.groq_client import get_groq_client
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    """A single damage detection result."""
    class_name: str
    confidence: float
    bbox_xyxy: List[int]     # [x1, y1, x2, y2] in pixels
    frame_id: str
    area_px2: float = 0.0

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        self.area_px2 = float((x2 - x1) * (y2 - y1))


class DamageDetector:
    """
    Runs YOLOv8 damage detection on RGB frames.

    Args:
        cfg: ``settings.damage`` config dict.
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        self.cfg = cfg or settings.damage
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model — use fine-tuned weights if available, else base model."""
        model_path = Path(self.cfg.get("model_path", "./models/damage_detector.pt"))
        model_name = self.cfg.get("model_name", "yolov8n.pt")

        try:
            from ultralytics import YOLO  # type: ignore

            if model_path.is_file():
                log.info("Loading fine-tuned damage detector from: {}", model_path)
                self._model = YOLO(str(model_path))
            else:
                log.warning(
                    "Fine-tuned model not found at '{}'. "
                    "Falling back to base YOLO '{}' — damage class names may not match.",
                    model_path, model_name,
                )
                self._model = YOLO(model_name)  # downloads on first run

        except ImportError:
            log.error(
                "ultralytics not installed — damage detection disabled. "
                "Run: pip install ultralytics"
            )
            self._model = None

    def detect_in_frame(
        self,
        bgr_frame: NDArray,
        frame_id: str,
    ) -> List[Detection]:
        """
        Run damage detection on a single BGR frame.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR image (OpenCV format).
            frame_id:  Identifier string for this frame.

        Returns:
            List of :class:`Detection` objects.
        """
        if self._model is None:
            return []

        conf_thresh: float = self.cfg.get("confidence_threshold", 0.45)
        iou_thresh: float  = self.cfg.get("iou_threshold", 0.5)
        groq_refine_below: float = conf_thresh + 0.1  # refine detections below this conf

        results = self._model.predict(
            source=bgr_frame,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

        groq = get_groq_client()
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                cls_name = result.names.get(cls_idx, f"class_{cls_idx}")
                confidence = float(box.conf.item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                # If Groq is available and confidence is borderline, ask LLM to confirm
                if groq.available and confidence < groq_refine_below:
                    refined = groq.classify_damage(
                        detected_class=cls_name,
                        context=f"Detected in frame {frame_id}, confidence={confidence:.2f}",
                    )
                    if refined and refined != cls_name:
                        log.debug(
                            "Groq refined '{}' → '{}' (conf={:.2f})",
                            cls_name, refined, confidence,
                        )
                        cls_name = refined

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=confidence,
                    bbox_xyxy=xyxy,
                    frame_id=frame_id,
                ))

        return detections

    def detect_in_video(
        self,
        video_path: str | Path,
        frame_stride: Optional[int] = None,
    ) -> List[Detection]:
        """
        Run detection across all frames of an RGB video.

        Args:
            video_path:   Path to the RGB video file.
            frame_stride: Sample every Nth frame. Defaults to config value.

        Returns:
            Aggregated list of detections from all sampled frames.
        """
        stride: int = frame_stride or self.cfg.get("frame_sample_rate", 10)
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"RGB video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        all_detections: List[Detection] = []
        idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    dets = self.detect_in_frame(frame, frame_id=f"video_frame_{idx:06d}")
                    all_detections.extend(dets)
                idx += 1
        finally:
            cap.release()

        log.info(
            "Damage detection complete: {} detections across {} sampled frames",
            len(all_detections), idx // stride,
        )
        return all_detections
