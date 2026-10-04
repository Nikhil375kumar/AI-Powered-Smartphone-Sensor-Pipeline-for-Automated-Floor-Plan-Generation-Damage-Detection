"""
src/damage/rules.py
~~~~~~~~~~~~~~~~~~~~
Rule-based concealed damage flagging.

Each rule is defined in config/default.yaml under concealed_damage_rules.
No rule logic lives in this file — only the evaluation engine.

A concealed flag fires when a visible damage detection occurs near a
structural risk zone (e.g. water stain near a plumbing route).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.config import settings
from src.damage.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ConcealedFlag:
    """A fired concealed-damage rule."""
    flag_id: str
    class_name: str          # e.g. "possible_leak_behind_wall"
    trigger_detection: Detection
    rule_fired: str          # human-readable rule description
    explanation: str
    location_description: str = ""


class ConcealedDamageEngine:
    """
    Evaluates visible damage detections against the rule set in config
    and emits :class:`ConcealedFlag` objects.

    Args:
        rules: List of rule dicts from ``settings.concealed_damage_rules``.
    """

    def __init__(self, rules: Optional[List[Dict]] = None) -> None:
        self.rules = rules if rules is not None else settings.concealed_damage_rules
        log.info("Loaded {} concealed damage rules", len(self.rules))

    def evaluate(self, detections: List[Detection]) -> List[ConcealedFlag]:
        """
        Evaluate all detections against every rule.

        Args:
            detections: Flat list of :class:`Detection` from the damage detector.

        Returns:
            List of fired :class:`ConcealedFlag` objects.
        """
        flags: List[ConcealedFlag] = []
        flag_idx = 0

        for det in detections:
            for rule in self.rules:
                trigger_cls: str = rule.get("trigger_class", "")
                flag_cls: str    = rule.get("flag_class", "")
                explanation: str = rule.get("explanation", "")

                if det.class_name.lower() != trigger_cls.lower():
                    continue

                # Rule fired — create a concealed flag
                flags.append(ConcealedFlag(
                    flag_id=f"flag_{flag_idx:04d}",
                    class_name=flag_cls,
                    trigger_detection=det,
                    rule_fired=(
                        f"Visible '{trigger_cls}' detected in frame {det.frame_id} "
                        f"→ concealed risk: '{flag_cls}'"
                    ),
                    explanation=explanation,
                    location_description=f"Near bbox {det.bbox_xyxy} in frame {det.frame_id}",
                ))
                flag_idx += 1
                log.debug("Rule fired: {} → {}", trigger_cls, flag_cls)

        log.info("Concealed damage engine: {} flags fired from {} detections",
                 len(flags), len(detections))
        return flags
