"""
src/config.py
~~~~~~~~~~~~~
Central configuration loader.

All config values come from:
  1. config/default.yaml  (base defaults)
  2. Environment variables (override via .env or shell)

Zero hardcoding in any other module — import `settings` from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

# Load .env file if it exists (silently skip if not present)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _env_override(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to the config dict.

    Convention: SECTION__KEY=value  → config['section']['key'] = value
    Example:    PLANE_DETECTION__DISTANCE_THRESHOLD=0.03
    """
    for env_key, env_val in os.environ.items():
        if "__" in env_key:
            parts = env_key.lower().split("__")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            # Attempt numeric coercion
            try:
                target[parts[-1]] = float(env_val) if "." in env_val else int(env_val)
            except (ValueError, TypeError):
                target[parts[-1]] = env_val
    return config


class _Config:
    """
    Singleton configuration object.

    Access any config value via attribute or dict-style access:
        settings.point_cloud["voxel_size"]
        settings.damage["confidence_threshold"]
    """

    def __init__(self) -> None:
        raw = _load_yaml(_CONFIG_PATH)
        raw = _env_override(raw)
        self._data: Dict[str, Any] = raw

        # Top-level shortcuts
        self.data: Dict[str, Any] = raw.get("data", {})
        self.camera: Dict[str, Any] = raw.get("camera", {})
        self.point_cloud: Dict[str, Any] = raw.get("point_cloud", {})
        self.plane_detection: Dict[str, Any] = raw.get("plane_detection", {})
        self.floor_plan: Dict[str, Any] = raw.get("floor_plan", {})
        self.stitching: Dict[str, Any] = raw.get("stitching", {})
        self.damage: Dict[str, Any] = raw.get("damage", {})
        self.concealed_damage_rules: List[Dict[str, Any]] = raw.get(
            "concealed_damage_rules", []
        )
        self.confidence: Dict[str, Any] = raw.get("confidence", {})
        self.output: Dict[str, Any] = raw.get("output", {})
        self.logging: Dict[str, Any] = raw.get("logging", {})

        # Environment-level overrides (readable shorthand)
        self.app_env: str = os.getenv("APP_ENV", "development")
        self.output_dir: Path = Path(
            os.getenv("OUTPUT_DIR", raw.get("output", {}).get("base_dir", "./outputs"))
        )
        self.log_level: str = os.getenv(
            "LOG_LEVEL", raw.get("logging", {}).get("level", "INFO")
        )

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"<Config env={self.app_env} output_dir={self.output_dir}>"


# ─── Singleton ────────────────────────────────────────────────────────────────
settings = _Config()
