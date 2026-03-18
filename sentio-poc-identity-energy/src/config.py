from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DetectorConfig:
    skip_n: int = 4
    max_frames: Optional[int] = None
    min_upscale_size: int = 112
    min_face_confidence: float = 0.80


@dataclass
class MatcherConfig:
    similarity_threshold: float = 0.50
    embedding_min_size: int = 112
    window_size: int = 5
    max_missed: int = 3
    decay: float = 0.88
    boost: float = 0.12
    min_track_confidence: float = 0.15
    association_min_score: float = 0.33
    iou_weight: float = 0.45
    center_weight: float = 0.25
    embedding_weight: float = 0.30


@dataclass
class EnergyConfig:
    brightness_weight: float = 0.35
    eye_weight: float = 0.30
    motion_weight: float = 0.35
    eye_fallback: float = 50.0


@dataclass
class RuntimeConfig:
    realtime: bool = False
    no_display: bool = False
    write_demo_video: bool = True
    port: int = 8000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(target: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(config_path: Optional[str] = None) -> AppConfig:
    cfg = AppConfig()
    merged = cfg.to_dict()

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix.lower() not in {".json"}:
            raise ValueError("Only JSON config override is supported for now. Use .json")

        with path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
        merged = _deep_update(merged, overrides)

    if os.getenv("LOG_LEVEL"):
        merged["logging"]["level"] = str(os.getenv("LOG_LEVEL")).upper()
    if os.getenv("PORT"):
        try:
            merged["runtime"]["port"] = int(os.getenv("PORT", "8000"))
        except ValueError:
            pass

    return AppConfig(
        detector=DetectorConfig(**merged["detector"]),
        matcher=MatcherConfig(**merged["matcher"]),
        energy=EnergyConfig(**merged["energy"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        logging=LoggingConfig(**merged["logging"]),
    )
