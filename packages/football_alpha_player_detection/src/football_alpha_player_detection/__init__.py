"""Person detection contracts and adapters for football-alpha."""

from football_alpha_player_detection.contracts import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
    PersonDetector,
)
from football_alpha_player_detection.rf_detr import RfDetrPersonDetector

__version__ = "0.1.0"

__all__ = [
    "FramePersonDetections",
    "ImageBoundingBox",
    "PersonDetection",
    "PersonDetector",
    "RfDetrPersonDetector",
]
