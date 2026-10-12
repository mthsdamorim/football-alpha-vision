"""Project-owned contracts for image-space person detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Protocol

from football_alpha_video import DecodedFrame


@dataclass(frozen=True, slots=True)
class ImageBoundingBox:
    """One bounding box in original-image continuous pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(not isfinite(value) or value < 0.0 for value in coordinates):
            raise ValueError("bounding box requires non-negative finite coordinates")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box must have positive width and height")


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """One visible human candidate in image coordinates."""

    bbox: ImageBoundingBox
    confidence: float

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class FramePersonDetections:
    """Normalized person detections for one decoded frame."""

    frame_index: int
    pts: int
    time_base: Fraction
    image_width: int
    image_height: int
    detections: tuple[PersonDetection, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(self.detections))
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("source image dimensions must be positive")
        if any(
            detection.bbox.x_max > self.image_width
            or detection.bbox.y_max > self.image_height
            for detection in self.detections
        ):
            raise ValueError("person detection is outside the source image")


class PersonDetector(Protocol):
    """Detect visible human candidates in explicit decoded-frame batches."""

    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        """Return exactly one ordered detection result per input frame."""
        ...
