"""Project-owned contracts for session-local person tracking."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Protocol

from football_alpha_player_detection import (
    FramePersonDetections,
    ImageBoundingBox,
)


@dataclass(frozen=True, slots=True)
class PersonTrack:
    """One currently observed person detection with a session-local ID."""

    track_id: int
    bbox: ImageBoundingBox
    detection_confidence: float

    def __post_init__(self) -> None:
        if self.track_id <= 0:
            raise ValueError("track_id must be positive")
        if not isfinite(self.detection_confidence) or not (
            0.0 <= self.detection_confidence <= 1.0
        ):
            raise ValueError("detection_confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class FramePersonTracks:
    """Current associated person observations for one decoded frame."""

    frame_index: int
    pts: int
    time_base: Fraction
    image_width: int
    image_height: int
    tracks: tuple[PersonTrack, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", tuple(self.tracks))
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("source image dimensions must be positive")
        if any(
            track.bbox.x_max > self.image_width or track.bbox.y_max > self.image_height
            for track in self.tracks
        ):
            raise ValueError("person track is outside the source image")


class PersonTracker(Protocol):
    """Associate ordered person detections within one in-memory session."""

    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        """Associate observations from the current frame."""
        ...

    def reset(self) -> None:
        """Start a new session and clear association and ordering state."""
        ...
