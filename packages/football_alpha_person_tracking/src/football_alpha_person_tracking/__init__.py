"""Session-local person tracking contracts and adapters for football-alpha."""

from football_alpha_person_tracking.byte_track import ByteTrackPersonTracker
from football_alpha_person_tracking.contracts import (
    FramePersonTracks,
    PersonTrack,
    PersonTracker,
)

__version__ = "0.1.0"

__all__ = [
    "ByteTrackPersonTracker",
    "FramePersonTracks",
    "PersonTrack",
    "PersonTracker",
]
