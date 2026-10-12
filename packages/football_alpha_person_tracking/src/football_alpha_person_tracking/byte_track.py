"""ByteTrack adapter for project-owned person tracking contracts."""

from __future__ import annotations

from fractions import Fraction
from importlib.metadata import version
from math import isfinite

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from football_alpha_person_tracking.contracts import FramePersonTracks, PersonTrack
from football_alpha_player_detection import FramePersonDetections

_DETECTION_INDEX_KEY = "football_alpha_detection_index"
_LOST_TRACK_BUFFER = 30
_TRACK_ACTIVATION_THRESHOLD = 0.7
_MINIMUM_CONSECUTIVE_FRAMES = 2
_MINIMUM_IOU_THRESHOLD = 0.1
_HIGH_CONF_DET_THRESHOLD = 0.6


class ByteTrackPersonTracker:
    """Associate current person observations within one resettable session."""

    def __init__(self, *, source_frame_rate: float) -> None:
        if not isfinite(source_frame_rate) or source_frame_rate <= 0.0:
            raise ValueError("source_frame_rate must be positive and finite")
        self._backend = ByteTrackTracker(
            frame_rate=source_frame_rate,
            lost_track_buffer=_LOST_TRACK_BUFFER,
            track_activation_threshold=_TRACK_ACTIVATION_THRESHOLD,
            minimum_consecutive_frames=_MINIMUM_CONSECUTIVE_FRAMES,
            minimum_iou_threshold=_MINIMUM_IOU_THRESHOLD,
            high_conf_det_threshold=_HIGH_CONF_DET_THRESHOLD,
        )
        self._last_frame_index: int | None = None
        self._last_timestamp: Fraction | None = None

    def provenance(self) -> dict[str, str | dict[str, float | int]]:
        """Describe the maintained backend and fixed untuned profile."""
        return {
            "backend": "trackers.ByteTrackTracker",
            "version": version("trackers"),
            "parameters": {
                "lost_track_buffer": _LOST_TRACK_BUFFER,
                "track_activation_threshold": _TRACK_ACTIVATION_THRESHOLD,
                "minimum_consecutive_frames": _MINIMUM_CONSECUTIVE_FRAMES,
                "minimum_iou_threshold": _MINIMUM_IOU_THRESHOLD,
                "high_conf_det_threshold": _HIGH_CONF_DET_THRESHOLD,
            },
        }

    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        """Associate one ordered frame without emitting predicted positions."""
        timestamp = frame.pts * frame.time_base
        self._validate_order(frame.frame_index, timestamp)

        tracked = self._backend.update(
            self._to_vendor_detections(frame),
            timestamp=float(timestamp),
        )
        tracks = self._project_tracks(frame, tracked)
        self._last_frame_index = frame.frame_index
        self._last_timestamp = timestamp
        return FramePersonTracks(
            frame_index=frame.frame_index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=frame.image_width,
            image_height=frame.image_height,
            tracks=tracks,
        )

    def reset(self) -> None:
        """Start a new session and clear backend and ordering state."""
        self._backend.reset()
        self._last_frame_index = None
        self._last_timestamp = None

    def _validate_order(self, frame_index: int, timestamp: Fraction) -> None:
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must strictly increase within a session")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("presentation timestamp must not regress within a session")

    @staticmethod
    def _to_vendor_detections(frame: FramePersonDetections) -> sv.Detections:
        boxes = np.asarray(
            [
                (
                    detection.bbox.x_min,
                    detection.bbox.y_min,
                    detection.bbox.x_max,
                    detection.bbox.y_max,
                )
                for detection in frame.detections
            ],
            dtype=np.float64,
        ).reshape((-1, 4))
        confidences = np.asarray(
            [detection.confidence for detection in frame.detections],
            dtype=np.float64,
        )
        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            data={
                _DETECTION_INDEX_KEY: np.arange(
                    len(frame.detections),
                    dtype=np.int64,
                )
            },
        )

    @staticmethod
    def _project_tracks(
        frame: FramePersonDetections,
        tracked: sv.Detections,
    ) -> tuple[PersonTrack, ...]:
        if len(tracked) == 0:
            return ()
        tracker_ids = tracked.tracker_id
        detection_indices = tracked.data.get(_DETECTION_INDEX_KEY)
        if tracker_ids is None or detection_indices is None:
            raise RuntimeError("ByteTrack returned malformed association output")
        if len(tracker_ids) != len(detection_indices):
            raise RuntimeError("ByteTrack returned misaligned association output")
        return tuple(
            ByteTrackPersonTracker._person_track(
                frame,
                int(detection_index),
                int(tracker_id),
            )
            for detection_index, tracker_id in zip(
                detection_indices,
                tracker_ids,
                strict=True,
            )
            if int(tracker_id) >= 0
        )

    @staticmethod
    def _person_track(
        frame: FramePersonDetections,
        detection_index: int,
        tracker_id: int,
    ) -> PersonTrack:
        try:
            detection = frame.detections[detection_index]
        except IndexError as error:
            raise RuntimeError(
                "ByteTrack returned an invalid detection reference"
            ) from error
        return PersonTrack(
            track_id=tracker_id + 1,
            bbox=detection.bbox,
            detection_confidence=detection.confidence,
        )
