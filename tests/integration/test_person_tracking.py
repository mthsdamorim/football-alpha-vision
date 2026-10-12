"""Behavior tests for session-local person tracking."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
from importlib.metadata import version

import football_alpha_person_tracking.byte_track as byte_track_module
import numpy as np
import pytest
import supervision as sv

from football_alpha_person_tracking import (
    ByteTrackPersonTracker,
    FramePersonTracks,
    PersonTrack,
    PersonTracker,
)
from football_alpha_player_detection import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
)


class EmptyPersonTracker:
    """Small implementation used to exercise the public protocol."""

    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        return FramePersonTracks(
            frame_index=frame.frame_index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=frame.image_width,
            image_height=frame.image_height,
            tracks=(),
        )

    def reset(self) -> None:
        pass


def test_tracking_contract_is_immutable_and_project_owned() -> None:
    box = ImageBoundingBox(x_min=1.0, y_min=2.0, x_max=8.0, y_max=9.0)
    track = PersonTrack(
        track_id=3,
        bbox=box,
        detection_confidence=0.75,
    )
    mutable_tracks = [track]
    frame = FramePersonTracks(
        frame_index=7,
        pts=3_584,
        time_base=Fraction(1, 15_360),
        image_width=20,
        image_height=10,
        tracks=mutable_tracks,  # type: ignore[arg-type]
    )
    mutable_tracks.clear()
    tracker: PersonTracker = EmptyPersonTracker()

    assert frame.tracks == (track,)
    assert (
        tracker.update(
            FramePersonDetections(
                frame_index=7,
                pts=3_584,
                time_base=Fraction(1, 15_360),
                image_width=20,
                image_height=10,
                detections=(PersonDetection(bbox=box, confidence=0.75),),
            )
        ).tracks
        == ()
    )
    with pytest.raises(FrozenInstanceError):
        track.track_id = 4  # type: ignore[misc]


@pytest.mark.parametrize("track_id", [0, -1])
def test_person_track_requires_a_positive_session_local_id(track_id: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        PersonTrack(
            track_id=track_id,
            bbox=ImageBoundingBox(1.0, 2.0, 8.0, 9.0),
            detection_confidence=0.75,
        )


@pytest.mark.parametrize("detection_confidence", [-0.1, 1.1, float("nan")])
def test_person_track_rejects_invalid_detection_confidence(
    detection_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="detection_confidence"):
        PersonTrack(
            track_id=1,
            bbox=ImageBoundingBox(1.0, 2.0, 8.0, 9.0),
            detection_confidence=detection_confidence,
        )


@pytest.mark.parametrize(("image_width", "image_height"), [(0, 10), (20, 0)])
def test_frame_tracks_requires_positive_dimensions(
    image_width: int,
    image_height: int,
) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        FramePersonTracks(
            frame_index=0,
            pts=0,
            time_base=Fraction(1, 30),
            image_width=image_width,
            image_height=image_height,
            tracks=(),
        )


@pytest.mark.parametrize(
    "bbox",
    [
        ImageBoundingBox(1.0, 2.0, 21.0, 9.0),
        ImageBoundingBox(1.0, 2.0, 8.0, 11.0),
    ],
)
def test_frame_tracks_rejects_observations_outside_the_image(
    bbox: ImageBoundingBox,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        FramePersonTracks(
            frame_index=0,
            pts=0,
            time_base=Fraction(1, 30),
            image_width=20,
            image_height=10,
            tracks=(PersonTrack(1, bbox, 0.8),),
        )


def _detections(
    frame_index: int,
    *,
    pts: int | None = None,
    time_base: Fraction = Fraction(1, 30),
    box: ImageBoundingBox | None = None,
    confidence: float = 0.9,
) -> FramePersonDetections:
    detections = (PersonDetection(bbox=box, confidence=confidence),) if box else ()
    return FramePersonDetections(
        frame_index=frame_index,
        pts=frame_index if pts is None else pts,
        time_base=time_base,
        image_width=100,
        image_height=60,
        detections=detections,
    )


def test_bytetrack_preserves_id_and_current_observation_across_frames() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    first_box = ImageBoundingBox(10.25, 10.5, 20.75, 30.5)
    second_box = ImageBoundingBox(11.125, 10.25, 21.625, 30.25)

    first = tracker.update(
        _detections(
            7,
            pts=1_536,
            time_base=Fraction(1, 15_360),
            box=first_box,
            confidence=0.91,
        )
    )
    second = tracker.update(
        _detections(
            8,
            pts=2_048,
            time_base=Fraction(1, 15_360),
            box=second_box,
            confidence=0.87,
        )
    )

    assert first.tracks == ()
    assert second == FramePersonTracks(
        frame_index=8,
        pts=2_048,
        time_base=Fraction(1, 15_360),
        image_width=100,
        image_height=60,
        tracks=(
            PersonTrack(
                track_id=1,
                bbox=second_box,
                detection_confidence=0.87,
            ),
        ),
    )


def test_bytetrack_activates_at_the_exact_fixed_threshold() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 20.0, 30.0)

    first = tracker.update(_detections(0, box=box, confidence=0.7))
    second = tracker.update(_detections(1, box=box, confidence=0.7))

    assert first.tracks == ()
    assert [track.track_id for track in second.tracks] == [1]


def test_bytetrack_preserves_continuous_box_precision_for_association() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 10.00000001, 30.0)

    first = tracker.update(_detections(0, box=box))
    second = tracker.update(_detections(1, box=box))

    assert first.tracks == ()
    assert [track.track_id for track in second.tracks] == [1]


def test_bytetrack_recovers_the_same_id_after_a_short_empty_gap() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 20.0, 30.0)

    tracker.update(_detections(0, box=box))
    established = tracker.update(_detections(1, box=box))
    gap = tracker.update(_detections(2))
    recovered = tracker.update(_detections(3, box=box))

    assert [track.track_id for track in established.tracks] == [1]
    assert gap.tracks == ()
    assert [track.track_id for track in recovered.tracks] == [1]


def test_bytetrack_expires_a_track_after_the_fixed_buffer() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 20.0, 30.0)

    tracker.update(_detections(0, box=box))
    established = tracker.update(_detections(1, box=box))
    empty_results = [tracker.update(_detections(index)) for index in range(2, 33)]
    reappeared = tracker.update(_detections(33, box=box))
    confirmed = tracker.update(_detections(34, box=box))

    assert [track.track_id for track in established.tracks] == [1]
    assert all(result.tracks == () for result in empty_results)
    assert reappeared.tracks == ()
    assert [track.track_id for track in confirmed.tracks] == [2]


def test_reset_starts_a_new_session_and_permits_numeric_id_reuse() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 20.0, 30.0)

    tracker.update(_detections(10, box=box))
    before_reset = tracker.update(_detections(11, box=box))
    tracker.reset()
    tracker.update(_detections(0, box=box))
    after_reset = tracker.update(_detections(1, box=box))

    assert [track.track_id for track in before_reset.tracks] == [1]
    assert [track.track_id for track in after_reset.tracks] == [1]


def test_ordering_rejection_does_not_mutate_association_state() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    box = ImageBoundingBox(10.0, 10.0, 20.0, 30.0)

    tracker.update(_detections(0, box=box))
    established = tracker.update(_detections(1, box=box))

    for invalid_frame_index in (1, 0):
        with pytest.raises(ValueError, match="frame_index"):
            tracker.update(_detections(invalid_frame_index))
    with pytest.raises(ValueError, match="timestamp"):
        tracker.update(_detections(2, pts=0))

    with pytest.warns(UserWarning, match="duplicate timestamp"):
        accepted = tracker.update(_detections(3, pts=1, box=box))
    after_temporal_gap = tracker.update(_detections(4, pts=5, box=box))

    assert [track.track_id for track in established.tracks] == [1]
    assert [track.track_id for track in accepted.tracks] == [1]
    assert [track.track_id for track in after_temporal_gap.tracks] == [1]


@pytest.mark.parametrize("source_frame_rate", [0.0, -1.0, float("nan")])
def test_bytetrack_rejects_invalid_source_frame_rate(
    source_frame_rate: float,
) -> None:
    with pytest.raises(ValueError, match="source_frame_rate"):
        ByteTrackPersonTracker(source_frame_rate=source_frame_rate)


class StubByteTrackBackend:
    def __init__(self, output: sv.Detections) -> None:
        self.output = output

    def update(
        self,
        detections: sv.Detections,
        *,
        timestamp: float,
    ) -> sv.Detections:
        return self.output

    def reset(self) -> None:
        pass


class CapturingByteTrackBackend(StubByteTrackBackend):
    def __init__(self) -> None:
        super().__init__(sv.Detections.empty())
        self.timestamps: list[float] = []

    def update(
        self,
        detections: sv.Detections,
        *,
        timestamp: float,
    ) -> sv.Detections:
        self.timestamps.append(timestamp)
        return self.output


def _vendor_output(
    *,
    tracker_ids: list[int] | None,
    detection_indices: list[int] | None,
) -> sv.Detections:
    output = sv.Detections(xyxy=np.asarray([[1.0, 2.0, 8.0, 9.0]]))
    if tracker_ids is not None:
        output.tracker_id = np.asarray(tracker_ids, dtype=int)
    if detection_indices is not None:
        output.data["football_alpha_detection_index"] = np.asarray(
            detection_indices,
            dtype=int,
        )
    return output


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            _vendor_output(tracker_ids=None, detection_indices=[0]),
            "malformed",
        ),
        (
            _vendor_output(tracker_ids=[0, 1], detection_indices=[0]),
            "misaligned",
        ),
        (
            _vendor_output(tracker_ids=[0], detection_indices=[2]),
            "invalid detection reference",
        ),
    ],
)
def test_bytetrack_encapsulates_malformed_vendor_results(
    monkeypatch: pytest.MonkeyPatch,
    output: sv.Detections,
    message: str,
) -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=30.0)
    monkeypatch.setattr(tracker, "_backend", StubByteTrackBackend(output))

    with pytest.raises(RuntimeError, match=message):
        tracker.update(
            _detections(
                0,
                box=ImageBoundingBox(1.0, 2.0, 8.0, 9.0),
            )
        )


def test_bytetrack_forwards_the_fixed_profile_fps_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CapturingByteTrackBackend()
    constructor_arguments: dict[str, float | int] = {}

    def build_backend(**arguments: float | int) -> CapturingByteTrackBackend:
        constructor_arguments.update(arguments)
        return backend

    monkeypatch.setattr(byte_track_module, "ByteTrackTracker", build_backend)
    tracker = ByteTrackPersonTracker(source_frame_rate=25.0)

    tracker.update(
        _detections(
            4,
            pts=512,
            time_base=Fraction(1, 15_360),
        )
    )

    assert constructor_arguments == {
        "frame_rate": 25.0,
        "lost_track_buffer": 30,
        "track_activation_threshold": 0.7,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "high_conf_det_threshold": 0.6,
    }
    assert backend.timestamps == [float(Fraction(1, 30))]


def test_bytetrack_reports_backend_and_the_exact_accepted_profile() -> None:
    tracker = ByteTrackPersonTracker(source_frame_rate=25.0)

    assert tracker.provenance() == {
        "backend": "trackers.ByteTrackTracker",
        "version": version("trackers"),
        "parameters": {
            "lost_track_buffer": 30,
            "track_activation_threshold": 0.7,
            "minimum_consecutive_frames": 2,
            "minimum_iou_threshold": 0.1,
            "high_conf_det_threshold": 0.6,
        },
    }
