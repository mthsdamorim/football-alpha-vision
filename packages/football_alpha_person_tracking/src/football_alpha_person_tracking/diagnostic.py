"""Frame-complete diagnostic artifacts for session-local person tracking."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, islice, pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast
from unicodedata import normalize

import av
import cv2
import numpy as np
from av.video.stream import VideoStream
from numpy.typing import NDArray

from football_alpha_person_tracking.contracts import (
    FramePersonTracks,
    PersonTrack,
    PersonTracker,
)
from football_alpha_person_tracking.fixture_calibration import evaluate_temporal_clip
from football_alpha_pitch_projection import (
    CalibrationStatus,
    CameraCalibration,
    CameraCalibrator,
    CorrespondenceHomographyCalibrator,
    FrameCalibrationInput,
    ImagePitchCorrespondence,
    ImagePoint,
    OpticalFlowCalibrator,
    PitchDimensions,
    PitchPoint,
    PointProjection,
    calibrate_frames,
    project_image_point,
)
from football_alpha_player_detection import FramePersonDetections, PersonDetector
from football_alpha_video import DecodedFrame, decode_video

type JsonScalar = bool | float | int | str | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type TrackerFactory = Callable[[float], tuple[PersonTracker, Mapping[str, JsonValue]]]
type PathIdentity = tuple[int, int, tuple[str, ...]]
DETECTOR_THRESHOLD = 0.10


@dataclass(frozen=True, slots=True)
class _ProjectedPersonObservation:
    track: PersonTrack
    projection: PointProjection


@dataclass(frozen=True, slots=True)
class _DiagnosticSummary:
    frame_count: int
    tracklets: list[JsonValue]
    accepted_calibration_frame_count: int
    abstained_calibration_frame_count: int
    calibration_abstention_counts: dict[str, int]
    current_track_observation_count: int
    projected_observation_count: int
    outside_pitch_observation_count: int


def create_tracking_diagnostic(
    source: Path,
    composite_path: Path,
    jsonl_path: Path,
    summary_path: Path,
    *,
    detector: PersonDetector,
    detector_provenance: Mapping[str, JsonValue],
    tracker_factory: TrackerFactory,
    calibrator: CameraCalibrator | None = None,
    pitch: PitchDimensions | None = None,
    correspondences_by_frame: Mapping[int, Sequence[ImagePitchCorrespondence]]
    | None = None,
    correspondence_source_sha256: str | None = None,
    batch_size: int,
    temporal: bool = False,
) -> dict[str, JsonValue]:
    """Stream one video session into a coherent diagnostic artifact set."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if calibrator is None:
        calibrator = CorrespondenceHomographyCalibrator()
    if pitch is None:
        pitch = PitchDimensions(105.0, 68.0)
    if correspondences_by_frame is None:
        correspondences_by_frame = {}
    validate_tracking_diagnostic_paths(
        source,
        composite_path,
        jsonl_path,
        summary_path,
    )
    for destination in (composite_path, jsonl_path, summary_path):
        destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_source, source_sha256 = _snapshot_source(source, composite_path.parent)
    temporary_composite: Path | None = None
    temporary_jsonl: Path | None = None
    temporary_summary: Path | None = None
    try:
        source_frame_rate = _source_frame_rate(temporary_source)
        tracker, tracker_provenance = tracker_factory(float(source_frame_rate))
        cue_profile_applied = (
            correspondence_source_sha256 is None
            or correspondence_source_sha256 == source_sha256
        )
        active_correspondences = correspondences_by_frame if cue_profile_applied else {}
        if temporal and not cue_profile_applied:
            raise ValueError("temporal experiment requires the versioned fixture")
        temporary_composite = _temporary_path(composite_path)
        temporary_jsonl = _temporary_path(jsonl_path)
        _process_frames(
            temporary_source,
            temporary_composite,
            temporary_jsonl,
            detector=detector,
            tracker=tracker,
            calibrator=calibrator,
            pitch=pitch,
            correspondences_by_frame=active_correspondences,
            source_frame_rate=source_frame_rate,
            batch_size=batch_size,
            temporal=temporal,
        )
        diagnostic_summary = _summarize_jsonl(temporary_jsonl)
        calibration_fixture: dict[str, JsonValue] = {
            "cue_source_sha256": correspondence_source_sha256,
            "cue_profile_applied": cue_profile_applied,
            "anchor_frame_indices": list(active_correspondences),
            "anchor_frame_count": len(active_correspondences),
        }
        raw_abstention_counts = diagnostic_summary.calibration_abstention_counts
        calibration_abstention_counts: dict[str, JsonValue] = {
            reason: count for reason, count in raw_abstention_counts.items()
        }
        summary: dict[str, JsonValue] = {
            "source": str(source),
            "source_sha256": source_sha256,
            "composite": str(composite_path),
            "jsonl": str(jsonl_path),
            "summary": str(summary_path),
            "frame_count": diagnostic_summary.frame_count,
            "source_frame_rate": str(source_frame_rate),
            "pitch": {
                "length_m": pitch.length_m,
                "width_m": pitch.width_m,
                "dimensions_source": "assumed_fixture_template",
            },
            "calibration_fixture": calibration_fixture,
            "detector": {
                **detector_provenance,
                "minimum_confidence": DETECTOR_THRESHOLD,
                "batch_size": batch_size,
            },
            "tracker": dict(tracker_provenance),
            "distinct_track_id_count": len(diagnostic_summary.tracklets),
            "tracklets": diagnostic_summary.tracklets,
            "accepted_calibration_frame_count": (
                diagnostic_summary.accepted_calibration_frame_count
            ),
            "abstained_calibration_frame_count": (
                diagnostic_summary.abstained_calibration_frame_count
            ),
            "calibration_abstention_counts": (calibration_abstention_counts),
            "current_track_observation_count": (
                diagnostic_summary.current_track_observation_count
            ),
            "projected_observation_count": (
                diagnostic_summary.projected_observation_count
            ),
            "outside_pitch_observation_count": (
                diagnostic_summary.outside_pitch_observation_count
            ),
        }
        if temporal:
            summary["frame_interval"] = [180, 225]
            summary["temporal_evaluation"] = cast(
                dict[str, JsonValue], evaluate_temporal_clip(temporary_jsonl)
            )
        temporary_summary = _write_summary(summary_path, summary)
        _validate_summary_against_jsonl(temporary_jsonl, temporary_summary)
        _publish_artifacts(
            (
                (temporary_composite, composite_path),
                (temporary_jsonl, jsonl_path),
                (temporary_summary, summary_path),
            )
        )
    except BaseException:
        if temporary_composite is not None:
            temporary_composite.unlink(missing_ok=True)
        if temporary_jsonl is not None:
            temporary_jsonl.unlink(missing_ok=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)
        raise
    finally:
        temporary_source.unlink(missing_ok=True)
    return summary


def validate_tracking_diagnostic_paths(
    source: Path,
    composite_path: Path,
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    """Reject destinations that could overwrite one another or the source."""
    identities = tuple(
        _path_identity(path)
        for path in (source, composite_path, jsonl_path, summary_path)
    )
    if len(set(identities)) != len(identities):
        raise ValueError("source and diagnostic artifact paths must be distinct")
    if any(
        _identity_contains(left, right) or _identity_contains(right, left)
        for left, right in combinations(identities[1:], 2)
    ):
        raise ValueError("diagnostic artifact paths must not contain one another")
    if any(path.is_dir() for path in (composite_path, jsonl_path, summary_path)):
        raise ValueError("diagnostic artifact paths must not be directories")


def _path_identity(path: Path) -> PathIdentity:
    ancestor = path.resolve()
    missing_components: list[str] = []
    while not ancestor.exists():
        missing_components.append(normalize("NFC", ancestor.name).casefold())
        ancestor = ancestor.parent
    status = ancestor.stat()
    return status.st_dev, status.st_ino, tuple(reversed(missing_components))


def _identity_contains(ancestor: PathIdentity, descendant: PathIdentity) -> bool:
    ancestor_device, ancestor_inode, ancestor_suffix = ancestor
    descendant_device, descendant_inode, descendant_suffix = descendant
    return (
        (ancestor_device, ancestor_inode) == (descendant_device, descendant_inode)
        and ancestor_suffix != descendant_suffix
        and descendant_suffix[: len(ancestor_suffix)] == ancestor_suffix
    )


def validate_tracking_diagnostic_source(source: Path) -> None:
    """Fail before model construction when the source is not readable video."""
    with av.open(source) as container:
        container.streams.video[0]


def _source_frame_rate(source: Path) -> Fraction:
    with av.open(source) as container:
        return cast(Fraction, container.streams.video[0].average_rate)


def _snapshot_source(source: Path, temporary_directory: Path) -> tuple[Path, str]:
    digest = sha256()
    temporary_source: Path | None = None
    try:
        with (
            source.open("rb") as source_file,
            NamedTemporaryFile(
                dir=temporary_directory,
                prefix=f".{source.stem}-",
                suffix=source.suffix,
                delete=False,
            ) as temporary_source_file,
        ):
            temporary_source = Path(temporary_source_file.name)
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
                temporary_source_file.write(chunk)
    except BaseException:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
        raise
    assert temporary_source is not None
    return temporary_source, digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _process_frames(
    source: Path,
    composite_path: Path,
    jsonl_path: Path,
    *,
    detector: PersonDetector,
    tracker: PersonTracker,
    calibrator: CameraCalibrator,
    pitch: PitchDimensions,
    correspondences_by_frame: Mapping[int, Sequence[ImagePitchCorrespondence]],
    source_frame_rate: Fraction,
    batch_size: int,
    temporal: bool = False,
) -> None:
    with (
        av.open(composite_path, mode="w") as container,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        stream: VideoStream | None = None
        source_frames = decode_video(source)
        camera = OpticalFlowCalibrator(pitch) if temporal else None
        if temporal:
            source_frames = islice(source_frames, 180, 226)
        for frames in _batches(source_frames, batch_size):
            detections = detector.detect(frames)
            if len(detections) != len(frames):
                raise ValueError("person detector violated frame/result cardinality")
            calibration_inputs = [
                FrameCalibrationInput.from_frame(
                    frame,
                    pitch=pitch,
                    correspondences=correspondences_by_frame.get(frame.index, ()),
                )
                for frame in frames
            ]
            calibrations = calibrate_frames(calibrator, calibration_inputs)
            for frame, frame_detections, calibration in zip(
                frames,
                detections,
                calibrations,
                strict=True,
            ):
                if camera is not None:
                    calibration = camera.update(
                        frame,
                        calibration
                        if frame.index in correspondences_by_frame
                        else None,
                    )
                _validate_detection_identity(frame, frame_detections)
                frame_tracks = tracker.update(frame_detections)
                _validate_track_identity(frame, frame_tracks)
                _validate_calibration_identity(frame_tracks, calibration)
                observations = _project_tracks(frame_tracks, calibration)
                annotated = _render_composite(
                    frame,
                    frame_tracks,
                    calibration,
                    observations,
                )
                if stream is None:
                    stream = container.add_stream("libx264", rate=source_frame_rate)
                    stream.width = annotated.shape[1]
                    stream.height = annotated.shape[0]
                    stream.pix_fmt = "yuv420p"
                    stream.time_base = frame.time_base
                    stream.codec_context.time_base = frame.time_base

                video_frame = av.VideoFrame.from_ndarray(annotated, format="rgb24")
                video_frame.pts = frame.pts
                video_frame.time_base = frame.time_base
                container.mux(stream.encode(video_frame))
                jsonl_file.write(
                    json.dumps(_frame_record(frame_tracks, calibration, observations))
                    + "\n"
                )

        if stream is None:
            raise ValueError("source video produced no decoded frames")
        container.mux(stream.encode())


def _batches(
    frames: Iterator[DecodedFrame],
    batch_size: int,
) -> Iterator[list[DecodedFrame]]:
    while batch := list(islice(frames, batch_size)):
        yield batch


def _validate_detection_identity(
    frame: DecodedFrame,
    detections: FramePersonDetections,
) -> None:
    expected = (
        frame.index,
        frame.pts,
        frame.time_base,
        frame.image.shape[1],
        frame.image.shape[0],
    )
    actual = (
        detections.frame_index,
        detections.pts,
        detections.time_base,
        detections.image_width,
        detections.image_height,
    )
    if actual != expected:
        raise ValueError("person detector returned a result for a different frame")


def _validate_track_identity(frame: DecodedFrame, tracks: FramePersonTracks) -> None:
    expected = (
        frame.index,
        frame.pts,
        frame.time_base,
        frame.image.shape[1],
        frame.image.shape[0],
    )
    actual = (
        tracks.frame_index,
        tracks.pts,
        tracks.time_base,
        tracks.image_width,
        tracks.image_height,
    )
    if actual != expected:
        raise ValueError("person tracker returned a result for a different frame")


def _validate_calibration_identity(
    tracks: FramePersonTracks,
    calibration: CameraCalibration,
) -> None:
    expected = (
        tracks.frame_index,
        tracks.pts,
        tracks.time_base,
        tracks.image_width,
        tracks.image_height,
    )
    actual = (
        calibration.frame_index,
        calibration.pts,
        calibration.time_base,
        calibration.image_width,
        calibration.image_height,
    )
    if actual != expected:
        raise ValueError("person tracks and calibration refer to different frames")


def _project_tracks(
    tracks: FramePersonTracks,
    calibration: CameraCalibration,
) -> tuple[_ProjectedPersonObservation, ...]:
    return tuple(
        _ProjectedPersonObservation(
            track=track,
            projection=project_image_point(
                calibration,
                ImagePoint(
                    u=(track.bbox.x_min + track.bbox.x_max) / 2.0,
                    v=track.bbox.y_max,
                ),
            ),
        )
        for track in tracks.tracks
    )


def _render_composite(
    frame: DecodedFrame,
    tracks: FramePersonTracks,
    calibration: CameraCalibration,
    observations: Sequence[_ProjectedPersonObservation],
) -> NDArray[np.uint8]:
    annotated = frame.image.copy()
    if calibration.status is CalibrationStatus.ACCEPTED:
        _draw_pitch_overlay(annotated, calibration)
    for observation in observations:
        track = observation.track
        box = track.bbox
        top_left = (round(box.x_min), round(box.y_min))
        bottom_right = (round(box.x_max), round(box.y_max))
        cv2.rectangle(annotated, top_left, bottom_right, (255, 64, 64), 3)
        label_top = max(0, top_left[1] - 26)
        label_right = min(annotated.shape[1] - 1, top_left[0] + 160)
        cv2.rectangle(
            annotated,
            (top_left[0], label_top),
            (label_right, label_top + 24),
            (255, 64, 64),
            -1,
        )
        cv2.putText(
            annotated,
            f"ID {track.track_id} {track.detection_confidence:.2f}",
            (top_left[0] + 3, label_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        support = observation.projection.image_point
        cv2.circle(
            annotated,
            (round(support.u), round(support.v)),
            5,
            (255, 220, 0),
            -1,
            cv2.LINE_AA,
        )
    status = (
        "calibration accepted"
        if calibration.status is CalibrationStatus.ACCEPTED
        else f"calibration abstained: {calibration.abstention_reason}"
    )
    cv2.putText(
        annotated,
        status,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (80, 255, 80)
        if calibration.status is CalibrationStatus.ACCEPTED
        else (255, 100, 100),
        2,
        cv2.LINE_AA,
    )
    minimap = _render_minimap(
        annotated.shape[0],
        max(annotated.shape[1] // 3, 16),
        calibration,
        observations,
    )
    composite = np.concatenate((annotated, minimap), axis=1)
    # YUV420 requires even dimensions; padding preserves source pixel geometry.
    return np.asarray(
        cv2.copyMakeBorder(
            composite,
            0,
            composite.shape[0] % 2,
            0,
            composite.shape[1] % 2,
            cv2.BORDER_CONSTANT,
        ),
        dtype=np.uint8,
    )


def _pitch_polylines(pitch: PitchDimensions) -> tuple[NDArray[np.float64], ...]:
    half_length = pitch.length_m / 2.0
    half_width = pitch.width_m / 2.0
    circle_angles = np.linspace(0.0, 2.0 * np.pi, 73)
    center_circle = np.column_stack(
        (9.15 * np.cos(circle_angles), 9.15 * np.sin(circle_angles))
    )
    lines: list[NDArray[np.float64]] = [
        np.asarray(
            [
                (-half_length, -half_width),
                (-half_length, half_width),
                (half_length, half_width),
                (half_length, -half_width),
                (-half_length, -half_width),
            ],
            dtype=np.float64,
        ),
        np.asarray(((0.0, -half_width), (0.0, half_width)), dtype=np.float64),
        center_circle,
    ]
    for direction in (-1.0, 1.0):
        goal_line_x = direction * half_length
        penalty_x = direction * (half_length - 16.5)
        goal_area_x = direction * (half_length - 5.5)
        lines.extend(
            (
                np.asarray(
                    (
                        (goal_line_x, -20.16),
                        (penalty_x, -20.16),
                        (penalty_x, 20.16),
                        (goal_line_x, 20.16),
                    ),
                    dtype=np.float64,
                ),
                np.asarray(
                    (
                        (goal_line_x, -9.16),
                        (goal_area_x, -9.16),
                        (goal_area_x, 9.16),
                        (goal_line_x, 9.16),
                    ),
                    dtype=np.float64,
                ),
            )
        )
    return tuple(lines)


def _draw_pitch_overlay(
    image: NDArray[np.uint8],
    calibration: CameraCalibration,
) -> None:
    homography = calibration.homography_img_to_pitch
    if homography is None:  # pragma: no cover - enforced by CameraCalibration
        raise ValueError("accepted calibration has no homography")
    pitch_to_image = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    for pitch_line in _pitch_polylines(calibration.pitch):
        image_line = cv2.perspectiveTransform(
            pitch_line.reshape(1, -1, 2),
            pitch_to_image,
        )[0]
        if np.isfinite(image_line).all():  # pragma: no branch - finite inputs
            cv2.polylines(
                image,
                [np.rint(image_line).astype(np.int32)],
                False,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )


def _render_minimap(
    height: int,
    width: int,
    calibration: CameraCalibration,
    observations: Sequence[_ProjectedPersonObservation],
) -> NDArray[np.uint8]:
    canvas = np.full((height, width, 3), (22, 92, 38), dtype=np.uint8)
    margin = max(2, min(width, height) // 16)

    def to_pixel(point: PitchPoint) -> tuple[int, int]:
        x_min, x_max = calibration.pitch.x_bounds_m
        y_min, y_max = calibration.pitch.y_bounds_m
        usable_width = max(1, width - 2 * margin)
        usable_height = max(1, height - 2 * margin)
        return (
            round(margin + (point.x_m - x_min) / (x_max - x_min) * usable_width),
            round(margin + (y_max - point.y_m) / (y_max - y_min) * usable_height),
        )

    for pitch_line in _pitch_polylines(calibration.pitch):
        pixels = np.asarray(
            [to_pixel(PitchPoint(float(x_m), float(y_m))) for x_m, y_m in pitch_line],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pixels], False, (230, 230, 230), 1, cv2.LINE_AA)
    if calibration.status is CalibrationStatus.ACCEPTED:
        for observation in observations:
            pitch_point = observation.projection.pitch_point
            if pitch_point is not None:
                cv2.circle(
                    canvas,
                    to_pixel(pitch_point),
                    4,
                    (255, 220, 0),
                    -1,
                    cv2.LINE_AA,
                )
    else:
        cv2.putText(
            canvas,
            "NO METRIC POSITION",
            (margin, max(12, margin * 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 100, 100),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _frame_record(
    frame: FramePersonTracks,
    calibration: CameraCalibration,
    observations: Sequence[_ProjectedPersonObservation],
) -> dict[str, JsonValue]:
    calibration_record: dict[str, JsonValue] = {
        "status": calibration.status.value,
        "abstention_reason": calibration.abstention_reason.value
        if calibration.abstention_reason is not None
        else None,
        "method": calibration.method,
        "provenance": calibration.provenance,
        "quality_score": calibration.quality_score,
        "quality_threshold": calibration.quality_threshold,
        "correspondence_count": calibration.correspondence_count,
        "inlier_count": calibration.inlier_count,
        "reprojection_rmse_px": calibration.reprojection_rmse_px,
        "homography_img_to_pitch": (
            [list(row) for row in calibration.homography_img_to_pitch]
            if calibration.homography_img_to_pitch is not None
            else None
        ),
    }
    track_records: list[JsonValue] = [
        {
            "track_id": observation.track.track_id,
            "bbox": {
                "x_min": observation.track.bbox.x_min,
                "y_min": observation.track.bbox.y_min,
                "x_max": observation.track.bbox.x_max,
                "y_max": observation.track.bbox.y_max,
            },
            "detection_confidence": observation.track.detection_confidence,
            "support_point_uv": {
                "u": observation.projection.image_point.u,
                "v": observation.projection.image_point.v,
            },
            "support_point_method": "bbox_bottom_center",
            "pitch_xy": {
                "x_m": observation.projection.pitch_point.x_m,
                "y_m": observation.projection.pitch_point.y_m,
            }
            if observation.projection.pitch_point is not None
            else None,
            "inside_pitch": observation.projection.inside_pitch,
            "projection_reason": observation.projection.failure_reason.value
            if observation.projection.failure_reason is not None
            else None,
        }
        for observation in observations
    ]
    return {
        "frame_index": frame.frame_index,
        "pts": frame.pts,
        "time_base": str(frame.time_base),
        "calibration": calibration_record,
        "tracks": track_records,
    }


def _summarize_jsonl(path: Path) -> _DiagnosticSummary:
    observations: dict[int, list[tuple[int, int, Fraction]]] = {}
    frame_count = 0
    accepted_calibration_frame_count = 0
    abstention_counts: Counter[str] = Counter()
    current_track_observation_count = 0
    projected_observation_count = 0
    outside_pitch_observation_count = 0
    with path.open(encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            record = cast(dict[str, object], json.loads(line))
            frame_index = cast(int, record["frame_index"])
            pts = cast(int, record["pts"])
            time_base = Fraction(cast(str, record["time_base"]))
            calibration = cast(dict[str, object], record["calibration"])
            if calibration["status"] == CalibrationStatus.ACCEPTED.value:
                accepted_calibration_frame_count += 1
            else:
                abstention_counts[cast(str, calibration["abstention_reason"])] += 1
            for raw_track in cast(Sequence[object], record["tracks"]):
                track = cast(dict[str, object], raw_track)
                track_id = cast(int, track["track_id"])
                observations.setdefault(track_id, []).append(
                    (frame_index, pts, time_base)
                )
                current_track_observation_count += 1
                if track["pitch_xy"] is not None:
                    projected_observation_count += 1
                    if track["inside_pitch"] is False:
                        outside_pitch_observation_count += 1
            frame_count += 1

    return _DiagnosticSummary(
        frame_count=frame_count,
        tracklets=[
            _tracklet_summary(track_id, track_observations)
            for track_id, track_observations in sorted(observations.items())
        ],
        accepted_calibration_frame_count=accepted_calibration_frame_count,
        abstained_calibration_frame_count=sum(abstention_counts.values()),
        calibration_abstention_counts=dict(sorted(abstention_counts.items())),
        current_track_observation_count=current_track_observation_count,
        projected_observation_count=projected_observation_count,
        outside_pitch_observation_count=outside_pitch_observation_count,
    )


def _tracklet_summary(
    track_id: int,
    observations: list[tuple[int, int, Fraction]],
) -> JsonValue:
    first = observations[0]
    last = observations[-1]
    gaps = [
        current[0] - previous[0] - 1 for previous, current in pairwise(observations)
    ]
    return {
        "track_id": track_id,
        "first_observation": _observation_record(first),
        "last_observation": _observation_record(last),
        "observed_frame_count": len(observations),
        "observed_time_span": str(last[1] * last[2] - first[1] * first[2]),
        "largest_observed_gap": max(gaps, default=0),
    }


def _observation_record(observation: tuple[int, int, Fraction]) -> JsonValue:
    frame_index, pts, time_base = observation
    return {
        "frame_index": frame_index,
        "pts": pts,
        "time_base": str(time_base),
    }


def _write_summary(destination: Path, summary: Mapping[str, JsonValue]) -> Path:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _validate_summary_against_jsonl(jsonl_path: Path, summary_path: Path) -> None:
    diagnostic_summary = _summarize_jsonl(jsonl_path)
    summary = cast(dict[str, object], json.loads(summary_path.read_text()))
    expected = (
        diagnostic_summary.frame_count,
        len(diagnostic_summary.tracklets),
        diagnostic_summary.tracklets,
        diagnostic_summary.accepted_calibration_frame_count,
        diagnostic_summary.abstained_calibration_frame_count,
        diagnostic_summary.calibration_abstention_counts,
        diagnostic_summary.current_track_observation_count,
        diagnostic_summary.projected_observation_count,
        diagnostic_summary.outside_pitch_observation_count,
    )
    actual = (
        summary.get("frame_count"),
        summary.get("distinct_track_id_count"),
        summary.get("tracklets"),
        summary.get("accepted_calibration_frame_count"),
        summary.get("abstained_calibration_frame_count"),
        summary.get("calibration_abstention_counts"),
        summary.get("current_track_observation_count"),
        summary.get("projected_observation_count"),
        summary.get("outside_pitch_observation_count"),
    )
    if actual != expected:
        raise ValueError("tracking summary does not reconcile with diagnostic JSONL")
    if "temporal_evaluation" in summary and summary[
        "temporal_evaluation"
    ] != evaluate_temporal_clip(jsonl_path):
        raise ValueError("temporal evaluation does not reconcile with diagnostic JSONL")


def _publish_artifacts(artifacts: Sequence[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, destination in artifacts:
            if destination.exists() or destination.is_symlink():
                with NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.stem}-backup-",
                    suffix=destination.suffix,
                    delete=False,
                ) as backup_file:
                    backup = Path(backup_file.name)
                backup.unlink()
                destination.replace(backup)
                backups.append((destination, backup))

        for temporary, destination in artifacts:
            temporary.replace(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            backup.replace(destination)
        for temporary, _ in artifacts:
            temporary.unlink(missing_ok=True)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)
