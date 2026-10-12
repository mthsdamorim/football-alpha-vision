"""Geometry boundary checks without vendor code or pretrained weights."""

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

_SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/pnlcalib-experiment.py")
)
_image_to_pitch = cast(
    Callable[[NDArray[np.float64]], NDArray[np.float64] | None],
    _SCRIPT["image_to_pitch"],
)
_project_tracks = cast(
    Callable[..., list[dict[str, object]]], _SCRIPT["project_tracks"]
)
_load_tracking_records = cast(
    Callable[[Path, Path, str], list[dict[str, object]]],
    _SCRIPT["load_tracking_records"],
)
_validate_tracking_identity = cast(
    Callable[[object, dict[str, object]], None],
    _SCRIPT["validate_tracking_identity"],
)


def test_camera_plane_roundtrip_preserves_metres_and_upper_touchline() -> None:
    # Known pinhole camera, including a nonzero z column which must be excluded.
    projection = np.array([[20.0, 2, 9, 960], [1, 8, 15, 540], [0.01, 0.02, 1, 1]])
    homography = _image_to_pitch(projection)
    assert homography is not None
    vendor_world = np.array([[0.0, -34, 0, 1], [36, 20.16, 0, 1], [-52.5, 0, 0, 1]])
    pixels = vendor_world @ projection.T
    pixels /= pixels[:, 2:3]
    recovered = pixels @ homography.T
    recovered /= recovered[:, 2:3]
    np.testing.assert_allclose(
        recovered[:, :2], vendor_world[:, :2] * [1, -1], atol=1e-10
    )


def test_degenerate_camera_cannot_generate_a_homography() -> None:
    assert _image_to_pitch(np.zeros((3, 4))) is None
    projection = np.eye(3, 4)
    projection[0, 0] = np.nan
    assert _image_to_pitch(projection) is None


def test_drawing_does_not_connect_across_the_projective_horizon() -> None:
    draw = cast(Callable[..., None], _SCRIPT["draw_projected"])
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    matrix = np.array([[1, 0, 10], [0, 1, 10], [1, 0, 0]], dtype=np.float64)
    draw(image, [(-1, 0), (1, 0)], matrix, (255, 255, 255))
    assert not image.any()
    draw(image, [(5, 5), (30, 30)], np.eye(3), (255, 255, 255))
    assert image.any()


def test_tracks_use_bbox_bottom_centre_and_current_homography() -> None:
    tracks = [
        {
            "track_id": 7,
            "bbox": {"x_min": 10.0, "y_min": 5.0, "x_max": 30.0, "y_max": 25.0},
            "detection_confidence": 0.9,
            "pitch_xy": {"x_m": 999.0, "y_m": 999.0},
        }
    ]
    projected = _project_tracks(tracks, np.diag([2.0, 3.0, 1.0]))
    assert projected == [
        {
            "track_id": 7,
            "detection_confidence": 0.9,
            "bbox": tracks[0]["bbox"],
            "support_point_uv": {"u": 20.0, "v": 25.0},
            "projection_reason": None,
            "pitch_xy": [40.0, 75.0],
            "inside_pitch": False,
        }
    ]
    absent = _project_tracks(tracks, None)
    assert len(absent) == 1
    assert absent[0]["bbox"] == tracks[0]["bbox"]
    assert absent[0]["pitch_xy"] is None
    assert absent[0]["projection_reason"] == "calibration_abstained"
    singular = _project_tracks(tracks, np.zeros((3, 3)))
    assert singular[0]["projection_reason"] == "non_finite_projection"
    assert singular[0]["pitch_xy"] is None


def test_tracking_records_require_source_and_exact_frame_count(tmp_path: Path) -> None:
    records = tmp_path / "tracks.jsonl"
    summary = tmp_path / "summary.json"
    records.write_text('{"frame_index":0,"pts":0,"time_base":"1/30","tracks":[]}\n')
    summary.write_text(json.dumps({"source_sha256": "source", "frame_count": 1}))
    assert _load_tracking_records(records, summary, "source")[0]["frame_index"] == 0

    summary.write_text(json.dumps({"source_sha256": "other", "frame_count": 1}))
    with np.testing.assert_raises_regex(ValueError, "source SHA"):
        _load_tracking_records(records, summary, "source")

    summary.write_text(json.dumps({"source_sha256": "source", "frame_count": 2}))
    with np.testing.assert_raises_regex(ValueError, "frame count"):
        _load_tracking_records(records, summary, "source")


def test_tracking_record_must_match_exact_source_frame() -> None:
    frame = type("Frame", (), {"index": 3, "pts": 512, "time_base": "1/30"})()
    _validate_tracking_identity(
        frame,
        {"frame_index": 3, "pts": 512, "time_base": "1/30"},
    )
    with np.testing.assert_raises_regex(ValueError, "different source frame"):
        _validate_tracking_identity(
            frame,
            {"frame_index": 3, "pts": 513, "time_base": "1/30"},
        )
