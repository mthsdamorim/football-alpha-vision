"""Observable checks for the frozen TCC projection/stability evaluator."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate-tcc-projection.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("evaluate_tcc_projection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int, x: float | None, *, track_id: int = 1) -> dict[str, Any]:
    return {
        "frame_index": index,
        "pts": index * 1000,
        "time_base": "1/25000",
        "tracks": [
            {
                "track_id": track_id,
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 20},
                "support_point_uv": {"u": 5, "v": 20},
                "pitch_xy": [x, 0] if x is not None else None,
                "inside_pitch": True if x is not None else None,
                "projection_reason": None if x is not None else "calibration_abstained",
            }
        ],
    }


def test_smooth_only_preserves_edges_gaps_ids_and_originals() -> None:
    rows = [
        _row(i, x) for i, x in enumerate((0, 1, 1.8, 3, 4, None, 9, 10, 11, 12, 13))
    ]
    smoothed = MODULE.smooth_records(rows)
    assert [r["tracks"][0]["pitch_xy"] for r in smoothed] == [
        r["tracks"][0]["pitch_xy"] for r in rows
    ]
    assert smoothed[0]["tracks"][0]["smoothed_pitch_xy"] == [0, 0]
    assert smoothed[1]["tracks"][0]["smoothed_pitch_xy"] == [1, 0]
    assert smoothed[2]["tracks"][0]["smoothed_pitch_xy"][0] == pytest.approx(
        1.9028571429
    )
    assert smoothed[5]["tracks"][0]["smoothed_pitch_xy"] is None
    assert smoothed[6]["tracks"][0]["smoothed_pitch_xy"] == [9, 0]
    rows[2]["tracks"][0]["track_id"] = 2
    broken = MODULE.smooth_records(rows)
    assert broken[2]["tracks"][0]["smoothed_pitch_xy"] == [1.8, 0]


def test_smooth_only_rejects_displacement_over_frozen_limit() -> None:
    rows = [_row(i, x) for i, x in enumerate((0, 0, 4, 0, 0))]
    smoothed = MODULE.smooth_records(rows)
    assert smoothed[2]["tracks"][0]["smoothed_pitch_xy"] == [4, 0]
    assert (
        smoothed[2]["tracks"][0]["smoothing_reason"]
        == "candidate_displacement_over_0.5m"
    )


def test_line_error_uses_held_out_image_pixels_and_abstains() -> None:
    lines = [{"Side line right": [{"x": 0.5, "y": 0.25}]}]
    matrix = [[1, 0, -50], [0, -1, 50], [0, 0, 1]]
    assert MODULE.line_errors_px(matrix, lines, 200, 200) == pytest.approx([2.5])
    assert MODULE.line_errors_px(None, lines, 200, 200) == []


def test_metric_cohort_requires_review_and_visible_support() -> None:
    person: dict[str, Any] = {
        "source_annotation_id": "1",
        "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 20},
        "support_image": {"x": 5, "y": 20, "visibility": "pending_review"},
        "bbox_pitch": {"x_bottom_middle": 5, "y_bottom_middle": 0},
    }
    row = _row(0, 5)
    row["homography_img_to_pitch"] = None
    reference = {
        "source_frame": 251,
        "metric_reference_status": "available_derived_reference",
        "people": [person],
    }
    assert MODULE.metric_pairs(row, reference, {251})["eligible"] == 1
    assert MODULE.metric_pairs(row, reference, set())["eligible"] == 0
    person["support_image"]["visibility"] = "image_edge_truncated"
    assert MODULE.metric_pairs(row, reference, {251})["eligible"] == 0


def test_matching_maximizes_cardinality_then_iou_with_stable_tie() -> None:
    box = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
    tracks = [{"bbox": box}, {"bbox": box}]
    people = [
        {"source_annotation_id": "b", "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10}},
        {"source_annotation_id": "a", "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]
    assert MODULE.match_boxes(tracks, people) == [(1, 0), (0, 1)]


def test_line_error_is_restricted_to_reviewed_frames() -> None:
    camera = [_row(0, 1), _row(1, 1)]
    for row in camera:
        row["homography_img_to_pitch"] = [[1, 0, -50], [0, -1, 50], [0, 0, 1]]
    references = [
        {
            "source_frame": source_frame,
            "derived_frame_index": index,
            "metric_reference_status": "unavailable",
            "people": [],
            "pitch_lines": [{"Side line right": [{"x": 0.5, "y": 0.25}]}],
            "image_width": 200,
            "image_height": 200,
        }
        for index, source_frame in enumerate((251, 256))
    ]
    result, frames, _, _ = MODULE.evaluate_clip("example", camera, references, {251})
    assert result["line_error_px"]["count"] == 1
    assert frames[0]["line_point_count"] == 1
    assert frames[1]["line_point_count"] == 0


def test_gaps_count_missing_metric_frames_without_filling_them() -> None:
    rows = [_row(i, x) for i, x in enumerate((1, None, None, 4, 5))]
    assert MODULE._gaps(rows) == {"intervals": 1, "missing_frames": 2}
    smoothed = MODULE.smooth_records(rows)
    assert MODULE._gaps(smoothed) == MODULE._gaps(rows)
    assert all(smoothed[i]["tracks"][0]["smoothed_pitch_xy"] is None for i in (1, 2))
