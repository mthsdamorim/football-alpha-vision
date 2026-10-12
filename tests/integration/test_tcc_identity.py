"""Behavior checks for the frozen TCC image-space MOT protocol."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate-tcc-identity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tcc_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
match_boxes = MODULE.match_boxes


def _frozen_reference_packet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, review_source: str
) -> None:
    """Point the evaluator at a self-consistent packet built in ``tmp_path``.

    The real packet is derived from SoccerNet-GSR and is not versioned, so the
    guard tests reproduce only its hash relationships with the frozen protocol.
    """
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "reference.jsonl").write_text("{}\n")
    packet = {
        "protocol_sha256": MODULE.digest(MODULE.PROTOCOL),
        "reference_sha256": MODULE.digest(reference / "reference.jsonl"),
    }
    (reference / "manifest.json").write_text(json.dumps(packet))
    review = json.loads(MODULE.REVIEW.read_text())
    review.update(packet, source=review_source)
    review_path = tmp_path / "tcc-reference-review.json"
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(MODULE, "REFERENCE", reference)
    monkeypatch.setattr(MODULE, "REVIEW", review_path)


def test_evaluator_refuses_changed_human_review_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _frozen_reference_packet(monkeypatch, tmp_path, "changed-after-freeze")
    monkeypatch.setattr(MODULE, "DEFAULT_TRACKING", tmp_path / "missing-tracking")
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate-tcc-identity.py", "--output", str(tmp_path / "evaluation")],
    )

    with pytest.raises(
        ValueError, match="frozen protocol, reference or human review differs"
    ):
        MODULE.main()


def test_evaluator_refuses_changed_tracking_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _frozen_reference_packet(monkeypatch, tmp_path, "frozen")
    monkeypatch.setattr(MODULE, "FROZEN_REVIEW_SHA256", MODULE.digest(MODULE.REVIEW))
    tracking_root = tmp_path / "tracking"
    run = tracking_root / "SNGS-116"
    run.mkdir(parents=True)
    (run / "tracks.jsonl").write_text("{}\n")
    (run / "tracking-summary.json").write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-tcc-identity.py",
            "--tracking-root",
            str(tracking_root),
            "--output",
            str(tmp_path / "evaluation"),
        ],
    )

    with pytest.raises(ValueError, match="tracking differs from frozen baseline"):
        MODULE.main()


def test_match_boxes_prioritizes_cardinality_then_unrounded_iou_then_stable_tie() -> (
    None
):
    box = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
    tracks = [{"bbox": box}, {"bbox": box}]
    people = [
        {"source_annotation_id": "b", "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10}},
        {"source_annotation_id": "a", "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10}},
    ]
    assert match_boxes(tracks, people) == [(1, 0), (0, 1)]
    person = [
        {"source_annotation_id": "a", "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10}},
        {"source_annotation_id": "b", "bbox_image": {"x": 3, "y": 0, "w": 10, "h": 10}},
    ]
    candidates = [
        {"bbox": box},
        {"bbox": {"x_min": -3, "y_min": 0, "x_max": 7, "y_max": 10}},
    ]
    assert match_boxes(candidates, person) == [(0, 1), (1, 0)]
    assert match_boxes([], person) == []
    assert match_boxes(candidates, []) == []
    near_identical = [
        {"bbox": {"x_min": 0.0001, "y_min": 0, "x_max": 10.0001, "y_max": 10}},
        {"bbox": box},
    ]
    assert match_boxes(near_identical, person[:1]) == [(0, 1)]


def test_evaluator_counts_missed_person_switch_fragment_and_false_positive() -> None:
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for sample, predicted_ids in enumerate(((10,), (), (20, 30))):
        index = sample * 5
        references.append(
            {
                "clip": "fixture",
                "source_frame": 251 + index,
                "derived_frame_index": index,
                "pts": index * 1000,
                "time_base": "1/25000",
                "ignore_regions_x": [],
                "ignore_regions_y": [],
                "people": [
                    {
                        "source_annotation_id": str(sample),
                        "track_id": 1,
                        "role": "player",
                        "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10},
                    }
                ],
            }
        )
        predictions.append(
            {
                "frame_index": index,
                "pts": index * 1000,
                "time_base": "1/25000",
                "tracks": [
                    {
                        "track_id": track_id,
                        "bbox": {
                            "x_min": 20 if track_id == 30 else 0,
                            "y_min": 0,
                            "x_max": 30 if track_id == 30 else 10,
                            "y_max": 10,
                        },
                    }
                    for track_id in predicted_ids
                ],
            }
        )
    summary, trace = MODULE.evaluate_sequence(references, predictions)
    assert summary["reference_people"] == 3
    assert summary["matched"] == 2
    assert summary["missed"] == 1
    assert summary["false_positives"] == 1
    assert summary["idf1"] == pytest.approx(1 / 3)
    assert summary["id_switches"] == 1
    assert summary["fragmentations"] == 1
    assert [row["event"] for row in trace if row["kind"] == "reference"] == [
        "MATCH",
        "MISS",
        "SWITCH",
    ]


def test_evaluator_retains_stable_identity_and_refuses_unarbitrated_ignores() -> None:
    references: list[dict[str, Any]] = [
        {
            "clip": "fixture",
            "source_frame": 251 + index,
            "derived_frame_index": index,
            "pts": index * 1000,
            "time_base": "1/25000",
            "ignore_regions_x": [],
            "ignore_regions_y": [],
            "people": [
                {
                    "source_annotation_id": str(index),
                    "track_id": 1,
                    "role": "goalkeeper",
                    "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 10},
                }
            ],
        }
        for index in (0, 5)
    ]
    predictions: list[dict[str, Any]] = [
        {
            "frame_index": reference["derived_frame_index"],
            "pts": reference["pts"],
            "time_base": reference["time_base"],
            "tracks": [
                {
                    "track_id": 7,
                    "bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
                }
            ],
        }
        for reference in references
    ]
    metrics, trace = MODULE.evaluate_sequence(references, predictions)
    assert metrics["idf1"] == 1
    assert metrics["id_switches"] == 0
    assert metrics["fragmentations"] == 0
    assert metrics["player_goalkeeper"]["coverage"] == 1
    assert [row["event"] for row in trace] == ["MATCH", "MATCH"]
    references[0]["ignore_regions_x"] = [[0, 10]]
    with pytest.raises(ValueError, match="nonempty ignore regions"):
        MODULE.evaluate_sequence(references, predictions)
    references[0]["ignore_regions_x"] = []
    predictions[0]["pts"] += 1
    with pytest.raises(ValueError, match="frame/PTS mismatch"):
        MODULE.evaluate_sequence(references, predictions)


def test_mot_reducer_cannot_override_frozen_lexical_assignment() -> None:
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index in (0, 5):
        references.append(
            {
                "clip": "fixture",
                "source_frame": 251 + index,
                "derived_frame_index": index,
                "pts": index * 1000,
                "time_base": "1/25000",
                "ignore_regions_x": [],
                "ignore_regions_y": [],
                "people": [
                    {
                        "source_annotation_id": label,
                        "track_id": track_id,
                        "role": "player",
                        "bbox_image": {
                            "x": x if index == 0 else 0,
                            "y": 0,
                            "w": 10,
                            "h": 10,
                        },
                    }
                    for label, track_id, x in (("a", 1, 0), ("b", 2, 20))
                ],
            }
        )
        predictions.append(
            {
                "frame_index": index,
                "pts": index * 1000,
                "time_base": "1/25000",
                "tracks": [
                    {
                        "track_id": track_id,
                        "bbox": {
                            "x_min": x if index == 0 else 0,
                            "y_min": 0,
                            "x_max": (x if index == 0 else 0) + 10,
                            "y_max": 10,
                        },
                    }
                    for track_id, x in ((10, 20), (20, 0))
                ],
            }
        )
    metrics, trace = MODULE.evaluate_sequence(references, predictions)
    assert metrics["id_switches"] == 2
    assert [
        (row["reference_track_id"], row["prediction_track_id"], row["event"])
        for row in trace
        if row["derived_frame_index"] == 5
    ] == [(1, 10, "SWITCH"), (2, 20, "SWITCH")]
