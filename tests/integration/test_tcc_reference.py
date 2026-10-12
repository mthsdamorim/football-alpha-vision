"""The TCC reference packet keeps source images, labels, and derived time aligned."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import hashlib
import json
import runpy
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from PIL import Image

_SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/prepare-tcc-reference.py")
)
prepare: Any = _SCRIPT["prepare"]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source" / "SNGS-1"
    images = source / "img1"
    images.mkdir(parents=True)
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    video = video_dir / "SNGS-1.mp4"
    image_rows = []
    annotation_rows = []
    files = {}

    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "veryfast"}
        for number in range(1, 6):
            pixels = np.full((16, 16, 3), number * 20, dtype=np.uint8)
            image = images / f"{number:06}.jpg"
            Image.fromarray(pixels).save(image, quality=100, subsampling=0)
            files[f"img1/{number:06}.jpg"] = {"sha256": _sha(image)}
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            image_rows.append(
                {
                    "image_id": str(number),
                    "file_name": image.name,
                    "width": 16,
                    "height": 16,
                    "is_labeled": True,
                    "has_labeled_person": True,
                    "has_labeled_pitch": True,
                    "has_labeled_camera": number != 3,
                    "ignore_regions_x": [],
                    "ignore_regions_y": [],
                }
            )
            annotation_rows.extend(
                [
                    {
                        "id": f"person-{number}",
                        "image_id": str(number),
                        "supercategory": "object",
                        "track_id": 7,
                        "attributes": {"role": "player"},
                        "bbox_image": {
                            "x": 1,
                            "y": 9 if number == 5 else 1,
                            "w": 4,
                            "h": 7 if number == 5 else 8,
                        },
                        "bbox_pitch": {"x_bottom_middle": 2.0, "y_bottom_middle": 3.0},
                    },
                    {
                        "id": f"pitch-{number}",
                        "image_id": str(number),
                        "supercategory": "pitch",
                        "lines": {"Middle line": [{"x": 0.1, "y": 0.5}]},
                    },
                ]
            )
        for packet in stream.encode():
            container.mux(packet)

    labels = source / "Labels-GameState.json"
    _write_json(
        labels,
        {
            "info": {
                "version": "1.3",
                "frame_rate": 25,
                "game_id": "A",
                "name": "SNGS-1",
            },
            "images": image_rows,
            "annotations": annotation_rows,
        },
    )
    files[labels.name] = {"sha256": _sha(labels)}
    source_provenance = {"dataset": "fixture", "revision": "fixed", "split": "test"}
    source_manifest = source / "candidate-manifest.json"
    _write_json(source_manifest, {**source_provenance, "files": files})
    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {
            "schema": "tcc-reference-protocol/v1",
            "source": source_provenance,
            "frame_rate": 25,
            "window": {
                "start": 1,
                "end": 5,
                "sample_step": 2,
                "review_frames": [1, 5],
            },
            "clips": [
                {
                    "id": "SNGS-1",
                    "game_id": "A",
                    "labels_sha256": _sha(labels),
                    "source_manifest_sha256": _sha(source_manifest),
                }
            ],
        },
    )
    return protocol, source.parent, video_dir, tmp_path / "output"


def test_prepares_reference_with_explicit_unavailable_camera(tmp_path: Path) -> None:
    protocol, source, videos, output = _packet(tmp_path)
    result = prepare(protocol, source, videos, output)

    rows = [
        json.loads(line)
        for line in (output / "reference.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert [row["source_frame"] for row in rows] == [1, 3, 5]
    assert [Fraction(row["pts"]) * Fraction(row["time_base"]) for row in rows] == [
        Fraction(0),
        Fraction(2, 25),
        Fraction(4, 25),
    ]
    assert rows[1]["metric_reference_status"] == "unavailable_camera"
    assert rows[1]["people"][0]["bbox_pitch"] is None
    assert rows[1]["people"][0]["bbox_image"]["x"] == 1
    assert rows[1]["people"][0]["support_image"] == {
        "x": 3.0,
        "y": 9,
        "visibility": "pending_review",
        "uncertainty_px": None,
    }
    assert rows[2]["people"][0]["support_image"]["visibility"] == "image_edge_truncated"
    assert result["clips"]["SNGS-1"]["sampled_frames"] == 3
    assert result["review_status"] == "source_template_awaits_review_overlay"
    review = json.loads((output / "review.json").read_text())
    assert set(review["frames"]) == {"SNGS-1:1", "SNGS-1:5"}
    assert all(item["status"] == "pending" for item in review["frames"].values())


def test_tracked_protocol_matches_completed_human_review() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol = repository / "evaluation/tcc-reference-protocol.json"
    review = json.loads(
        (repository / "evaluation/tcc-reference-review.json").read_text()
    )

    assert _sha(protocol) == review["protocol_sha256"]
    assert review["reviewed_frame_count"] == len(review["clips_reviewed"]) * len(
        review["reviewed_frames_per_clip"]
    )
    assert review["unreviewed_frames_support_status"] == "not_qualified"
    assert review["metric_reference_release"] == (
        "reviewed_frame_subset_only_for_descriptive_gsr_agreement"
    )


def test_tracked_protocol_freezes_matching_and_smooth_only_variant() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol = json.loads(
        (repository / "evaluation/tcc-reference-protocol.json").read_text()
    )

    assert protocol["matching"] == {
        "iou_threshold": 0.5,
        "algorithm": "maximum-cardinality-maximum-sum-iou-bipartite",
        "tie_break": (
            "lexicographically-smallest-reference-source-annotation-id-"
            "prediction-ordinal"
        ),
    }
    assert protocol["smoothing"] == {
        "variant": "smooth-only-savgol-5f-degree-2-25fps",
        "coordinate": "metric-support-point",
        "segment": "contiguous-valid-same-local-id-metric-observations",
        "window_frames": 5,
        "degree": 2,
        "edge_policy": "unchanged",
        "gap_filling": "forbidden",
        "id_merging": "forbidden",
        "max_candidate_displacement_m": 0.5,
    }


def test_rejects_missing_source_frame_before_writing_packet(tmp_path: Path) -> None:
    protocol, source, videos, output = _packet(tmp_path)
    (source / "SNGS-1" / "img1" / "000003.jpg").unlink()

    with pytest.raises(ValueError, match="source image missing"):
        prepare(protocol, source, videos, output)
    assert not output.exists()


def test_rejects_changed_source_manifest(tmp_path: Path) -> None:
    protocol, source, videos, output = _packet(tmp_path)
    manifest = source / "SNGS-1" / "candidate-manifest.json"
    manifest.write_text(manifest.read_text() + "\n")

    with pytest.raises(ValueError, match="source manifest checksum"):
        prepare(protocol, source, videos, output)
    assert not output.exists()
