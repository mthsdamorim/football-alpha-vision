"""Validate the approved GSR sample and prepare read-only TCC references."""

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from football_alpha_video import decode_video


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _original_rgb(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def _reference_row(
    clip: str,
    number: int,
    image: dict,
    annotations: list[dict],
    frame,
    image_path: Path,
    image_sha256: str,
) -> dict:
    people = []
    distractors = []
    lines = []
    camera_available = image["has_labeled_camera"]
    for annotation in annotations:
        kind = annotation["supercategory"]
        if kind == "pitch":
            lines.append(annotation["lines"])
        elif kind == "object":
            role = annotation["attributes"]["role"]
            if role in {"player", "goalkeeper", "referee"}:
                box = annotation["bbox_image"]
                bottom = box["y"] + box["h"]
                people.append(
                    {
                        "source_annotation_id": annotation["id"],
                        "track_id": annotation["track_id"],
                        "role": role,
                        "team": annotation["attributes"].get("team"),
                        "bbox_image": box,
                        "support_image": {
                            "x": box["x"] + box["w"] / 2,
                            "y": bottom,
                            "visibility": (
                                "image_edge_truncated"
                                if bottom >= image["height"] - 1
                                else "pending_review"
                            ),
                            "uncertainty_px": None,
                        },
                        "bbox_pitch": (
                            annotation.get("bbox_pitch") if camera_available else None
                        ),
                    }
                )
            else:
                distractors.append(
                    {"source_annotation_id": annotation["id"], "role": role}
                )
    if not lines or not people:
        raise ValueError(f"{clip}:{number} lacks pitch or person references")
    return {
        "clip": clip,
        "source_frame": number,
        "derived_frame_index": frame.index,
        "pts": frame.pts,
        "time_base": str(frame.time_base),
        "time_semantics": "derived_mp4_not_broadcast_pts",
        "image_id": image["image_id"],
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "image_width": image["width"],
        "image_height": image["height"],
        "metric_reference_status": (
            "available_derived_reference" if camera_available else "unavailable_camera"
        ),
        "ignore_regions_x": image["ignore_regions_x"],
        "ignore_regions_y": image["ignore_regions_y"],
        "people": people,
        "distractors": distractors,
        "pitch_lines": lines,
    }


def prepare(
    protocol_path: Path, source_root: Path, video_root: Path, output: Path
) -> dict:
    """Reject misaligned inputs before writing a sampled reference packet."""
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != "tcc-reference-protocol/v1":
        raise ValueError("unsupported reference protocol")
    window = protocol["window"]
    start, end = window["start"], window["end"]
    step = window["sample_step"]
    rate = protocol["frame_rate"]
    review_frames = set(window["review_frames"])
    sample_frames = set(range(start, end + 1, step))
    if not review_frames <= sample_frames or step <= 0 or start > end or rate <= 0:
        raise ValueError("invalid sampling or review frames")

    rows = []
    review = {}
    clip_results = {}
    for chosen in protocol["clips"]:
        clip = chosen["id"]
        source_dir = source_root / clip
        labels_path = source_dir / "Labels-GameState.json"
        video_path = video_root / f"{clip}.mp4"
        source_manifest_path = source_dir / "candidate-manifest.json"
        source_manifest = _read_json(source_manifest_path)
        if _sha(source_manifest_path) != chosen["source_manifest_sha256"]:
            raise ValueError(f"{clip}: source manifest checksum differs from protocol")
        if any(
            source_manifest.get(key) != value
            for key, value in protocol["source"].items()
        ):
            raise ValueError(f"{clip}: source provenance differs from protocol")
        if _sha(labels_path) != chosen["labels_sha256"]:
            raise ValueError(f"{clip}: label checksum differs from protocol")
        if source_manifest["files"][labels_path.name]["sha256"] != _sha(labels_path):
            raise ValueError(f"{clip}: label checksum differs from source manifest")
        labels = _read_json(labels_path)
        info = labels["info"]
        if (
            str(info["game_id"]) != str(chosen["game_id"])
            or info["name"] != clip
            or info["frame_rate"] != rate
            or info["version"] != "1.3"
        ):
            raise ValueError(f"{clip}: source metadata differs from protocol")
        images = {
            int(Path(image["file_name"]).stem): image for image in labels["images"]
        }
        if len(images) != len(labels["images"]):
            raise ValueError(f"{clip}: duplicate image numbers")
        by_image = defaultdict(list)
        for annotation in labels["annotations"]:
            by_image[annotation["image_id"]].append(annotation)

        sampled = 0
        available = 0
        decoded = 0
        for frame in decode_video(video_path):
            number = start + frame.index
            if number > end:
                raise ValueError(f"{clip}: derived video has extra frames")
            image = images.get(number)
            image_path = source_dir / "img1" / f"{number:06}.jpg"
            if image is None or not image_path.is_file():
                raise ValueError(f"{clip}:{number}: source image missing")
            if image["file_name"] != image_path.name:
                raise ValueError(f"{clip}:{number}: image name differs from label")
            expected_sha = source_manifest["files"][f"img1/{number:06}.jpg"]["sha256"]
            if _sha(image_path) != expected_sha:
                raise ValueError(f"{clip}:{number}: source image checksum differs")
            if frame.timestamp != Fraction(frame.index, rate):
                raise ValueError(
                    f"{clip}:{number}: derived PTS differs from frame order"
                )
            if frame.image.shape != (image["height"], image["width"], 3):
                raise ValueError(f"{clip}:{number}: derived image dimensions differ")
            if not np.array_equal(frame.image, _original_rgb(image_path)):
                raise ValueError(f"{clip}:{number}: derived pixels differ from source")
            if not image["is_labeled"] or not image["has_labeled_person"]:
                raise ValueError(f"{clip}:{number}: person reference is unlabeled")
            decoded += 1
            if number not in sample_frames:
                continue
            if not image["has_labeled_pitch"]:
                raise ValueError(f"{clip}:{number}: pitch reference is unlabeled")
            row = _reference_row(
                clip,
                number,
                image,
                by_image[image["image_id"]],
                frame,
                image_path,
                expected_sha,
            )
            rows.append(row)
            sampled += 1
            available += image["has_labeled_camera"]
            if number in review_frames:
                review[f"{clip}:{number}"] = {
                    "status": "pending",
                    "source_image_sha256": expected_sha,
                    "source_label_sha256": chosen["labels_sha256"],
                    "annotator": None,
                    "prior_exposure": None,
                    "disagreements": [],
                    "corrections": [],
                    "visible_unlabeled_people": [],
                    "ignored_instances_or_regions": [],
                    "person_boxes_and_ids_checked": False,
                    "pitch_lines_checked": False,
                    "notes": None,
                }
        if decoded != end - start + 1 or sampled != len(sample_frames):
            raise ValueError(f"{clip}: derived video or sample is incomplete")
        clip_results[clip] = {
            "source_frames": decoded,
            "sampled_frames": sampled,
            "review_frames": len(review_frames),
            "camera_reference_available_frames": available,
            "camera_reference_unavailable_frames": sampled - available,
            "labels_sha256": chosen["labels_sha256"],
            "video_sha256": _sha(video_path),
        }

    output.mkdir(parents=True)
    reference = output / "reference.jsonl"
    reference.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    _write_json(
        output / "review.json",
        {"schema": "tcc-reference-review/v1", "frames": review},
    )
    result = {
        "schema": "tcc-reference-packet/v1",
        "protocol_sha256": _sha(protocol_path),
        "source": protocol["source"],
        "clips": clip_results,
        "reference_sha256": _sha(reference),
        "review_sha256": _sha(output / "review.json"),
        "review_status": "source_template_awaits_review_overlay",
    }
    _write_json(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.protocol, args.source_root, args.video_root, args.output),
            indent=2,
        )
    )
