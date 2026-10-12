"""Evaluate the frozen issue #19 projection and smooth-only variant offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from statistics import median

import av
import cv2
import numpy as np
from scipy.signal import savgol_filter
from tcc_protocol_matching import match_boxes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evaluation/tcc-reference-protocol.json"
REVIEW = ROOT / "evaluation/tcc-reference-review.json"
REFERENCE = ROOT / "artifacts/issue18/reference"
INPUT = ROOT / "artifacts/issue19"
LINE_AXES = {
    "Side line left": ("x", -52.5),
    "Side line right": ("x", 52.5),
    "Middle line": ("x", 0.0),
    "Big rect. left main": ("x", -36.0),
    "Big rect. right main": ("x", 36.0),
    "Small rect. left main": ("x", -47.0),
    "Small rect. right main": ("x", 47.0),
    "Side line top": ("y", 34.0),
    "Side line bottom": ("y", -34.0),
    "Big rect. left top": ("y", 20.16),
    "Big rect. right top": ("y", 20.16),
    "Big rect. left bottom": ("y", -20.16),
    "Big rect. right bottom": ("y", -20.16),
    "Small rect. left top": ("y", 9.16),
    "Small rect. right top": ("y", 9.16),
    "Small rect. left bottom": ("y", -9.16),
    "Small rect. right bottom": ("y", -9.16),
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _point(value: dict | list | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return np.asarray((value["x_m"], value["y_m"]), dtype=float)
    return np.asarray(value, dtype=float)


def _project(matrix: list | None, u: float, v: float) -> np.ndarray | None:
    if matrix is None:
        return None
    point = np.asarray(matrix, dtype=float) @ np.asarray((u, v, 1.0))
    if not np.isfinite(point).all() or abs(point[2]) < 1e-12:
        return None
    result = point[:2] / point[2]
    return result if np.isfinite(result).all() else None


def line_errors_px(
    matrix: list | None, lines: list[dict], width: int, height: int
) -> list[float]:
    """Distance of labeled straight-line pixels to the predicted image line."""
    if matrix is None:
        return []
    image_to_pitch = np.asarray(matrix, dtype=float)
    errors = []
    for item in lines:
        name, points = next(iter(item.items()))
        if name not in LINE_AXES:
            continue  # circles, elevated goals and unknown marks are not planar lines
        axis, coordinate = LINE_AXES[name]
        world_line = (
            np.asarray((1, 0, -coordinate), dtype=float)
            if axis == "x"
            else np.asarray((0, 1, -coordinate), dtype=float)
        )
        image_line = image_to_pitch.T @ world_line
        norm = float(np.hypot(*image_line[:2]))
        if not np.isfinite(image_line).all() or norm < 1e-12:
            continue
        for point in points:
            u, v = float(point["x"]) * width, float(point["y"]) * height
            errors.append(abs(float(image_line @ (u, v, 1))) / norm)
    return errors


def smooth_records(records: list[dict]) -> list[dict]:
    """Apply only the frozen centered five-frame degree-two policy."""
    result = deepcopy(records)
    segments: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    active: dict[int, list[tuple[int, int]]] = {}
    for frame_number, row in enumerate(result):
        present = set()
        for ordinal, track in enumerate(row["tracks"]):
            value = _point(track["pitch_xy"])
            track["smoothed_pitch_xy"] = value.tolist() if value is not None else None
            track["smoothing_reason"] = "unchanged"
            if value is None:
                continue
            track_id = track["track_id"]
            present.add(track_id)
            if track_id not in active:
                active[track_id] = []
                segments[track_id].append(active[track_id])
            active[track_id].append((frame_number, ordinal))
        for missing in active.keys() - present:
            del active[missing]
    for runs in segments.values():
        for run in runs:
            if len(run) < 5:
                continue
            values = np.asarray(
                [_point(result[f]["tracks"][o]["pitch_xy"]) for f, o in run]
            )
            candidates = savgol_filter(values, 5, 2, axis=0, mode="interp")
            for index in range(2, len(run) - 2):
                frame_number, ordinal = run[index]
                track = result[frame_number]["tracks"][ordinal]
                candidate = candidates[index]
                if float(np.linalg.norm(candidate - values[index])) > 0.5:
                    track["smoothing_reason"] = "candidate_displacement_over_0.5m"
                else:
                    track["smoothed_pitch_xy"] = candidate.tolist()
                    track["smoothing_reason"] = "savgol_5f_degree_2"
    return result


def metric_pairs(row: dict, reference: dict, review_frames: set[int]) -> dict:
    """Return only qualified reviewed GSR agreement pairs, with denominators."""
    if reference["source_frame"] not in review_frames:
        return {"eligible": 0, "pairs": [], "excluded": "unreviewed"}
    if reference["metric_reference_status"] != "available_derived_reference":
        return {"eligible": 0, "pairs": [], "excluded": "reference_camera_absent"}
    people = reference["people"]
    qualified = [
        i
        for i, person in enumerate(people)
        if person["support_image"]["visibility"] != "image_edge_truncated"
        and person["bbox_pitch"] is not None
    ]
    matches = dict(match_boxes(row["tracks"], people))
    pairs = []
    for i in qualified:
        if i not in matches:
            continue
        person = people[i]
        track = row["tracks"][matches[i]]
        support = person["support_image"]
        reference_pitch = np.asarray(
            (
                person["bbox_pitch"]["x_bottom_middle"],
                -person["bbox_pitch"]["y_bottom_middle"],
            )
        )
        camera_only = _project(
            row["homography_img_to_pitch"], support["x"], support["y"]
        )
        original = _point(track["pitch_xy"])
        smooth = _point(track.get("smoothed_pitch_xy", track["pitch_xy"]))
        predicted_support = track["support_point_uv"]
        pairs.append(
            {
                "source_annotation_id": person["source_annotation_id"],
                "prediction_ordinal": matches[i],
                "source_frame": reference["source_frame"],
                "track_id": track["track_id"],
                "camera_only_error_m": float(
                    np.linalg.norm(camera_only - reference_pitch)
                )
                if camera_only is not None
                else None,
                "combined_original_error_m": float(
                    np.linalg.norm(original - reference_pitch)
                )
                if original is not None
                else None,
                "combined_smoothed_error_m": float(
                    np.linalg.norm(smooth - reference_pitch)
                )
                if smooth is not None
                else None,
                "support_error_px": float(
                    np.hypot(
                        predicted_support["u"] - support["x"],
                        predicted_support["v"] - support["y"],
                    )
                ),
                "smoothing_displacement_m": float(np.linalg.norm(smooth - original))
                if smooth is not None and original is not None
                else None,
            }
        )
    return {
        "eligible": len(qualified),
        "matched": len(pairs),
        "pairs": pairs,
        "excluded": None,
    }


def _distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "p90": float(np.percentile(values, 90)) if values else None,
        "maximum": max(values) if values else None,
    }


def _jitter(records: list[dict], key: str) -> list[float]:
    by_id: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row in records:
        for track in row["tracks"]:
            point = _point(track[key])
            if point is not None:
                by_id[track["track_id"]].append((row["frame_index"], point))
    errors = []
    for positions in by_id.values():
        for a, b, c in zip(positions, positions[1:], positions[2:], strict=False):
            if a[0] + 1 == b[0] and b[0] + 1 == c[0]:
                errors.append(float(np.linalg.norm(b[1] - (a[1] + c[1]) / 2)))
    return errors


def _gaps(records: list[dict]) -> dict[str, int]:
    indices: dict[int, list[int]] = defaultdict(list)
    for row in records:
        for track in row["tracks"]:
            if track["pitch_xy"] is not None:
                indices[track["track_id"]].append(row["frame_index"])
    spans = [
        b - a - 1
        for frames in indices.values()
        for a, b in pairwise(frames)
        if b - a > 1
    ]
    return {"intervals": len(spans), "missing_frames": sum(spans)}


def _validate_clip(
    clip: str,
    source: Path,
    tracks: list[dict],
    tracking_summary: dict,
    camera: list[dict],
    camera_summary: dict,
    reference: list[dict],
    packet: dict,
) -> None:
    if digest(source) != packet["clips"][clip]["video_sha256"]:
        raise ValueError(f"{clip}: source video hash differs from approved packet")
    if tracking_summary["source_sha256"] != digest(source) or camera_summary[
        "source_sha256"
    ] != digest(source):
        raise ValueError(f"{clip}: model outputs refer to a different source")
    if len(tracks) != len(camera) or len(tracks) != 250 or len(reference) != 50:
        raise ValueError(f"{clip}: frame count differs from frozen window")
    if (
        tracking_summary["detector"]["checkpoint_sha256"]
        != "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
    ):
        raise ValueError(f"{clip}: RF-DETR checkpoint differs from frozen baseline")
    if (
        tracking_summary["detector"]["rfdetr_version"] != "1.9.4"
        or tracking_summary["detector"]["minimum_confidence"] != 0.1
        or tracking_summary["detector"]["batch_size"] != 4
    ):
        raise ValueError(f"{clip}: detector parameters differ from frozen baseline")
    if tracking_summary["tracker"]["version"] != "2.6.0" or tracking_summary["tracker"][
        "parameters"
    ] != {
        "high_conf_det_threshold": 0.6,
        "lost_track_buffer": 30,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "track_activation_threshold": 0.7,
    }:
        raise ValueError(f"{clip}: tracker parameters differ from frozen baseline")
    if camera_summary["upstream_sha"] != "8c87391d6f4ea40c5e4d65e61529916c7a49ce62":
        raise ValueError(f"{clip}: PnLCalib upstream differs from frozen baseline")
    if camera_summary["weights_sha256"] != {
        "SV_kp": "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113",
        "SV_lines": "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455",
    } or camera_summary["parameters"] != {
        "keypoint_threshold": 0.3434,
        "line_threshold": 0.7867,
        "pnl_refine": True,
        "temporal_smoothing": False,
    }:
        raise ValueError(f"{clip}: PnLCalib baseline differs from frozen protocol")
    with av.open(str(source)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            expected = (index, frame.pts, str(frame.time_base))
            for item in (tracks[index], camera[index]):
                if (item["frame_index"], item["pts"], item["time_base"]) != expected:
                    raise ValueError(
                        f"{clip}: source/model frame identity mismatch at {index}"
                    )
            if index % 5 == 0:
                ref = reference[index // 5]
                if (
                    ref["derived_frame_index"],
                    ref["pts"],
                    ref["time_base"],
                ) != expected:
                    raise ValueError(
                        f"{clip}: reference frame identity mismatch at {index}"
                    )
        if index != 249:
            raise ValueError(f"{clip}: source frame count differs")
    for original, projected in zip(tracks, camera, strict=True):
        if len(original["tracks"]) != len(projected["tracks"]):
            raise ValueError(f"{clip}: camera dropped an image observation")
        for a, b in zip(original["tracks"], projected["tracks"], strict=True):
            if any(
                a[key] != b[key] for key in ("track_id", "bbox", "support_point_uv")
            ):
                raise ValueError(f"{clip}: camera changed ID, box or support")
            if (
                projected["homography_img_to_pitch"] is None
                and b["pitch_xy"] is not None
            ):
                raise ValueError(f"{clip}: metric position on camera abstention")
    for video_path in (
        INPUT / clip / "tracking.mp4",
        INPUT / clip / "camera/preview.mp4",
    ):
        with av.open(str(video_path)) as video:
            times = [
                (frame.pts, str(frame.time_base)) for frame in video.decode(video=0)
            ]
        if times != [(row["pts"], row["time_base"]) for row in camera]:
            raise ValueError(f"{clip}: diagnostic video differs in PTS or frame count")


def evaluate_clip(
    clip: str, camera: list[dict], reference: list[dict], review_frames: set[int]
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    smooth = smooth_records(camera)
    frame_rows = []
    all_pairs = []
    for ref in reference:
        index = ref["derived_frame_index"]
        row = smooth[index]
        line_errors = (
            line_errors_px(
                row["homography_img_to_pitch"],
                ref["pitch_lines"],
                ref["image_width"],
                ref["image_height"],
            )
            if ref["source_frame"] in review_frames
            else []
        )
        metric = metric_pairs(row, ref, review_frames)
        all_pairs.extend(metric["pairs"])
        frame_rows.append(
            {
                "source_frame": ref["source_frame"],
                "derived_frame_index": index,
                "reference_camera_available": ref["metric_reference_status"]
                == "available_derived_reference",
                "camera_accepted": row["homography_img_to_pitch"] is not None,
                "line_point_count": len(line_errors),
                "line_error_px": _distribution(line_errors),
                "metric_eligible_people": metric["eligible"],
                "metric_matched_people": metric.get("matched", 0),
                "metric_exclusion": metric["excluded"],
            }
        )
    sampled = [smooth[r["derived_frame_index"]] for r in reference]
    projected = sum(t["pitch_xy"] is not None for row in camera for t in row["tracks"])
    observations = sum(len(row["tracks"]) for row in camera)
    metrics = {
        key: _distribution([p[key] for p in all_pairs if p[key] is not None])
        for key in (
            "camera_only_error_m",
            "combined_original_error_m",
            "combined_smoothed_error_m",
            "support_error_px",
            "smoothing_displacement_m",
        )
    }
    error_deltas = [
        p["combined_smoothed_error_m"] - p["combined_original_error_m"]
        for p in all_pairs
        if p["combined_smoothed_error_m"] is not None
        and p["combined_original_error_m"] is not None
    ]
    result = {
        "clip": clip,
        "source_frames": len(camera),
        "sampled_frames": len(reference),
        "camera_accepted_all_frames": sum(
            r["homography_img_to_pitch"] is not None for r in camera
        ),
        "reference_camera_available_sampled_frames": sum(
            r["reference_camera_available"] for r in frame_rows
        ),
        "camera_accepted_sampled_frames": sum(r["camera_accepted"] for r in frame_rows),
        "sampled_line_frames_with_error": sum(
            r["line_point_count"] > 0 for r in frame_rows
        ),
        "line_error_px": _distribution(
            [
                e
                for ref in reference
                if ref["source_frame"] in review_frames
                for e in line_errors_px(
                    camera[ref["derived_frame_index"]]["homography_img_to_pitch"],
                    ref["pitch_lines"],
                    ref["image_width"],
                    ref["image_height"],
                )
            ]
        ),
        "observations_all_frames": observations,
        "projected_all_frames": projected,
        "outside_pitch_projected_all_frames": sum(
            track["inside_pitch"] is False for row in camera for track in row["tracks"]
        ),
        "metric_eligible_reviewed_people": sum(
            r["metric_eligible_people"] for r in frame_rows
        ),
        "metric_matched_reviewed_people": len(all_pairs),
        "metric_projected_reviewed_people": sum(
            p["combined_original_error_m"] is not None for p in all_pairs
        ),
        "reviewed_reference_people": sum(
            len(ref["people"])
            for ref in reference
            if ref["source_frame"] in review_frames
        ),
        "reviewed_edge_truncated_people": sum(
            person["support_image"]["visibility"] == "image_edge_truncated"
            for ref in reference
            if ref["source_frame"] in review_frames
            for person in ref["people"]
        ),
        "reviewed_people_without_reference_camera": sum(
            len(ref["people"])
            for ref in reference
            if ref["source_frame"] in review_frames
            and ref["metric_reference_status"] != "available_derived_reference"
        ),
        "jitter_original_m": _distribution(_jitter(camera, "pitch_xy")),
        "jitter_smoothed_m": _distribution(_jitter(smooth, "smoothed_pitch_xy")),
        "metric": metrics,
        "smoothing_error_change": {
            "paired_count": len(error_deltas),
            "worsened_count": sum(value > 1e-9 for value in error_deltas),
            "improved_count": sum(value < -1e-9 for value in error_deltas),
            "unchanged_count": sum(abs(value) <= 1e-9 for value in error_deltas),
            "maximum_worsening_m": max(
                (value for value in error_deltas if value > 0), default=None
            ),
            "delta_m": _distribution(error_deltas),
        },
        "original_metric_gaps": _gaps(camera),
        "gaps_filled": 0,
        "smoothing_rejected_observations": sum(
            t["smoothing_reason"] == "candidate_displacement_over_0.5m"
            for r in smooth
            for t in r["tracks"]
        ),
        "sampled_observations": sum(len(r["tracks"]) for r in sampled),
    }
    return result, frame_rows, smooth, all_pairs


def _render_example(
    reference: dict, camera_row: dict, destination: Path, caption: str
) -> None:
    image = cv2.imread(str(ROOT / reference["image_path"]))
    if image is None:
        raise ValueError("reference image unavailable for visual example")
    homography = camera_row["homography_img_to_pitch"]
    try:
        pitch_to_image = (
            np.linalg.inv(np.asarray(homography, dtype=float))
            if homography is not None
            else None
        )
    except np.linalg.LinAlgError:
        pitch_to_image = None
    for item in reference["pitch_lines"]:
        name, points = next(iter(item.items()))
        if name not in LINE_AXES:
            continue
        labelled = np.asarray(
            [
                (round(p["x"] * image.shape[1]), round(p["y"] * image.shape[0]))
                for p in points
            ],
            dtype=np.int32,
        )
        if len(labelled) > 1:
            cv2.polylines(image, [labelled], False, (0, 220, 0), 3)
        for point in labelled:
            cv2.circle(image, tuple(int(v) for v in point), 4, (0, 220, 0), -1)
        if pitch_to_image is None:
            continue
        axis, coordinate = LINE_AXES[name]
        endpoints = (
            ((coordinate, -34), (coordinate, 34))
            if axis == "x"
            else ((-52.5, coordinate), (52.5, coordinate))
        )
        projected = [pitch_to_image @ np.asarray((*point, 1)) for point in endpoints]
        if any(
            not np.isfinite(point).all() or abs(point[2]) < 1e-12 for point in projected
        ):
            continue
        pixels = [point[:2] / point[2] for point in projected]
        if max(abs(float(value)) for point in pixels for value in point) > 1e7:
            continue
        start, end = (tuple(round(value) for value in point) for point in pixels)
        visible, start, end = cv2.clipLine(
            (0, 0, image.shape[1], image.shape[0]), start, end
        )
        if visible:
            cv2.line(image, start, end, (0, 0, 255), 2, cv2.LINE_AA)
    for person in reference["people"]:
        box = person["bbox_image"]
        cv2.rectangle(
            image,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            (0, 220, 0),
            2,
        )
    for track in camera_row["tracks"]:
        box = track["bbox"]
        cv2.rectangle(
            image,
            (round(box["x_min"]), round(box["y_min"])),
            (round(box["x_max"]), round(box["y_max"])),
            (0, 0, 255),
            2,
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(
        image,
        caption,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), image):
        raise ValueError("could not write visual example")


def _number(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _readout(summary: dict) -> str:
    lines = [
        "# Issue #19: frozen projection and stability evaluation",
        "",
        "All timestamps are derived from JPEG order at 25 fps, not broadcast PTS.",
        "Metres below measure agreement with the GSR-derived reference on reviewed",
        "support points; they are not independent physical-position accuracy.",
        "Unreviewed support points do not enter metre denominators.",
        "",
        "| Clip | Camera/250 | Lines/10 | Line median/p90 px | "
        "Support projected/matched/eligible | Camera median m | Combined median m | "
        "Combined p90 m | Jitter p90 m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in summary["clips"]:
        metric = clip["metric"]
        line = clip["line_error_px"]
        jitter_before = clip["jitter_original_m"]
        jitter_after = clip["jitter_smoothed_m"]
        lines.append(
            f"| {clip['clip']} | {clip['camera_accepted_all_frames']}/250 | "
            f"{clip['sampled_line_frames_with_error']}/10 | "
            f"{_number(line['median'])} / {_number(line['p90'])} | "
            f"{clip['metric_projected_reviewed_people']}/"
            f"{clip['metric_matched_reviewed_people']}/"
            f"{clip['metric_eligible_reviewed_people']} | "
            f"{_number(metric['camera_only_error_m']['median'])} | "
            f"{_number(metric['combined_original_error_m']['median'])} / "
            f"{_number(metric['combined_smoothed_error_m']['median'])} | "
            f"{_number(metric['combined_original_error_m']['p90'])} / "
            f"{_number(metric['combined_smoothed_error_m']['p90'])} | "
            f"{_number(jitter_before['p90'])} / {_number(jitter_after['p90'])} |"
        )
    lines.extend(
        [
            "",
            "| Clip | Reviewed people | Edge excluded | No GSR camera | "
            "Metric eligible | Unmatched | Matched without projection |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for clip in summary["clips"]:
        eligible = clip["metric_eligible_reviewed_people"]
        matched = clip["metric_matched_reviewed_people"]
        projected = clip["metric_projected_reviewed_people"]
        lines.append(
            f"| {clip['clip']} | {clip['reviewed_reference_people']} | "
            f"{clip['reviewed_edge_truncated_people']} | "
            f"{clip['reviewed_people_without_reference_camera']} | "
            f"{eligible} | {eligible - matched} | {matched - projected} |"
        )
    lines.extend(
        [
            "",
            "| Clip | Worse support error / paired | Max worsening m | "
            "Rejected smoothing candidates | Metric gap intervals / frames | "
            "Projected / outside pitch / observed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for clip in summary["clips"]:
        change = clip["smoothing_error_change"]
        gaps = clip["original_metric_gaps"]
        lines.append(
            f"| {clip['clip']} | {change['worsened_count']}/{change['paired_count']} | "
            f"{_number(change['maximum_worsening_m'])} | "
            f"{clip['smoothing_rejected_observations']} | "
            f"{gaps['intervals']}/{gaps['missing_frames']} | "
            f"{clip['projected_all_frames']}/"
            f"{clip['outside_pitch_projected_all_frames']}/"
            f"{clip['observations_all_frames']} |"
        )
    lines.extend(
        [
            "",
            "Jitter is the centre point's deviation from its two contiguous",
            "neighbours. It is paired across original and smoothed points; no gaps",
            "or IDs are filled. The smoother is descriptive and has no promotion gate.",
            "Camera-only uses the reviewed reference image support with the predicted",
            "camera. Combined uses the tracked box support with the same camera.",
            "A support-only metre result cannot be isolated: the GSR packet exposes",
            "derived `bbox_pitch`, but no independently qualified camera homography.",
            "The separate pixel support error is in summary.json. Missing cameras,",
            "unmatched people and edge-truncated supports have explicit counts.",
            "",
            "## Visual examples",
            "",
        ]
    )
    for clip in summary["clips"]:
        clip_id = clip["clip"]
        lines.append(
            f"- {clip_id}: [lower line error]({clip_id}/lowest-line-error.jpg), "
            f"[higher line error]({clip_id}/highest-line-error.jpg); "
            "green boxes are GSR, red boxes are predictions."
        )
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "After reproducing the tracking and PnLCalib outputs, run:",
            "",
            "```bash",
            "uv run python scripts/evaluate-tcc-projection.py \\",
            "  --output artifacts/issue19/evaluation-replay",
            "```",
            "",
            "This verifies source, reference, review,",
            "model provenance, frame/PTS identity and both videos before calculating",
            "these tables. `summary.json`, per-clip `frames.json`,",
            "`metric-pairs.jsonl` and `smoothed.jsonl` are the evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=INPUT / "evaluation")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    review = json.loads(REVIEW.read_text())
    packet = json.loads((REFERENCE / "manifest.json").read_text())
    if (
        digest(PROTOCOL) != packet["protocol_sha256"]
        or digest(REFERENCE / "reference.jsonl") != packet["reference_sha256"]
    ):
        raise ValueError("issue #18 packet differs from frozen protocol/reference")
    if (review["protocol_sha256"], review["reference_sha256"]) != (
        packet["protocol_sha256"],
        packet["reference_sha256"],
    ):
        raise ValueError("human review does not cover current packet")
    references = read_jsonl(REFERENCE / "reference.jsonl")
    review_frames = set(review["reviewed_frames_per_clip"])
    results = []
    args.output.mkdir(parents=True, exist_ok=False)
    for item in protocol["clips"]:
        clip = item["id"]
        source = ROOT / f"artifacts/issue18/derived/{clip}.mp4"
        run = INPUT / clip
        tracks = read_jsonl(run / "tracks.jsonl")
        camera = read_jsonl(run / "camera/frames.jsonl")
        tracking_summary = json.loads((run / "tracking-summary.json").read_text())
        camera_summary = json.loads((run / "camera/summary.json").read_text())
        clip_reference = [r for r in references if r["clip"] == clip]
        _validate_clip(
            clip,
            source,
            tracks,
            tracking_summary,
            camera,
            camera_summary,
            clip_reference,
            packet,
        )
        result, frame_rows, smooth, pairs = evaluate_clip(
            clip, camera, clip_reference, review_frames
        )
        result["input_sha256"] = {
            name: digest(path)
            for name, path in {
                "source": source,
                "tracking_jsonl": run / "tracks.jsonl",
                "tracking_summary": run / "tracking-summary.json",
                "camera_jsonl": run / "camera/frames.jsonl",
                "camera_summary": run / "camera/summary.json",
                "camera_video": run / "camera/preview.mp4",
            }.items()
        }
        results.append(result)
        output = args.output / clip
        output.mkdir()
        (output / "frames.json").write_text(json.dumps(frame_rows, indent=2) + "\n")
        with (output / "metric-pairs.jsonl").open("w") as file:
            for pair in pairs:
                file.write(json.dumps(pair, allow_nan=False) + "\n")
        with (output / "smoothed.jsonl").open("w") as file:
            for row in smooth:
                file.write(json.dumps(row, allow_nan=False) + "\n")
        ranked = [row for row in frame_rows if row["line_point_count"]]
        if ranked:
            for label, selected in (
                (
                    "lowest-line-error",
                    min(ranked, key=lambda row: row["line_error_px"]["median"]),
                ),
                (
                    "highest-line-error",
                    max(ranked, key=lambda row: row["line_error_px"]["median"]),
                ),
            ):
                ref = clip_reference[selected["derived_frame_index"] // 5]
                _render_example(
                    ref,
                    camera[selected["derived_frame_index"]],
                    output / f"{label}.jpg",
                    f"{clip}:{selected['source_frame']} {label} | green ref, red pred",
                )
    summary = {
        "schema": "tcc-projection-evaluation/v1",
        "issue": 19,
        "protocol_sha256": digest(PROTOCOL),
        "reference_sha256": digest(REFERENCE / "reference.jsonl"),
        "review_sha256": digest(REVIEW),
        "time_semantics": "derived_25fps_not_broadcast_pts",
        "metre_claim": "descriptive_agreement_with_gsr_derived_reference_only",
        "variant": protocol["smoothing"]["variant"],
        "clips": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "readout.md").write_text(_readout(summary))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
