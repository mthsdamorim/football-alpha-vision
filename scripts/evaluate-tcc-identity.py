"""Evaluate frozen local identities on the approved GSR sample at 5 fps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import av
import cv2
import motmetrics as mm
import numpy as np
from tcc_protocol_matching import box_iou, match_boxes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evaluation/tcc-reference-protocol.json"
REVIEW = ROOT / "evaluation/tcc-reference-review.json"
REFERENCE = ROOT / "artifacts/issue18/reference"
DEFAULT_TRACKING = ROOT / "artifacts/issue19"
FROZEN_REVIEW_SHA256 = (
    "9d37c8a25088a75dd77e52a8ac721d59f0c89b0b3f325a1e4b651c9721909ba0"
)
FROZEN_TRACKING_SHA256 = {
    "SNGS-116": (
        "8298a96f6588af04b68be5464fe3aff0ac45c0001227491bfbb7e4a8cfb4269c",
        "0d2c90cbbb72c5ecbada3af1a41dfb3b2d1d3c35d8f83c072bd31c7030c8577c",
    ),
    "SNGS-132": (
        "580c5c82b562998050ca134e192d56601bfc16e1eae7ba0410dd68a4169109c2",
        "4e1cfab71ce75439722126eed64d1a15bd70bf56e4c921ca68cd3d7a881bc4da",
    ),
    "SNGS-187": (
        "d1e5a022903392ff767f23d354d0fd5cc2df7f552bf64e26b9132d0bc77dd7ac",
        "b4a0d4bdd8ee74e6694382d82b8922faa33f714f3424865424ee093c8324bd63",
    ),
}
METRICS = [
    "idf1",
    "idtp",
    "idfp",
    "idfn",
    "num_switches",
    "num_fragmentations",
    "num_misses",
    "num_false_positives",
    "num_objects",
    "num_predictions",
    "num_unique_objects",
    "num_frames",
]


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def evaluate_sequence(
    references: list[dict], predictions: list[dict]
) -> tuple[dict, list[dict]]:
    """Feed only frozen per-frame assignments to the standard MOT reducer."""
    if len(references) != len(predictions):
        raise ValueError("sampled reference and prediction counts differ")
    accumulator = mm.MOTAccumulator(auto_id=False)
    trace: list[dict] = []
    per_identity: dict[int, dict] = defaultdict(
        lambda: {"reference_samples": 0, "matched_samples": 0, "predicted_ids": set()}
    )
    player_goalkeeper = {"reference_people": 0, "matched": 0}
    distractors = 0
    for reference, prediction in zip(references, predictions, strict=True):
        frame = reference["derived_frame_index"]
        if (prediction["frame_index"], prediction["pts"], prediction["time_base"]) != (
            frame,
            reference["pts"],
            reference["time_base"],
        ):
            raise ValueError(f"frame/PTS mismatch at {frame}")
        if reference["ignore_regions_x"] or reference["ignore_regions_y"]:
            raise ValueError("nonempty ignore regions require explicit arbitration")
        people = reference["people"]
        tracks = prediction["tracks"]
        gt_ids = [person["track_id"] for person in people]
        pred_ids = [track["track_id"] for track in tracks]
        if len(set(gt_ids)) != len(gt_ids) or len(set(pred_ids)) != len(pred_ids):
            raise ValueError(f"duplicate local identity within frame {frame}")
        pairs = match_boxes(tracks, people)
        distances = np.full((len(people), len(tracks)), np.nan)
        for ref_index, pred_index in pairs:
            distances[ref_index, pred_index] = 1.0 - box_iou(
                tracks[pred_index]["bbox"], people[ref_index]["bbox_image"]
            )
        accumulator.update(gt_ids, pred_ids, distances, frameid=frame)
        matched = dict(pairs)
        paired_predictions = set(matched.values())
        distractors += len(reference.get("distractors", []))
        for ref_index, person in enumerate(people):
            pred_index = matched.get(ref_index)
            track = tracks[pred_index] if pred_index is not None else None
            identity = per_identity[person["track_id"]]
            identity["reference_samples"] += 1
            if track is not None:
                identity["matched_samples"] += 1
                identity["predicted_ids"].add(track["track_id"])
            if person["role"] in {"player", "goalkeeper"}:
                player_goalkeeper["reference_people"] += 1
                player_goalkeeper["matched"] += track is not None
            trace.append(
                {
                    "kind": "reference",
                    "clip": reference["clip"],
                    "source_frame": reference["source_frame"],
                    "derived_frame_index": frame,
                    "pts": reference["pts"],
                    "time_base": reference["time_base"],
                    "source_annotation_id": person["source_annotation_id"],
                    "reference_track_id": person["track_id"],
                    "role": person["role"],
                    "reference_bbox": person["bbox_image"],
                    "prediction_ordinal": pred_index,
                    "prediction_track_id": track["track_id"] if track else None,
                    "prediction_bbox": track["bbox"] if track else None,
                    "iou": float(1.0 - distances[ref_index, pred_index])
                    if track is not None
                    else None,
                }
            )
        for pred_index, track in enumerate(tracks):
            if pred_index not in paired_predictions:
                trace.append(
                    {
                        "kind": "false_positive",
                        "clip": reference["clip"],
                        "source_frame": reference["source_frame"],
                        "derived_frame_index": frame,
                        "pts": reference["pts"],
                        "time_base": reference["time_base"],
                        "source_annotation_id": None,
                        "reference_track_id": None,
                        "role": None,
                        "reference_bbox": None,
                        "prediction_ordinal": pred_index,
                        "prediction_track_id": track["track_id"],
                        "prediction_bbox": track["bbox"],
                        "iou": None,
                    }
                )
    events = accumulator.events.reset_index()
    event_by_reference = {
        (int(row.FrameId), int(row.OId)): row.Type
        for row in events.itertuples()
        if row.Type in {"MATCH", "SWITCH", "MISS"}
    }
    for row in trace:
        row["event"] = (
            event_by_reference[(row["derived_frame_index"], row["reference_track_id"])]
            if row["kind"] == "reference"
            else "FP"
        )
    result = mm.metrics.create().compute(accumulator, metrics=METRICS, name="clip")
    metrics = {name: result.loc["clip", name] for name in METRICS}
    reference_count = len([row for row in trace if row["kind"] == "reference"])
    matched_count = len(
        [row for row in trace if row["kind"] == "reference" and row["event"] != "MISS"]
    )
    false_positives = len([row for row in trace if row["kind"] == "false_positive"])
    if (
        metrics["num_objects"] != reference_count
        or metrics["num_misses"] != reference_count - matched_count
        or metrics["num_false_positives"] != false_positives
    ):
        raise ValueError("MOT reducer disagrees with frozen assignment counts")
    identities = {
        str(identity): {
            "reference_samples": value["reference_samples"],
            "matched_samples": value["matched_samples"],
            "coverage": value["matched_samples"] / value["reference_samples"],
            "predicted_local_ids": sorted(value["predicted_ids"]),
        }
        for identity, value in sorted(per_identity.items())
    }
    summary = {
        "sampled_frames": len(references),
        "reference_people": reference_count,
        "matched": matched_count,
        "missed": reference_count - matched_count,
        "false_positives": false_positives,
        "detection_coverage": matched_count / reference_count
        if reference_count
        else None,
        "player_goalkeeper": {
            **player_goalkeeper,
            "coverage": player_goalkeeper["matched"]
            / player_goalkeeper["reference_people"]
            if player_goalkeeper["reference_people"]
            else None,
        },
        "distractor_labels": distractors,
        "ignored_regions": 0,
        "reference_local_ids": len(identities),
        "predicted_local_ids": len(
            {t["track_id"] for row in predictions for t in row["tracks"]}
        ),
        "idf1": float(metrics["idf1"]),
        "idtp": int(metrics["idtp"]),
        "idfp": int(metrics["idfp"]),
        "idfn": int(metrics["idfn"]),
        "id_switches": int(metrics["num_switches"]),
        "fragmentations": int(metrics["num_fragmentations"]),
        "per_reference_identity": identities,
    }
    return summary, trace


def validate_inputs(
    clip: str,
    source: Path,
    packet: dict,
    references: list[dict],
    predictions: list[dict],
    tracking_summary: dict,
) -> list[dict]:
    if digest(source) != packet["clips"][clip]["video_sha256"]:
        raise ValueError(f"{clip}: source differs from approved packet")
    if tracking_summary["source_sha256"] != digest(source):
        raise ValueError(f"{clip}: tracking summary refers to another source")
    if (
        len(predictions) != 250
        or len(references) != 50
        or tracking_summary["frame_count"] != 250
    ):
        raise ValueError(f"{clip}: frozen frame count differs")
    detector = tracking_summary["detector"]
    tracker = tracking_summary["tracker"]
    if (
        detector["checkpoint_sha256"]
        != "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
        or detector["rfdetr_version"] != "1.9.4"
        or detector["minimum_confidence"] != 0.1
        or detector["batch_size"] != 4
    ):
        raise ValueError(f"{clip}: detector profile differs from frozen baseline")
    if tracker["version"] != "2.6.0" or tracker["parameters"] != {
        "high_conf_det_threshold": 0.6,
        "lost_track_buffer": 30,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "track_activation_threshold": 0.7,
    }:
        raise ValueError(f"{clip}: tracker profile differs from frozen baseline")
    with av.open(str(source)) as video:
        source_times = [(f.pts, str(f.time_base)) for f in video.decode(video=0)]
    if len(source_times) != 250:
        raise ValueError(f"{clip}: source video frame count differs")
    for index, row in enumerate(predictions):
        if (row["frame_index"], row["pts"], row["time_base"]) != (
            index,
            *source_times[index],
        ):
            raise ValueError(f"{clip}: tracking frame/PTS differs at {index}")
    sampled = []
    for ordinal, reference in enumerate(references):
        index = ordinal * 5
        if (
            reference["clip"],
            reference["source_frame"],
            reference["derived_frame_index"],
            reference["pts"],
            reference["time_base"],
        ) != (clip, 251 + index, index, *source_times[index]):
            raise ValueError(f"{clip}: reference frame/PTS differs at {index}")
        sampled.append(predictions[index])
    return sampled


def render_example(reference: dict, trace_row: dict, output: Path) -> None:
    image = cv2.imread(str(ROOT / reference["image_path"]))
    if image is None:
        raise ValueError(
            f"missing source JPEG for {reference['clip']}:{reference['source_frame']}"
        )
    rb = trace_row["reference_bbox"]
    if rb is not None:
        cv2.rectangle(
            image,
            (int(rb["x"]), int(rb["y"])),
            (int(rb["x"] + rb["w"]), int(rb["y"] + rb["h"])),
            (0, 200, 0),
            2,
        )
    pb = trace_row["prediction_bbox"]
    if pb is not None:
        cv2.rectangle(
            image,
            (int(pb["x_min"]), int(pb["y_min"])),
            (int(pb["x_max"]), int(pb["y_max"])),
            (0, 0, 220),
            2,
        )
    caption = (
        f"{reference['clip']} source {reference['source_frame']} "
        f"{trace_row['event']} GSR {trace_row['reference_track_id']} "
        f"pred {trace_row['prediction_track_id']}"
    )
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 950), 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        caption,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"could not write {output}")


def readout(summary: dict) -> str:
    lines = [
        "# Issue #20: local identity and visible-person coverage",
        "",
        "All metrics use the 50 prespecified 5 fps frames per clip. PTS are derived",
        "from JPEG order, not broadcast timestamps. Identities are local to each",
        "clip; these correlated frames are not independent match trials.",
        "",
        "| Clip | GT | TP | FN | FP | Cov. | IDF1 | IDSW | Frag | Player/GK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in summary["clips"]:
        m = clip["metrics"]
        pg = m["player_goalkeeper"]
        lines.append(
            f"| {clip['clip']} | {m['reference_people']} | {m['matched']} | "
            f"{m['missed']} | {m['false_positives']} | "
            f"{m['detection_coverage']:.3f} | {m['idf1']:.3f} | "
            f"{m['id_switches']} | {m['fragmentations']} | "
            f"{pg['matched']}/{pg['reference_people']} |"
        )
    lines += [
        "",
        "A per-clip `trace.jsonl` lists every labeled person and unmatched prediction,",
        "its exact frame/PTS, box, local ID, source annotation ID, frozen match and",
        "MOT event. `metrics.json` includes per-ID coverage and IDTP/IDFP/IDFN.",
        "The source packet has zero ignore regions; nonempty ones fail evaluation",
        "until arbitrated. Source-labeled distractors are counted, not silently",
        "converted into GT people. All unmatched predictions are FP.",
        "",
        "Identity events are measured from image boxes and IDs only. Camera",
        "projection does not enter matching, so an identity error is separate",
        "from a projection error. A missed visible person can still be caused",
        "by detection or association; this packet does not isolate that cause.",
        "Switches between 5 fps samples may be missed. Fragmentation uses the",
        "pinned motmetrics matched-missed-matched definition; a reappearance with",
        "a new predicted ID also counts as a switch. IDF1 is global identity",
        "assignment within each clip after the frozen frame matches.",
        "",
        "## Visual examples",
        "",
    ]
    for clip in summary["clips"]:
        examples = clip["examples"]
        parts = [
            f"{label}: [{label}]({clip['clip']}/{path})"
            for label, path in examples.items()
        ]
        lines.append(
            f"- {clip['clip']}: "
            + (", ".join(parts) if parts else "no sampled loss or switch")
        )
    lines += [
        "",
        "Green boxes are source labels and red boxes are predictions. A switch",
        "example shows the changed local predicted ID; a loss has no red match.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    packet = json.loads((REFERENCE / "manifest.json").read_text())
    review = json.loads(REVIEW.read_text())
    if (
        digest(PROTOCOL) != packet["protocol_sha256"]
        or digest(REFERENCE / "reference.jsonl") != packet["reference_sha256"]
        or digest(REVIEW) != FROZEN_REVIEW_SHA256
        or (review["protocol_sha256"], review["reference_sha256"])
        != (packet["protocol_sha256"], packet["reference_sha256"])
        or protocol["approval"]["annotation_review"]
        != "completed_by_user_in_session_2026-09-26"
    ):
        raise ValueError("frozen protocol, reference or human review differs")
    reference_rows = read_jsonl(REFERENCE / "reference.jsonl")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output directory must be new to preserve earlier evidence")
    output.mkdir(parents=True)
    clips = []
    for clip_spec in protocol["clips"]:
        clip = clip_spec["id"]
        run = args.tracking_root / clip
        source = ROOT / "artifacts/issue18/derived" / f"{clip}.mp4"
        tracking_path = run / "tracks.jsonl"
        tracking_summary_path = run / "tracking-summary.json"
        if (
            digest(tracking_path),
            digest(tracking_summary_path),
        ) != FROZEN_TRACKING_SHA256[clip]:
            raise ValueError(f"{clip}: tracking differs from frozen baseline")
        predictions = read_jsonl(tracking_path)
        tracking_summary = json.loads(tracking_summary_path.read_text())
        references = [row for row in reference_rows if row["clip"] == clip]
        sampled = validate_inputs(
            clip, source, packet, references, predictions, tracking_summary
        )
        metrics, trace = evaluate_sequence(references, sampled)
        clip_output = output / clip
        clip_output.mkdir()
        (clip_output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (clip_output / "trace.jsonl").open("w") as file:
            for row in trace:
                file.write(json.dumps(row) + "\n")
        examples = {}
        for event, name in (("MISS", "loss"), ("SWITCH", "switch")):
            selected = next((row for row in trace if row["event"] == event), None)
            if selected is not None:
                reference = references[selected["derived_frame_index"] // 5]
                image_name = f"{name}-frame-{selected['source_frame']}.jpg"
                render_example(reference, selected, clip_output / image_name)
                examples[name] = image_name
        clips.append(
            {
                "clip": clip,
                "metrics": metrics,
                "examples": examples,
                "input_sha256": {
                    "source_video": digest(source),
                    "tracking_jsonl": digest(tracking_path),
                    "tracking_summary": digest(tracking_summary_path),
                },
            }
        )
    summary = {
        "schema": "tcc-identity-evaluation/v1",
        "issue": 20,
        "protocol_sha256": digest(PROTOCOL),
        "reference_sha256": digest(REFERENCE / "reference.jsonl"),
        "review_sha256": digest(REVIEW),
        "frame_rate": 25,
        "evaluation_fps": 5,
        "time_semantics": "derived_25fps_not_broadcast_pts",
        "matching": protocol["matching"],
        "evaluator": {"package": "motmetrics", "version": mm.__version__},
        "clips": clips,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "readout.md").write_text(readout(summary))
    print(json.dumps({"output": str(output), "clips": len(clips)}))


if __name__ == "__main__":
    main()
