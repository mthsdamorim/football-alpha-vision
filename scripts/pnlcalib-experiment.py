"""Local pretrained PnLCalib experiment; setup in README.md."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms
import yaml

from football_alpha_video import decode_video

UPSTREAM_SHA = "8c87391d6f4ea40c5e4d65e61529916c7a49ce62"
# PnL uses +y towards the bottom touchline; project coordinates use +y upwards.
PNL_TO_PITCH = np.diag([1.0, -1.0, 1.0])
PITCH_TO_PANEL = np.array([[6.0, 0, 335], [0, -6.0, 254], [0, 0, 1]])
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def image_to_pitch(projection):
    """Extract z=0 from a 3D camera and convert to project pitch coordinates."""
    plane = np.asarray(projection, dtype=np.float64)[:, [0, 1, 3]]
    if not np.isfinite(plane).all() or np.linalg.matrix_rank(plane) < 3:
        return None
    result = PNL_TO_PITCH @ np.linalg.inv(plane)
    if not np.isfinite(result).all() or abs(result[2, 2]) < 1e-12:
        return None
    return result / result[2, 2]


def load_tracking_records(jsonl_path, summary_path, source_sha):
    summary = json.loads(summary_path.read_text())
    if summary.get("source_sha256") != source_sha:
        raise ValueError("tracking source SHA does not match the input video")
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    if summary.get("frame_count") != len(records):
        raise ValueError("tracking summary and JSONL frame counts differ")
    if [record.get("frame_index") for record in records] != list(range(len(records))):
        raise ValueError("tracking records must contain ordered contiguous frames")
    return records


def validate_tracking_identity(frame, record):
    expected = (frame.index, frame.pts, str(frame.time_base))
    actual = (record.get("frame_index"), record.get("pts"), record.get("time_base"))
    if actual != expected:
        raise ValueError("tracking record refers to a different source frame")


def project_tracks(tracks, homography):
    """Keep every observation even when geometry is absent or non-finite."""
    projected = []
    for track in tracks:
        box = track["bbox"]
        u, v = (box["x_min"] + box["x_max"]) / 2.0, box["y_max"]
        support = {"u": u, "v": v}
        if track.get("support_point_uv", support) != support:
            raise ValueError("tracking support point is not bbox_bottom_center")
        result = {
            **track,
            "support_point_uv": support,
            "pitch_xy": None,
            "inside_pitch": None,
            "projection_reason": "calibration_abstained",
        }
        if homography is not None:
            pitch = homography @ np.asarray((u, v, 1.0))
            result["projection_reason"] = "non_finite_projection"
            if np.isfinite(pitch).all() and abs(pitch[2]) >= 1e-12:
                x_m, y_m = (float(value) for value in pitch[:2] / pitch[2])
                result.update(
                    pitch_xy=[x_m, y_m],
                    inside_pitch=abs(x_m) <= PITCH_LENGTH_M / 2
                    and abs(y_m) <= PITCH_WIDTH_M / 2,
                    projection_reason=None,
                )
        projected.append(result)
    return projected


def track_color(track_id):
    return (
        80 + (track_id * 73) % 176,
        80 + (track_id * 131) % 176,
        80 + (track_id * 197) % 176,
    )


def draw_projected(image, points, matrix, color, thickness=2):
    points = np.asarray(points)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    for a, b in pairwise(homogeneous):
        if not np.isfinite([a, b]).all() or a[2] * b[2] <= 0:
            continue
        pixels = np.array([a[:2] / a[2], b[:2] / b[2]])
        if np.max(np.abs(pixels)) > 1e7:
            continue
        start, end = (tuple(int(v) for v in p) for p in pixels)
        visible, start, end = cv2.clipLine(
            (0, 0, image.shape[1], image.shape[0]), start, end
        )
        if visible:
            cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def field_polylines(upstream_lines):
    lines = [
        [(a[0], -a[1]), (b[0], -b[1])] for a, b in upstream_lines if a[2] == b[2] == 0
    ]
    for center, low, high in [(0, 0, 360), (-41.5, -53.1, 53.1), (41.5, 126.9, 233.1)]:
        angle = np.deg2rad(np.linspace(low, high, 121))
        lines.append(
            np.column_stack((center + 9.15 * np.cos(angle), 9.15 * np.sin(angle)))
        )
    return lines


def draw_track_boxes(image, tracks):
    for track in tracks:
        box = track["bbox"]
        top_left = (round(box["x_min"]), round(box["y_min"]))
        bottom_right = (round(box["x_max"]), round(box["y_max"]))
        color = track_color(track["track_id"])
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(
            image,
            f"ID {track['track_id']}",
            (top_left[0], max(20, top_left[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            (round((box["x_min"] + box["x_max"]) / 2), round(box["y_max"])),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )


def render_tracking_minimap(width, height, polylines, projected_tracks, has_solution):
    canvas = np.full((height, width, 3), (22, 92, 38), dtype=np.uint8)
    margin = 20
    scale = min(
        (width - 2 * margin) / PITCH_LENGTH_M,
        (height - 2 * margin) / PITCH_WIDTH_M,
    )
    centre = np.asarray((width / 2, height / 2), dtype=np.float64)
    pitch_to_minimap = np.asarray(
        ((scale, 0, centre[0]), (0, -scale, centre[1]), (0, 0, 1)),
        dtype=np.float64,
    )
    for line in polylines:
        draw_projected(canvas, line, pitch_to_minimap, (235, 235, 235), 1)
    if not has_solution:
        cv2.putText(
            canvas,
            "NO METRIC POSITIONS",
            (margin, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 100, 100),
            2,
            cv2.LINE_AA,
        )
        return canvas
    for track in projected_tracks:
        if not track["inside_pitch"]:
            continue
        x_m, y_m = track["pitch_xy"]
        pixel = pitch_to_minimap @ np.asarray((x_m, y_m, 1.0))
        point = tuple(round(value) for value in pixel[:2])
        color = track_color(track["track_id"])
        cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(track["track_id"]),
            (point[0] + 8, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def render(
    frame,
    keypoints,
    detected_lines,
    homography,
    polylines,
    tracks=(),
    projected_tracks=(),
    camera_origin=None,
):
    image = frame.image.copy()
    for point in keypoints.values():
        cv2.circle(image, (int(point["x"]), int(point["y"])), 4, (255, 210, 40), -1)
    for line in detected_lines.values():
        draw_projected(
            image,
            [(line["x_1"], line["y_1"]), (line["x_2"], line["y_2"])],
            np.eye(3),
            (0, 230, 255),
            3,
        )
    draw_track_boxes(image, tracks)
    panel = np.full((image.shape[0], 670, 3), 24, dtype=np.uint8)
    if homography is not None:
        panel[:460] = cv2.warpPerspective(
            frame.image, PITCH_TO_PANEL @ homography, (670, 460)
        )
        for line in polylines:
            draw_projected(image, line, np.linalg.inv(homography), (80, 255, 80))
            draw_projected(panel, line, PITCH_TO_PANEL, (255, 255, 255), 1)
        matrix_label = " | ".join(
            " ".join(f"{value:.3g}" for value in row) for row in homography
        )
    else:
        matrix_label = "NO CAMERA SOLUTION | no previous homography reused"
    cv2.putText(
        panel,
        matrix_label,
        (12, 490),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "SESSION-LOCAL PERSON TRACKS",
        (12, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    panel[548:1068] = render_tracking_minimap(
        panel.shape[1],
        520,
        polylines,
        projected_tracks,
        homography is not None,
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (20, 20, 20), -1)
    status = camera_origin or ("estimated" if homography is not None else "NO SOLUTION")
    count = sum(t["pitch_xy"] is not None for t in projected_tracks)
    status += f" | people {len(tracks)} projected {count} absent {len(tracks) - count}"
    cv2.putText(
        image,
        f"PnLCalib | frame {frame.index} | {float(frame.timestamp):.3f}s | {status}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((image, panel), axis=1)


def evaluate(records, source_sha):
    # Evaluation labels are loaded only after all inference has finished.
    from football_alpha_person_tracking.fixture_calibration import (
        CLIP_HELD_OUT_MARKINGS,
        CLIP_SHA256,
        _image_line_error_px,
    )

    if source_sha != CLIP_SHA256:
        return None
    errors = {}
    for index, markings in CLIP_HELD_OUT_MARKINGS.items():
        homography = records[index]["homography_img_to_pitch"]
        residuals = (
            [
                _image_line_error_px(homography, marking, point)
                for marking in markings
                for point in marking.image_points
            ]
            if homography is not None
            else []
        )
        errors[str(index)] = (
            max(residuals) if residuals and None not in residuals else None
        )
    return {"maximum_line_error_px_by_frame": errors, "labels_used_in_inference": False}


def independent_cues(points, world):
    """Exclude elevated, auxiliary and geometrically completed PnL keypoints."""
    from football_alpha_pitch_projection.contracts import (
        ImagePitchCorrespondence,
        ImagePoint,
        PitchPoint,
    )

    return [
        ImagePitchCorrespondence(
            ImagePoint(float(point["x"]), float(point["y"])),
            PitchPoint(float(world[int(key) - 1][0]), -float(world[int(key) - 1][1])),
        )
        for key, point in points.items()
        if 1 <= int(key) <= 57
        and int(key) not in (12, 15, 16, 19)
        and 0 < point["p"] < 1
    ]


def load_cached_records(directory, source_sha, weights):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    records = load_tracking_records(
        directory / "frames.jsonl", summary_path, source_sha
    )
    if summary["upstream_sha"] != UPSTREAM_SHA:
        raise ValueError("baseline upstream differs")
    if summary["weights_sha256"] != {
        name: sha256(weights / name) for name in ("SV_kp", "SV_lines")
    }:
        raise ValueError("baseline weights differ")
    # The historical JSONL omitted dimensions. Recover and verify its paired preview;
    # the pinned render adds exactly 670 px on the right, without resizing the source.
    with av.open(str(directory / "preview.mp4")) as video:
        stream = video.streams.video[0]
        size = [stream.width - 670, stream.height]
    verify_preview(directory / "preview.mp4", records)
    return [{**row, "image_size": size} for row in records]


def verify_preview(path, records):
    with av.open(str(path)) as video:
        count = 0
        for count, decoded in enumerate(video.decode(video=0), start=1):
            if count > len(records):
                raise ValueError("preview has extra frames")
            row = records[count - 1]
            if (decoded.pts, str(decoded.time_base)) != (row["pts"], row["time_base"]):
                raise ValueError("preview PTS/time base differs from source records")
        if count != len(records):
            raise ValueError("preview frame count differs from source records")


def automatic_summary(args, records, calibrator):
    import football_alpha_pitch_projection.automatic as automatic

    projected = sum(t["pitch_xy"] is not None for r in records for t in r["tracks"])
    observed = sum(len(r["tracks"]) for r in records)
    return {
        "automatic": {
            "origin_counts": dict(Counter(r["calibration"]["origin"] for r in records)),
            "abstention_counts": dict(
                Counter(
                    r["calibration"]["output_gate"]["reason"]
                    for r in records
                    if r["status"] == "abstained"
                )
            ),
            "observed_person_count": observed,
            "projected_person_count": projected,
            "unprojected_person_count": observed - projected,
            "refinement_attempt_count": sum(
                r["calibration"]["refinement"] is not None for r in records
            ),
            "qualified_refinement_count": sum(
                r["calibration"]["refinement"] is not None
                and r["calibration"]["refinement"]["gate"]["reason"] is None
                for r in records
            ),
            "used_refinement_count": sum(
                r["calibration"]["calibration"]["method"]
                == "image_marking_refinement+image_gate"
                and r["status"] == "accepted"
                for r in records
            ),
            "parameters": asdict(calibrator.parameters),
            "implementation_sha256": sha256(Path(automatic.__file__)),
            "orchestration_sha256": sha256(Path(__file__)),
            "opencv_version": cv2.__version__,
            "opencv_rng_seed": 0,
            "baseline_directory": str(args.cached),
            "input_sha256": {
                str(path): sha256(path)
                for path in (
                    args.cached / "frames.jsonl",
                    args.cached / "summary.json",
                    args.tracks,
                    args.tracking_summary,
                )
            },
            "cue_coordinate_system": "original_image_pixels;completed_p1_excluded",
            "solver_route": (
                "cached_FramebyFrameCalib.heuristic_voting(refine_lines=True)"
            ),
            "camera_parameters": "unavailable_in_historical_cache",
            "solver_selected_mode": "unavailable_in_historical_cache",
            "distortion": "none;cached_planar_homography",
            "evaluation_labels_loaded": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/clip.mp4"))
    parser.add_argument(
        "--upstream", type=Path, default=Path("artifacts/pnlcalib/vendor/PnLCalib")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("artifacts/pnlcalib/weights")
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("artifacts/person-tracking/tracks.jsonl"),
    )
    parser.add_argument(
        "--tracking-summary",
        type=Path,
        default=Path("artifacts/person-tracking/summary.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pnlcalib/run"))
    parser.add_argument(
        "--cached",
        type=Path,
        help="Reuse a baseline run directory and apply causal automatic calibration",
    )
    args = parser.parse_args()
    upstream_sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_sha != UPSTREAM_SHA:
        raise ValueError(f"Expected PnLCalib revision {UPSTREAM_SHA}")
    sys.path.insert(0, str(args.upstream.resolve()))
    geometry = importlib.import_module("utils.utils_calib")
    cv2.setRNGSeed(0)
    cv2.setNumThreads(4)
    if args.cached is None:
        heatmap = importlib.import_module("utils.utils_heatmap")
        geometry = importlib.import_module("utils.utils_calib")
        demo = importlib.import_module("inference")
        torch.set_num_threads(4)
        torch.manual_seed(0)
        cv2.setRNGSeed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for module, config, weights in [
            ("model.cls_hrnet", "hrnetv2_w48.yaml", "SV_kp"),
            ("model.cls_hrnet_l", "hrnetv2_w48_l.yaml", "SV_lines"),
        ]:
            cfg = yaml.safe_load((args.upstream / "config" / config).read_text())
            model = importlib.import_module(module).get_cls_net(cfg)
            model.load_state_dict(
                torch.load(
                    args.weights / weights, map_location="cpu", weights_only=True
                )
            )
            models.append(model.to(device).eval())
    else:
        device = "cached;opencv_cpu"
    polylines = field_polylines(geometry.line_world_coords_3D)
    source_sha = sha256(args.source)
    tracking_records = load_tracking_records(
        args.tracks,
        args.tracking_summary,
        source_sha,
    )
    tracking_summary = json.loads(args.tracking_summary.read_text())
    cached = None
    calibrator = None
    if args.cached is not None:
        from football_alpha_pitch_projection.automatic import AutomaticCameraCalibrator
        from football_alpha_pitch_projection.contracts import PitchDimensions

        cached = load_cached_records(args.cached, source_sha, args.weights)
        calibrator = AutomaticCameraCalibrator(PitchDimensions(105, 68), polylines)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.monotonic()
    with av.open(str(args.source)) as source:
        rate = source.streams.video[0].average_rate
    with (
        av.open(
            str(args.output / "preview.mp4"), "w", options={"movflags": "+faststart"}
        ) as output,
        (args.output / "frames.jsonl").open("w") as evidence,
    ):
        stream = output.add_stream("libx264", rate=rate)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        for frame in decode_video(args.source):
            if frame.index >= len(tracking_records):
                raise ValueError("source has more frames than the tracking artifact")
            tracking = tracking_records[frame.index]
            validate_tracking_identity(frame, tracking)
            if frame.index == 0:
                stream.width = frame.image.shape[1] + 670
                stream.height = frame.image.shape[0]
                stream.time_base = frame.time_base
            step = None
            if cached is None:
                tensor = transforms.to_tensor(frame.image).unsqueeze(0)
                tensor = transforms.resize(tensor, [540, 960]).to(device)
                with torch.inference_mode():
                    points = heatmap.get_keypoints_from_heatmap_batch_maxpool(
                        models[0](tensor)[:, :-1]
                    )
                    lines = heatmap.get_keypoints_from_heatmap_batch_maxpool_l(
                        models[1](tensor)[:, :-1]
                    )
                points = heatmap.coords_to_dict(points, threshold=0.3434)[0]
                lines = heatmap.coords_to_dict(lines, threshold=0.7867)[0]
                points, lines = heatmap.complete_keypoints(
                    points, lines, w=960, h=540, normalize=True
                )
                # Fresh solver per frame: failed inference cannot inherit camera state.
                camera = geometry.FramebyFrameCalib(
                    iwidth=frame.image.shape[1],
                    iheight=frame.image.shape[0],
                    denormalize=True,
                )
                camera.update(points, lines)
                result = camera.heuristic_voting(refine_lines=True)
                homography = (
                    image_to_pitch(demo.projection_from_cam_params(result))
                    if result is not None
                    else None
                )
            else:
                if frame.index >= len(cached):
                    raise ValueError("source has more frames than the baseline")
                raw = cached[frame.index]
                validate_tracking_identity(frame, raw)
                if [frame.image.shape[1], frame.image.shape[0]] != raw["image_size"]:
                    raise ValueError("cached cue dimensions differ from source")
                points, lines = raw["keypoints"], raw["lines"]
                hypothesis = raw["homography_img_to_pitch"]
                hypothesis = np.asarray(hypothesis) if hypothesis is not None else None
                excluded = [
                    tuple(t["bbox"][k] for k in ("x_min", "y_min", "x_max", "y_max"))
                    for t in tracking["tracks"]
                ]
                step = calibrator.update(
                    frame,
                    hypothesis,
                    independent_cues(points, geometry.keypoint_world_coords_2D),
                    excluded,
                )
                accepted = step.calibration.homography_img_to_pitch
                homography = np.asarray(accepted) if accepted is not None else None
                result = None
            record = {
                "frame_index": frame.index,
                "pts": frame.pts,
                "time_base": str(frame.time_base),
                "status": "estimated" if homography is not None else "no_solution",
                "homography_img_to_pitch": homography.tolist()
                if homography is not None
                else None,
                "keypoints": points,
                "lines": lines,
                "solver_reprojection_error_px": float(result["rep_err"])
                if result is not None
                else None,
            }
            if step is not None:
                record.update(
                    status=step.calibration.status.value,
                    image_width=frame.image.shape[1],
                    image_height=frame.image.shape[0],
                    calibration=asdict(step),
                    raw_hypothesis={k: v for k, v in raw.items() if k != "tracks"},
                )
                record["calibration"]["calibration"]["time_base"] = str(frame.time_base)
            projected_tracks = project_tracks(tracking["tracks"], homography)
            record["tracks"] = projected_tracks
            evidence.write(json.dumps(record, allow_nan=False) + "\n")
            evidence.flush()
            records.append(record)
            rendered = render(
                frame,
                points,
                lines,
                homography,
                polylines,
                tracking["tracks"],
                projected_tracks,
                step.origin if step else None,
            )
            encoded = av.VideoFrame.from_ndarray(rendered, format="rgb24")
            encoded.pts, encoded.time_base = frame.pts, frame.time_base
            for packet in stream.encode(encoded):
                output.mux(packet)
            if frame.index in (0, 90, 105, 163, 180, 195, 210, 225, 301):
                cv2.imwrite(
                    str(args.output / f"frame-{frame.index:03}.png"),
                    cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                )
            if frame.index % 10 == 0:
                print(
                    f"frame={frame.index} status={record['status']} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        for packet in stream.encode():
            output.mux(packet)
    if len(records) != len(tracking_records):
        raise ValueError("source and tracking artifact frame counts differ")
    if cached is not None and len(records) != len(cached):
        raise ValueError("source and baseline frame counts differ")
    verify_preview(args.output / "preview.mp4", records)
    estimated = sum(row["homography_img_to_pitch"] is not None for row in records)
    summary = {
        "source_sha256": source_sha,
        "upstream_sha": upstream_sha,
        "weights_sha256": {
            name: sha256(args.weights / name) for name in ("SV_kp", "SV_lines")
        },
        "device": device,
        "frame_count": len(records),
        "estimated_frame_count": estimated,
        "solution_coverage": estimated / len(records),
        "all_frames_have_solution": estimated == len(records),
        "tracking": {
            "source_jsonl": str(args.tracks),
            "source_summary": str(args.tracking_summary),
            "support_point_method": "bbox_bottom_center",
            "distinct_track_id_count": tracking_summary.get("distinct_track_id_count"),
            "detector": tracking_summary.get("detector"),
            "tracker": tracking_summary.get("tracker"),
            "observation_count": sum(len(row["tracks"]) for row in records),
            "metric_observation_count": sum(
                t["pitch_xy"] is not None for row in records for t in row["tracks"]
            ),
            "inside_pitch_observation_count": sum(
                bool(track["inside_pitch"])
                for row in records
                for track in row["tracks"]
            ),
            "no_metric_position_without_homography": True,
        },
        "evaluation": evaluate(records, source_sha) if cached is None else None,
        "elapsed_seconds": time.monotonic() - started,
        "pitch": {
            "length_m": 105,
            "width_m": 68,
            "origin": "centre",
            "positive_y": "upper_touchline",
            "dimensions_assumed": True,
        },
        "parameters": {
            "keypoint_threshold": 0.3434,
            "line_threshold": 0.7867,
            "pnl_refine": True,
            "temporal_smoothing": False,
        },
    }
    if cached is not None:
        summary.update(automatic_summary(args, records, calibrator))
    summary["completed"] = True
    (args.output / "summary.json.tmp").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "summary.json.tmp").replace(args.output / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
