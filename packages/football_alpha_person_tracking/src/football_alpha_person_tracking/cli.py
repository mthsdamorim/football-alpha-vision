"""Command-line diagnostics for session-local person tracking."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from football_alpha_person_tracking.byte_track import ByteTrackPersonTracker
from football_alpha_person_tracking.diagnostic import (
    DETECTOR_THRESHOLD,
    JsonValue,
    create_tracking_diagnostic,
    validate_tracking_diagnostic_paths,
    validate_tracking_diagnostic_source,
)
from football_alpha_person_tracking.fixture_calibration import (
    CLIP_ANCHOR_CORRESPONDENCES,
    CLIP_PITCH,
    CLIP_SHA256,
    create_clip_calibrator,
)
from football_alpha_player_detection import RfDetrPersonDetector


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-alpha-person-tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose = subparsers.add_parser(
        "diagnose",
        help="track people and write bounded diagnostic artifacts",
    )
    diagnose.add_argument("source", type=Path)
    diagnose.add_argument("--composite", required=True, type=Path)
    diagnose.add_argument("--jsonl", required=True, type=Path)
    diagnose.add_argument("--summary", required=True, type=Path)
    diagnose.add_argument("--batch-size", type=_positive_int, default=4)
    diagnose.add_argument("--device", default="auto")
    diagnose.add_argument(
        "--temporal",
        action="store_true",
        help="run the frozen 180..225 temporal calibration experiment",
    )
    return parser


def _create_tracker(
    source_frame_rate: float,
) -> tuple[ByteTrackPersonTracker, dict[str, JsonValue]]:
    tracker = ByteTrackPersonTracker(source_frame_rate=source_frame_rate)
    provenance = cast(dict[str, JsonValue], tracker.provenance())
    return tracker, provenance


def main(argv: Sequence[str] | None = None) -> int:
    """Run the person-tracking diagnostic command."""
    arguments = _parser().parse_args(argv)
    validate_tracking_diagnostic_paths(
        arguments.source,
        arguments.composite,
        arguments.jsonl,
        arguments.summary,
    )
    validate_tracking_diagnostic_source(arguments.source)
    detector = RfDetrPersonDetector(
        minimum_confidence=DETECTOR_THRESHOLD,
        device=arguments.device,
    )
    anchor_frames = (
        (180, 210) if arguments.temporal else tuple(CLIP_ANCHOR_CORRESPONDENCES)
    )
    summary = create_tracking_diagnostic(
        arguments.source,
        arguments.composite,
        arguments.jsonl,
        arguments.summary,
        detector=detector,
        detector_provenance=detector.provenance(),
        tracker_factory=_create_tracker,
        calibrator=create_clip_calibrator(anchor_frames),
        pitch=CLIP_PITCH,
        correspondences_by_frame={
            i: CLIP_ANCHOR_CORRESPONDENCES[i] for i in anchor_frames
        },
        correspondence_source_sha256=CLIP_SHA256,
        batch_size=arguments.batch_size,
        temporal=arguments.temporal,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.temporal:
        evaluation = cast(dict[str, JsonValue], summary["temporal_evaluation"])
        return 0 if evaluation["passed"] else 1
    return 0
