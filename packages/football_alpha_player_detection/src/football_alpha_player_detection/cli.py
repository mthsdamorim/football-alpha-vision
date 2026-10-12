"""Command-line diagnostics for person detection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from football_alpha_player_detection.contracts import PersonDetector
from football_alpha_player_detection.preview import (
    JsonScalar,
    create_detection_preview,
    validate_detection_preview_paths,
    validate_detection_preview_source,
)
from football_alpha_player_detection.rf_detr import RfDetrPersonDetector


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-alpha-person-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser(
        "preview",
        help="detect people and render a diagnostic video",
    )
    preview.add_argument("source", type=Path)
    preview.add_argument("--output", required=True, type=Path)
    preview.add_argument("--summary", required=True, type=Path)
    preview.add_argument("--batch-size", type=_positive_int, default=4)
    preview.add_argument("--minimum-confidence", type=float, default=0.10)
    preview.add_argument("--device", default="auto")
    return parser


def _create_detector(
    *,
    minimum_confidence: float,
    device: str,
) -> tuple[PersonDetector, Mapping[str, JsonScalar]]:
    detector = RfDetrPersonDetector(
        minimum_confidence=minimum_confidence,
        device=device,
    )
    return detector, detector.provenance()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the person-detection diagnostic command."""
    arguments = _parser().parse_args(argv)
    validate_detection_preview_paths(
        arguments.source,
        arguments.output,
        arguments.summary,
    )
    validate_detection_preview_source(arguments.source)
    detector, provenance = _create_detector(
        minimum_confidence=arguments.minimum_confidence,
        device=arguments.device,
    )
    summary = create_detection_preview(
        arguments.source,
        arguments.output,
        arguments.summary,
        detector=detector,
        provenance=provenance,
        batch_size=arguments.batch_size,
        minimum_confidence=arguments.minimum_confidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
