"""Command-line inspection for timestamp-correct video decode."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

from football_alpha_video import __version__
from football_alpha_video.decode import DecodedFrame, decode_video


class _Arguments(argparse.Namespace):
    path: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the video package command-line parser."""
    parser = argparse.ArgumentParser(prog="football-alpha-video")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(required=True)
    decode_parser = subparsers.add_parser(
        "decode",
        help="decode a video and report its observed frame timeline",
    )
    decode_parser.add_argument("path", type=Path)
    return parser


def _summary(
    first: DecodedFrame, remaining: Iterable[DecodedFrame]
) -> dict[str, object]:
    last = first
    frame_count = 1
    timestamps_monotonic = True
    for frame in remaining:
        frame_count += 1
        timestamps_monotonic &= frame.timestamp >= last.timestamp
        last = frame

    height, width, _channels = first.image.shape
    return {
        "first_pts": first.pts,
        "first_time_base": str(first.time_base),
        "first_timestamp": str(first.timestamp),
        "frame_count": frame_count,
        "height": height,
        "last_pts": last.pts,
        "last_time_base": str(last.time_base),
        "last_timestamp": str(last.timestamp),
        "timestamps_monotonic": timestamps_monotonic,
        "width": width,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Decode the requested video and print a compact JSON summary."""
    arguments = cast(_Arguments, build_parser().parse_args(argv))
    frames = iter(decode_video(arguments.path))
    print(json.dumps(_summary(next(frames), frames), sort_keys=True))
    return 0
