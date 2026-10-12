"""Diagnostic rendering for person detections on a decoded video."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from fractions import Fraction
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import av
import numpy as np
from av.video.stream import VideoStream
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from football_alpha_player_detection.contracts import (
    FramePersonDetections,
    PersonDetector,
)
from football_alpha_video import DecodedFrame, decode_video

type JsonScalar = bool | float | int | str


def create_detection_preview(
    source: Path,
    output: Path,
    summary_path: Path,
    *,
    detector: PersonDetector,
    provenance: Mapping[str, JsonScalar],
    batch_size: int,
    minimum_confidence: float,
) -> dict[str, JsonScalar]:
    """Process a complete video and write diagnostic overlay and provenance."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    validate_detection_preview_paths(source, output, summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_source, source_sha256 = _snapshot_source(source, output.parent)
    try:
        source_frame_rate = _source_frame_rate(temporary_source)
        with NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.stem}-",
            suffix=output.suffix,
            delete=False,
        ) as temporary_file:
            temporary_output = Path(temporary_file.name)
    except BaseException:
        temporary_source.unlink(missing_ok=True)
        raise

    frame_count = 0
    detection_count = 0
    first_frame: DecodedFrame | None = None
    last_frame: DecodedFrame | None = None
    try:
        with av.open(temporary_output, mode="w") as container:
            stream: VideoStream | None = None
            for frames in _batches(decode_video(temporary_source), batch_size):
                results = detector.detect(frames)
                if len(results) != len(frames):
                    raise ValueError(
                        "person detector violated frame/result cardinality"
                    )

                for frame, result in zip(frames, results, strict=True):
                    _validate_frame_identity(frame, result)
                    if stream is None:
                        stream = container.add_stream("mpeg4", rate=source_frame_rate)
                        stream.width = result.image_width
                        stream.height = result.image_height
                        stream.pix_fmt = "yuv420p"
                        stream.time_base = frame.time_base
                        stream.codec_context.time_base = frame.time_base

                    annotated = _annotate(frame, result)
                    video_frame = av.VideoFrame.from_ndarray(annotated, format="rgb24")
                    video_frame.pts = frame.pts
                    video_frame.time_base = frame.time_base
                    container.mux(stream.encode(video_frame))

                    frame_count += 1
                    detection_count += len(result.detections)
                    first_frame = frame if first_frame is None else first_frame
                    last_frame = frame

            if stream is None:
                raise ValueError("source video produced no decoded frames")
            container.mux(stream.encode())
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        temporary_source.unlink(missing_ok=True)
        raise
    temporary_source.unlink(missing_ok=True)
    assert first_frame is not None
    assert last_frame is not None
    summary: dict[str, JsonScalar] = {
        **provenance,
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "frame_count": frame_count,
        "detection_count": detection_count,
        "first_pts": first_frame.pts,
        "last_pts": last_frame.pts,
        "first_time_base": str(first_frame.time_base),
        "last_time_base": str(last_frame.time_base),
        "batch_size": batch_size,
        "minimum_confidence": minimum_confidence,
    }
    temporary_summary: Path | None = None
    try:
        serialized_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        with NamedTemporaryFile(
            dir=summary_path.parent,
            prefix=f".{summary_path.stem}-",
            suffix=summary_path.suffix,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary_summary_file:
            temporary_summary_file.write(serialized_summary)
            temporary_summary = Path(temporary_summary_file.name)
        _publish_diagnostic(temporary_output, output, temporary_summary, summary_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)
        raise
    return summary


def validate_detection_preview_paths(
    source: Path,
    output: Path,
    summary_path: Path,
) -> None:
    resolved = (source.resolve(), output.resolve(), summary_path.resolve())
    if len(set(resolved)) != len(resolved):
        raise ValueError("source, output, and summary paths must be distinct")
    if output.is_dir() or summary_path.is_dir():
        raise ValueError("output and summary paths must not be directories")


def validate_detection_preview_source(source: Path) -> None:
    """Fail before detector construction when the source is not readable video."""
    with av.open(source) as container:
        container.streams.video[0]


def _snapshot_source(source: Path, temporary_directory: Path) -> tuple[Path, str]:
    digest = sha256()
    temporary_source: Path | None = None
    try:
        with (
            source.open("rb") as source_file,
            NamedTemporaryFile(
                dir=temporary_directory,
                prefix=f".{source.stem}-",
                suffix=source.suffix,
                delete=False,
            ) as temporary_source_file,
        ):
            temporary_source = Path(temporary_source_file.name)
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
                temporary_source_file.write(chunk)
    except BaseException:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
        raise
    assert temporary_source is not None
    return temporary_source, digest.hexdigest()


def _source_frame_rate(source: Path) -> Fraction:
    with av.open(source) as container:
        return cast(Fraction, container.streams.video[0].average_rate)


def _publish_diagnostic(
    temporary_output: Path,
    output: Path,
    temporary_summary: Path,
    summary_path: Path,
) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination in (output, summary_path):
            if destination.exists():
                with NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.stem}-backup-",
                    suffix=destination.suffix,
                    delete=False,
                ) as backup_file:
                    backup = Path(backup_file.name)
                backup.unlink()
                destination.replace(backup)
                backups.append((destination, backup))

        for temporary, destination in (
            (temporary_output, output),
            (temporary_summary, summary_path),
        ):
            temporary.replace(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            backup.replace(destination)
        temporary_output.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def _batches(
    frames: Iterator[DecodedFrame],
    batch_size: int,
) -> Iterator[list[DecodedFrame]]:
    while batch := list(islice(frames, batch_size)):
        yield batch


def _validate_frame_identity(
    frame: DecodedFrame,
    result: FramePersonDetections,
) -> None:
    expected_identity = (
        frame.index,
        frame.pts,
        frame.time_base,
        frame.image.shape[1],
        frame.image.shape[0],
    )
    result_identity = (
        result.frame_index,
        result.pts,
        result.time_base,
        result.image_width,
        result.image_height,
    )
    if result_identity != expected_identity:
        raise ValueError("person detector returned a result for a different frame")


def _annotate(
    frame: DecodedFrame,
    result: FramePersonDetections,
) -> NDArray[np.uint8]:
    image = Image.fromarray(frame.image)
    draw = ImageDraw.Draw(image)
    for detection in result.detections:
        box = detection.bbox
        draw.rectangle(
            (box.x_min, box.y_min, box.x_max, box.y_max),
            outline=(255, 64, 64),
            width=3,
        )
        draw.text(
            (box.x_min, max(0.0, box.y_min - 12.0)),
            f"person {detection.confidence:.2f}",
            fill=(255, 64, 64),
        )
    return np.asarray(image, dtype=np.uint8)
