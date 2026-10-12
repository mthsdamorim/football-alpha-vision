"""Behavior tests for the person-detection diagnostic command."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import av
import numpy as np
import pytest
from football_alpha_player_detection.cli import main
from football_alpha_player_detection.preview import create_detection_preview

from football_alpha_player_detection import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
)
from football_alpha_video import DecodedFrame, decode_video


class PreviewDetector:
    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        return [
            FramePersonDetections(
                frame_index=frame.index,
                pts=frame.pts,
                time_base=frame.time_base,
                image_width=frame.image.shape[1],
                image_height=frame.image.shape[0],
                detections=(
                    PersonDetection(
                        bbox=ImageBoundingBox(
                            x_min=1.0,
                            y_min=1.0,
                            x_max=10.0,
                            y_max=10.0,
                        ),
                        confidence=0.8,
                    ),
                ),
            )
            for frame in frames
        ]


class DiagnosticRfDetrPersonDetector(PreviewDetector):
    def __init__(self, minimum_confidence: float, *, device: str) -> None:
        assert minimum_confidence == 0.10
        assert device == "auto"

    def provenance(self) -> dict[str, str]:
        return {
            "rfdetr_version": "test-version",
            "model_name": "RFDETRLarge",
            "checkpoint_name": "test-checkpoint.pth",
            "checkpoint_sha256": "abc123",
            "device": "cpu",
        }


def test_preview_command_processes_the_complete_video(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preview_path = tmp_path / "preview.mp4"
    summary_path = tmp_path / "summary.json"
    source_frames = list(decode_video(three_frame_video))
    monkeypatch.setattr(
        "football_alpha_player_detection.cli.RfDetrPersonDetector",
        DiagnosticRfDetrPersonDetector,
    )

    assert (
        main(
            [
                "preview",
                str(three_frame_video),
                "--output",
                str(preview_path),
                "--summary",
                str(summary_path),
                "--batch-size",
                "2",
            ]
        )
        == 0
    )

    summary = cast(dict[str, object], json.loads(summary_path.read_text()))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["frame_count"] == 3
    assert summary["detection_count"] == 3
    assert summary["first_pts"] == source_frames[0].pts
    assert summary["last_pts"] == source_frames[-1].pts
    assert summary["first_time_base"] == str(source_frames[0].time_base)
    assert summary["last_time_base"] == str(source_frames[-1].time_base)
    assert summary["batch_size"] == 2
    assert summary["minimum_confidence"] == 0.1
    assert summary["rfdetr_version"] == "test-version"
    assert summary["model_name"] == "RFDETRLarge"
    assert preview_path.is_file()
    preview_frames = list(decode_video(preview_path))
    assert [frame.pts * frame.time_base for frame in preview_frames] == [
        frame.pts * frame.time_base for frame in source_frames
    ]
    with (
        av.open(three_frame_video) as source_container,
        av.open(preview_path) as preview_container,
    ):
        assert (
            preview_container.streams.video[0].average_rate
            == source_container.streams.video[0].average_rate
        )


def test_preview_command_rejects_non_positive_batch_size(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "preview",
                str(three_frame_video),
                "--output",
                str(tmp_path / "preview.mp4"),
                "--summary",
                str(tmp_path / "summary.json"),
                "--batch-size",
                "0",
            ]
        )

    assert error.value.code == 2


def test_preview_function_rejects_non_positive_batch_size(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        create_detection_preview(
            three_frame_video,
            tmp_path / "preview.mp4",
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={},
            batch_size=0,
            minimum_confidence=0.1,
        )


class WrongCardinalityDetector:
    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        return []


class WrongIdentityDetector(PreviewDetector):
    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        results = super().detect(frames)
        first = results[0]
        results[0] = FramePersonDetections(
            frame_index=first.frame_index + 1,
            pts=first.pts,
            time_base=first.time_base,
            image_width=first.image_width,
            image_height=first.image_height,
            detections=first.detections,
        )
        return results


def _decode_no_frames(_source: Path) -> Iterator[DecodedFrame]:
    return iter(())


@pytest.mark.parametrize(
    ("detector", "message"),
    [
        (WrongCardinalityDetector(), "cardinality"),
        (WrongIdentityDetector(), "different frame"),
    ],
)
def test_preview_removes_partial_output_when_detector_contract_fails(
    three_frame_video: Path,
    tmp_path: Path,
    detector: WrongCardinalityDetector | WrongIdentityDetector,
    message: str,
) -> None:
    preview_path = tmp_path / "preview.mp4"
    with pytest.raises(ValueError, match=message):
        create_detection_preview(
            three_frame_video,
            preview_path,
            tmp_path / "summary.json",
            detector=detector,
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert not preview_path.exists()


def test_preview_rejects_a_video_without_decoded_frames(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "football_alpha_player_detection.preview.decode_video",
        _decode_no_frames,
    )

    with pytest.raises(ValueError, match="no decoded frames"):
        create_detection_preview(
            three_frame_video,
            tmp_path / "preview.mp4",
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )


@pytest.mark.parametrize("aliased_artifact", ["output", "summary"])
def test_preview_rejects_source_and_artifact_path_aliases(
    three_frame_video: Path,
    tmp_path: Path,
    aliased_artifact: str,
) -> None:
    output = three_frame_video if aliased_artifact == "output" else tmp_path / "out.mp4"
    summary = (
        three_frame_video
        if aliased_artifact == "summary"
        else tmp_path / "summary.json"
    )

    with pytest.raises(ValueError, match="must be distinct"):
        create_detection_preview(
            three_frame_video,
            output,
            summary,
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )


def test_preview_rejects_aliased_output_and_summary_paths(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"

    with pytest.raises(ValueError, match="must be distinct"):
        create_detection_preview(
            three_frame_video,
            artifact,
            artifact,
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )


@pytest.mark.parametrize("directory_artifact", ["output", "summary"])
def test_preview_rejects_directory_artifact_destinations(
    three_frame_video: Path,
    tmp_path: Path,
    directory_artifact: str,
) -> None:
    directory = tmp_path / "existing-directory"
    directory.mkdir()
    output = directory if directory_artifact == "output" else tmp_path / "preview.mp4"
    summary = (
        directory if directory_artifact == "summary" else tmp_path / "summary.json"
    )

    with pytest.raises(ValueError, match="must not be directories"):
        create_detection_preview(
            three_frame_video,
            output,
            summary,
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert directory.is_dir()


def test_preview_cli_validates_paths_before_loading_the_detector(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_detector_creation(*args: object, **kwargs: object) -> None:
        raise AssertionError("detector must not be created")

    monkeypatch.setattr(
        "football_alpha_player_detection.cli.RfDetrPersonDetector",
        fail_detector_creation,
    )

    with pytest.raises(ValueError, match="must be distinct"):
        main(
            [
                "preview",
                str(three_frame_video),
                "--output",
                str(three_frame_video),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )


def test_preview_cli_validates_source_before_loading_the_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_detector_creation(*args: object, **kwargs: object) -> None:
        raise AssertionError("detector must not be created")

    monkeypatch.setattr(
        "football_alpha_player_detection.cli.RfDetrPersonDetector",
        fail_detector_creation,
    )

    with pytest.raises(FileNotFoundError):
        main(
            [
                "preview",
                str(tmp_path / "missing.mp4"),
                "--output",
                str(tmp_path / "preview.mp4"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )


def test_preview_restores_existing_artifacts_when_publication_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "preview.mp4"
    summary = tmp_path / "summary.json"
    output.write_bytes(b"old preview")
    summary.write_text("old summary", encoding="utf-8")
    original_replace = Path.replace

    def fail_summary_publication(path: Path, target: Path) -> Path:
        if (
            path.name.startswith(".summary-")
            and "-backup-" not in path.name
            and target == summary
        ):
            raise OSError("summary publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_summary_publication)

    with pytest.raises(OSError, match="summary publication failed"):
        create_detection_preview(
            three_frame_video,
            output,
            summary,
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert output.read_bytes() == b"old preview"
    assert summary.read_text(encoding="utf-8") == "old summary"


def test_preview_replaces_an_existing_diagnostic_pair(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "preview.mp4"
    summary_path = tmp_path / "summary.json"
    output.write_bytes(b"old preview")
    summary_path.write_text("old summary", encoding="utf-8")

    summary = create_detection_preview(
        three_frame_video,
        output,
        summary_path,
        detector=PreviewDetector(),
        provenance={},
        batch_size=2,
        minimum_confidence=0.1,
    )

    assert len(list(decode_video(output))) == 3
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_preview_cleans_staged_video_when_summary_serialization_fails(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "preview.mp4"

    with pytest.raises(TypeError, match="JSON serializable"):
        create_detection_preview(
            three_frame_video,
            output,
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={"invalid": object()},  # type: ignore[dict-item]
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".preview-*")) == []


def test_preview_cleans_source_snapshot_when_output_staging_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_named_temporary_file = cast(Callable[..., Any], NamedTemporaryFile)
    call_count = 0

    def fail_second_temporary_file(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("output staging failed")
        return original_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        "football_alpha_player_detection.preview.NamedTemporaryFile",
        fail_second_temporary_file,
    )

    with pytest.raises(OSError, match="output staging failed"):
        create_detection_preview(
            three_frame_video,
            tmp_path / "preview.mp4",
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert list(tmp_path.glob(f".{three_frame_video.stem}-*")) == []


def test_preview_cleans_partial_snapshot_when_source_copy_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = cast(Callable[..., Any], Path.open)

    class FailingReader:
        def __init__(self, source_file: Any) -> None:
            self.source_file = source_file
            self.read_once = False

        def __enter__(self) -> FailingReader:
            return self

        def __exit__(self, *args: object) -> None:
            self.source_file.close()

        def read(self, size: int = -1) -> bytes:
            if self.read_once:
                raise OSError("source copy failed")
            self.read_once = True
            return self.source_file.read(size)

    def fail_second_source_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened = original_open(path, *args, **kwargs)
        if path == three_frame_video and args and args[0] == "rb":
            return FailingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", fail_second_source_read)

    with pytest.raises(OSError, match="source copy failed"):
        create_detection_preview(
            three_frame_video,
            tmp_path / "preview.mp4",
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert list(tmp_path.glob(f".{three_frame_video.stem}-*")) == []


def test_preview_propagates_source_open_failure_without_staging_files(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        create_detection_preview(
            missing_source,
            tmp_path / "preview.mp4",
            tmp_path / "summary.json",
            detector=PreviewDetector(),
            provenance={},
            batch_size=2,
            minimum_confidence=0.1,
        )

    assert list(tmp_path.glob(f".{missing_source.stem}-*")) == []


def test_preview_provenance_identifies_the_snapshotted_source_bytes(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with three_frame_video.open("rb") as source_file:
        expected_sha256 = file_digest(source_file, "sha256").hexdigest()

    class SourceMutatingDetector(PreviewDetector):
        mutated = False

        def detect(
            self,
            frames: Sequence[DecodedFrame],
        ) -> list[FramePersonDetections]:
            if not self.mutated:
                three_frame_video.write_bytes(b"source changed after snapshot")
                self.mutated = True
            return super().detect(frames)

    summary = create_detection_preview(
        three_frame_video,
        tmp_path / "preview.mp4",
        tmp_path / "summary.json",
        detector=SourceMutatingDetector(),
        provenance={},
        batch_size=2,
        minimum_confidence=0.1,
    )

    assert summary["source_sha256"] == expected_sha256


def test_preview_preserves_variable_frame_presentation_times(tmp_path: Path) -> None:
    source = tmp_path / "variable-rate.mp4"
    with av.open(source, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for frame_index, pts in enumerate((0, 1, 3)):
            image = np.full((16, 16, 3), frame_index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())

    output = tmp_path / "preview.mp4"
    source_frames = list(decode_video(source))
    create_detection_preview(
        source,
        output,
        tmp_path / "summary.json",
        detector=PreviewDetector(),
        provenance={},
        batch_size=2,
        minimum_confidence=0.1,
    )

    assert [frame.pts * frame.time_base for frame in decode_video(output)] == [
        frame.pts * frame.time_base for frame in source_frames
    ]


def test_preview_canonical_fields_override_detector_provenance(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    summary = create_detection_preview(
        three_frame_video,
        tmp_path / "preview.mp4",
        tmp_path / "summary.json",
        detector=PreviewDetector(),
        provenance={"source": "wrong", "frame_count": 999},
        batch_size=2,
        minimum_confidence=0.1,
    )

    assert summary["source"] == str(three_frame_video)
    assert summary["frame_count"] == 3


def test_default_lock_uses_cpu_only_pytorch_on_linux() -> None:
    repository_root = Path(__file__).parents[2]
    lock = tomllib.loads((repository_root / "uv.lock").read_text(encoding="utf-8"))
    packages = cast(list[dict[str, object]], lock["package"])
    names = {cast(str, package["name"]) for package in packages}

    assert not any(
        name.startswith(("cuda-", "nvidia-")) or name == "triton" for name in names
    )
    pytorch_sources = [
        cast(dict[str, str], package["source"])
        for package in packages
        if package["name"] in {"torch", "torchvision"}
    ]
    assert any(
        source.get("registry") == "https://download.pytorch.org/whl/cpu"
        for source in pytorch_sources
    )
