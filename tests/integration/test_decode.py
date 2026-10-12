"""Behavior tests for timestamp-correct video decoding."""

from __future__ import annotations

import json
from fractions import Fraction
from hashlib import file_digest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from football_alpha_video.cli import main
from football_alpha_video.decode import decode_video


def test_decode_video_preserves_presentation_timestamps(
    three_frame_video: Path,
) -> None:
    frames = list(decode_video(three_frame_video))

    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp for frame in frames] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
    ]
    assert all(frame.timestamp == frame.pts * frame.time_base for frame in frames)
    assert all(frame.image.shape == (16, 16, 3) for frame in frames)
    assert all(frame.image.dtype == np.uint8 for frame in frames)


def test_real_fixture_matches_acceptance_contract(
    capsys: pytest.CaptureFixture[str],
    development_clip: Path,
) -> None:
    fixture = development_clip
    with fixture.open("rb") as binary_fixture:
        assert file_digest(binary_fixture, "sha256").hexdigest() == (
            "0ad6f9a7c5facf49e58bcb9a2afaf579136e8d45153f8860e3ee1b85a947b001"
        )

    assert main(["decode", str(fixture)]) == 0
    summary = cast(dict[str, bool | int | str], json.loads(capsys.readouterr().out))
    assert summary == {
        "first_pts": 0,
        "first_time_base": "1/15360",
        "first_timestamp": "0",
        "frame_count": 302,
        "height": 1080,
        "last_pts": 155648,
        "last_time_base": "1/15360",
        "last_timestamp": "152/15",
        "timestamps_monotonic": True,
        "width": 1920,
    }


def test_decode_video_rejects_a_frame_without_pts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [object()]
    frame = MagicMock()
    frame.pts = None
    frame.time_base = Fraction(1, 30)
    container.decode.return_value = iter([frame])
    monkeypatch.setattr(
        "football_alpha_video.decode.av.open", MagicMock(return_value=container)
    )

    with pytest.raises(ValueError, match="frame 0 has no presentation timestamp"):
        list(decode_video("missing-pts.mp4"))
