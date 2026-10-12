"""Structural behavior tests for the video package CLI."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest
from football_alpha_video.cli import main

from football_alpha_video import __version__


def test_decode_command_reports_exact_timing_summary(
    three_frame_video: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["decode", str(three_frame_video)]) == 0

    summary = cast(dict[str, bool | int | str], json.loads(capsys.readouterr().out))
    expected_without_encoding_details = {
        "first_timestamp": "0",
        "frame_count": 3,
        "height": 16,
        "last_timestamp": "1",
        "timestamps_monotonic": True,
        "width": 16,
    }
    assert {
        key: summary[key] for key in expected_without_encoding_details
    } == expected_without_encoding_details
    assert int(summary["first_pts"]) * Fraction(
        str(summary["first_time_base"])
    ) == Fraction(str(summary["first_timestamp"]))
    assert int(summary["last_pts"]) * Fraction(
        str(summary["last_time_base"])
    ) == Fraction(str(summary["last_timestamp"]))


def test_version_flag_reports_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out == f"football-alpha-video {__version__}\n"
