"""Shared deterministic video fixtures."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


@pytest.fixture
def three_frame_video(tmp_path: Path) -> Path:
    """Create a tiny video with known presentation timestamps."""
    path = tmp_path / "three-frames.mp4"
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"

        for frame_index in range(3):
            image = np.full((16, 16, 3), frame_index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 2)
            container.mux(stream.encode(frame))

        container.mux(stream.encode())

    return path


DEVELOPMENT_CLIP = Path(__file__).resolve().parents[1] / "data" / "clip.mp4"


@pytest.fixture
def development_clip() -> Path:
    """Return the unversioned development clip, skipping when it is absent."""
    if not DEVELOPMENT_CLIP.is_file():
        pytest.skip("data/clip.mp4 is not versioned; see data/README.md")
    return DEVELOPMENT_CLIP
