"""Streaming video decode with exact presentation timestamps."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import av
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One RGB frame in presentation order with its original timing."""

    index: int
    pts: int
    time_base: Fraction
    image: NDArray[np.uint8]

    @property
    def timestamp(self) -> Fraction:
        """Return the exact presentation timestamp in seconds."""
        return self.pts * self.time_base


def decode_video(path: str | os.PathLike[str]) -> Iterator[DecodedFrame]:
    """Yield RGB frames from the first video stream in presentation order.

    A missing PTS or time base is rejected because reconstructing either value
    from an advertised frame rate would silently change the video's timeline.
    """
    with av.open(os.fspath(path), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                message = f"decoded frame {index} has no presentation timestamp"
                raise ValueError(message)

            time_base = Fraction(
                frame.time_base.numerator,
                frame.time_base.denominator,
            )
            image = cast(NDArray[np.uint8], frame.to_ndarray(format="rgb24"))
            yield DecodedFrame(
                index=index,
                pts=frame.pts,
                time_base=time_base,
                image=image,
            )
