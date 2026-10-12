"""Temporal geometry proved using known image motion, without model weights."""

from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np
import pytest
from football_alpha_pitch_projection.temporal import OpticalFlowCalibrator
from numpy.typing import NDArray

from football_alpha_pitch_projection import (
    CalibrationStatus,
    CameraCalibration,
    CorrespondenceHomographyCalibrator,
    FrameCalibrationInput,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
    project_image_point,
)
from football_alpha_video import DecodedFrame

PITCH = PitchDimensions(105, 68)


def field_image() -> NDArray[np.uint8]:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 1] = np.random.default_rng(42).integers(
        50, 220, size=(240, 320), dtype=np.uint8
    )
    return image


def frame(index: int, image: NDArray[np.uint8]) -> DecodedFrame:
    return DecodedFrame(index, index * 512, Fraction(1, 15360), image)


def anchor_for(item: DecodedFrame) -> CameraCalibration:
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, v / 10))
        for u, v in ((20, 20), (280, 20), (280, 220), (20, 220))
    )
    return CorrespondenceHomographyCalibrator().calibrate(
        [FrameCalibrationInput.from_frame(item, pitch=PITCH, correspondences=cues)]
    )[0]


def test_propagates_a_known_camera_translation_without_new_field_labels() -> None:
    image = field_image()
    first = frame(0, image)
    moved = np.asarray(
        cv2.warpAffine(
            image, np.asarray([[1, 0, 4], [0, 1, 2]], dtype=np.float32), (320, 240)
        ),
        dtype=np.uint8,
    )
    camera = OpticalFlowCalibrator(PITCH)
    camera.update(first, anchor_for(first))

    result = camera.update(frame(1, moved))

    assert result.status is CalibrationStatus.ACCEPTED
    assert (result.frame_index, result.pts, result.time_base) == (
        1,
        512,
        Fraction(1, 15360),
    )
    projected = project_image_point(result, ImagePoint(84, 82)).pitch_point
    assert projected is not None
    assert projected.x_m == pytest.approx(8, abs=0.05)
    assert projected.y_m == pytest.approx(8, abs=0.05)


def test_lost_flow_abstains_until_an_explicit_reanchor() -> None:
    image = field_image()
    camera = OpticalFlowCalibrator(PITCH)
    first = frame(0, image)
    camera.update(first, anchor_for(first))

    lost = camera.update(frame(1, np.zeros_like(image)))
    assert lost.status is CalibrationStatus.ABSTAINED
    assert lost.homography_img_to_pitch is None
    assert camera.update(frame(2, image)).status is CalibrationStatus.ABSTAINED

    restored = frame(3, image)
    assert (
        camera.update(restored, anchor_for(restored)).status
        is CalibrationStatus.ACCEPTED
    )
    assert camera.update(frame(4, image)).status is CalibrationStatus.ACCEPTED


@pytest.mark.parametrize(
    "boundary", ["gap", "time", "time_base", "size", "age", "reset", "rejected_anchor"]
)
def test_memory_does_not_cross_a_session_or_evidence_boundary(boundary: str) -> None:
    image = field_image()
    camera = OpticalFlowCalibrator(PITCH)
    first = frame(0, image)
    camera.update(first, anchor_for(first))
    next_frame = frame(1, image)
    if boundary == "gap":
        next_frame = frame(2, image)
    elif boundary == "time":
        next_frame = replace(next_frame, pts=0)
    elif boundary == "time_base":
        next_frame = replace(next_frame, time_base=Fraction(1, 1000))
    elif boundary == "size":
        next_frame = frame(1, image[:200])
    elif boundary == "age":
        for index in range(1, 31):
            assert (
                camera.update(frame(index, image)).status is CalibrationStatus.ACCEPTED
            )
        next_frame = frame(31, image)
    elif boundary == "reset":
        camera.reset()
    else:
        rejected = CorrespondenceHomographyCalibrator().calibrate(
            [
                FrameCalibrationInput.from_frame(
                    next_frame, pitch=PITCH, correspondences=()
                )
            ]
        )[0]
        assert camera.update(next_frame, rejected) == rejected
        next_frame = frame(2, image)

    assert camera.update(next_frame).status is CalibrationStatus.ABSTAINED
    assert (
        camera.update(frame(next_frame.index + 1, image)).status
        is CalibrationStatus.ABSTAINED
    )


def test_rejects_an_anchor_for_a_different_frame_and_clears_memory() -> None:
    image = field_image()
    first = frame(0, image)
    camera = OpticalFlowCalibrator(PITCH)
    camera.update(first, anchor_for(first))
    with pytest.raises(ValueError, match=r"anchor.*frame"):
        camera.update(frame(1, image), anchor_for(first))
    assert camera.update(frame(2, image)).status is CalibrationStatus.ABSTAINED


def test_an_untextured_anchor_cannot_seed_temporal_evidence() -> None:
    image = np.zeros_like(field_image())
    first = frame(0, image)
    camera = OpticalFlowCalibrator(PITCH)
    assert camera.update(first, anchor_for(first)).status is CalibrationStatus.ACCEPTED
    assert camera.update(frame(1, field_image())).status is CalibrationStatus.ABSTAINED
