"""Batch calibration boundary and result validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from football_alpha_pitch_projection.contracts import (
    CameraCalibration,
    CameraCalibrator,
    FrameCalibrationInput,
)


def calibrate_frames(
    calibrator: CameraCalibrator,
    inputs: Sequence[FrameCalibrationInput],
) -> list[CameraCalibration]:
    """Calibrate a batch and reject incomplete or realigned provider output."""
    input_sequence = tuple(inputs)
    frame_identities = [
        (
            item.frame_index,
            item.pts,
            item.time_base,
            item.image_width,
            item.image_height,
        )
        for item in input_sequence
    ]
    if len(set(frame_identities)) != len(frame_identities):
        raise ValueError("calibration inputs contain duplicate frame identities")
    results = cast(object, calibrator.calibrate(input_sequence))
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError("calibrator returned a malformed result sequence")
    result_sequence = cast(Sequence[object], results)
    if len(result_sequence) != len(input_sequence):
        raise ValueError("calibrator result count does not match input count")

    validated: list[CameraCalibration] = []
    for calibration_input, result in zip(input_sequence, result_sequence, strict=True):
        if not isinstance(result, CameraCalibration):
            raise ValueError("calibrator returned a malformed result")
        expected_identity = (
            calibration_input.frame_index,
            calibration_input.pts,
            calibration_input.time_base,
            calibration_input.image_width,
            calibration_input.image_height,
            calibration_input.pitch,
        )
        result_identity = (
            result.frame_index,
            result.pts,
            result.time_base,
            result.image_width,
            result.image_height,
            result.pitch,
        )
        if result_identity != expected_identity:
            raise ValueError("calibrator changed frame identity or order")
        if result.correspondence_count != len(calibration_input.correspondences):
            raise ValueError("calibrator changed correspondence evidence count")
        validated.append(result)
    return validated
