"""Finite homogeneous projection through accepted frame calibration."""

from __future__ import annotations

from math import isfinite

from football_alpha_pitch_projection.contracts import (
    CalibrationStatus,
    CameraCalibration,
    ImagePoint,
    PitchPoint,
    PointProjection,
    ProjectionFailureReason,
)


def project_image_point(
    calibration: CameraCalibration,
    image_point: ImagePoint,
) -> PointProjection:
    """Project a finite image point or preserve an explicit no-coordinate result."""
    if calibration.status is CalibrationStatus.ABSTAINED:
        return _result(
            calibration,
            image_point,
            pitch_point=None,
            inside_pitch=None,
            failure_reason=ProjectionFailureReason.CALIBRATION_ABSTAINED,
        )

    homography = calibration.homography_img_to_pitch
    if homography is None:  # pragma: no cover - enforced by CameraCalibration
        raise ValueError("accepted calibration has no homography")
    numerator_x = (
        homography[0][0] * image_point.u
        + homography[0][1] * image_point.v
        + homography[0][2]
    )
    numerator_y = (
        homography[1][0] * image_point.u
        + homography[1][1] * image_point.v
        + homography[1][2]
    )
    denominator = (
        homography[2][0] * image_point.u
        + homography[2][1] * image_point.v
        + homography[2][2]
    )
    if (
        not all(isfinite(value) for value in (numerator_x, numerator_y, denominator))
        or denominator == 0.0
    ):
        return _result(
            calibration,
            image_point,
            pitch_point=None,
            inside_pitch=None,
            failure_reason=ProjectionFailureReason.NON_FINITE_PROJECTION,
        )

    x_m = numerator_x / denominator
    y_m = numerator_y / denominator
    if not isfinite(x_m) or not isfinite(y_m):
        return _result(
            calibration,
            image_point,
            pitch_point=None,
            inside_pitch=None,
            failure_reason=ProjectionFailureReason.NON_FINITE_PROJECTION,
        )
    pitch_point = PitchPoint(x_m=x_m, y_m=y_m)
    x_min, x_max = calibration.pitch.x_bounds_m
    y_min, y_max = calibration.pitch.y_bounds_m
    return _result(
        calibration,
        image_point,
        pitch_point=pitch_point,
        inside_pitch=x_min <= x_m <= x_max and y_min <= y_m <= y_max,
        failure_reason=None,
    )


def _result(
    calibration: CameraCalibration,
    image_point: ImagePoint,
    *,
    pitch_point: PitchPoint | None,
    inside_pitch: bool | None,
    failure_reason: ProjectionFailureReason | None,
) -> PointProjection:
    return PointProjection(
        frame_index=calibration.frame_index,
        pts=calibration.pts,
        time_base=calibration.time_base,
        image_width=calibration.image_width,
        image_height=calibration.image_height,
        image_point=image_point,
        pitch_point=pitch_point,
        inside_pitch=inside_pitch,
        failure_reason=failure_reason,
    )
