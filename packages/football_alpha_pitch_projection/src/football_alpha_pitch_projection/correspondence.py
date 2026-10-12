"""OpenCV correspondence control for framewise pitch calibration."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from football_alpha_pitch_projection.contracts import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    FrameCalibrationInput,
    Homography,
)

_QUALITY_THRESHOLD = 0.75
_MINIMUM_INLIERS = 4


class CorrespondenceHomographyCalibrator:
    """Estimate framewise image-to-pitch homographies from explicit cues."""

    def __init__(self, *, ransac_reprojection_threshold_px: float = 3.0) -> None:
        if (
            not isfinite(ransac_reprojection_threshold_px)
            or ransac_reprojection_threshold_px <= 0.0
        ):
            raise ValueError("ransac_reprojection_threshold_px must be positive")
        self._ransac_reprojection_threshold_px = ransac_reprojection_threshold_px

    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        """Return one independent accepted or abstained result per input."""
        return [self._calibrate_one(item) for item in inputs]

    def _calibrate_one(self, item: FrameCalibrationInput) -> CameraCalibration:
        count = len(item.correspondences)
        if count == 0:
            return self._abstain(item, AbstentionReason.NO_FIELD_CUES)
        if count < _MINIMUM_INLIERS:
            return self._abstain(
                item,
                AbstentionReason.INSUFFICIENT_CORRESPONDENCES,
                quality_score=0.0,
            )

        image_points = np.asarray(
            [
                (correspondence.image_point.u, correspondence.image_point.v)
                for correspondence in item.correspondences
            ],
            dtype=np.float64,
        )
        pitch_points = np.asarray(
            [
                (
                    correspondence.pitch_point.x_m,
                    correspondence.pitch_point.y_m,
                )
                for correspondence in item.correspondences
            ],
            dtype=np.float64,
        )
        try:
            pitch_to_image_object, inlier_mask_object = cast(
                tuple[object, object],
                cv2.findHomography(
                    pitch_points,
                    image_points,
                    cv2.RANSAC,
                    self._ransac_reprojection_threshold_px,
                ),
            )
        except cv2.error:
            return self._abstain(item, AbstentionReason.PROVIDER_FAILURE)

        if pitch_to_image_object is None or inlier_mask_object is None:
            return self._abstain(
                item,
                AbstentionReason.DEGENERATE_GEOMETRY,
                quality_score=0.0,
            )
        pitch_to_image = np.asarray(pitch_to_image_object, dtype=np.float64)
        inlier_mask = np.asarray(inlier_mask_object).reshape(-1)
        if pitch_to_image.shape != (3, 3) or inlier_mask.shape != (count,):
            return self._abstain(item, AbstentionReason.PROVIDER_FAILURE)

        inlier_count = int(np.count_nonzero(inlier_mask))
        quality_score = inlier_count / count
        if not np.isfinite(pitch_to_image).all():
            return self._abstain(
                item,
                AbstentionReason.NON_FINITE_CALIBRATION,
                quality_score=quality_score,
                inlier_count=inlier_count,
            )
        if np.linalg.matrix_rank(pitch_to_image) < 3:
            return self._abstain(
                item,
                AbstentionReason.DEGENERATE_GEOMETRY,
                quality_score=quality_score,
                inlier_count=inlier_count,
            )
        rmse_px = _reprojection_rmse_px(
            pitch_to_image,
            pitch_points,
            image_points,
            inlier_mask,
        )
        if inlier_count < _MINIMUM_INLIERS or quality_score < _QUALITY_THRESHOLD:
            return self._abstain(
                item,
                AbstentionReason.INSUFFICIENT_INLIERS,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )
        if (
            _homography_design_rank(
                pitch_points[inlier_mask.astype(bool)],
                image_points[inlier_mask.astype(bool)],
            )
            < 8
        ):
            return self._abstain(
                item,
                AbstentionReason.DEGENERATE_GEOMETRY,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )

        try:
            image_to_pitch = np.linalg.inv(pitch_to_image)
        except np.linalg.LinAlgError:
            return self._abstain(
                item,
                AbstentionReason.DEGENERATE_GEOMETRY,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )
        if not np.isfinite(image_to_pitch).all():
            return self._abstain(
                item,
                AbstentionReason.NON_FINITE_CALIBRATION,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )

        homography = cast(
            Homography,
            tuple(tuple(float(value) for value in row) for row in image_to_pitch),
        )
        try:
            return self._result(
                item,
                status=CalibrationStatus.ACCEPTED,
                homography_img_to_pitch=homography,
                abstention_reason=None,
                quality_score=quality_score,
                correspondence_count=count,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )
        except ValueError:
            return self._abstain(
                item,
                AbstentionReason.DEGENERATE_GEOMETRY,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=rmse_px,
            )

    def _abstain(
        self,
        item: FrameCalibrationInput,
        reason: AbstentionReason,
        *,
        quality_score: float | None = None,
        inlier_count: int = 0,
        reprojection_rmse_px: float | None = None,
    ) -> CameraCalibration:
        return self._result(
            item,
            status=CalibrationStatus.ABSTAINED,
            homography_img_to_pitch=None,
            abstention_reason=reason,
            quality_score=quality_score,
            correspondence_count=len(item.correspondences),
            inlier_count=inlier_count,
            reprojection_rmse_px=reprojection_rmse_px,
        )

    def _result(
        self,
        item: FrameCalibrationInput,
        *,
        status: CalibrationStatus,
        homography_img_to_pitch: Homography | None,
        abstention_reason: AbstentionReason | None,
        quality_score: float | None,
        correspondence_count: int,
        inlier_count: int,
        reprojection_rmse_px: float | None,
    ) -> CameraCalibration:
        return CameraCalibration(
            frame_index=item.frame_index,
            pts=item.pts,
            time_base=item.time_base,
            image_width=item.image_width,
            image_height=item.image_height,
            pitch=item.pitch,
            status=status,
            homography_img_to_pitch=homography_img_to_pitch,
            abstention_reason=abstention_reason,
            method="correspondence_homography_ransac",
            provenance=(
                f"opencv:{cv2.__version__};"
                f"ransac_reprojection_threshold_px:"
                f"{self._ransac_reprojection_threshold_px!r}"
            ),
            quality_score=quality_score,
            quality_threshold=_QUALITY_THRESHOLD,
            correspondence_count=correspondence_count,
            inlier_count=inlier_count,
            reprojection_rmse_px=reprojection_rmse_px,
        )


def _reprojection_rmse_px(
    pitch_to_image: NDArray[np.float64],
    pitch_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    inlier_mask: NDArray[np.generic],
) -> float | None:
    inliers = inlier_mask.astype(bool)
    if not np.any(inliers):
        return None
    homogeneous_pitch = np.column_stack(
        (pitch_points[inliers], np.ones(np.count_nonzero(inliers)))
    )
    projected = homogeneous_pitch @ pitch_to_image.T
    denominators = projected[:, 2]
    if not np.isfinite(projected).all() or np.any(denominators == 0.0):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        projected_image = projected[:, :2] / denominators[:, None]
        squared_errors = np.sum(
            (projected_image - image_points[inliers]) ** 2,
            axis=1,
        )
    rmse = sqrt(float(np.mean(squared_errors)))
    return rmse if isfinite(rmse) else None


def _homography_design_rank(
    pitch_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
) -> int:
    pitch_scale = float(np.max(np.abs(pitch_points)))
    image_scale = float(np.max(np.abs(image_points)))
    if pitch_scale > 0.0:
        pitch_points = pitch_points / pitch_scale
    if image_scale > 0.0:
        image_points = image_points / image_scale
    rows: list[list[float]] = []
    for (x_m, y_m), (u, v) in zip(pitch_points, image_points, strict=True):
        rows.extend(
            (
                [-x_m, -y_m, -1.0, 0.0, 0.0, 0.0, u * x_m, u * y_m, u],
                [0.0, 0.0, 0.0, -x_m, -y_m, -1.0, v * x_m, v * y_m, v],
            )
        )
    return int(np.linalg.matrix_rank(np.asarray(rows, dtype=np.float64)))
