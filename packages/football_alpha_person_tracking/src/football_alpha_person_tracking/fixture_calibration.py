"""Manual calibration cues for the bounded broadcast acceptance fixture."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as scipy_optimize  # pyright: ignore[reportMissingTypeStubs]

from football_alpha_pitch_projection import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    CameraCalibrator,
    CorrespondenceHomographyCalibrator,
    FrameCalibrationInput,
    Homography,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
    calibrate_frames,
)

CLIP_SHA256 = "0ad6f9a7c5facf49e58bcb9a2afaf579136e8d45153f8860e3ee1b85a947b001"
CLIP_PITCH = PitchDimensions(105.0, 68.0)
_MINIMUM_CORRESPONDENCE_INLIERS = 4


class _LeastSquaresResult(Protocol):
    x: NDArray[np.float64]
    success: bool


_least_squares = cast(
    Callable[..., _LeastSquaresResult],
    scipy_optimize.least_squares,  # pyright: ignore[reportUnknownMemberType]
)


@dataclass(frozen=True, slots=True)
class HeldOutPitchMarking:
    """Fixture-local image samples sharing one known pitch-axis coordinate."""

    axis: Literal["x", "y"]
    coordinate_m: float
    image_points: tuple[ImagePoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_points", tuple(self.image_points))
        if self.axis not in ("x", "y"):
            raise ValueError("held-out pitch marking axis must be x or y")
        if not isfinite(self.coordinate_m):
            raise ValueError("held-out pitch marking coordinate must be finite")
        if not self.image_points:
            raise ValueError("held-out pitch marking requires image samples")


class FixtureCalibrationValidator:
    """Fail closed when a fitted fixture anchor misses reserved field lines."""

    def __init__(
        self,
        calibrator: CameraCalibrator,
        *,
        held_out_markings_by_frame: Mapping[int, Sequence[HeldOutPitchMarking]],
        maximum_error_px: float,
        fit_markings_by_frame: Mapping[int, Sequence[HeldOutPitchMarking]]
        | None = None,
        correspondence_reprojection_threshold_px: float = 3.0,
    ) -> None:
        if not isfinite(maximum_error_px) or maximum_error_px <= 0.0:
            raise ValueError("maximum_error_px must be positive and finite")
        if (
            not isfinite(correspondence_reprojection_threshold_px)
            or correspondence_reprojection_threshold_px <= 0.0
        ):
            raise ValueError(
                "correspondence_reprojection_threshold_px must be positive and finite"
            )
        self._calibrator = calibrator
        self._held_out_markings_by_frame = {
            frame_index: tuple(markings)
            for frame_index, markings in held_out_markings_by_frame.items()
        }
        self._fit_markings_by_frame = {
            frame_index: tuple(markings)
            for frame_index, markings in (fit_markings_by_frame or {}).items()
        }
        self._maximum_error_px = maximum_error_px
        self._correspondence_reprojection_threshold_px = (
            correspondence_reprojection_threshold_px
        )

    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        """Refine fixture anchors and validate them against held-out lines."""
        baseline = calibrate_frames(self._calibrator, inputs)
        return [
            self._validate_one(calibration_input, calibration)
            for calibration_input, calibration in zip(inputs, baseline, strict=True)
        ]

    def _validate_one(
        self,
        calibration_input: FrameCalibrationInput,
        calibration: CameraCalibration,
    ) -> CameraCalibration:
        fit_markings = self._fit_markings_by_frame.get(calibration.frame_index, ())
        if calibration.status is CalibrationStatus.ABSTAINED and not fit_markings:
            return calibration
        if fit_markings:
            refined_homography = _fit_point_line_homography(
                calibration_input, fit_markings
            )
            if refined_homography is None:
                return self._abstain(calibration, None, 0)
            calibration = self._apply_final_correspondence_gate(
                calibration_input,
                calibration,
                refined_homography,
                sum(len(marking.image_points) for marking in fit_markings),
            )
            if calibration.status is CalibrationStatus.ABSTAINED:
                return calibration

        markings = self._held_out_markings_by_frame.get(calibration.frame_index, ())
        homography = cast(Homography, calibration.homography_img_to_pitch)
        errors_px: list[float] = []
        for marking in markings:
            for image_point in marking.image_points:
                if not _inside_source(calibration_input, image_point):
                    return self._abstain(calibration, None, len(errors_px))
                error_px = _image_line_error_px(
                    homography,
                    marking,
                    image_point,
                )
                if error_px is None:
                    return self._abstain(calibration, None, len(errors_px))
                errors_px.append(error_px)

        maximum_observed_error_px = max(errors_px, default=None)
        if (
            maximum_observed_error_px is None
            or maximum_observed_error_px > self._maximum_error_px
        ):
            return self._abstain(
                calibration,
                maximum_observed_error_px,
                len(errors_px),
            )
        return replace(
            calibration,
            method=f"{calibration.method}+fixture_held_out_lines",
            provenance=self._provenance(
                calibration,
                maximum_observed_error_px,
                len(errors_px),
            ),
        )

    def _apply_final_correspondence_gate(
        self,
        calibration_input: FrameCalibrationInput,
        calibration: CameraCalibration,
        homography: Homography,
        point_line_sample_count: int,
    ) -> CameraCalibration:
        evidence = _final_correspondence_evidence(
            calibration_input,
            homography,
            self._correspondence_reprojection_threshold_px,
        )
        method = f"{calibration.method}+fixture_point_line"
        provenance = (
            f"{calibration.provenance};"
            f"fixture_point_line_sample_count:{point_line_sample_count};"
            "fixture_final_correspondence_gate:true;"
            "fixture_final_correspondence_reprojection_threshold_px:"
            f"{self._correspondence_reprojection_threshold_px!r}"
        )
        if evidence is None:
            return replace(
                calibration,
                status=CalibrationStatus.ABSTAINED,
                homography_img_to_pitch=None,
                abstention_reason=AbstentionReason.NON_FINITE_CALIBRATION,
                method=method,
                provenance=provenance,
                quality_score=None,
                inlier_count=0,
                reprojection_rmse_px=None,
            )

        inlier_count, quality_score, reprojection_rmse_px = evidence
        if (
            inlier_count < _MINIMUM_CORRESPONDENCE_INLIERS
            or quality_score < calibration.quality_threshold
        ):
            return replace(
                calibration,
                status=CalibrationStatus.ABSTAINED,
                homography_img_to_pitch=None,
                abstention_reason=AbstentionReason.INSUFFICIENT_INLIERS,
                method=method,
                provenance=provenance,
                quality_score=quality_score,
                inlier_count=inlier_count,
                reprojection_rmse_px=reprojection_rmse_px,
            )
        return replace(
            calibration,
            status=CalibrationStatus.ACCEPTED,
            homography_img_to_pitch=homography,
            abstention_reason=None,
            method=method,
            provenance=provenance,
            quality_score=quality_score,
            inlier_count=inlier_count,
            reprojection_rmse_px=reprojection_rmse_px,
        )

    def _abstain(
        self,
        calibration: CameraCalibration,
        maximum_observed_error_px: float | None,
        sample_count: int,
    ) -> CameraCalibration:
        return replace(
            calibration,
            status=CalibrationStatus.ABSTAINED,
            homography_img_to_pitch=None,
            abstention_reason=AbstentionReason.INDEPENDENT_VALIDATION_FAILED,
            method=f"{calibration.method}+fixture_held_out_lines",
            provenance=self._provenance(
                calibration,
                maximum_observed_error_px,
                sample_count,
            ),
        )

    def _provenance(
        self,
        calibration: CameraCalibration,
        maximum_observed_error_px: float | None,
        sample_count: int,
    ) -> str:
        observed = (
            "unavailable"
            if maximum_observed_error_px is None
            else repr(maximum_observed_error_px)
        )
        return (
            f"{calibration.provenance};"
            f"fixture_held_out_line_sample_count:{sample_count};"
            f"fixture_held_out_maximum_error_px:{observed};"
            f"fixture_held_out_threshold_px:{self._maximum_error_px!r}"
        )


def _fit_point_line_homography(
    calibration_input: FrameCalibrationInput,
    markings: Sequence[HeldOutPitchMarking],
) -> Homography | None:
    image_scale = np.asarray(
        (
            (1.0 / (calibration_input.image_width / 2.0), 0.0, -1.0),
            (0.0, 1.0 / (calibration_input.image_height / 2.0), -1.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    pitch_scale = np.asarray(
        (
            (1.0 / (calibration_input.pitch.length_m / 2.0), 0.0, 0.0),
            (0.0, 1.0 / (calibration_input.pitch.width_m / 2.0), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    rows: list[list[float]] = []

    def add_row(axis: Literal["x", "y"], image_point: ImagePoint, value: float) -> None:
        u, v, _ = image_scale @ np.asarray(
            (image_point.u, image_point.v, 1.0), dtype=np.float64
        )
        scale = (
            calibration_input.pitch.length_m / 2.0
            if axis == "x"
            else calibration_input.pitch.width_m / 2.0
        )
        coordinate = value / scale
        if axis == "x":
            rows.append(
                [
                    -u,
                    -v,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    coordinate * u,
                    coordinate * v,
                    coordinate,
                ]
            )
        else:
            rows.append(
                [
                    0.0,
                    0.0,
                    0.0,
                    -u,
                    -v,
                    -1.0,
                    coordinate * u,
                    coordinate * v,
                    coordinate,
                ]
            )

    for correspondence in calibration_input.correspondences:
        add_row("x", correspondence.image_point, correspondence.pitch_point.x_m)
        add_row("y", correspondence.image_point, correspondence.pitch_point.y_m)
    for marking in markings:
        for image_point in marking.image_points:
            if not _inside_source(calibration_input, image_point):
                return None
            add_row(marking.axis, image_point, marking.coordinate_m)

    design = np.asarray(rows, dtype=np.float64)
    if design.shape[0] < 8 or np.linalg.matrix_rank(design) < 8:
        return None
    try:
        _, _, right_singular_vectors = np.linalg.svd(design)
        normalized = right_singular_vectors[-1].reshape(3, 3)
        homography = np.linalg.inv(pitch_scale) @ normalized @ image_scale
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        return None
    denominator = homography[2, 2]
    if denominator == 0.0:
        return None
    homography /= denominator
    return _refine_point_line_homography(
        calibration_input,
        markings,
        homography,
    )


def _refine_point_line_homography(
    calibration_input: FrameCalibrationInput,
    markings: Sequence[HeldOutPitchMarking],
    initial_homography: NDArray[np.float64],
) -> Homography | None:
    try:
        pitch_to_image = np.linalg.inv(initial_homography)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by rank check above
        return None
    scale = pitch_to_image[2, 2]
    if scale == 0.0 or not isfinite(float(scale)):
        return None
    pitch_to_image /= scale
    try:
        result = _least_squares(
            _point_line_residuals,
            pitch_to_image.reshape(-1)[:8],
            args=(calibration_input, markings),
            max_nfev=200,
        )
        refined_pitch_to_image = np.append(result.x, 1.0).reshape(3, 3)
        homography = np.linalg.inv(refined_pitch_to_image)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not result.success or not np.isfinite(homography).all():
        return None
    denominator = homography[2, 2]
    if denominator == 0.0 or not isfinite(float(denominator)):
        return None
    homography /= denominator
    return cast(
        Homography,
        tuple(tuple(float(value) for value in row) for row in homography),
    )


def _point_line_residuals(
    parameters: NDArray[np.float64],
    calibration_input: FrameCalibrationInput,
    markings: Sequence[HeldOutPitchMarking],
) -> NDArray[np.float64]:
    pitch_to_image = np.append(parameters, 1.0).reshape(3, 3)
    residual_count = 2 * len(calibration_input.correspondences) + sum(
        len(marking.image_points) for marking in markings
    )
    pitch_points, image_points = _correspondence_arrays(calibration_input)
    projected = pitch_points @ pitch_to_image.T
    denominators = projected[:, 2]
    if not np.isfinite(projected).all() or np.any(denominators == 0.0):
        return np.full(residual_count, 1e6, dtype=np.float64)
    projected_image = projected[:, :2] / denominators[:, None]
    residuals = [float(value) for value in (projected_image - image_points).flat]
    try:
        homography = np.linalg.inv(pitch_to_image)
    except np.linalg.LinAlgError:
        return np.full(residual_count, 1e6, dtype=np.float64)
    for marking in markings:
        pitch_line = np.asarray(
            (1.0, 0.0, -marking.coordinate_m)
            if marking.axis == "x"
            else (0.0, 1.0, -marking.coordinate_m),
            dtype=np.float64,
        )
        image_line = homography.T @ pitch_line
        normal = float(np.hypot(image_line[0], image_line[1]))
        if not isfinite(normal) or normal == 0.0:
            return np.full(residual_count, 1e6, dtype=np.float64)
        residuals.extend(
            float(image_line @ np.asarray((point.u, point.v, 1.0))) / normal
            for point in marking.image_points
        )
    return np.asarray(residuals, dtype=np.float64)


def _final_correspondence_evidence(
    calibration_input: FrameCalibrationInput,
    homography: Homography,
    threshold_px: float,
) -> tuple[int, float, float | None] | None:
    try:
        pitch_to_image = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    pitch_points, image_points = _correspondence_arrays(calibration_input)
    projected = pitch_points @ pitch_to_image.T
    denominators = projected[:, 2]
    if not np.isfinite(projected).all() or np.any(denominators == 0.0):
        return None
    projected_image = projected[:, :2] / denominators[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        errors_px = np.linalg.norm(projected_image - image_points, axis=1)
    if not np.isfinite(errors_px).all():
        return None
    inliers = errors_px <= threshold_px
    inlier_count = int(np.count_nonzero(inliers))
    quality_score = inlier_count / len(calibration_input.correspondences)
    reprojection_rmse_px = (
        sqrt(float(np.mean(errors_px[inliers] ** 2))) if inlier_count else None
    )
    return inlier_count, quality_score, reprojection_rmse_px


def _correspondence_arrays(
    calibration_input: FrameCalibrationInput,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pitch_points = np.asarray(
        [
            (
                correspondence.pitch_point.x_m,
                correspondence.pitch_point.y_m,
                1.0,
            )
            for correspondence in calibration_input.correspondences
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [
            (correspondence.image_point.u, correspondence.image_point.v)
            for correspondence in calibration_input.correspondences
        ],
        dtype=np.float64,
    )
    return pitch_points, image_points


def _image_line_error_px(
    homography: Homography,
    marking: HeldOutPitchMarking,
    image_point: ImagePoint,
) -> float | None:
    pitch_line = np.asarray(
        (1.0, 0.0, -marking.coordinate_m)
        if marking.axis == "x"
        else (0.0, 1.0, -marking.coordinate_m),
        dtype=np.float64,
    )
    image_line = np.asarray(homography, dtype=np.float64).T @ pitch_line
    normal = float(np.hypot(image_line[0], image_line[1]))
    if not isfinite(normal) or normal == 0.0:
        return None
    numerator = abs(float(image_line @ np.asarray((image_point.u, image_point.v, 1.0))))
    error_px = numerator / normal
    return error_px if isfinite(error_px) else None


def _inside_source(
    calibration_input: FrameCalibrationInput,
    image_point: ImagePoint,
) -> bool:
    return (
        0.0 <= image_point.u <= calibration_input.image_width
        and 0.0 <= image_point.v <= calibration_input.image_height
    )


def _cues(
    values: Sequence[tuple[float, float, float, float]],
) -> tuple[ImagePitchCorrespondence, ...]:
    return tuple(
        ImagePitchCorrespondence(
            image_point=ImagePoint(u=u, v=v),
            pitch_point=PitchPoint(x_m=x_m, y_m=y_m),
        )
        for u, v, x_m, y_m in values
    )


# These five frame-local cue sets are acceptance metadata for data/clip.mp4.
# Pitch dimensions are assumed, not venue metadata. The points are not a gold
# calibration dataset and must not be propagated to frames without cues.
CLIP_ANCHOR_CORRESPONDENCES: Mapping[int, tuple[ImagePitchCorrespondence, ...]] = (
    MappingProxyType(
        {
            0: _cues(
                (
                    (728.000, 329.000, 0.000000, 34.000000),
                    (733.000, 712.000, 0.000000, -9.150000),
                    (164.000, 609.000, -9.150000, 0.000000),
                    (1309.000, 607.000, 9.150000, 0.000000),
                    (731.642, 608.009, 0.000000, 0.000000),
                    (730.475, 518.612, 0.000000, 9.150000),
                )
            ),
            180: _cues(
                (
                    (484.000, 500.000, 36.000000, 20.160000),
                    (1512.000, 980.000, 36.000000, -20.160000),
                    (623.000, 674.000, 32.350000, 0.000000),
                    (1245.000, 640.000, 41.500000, 0.000000),
                    (682.631, 592.746, 36.000000, 7.312489),
                    (1021.334, 750.895, 36.000000, -7.312489),
                )
            ),
            195: _cues(
                (
                    (307.000, 500.000, 36.000000, 20.160000),
                    (1360.000, 989.000, 36.000000, -20.160000),
                    (1423.000, 471.000, 52.500000, 20.160000),
                    (1100.000, 680.000, 41.500000, 0.000000),
                    (560.000, 620.000, 36.000000, 7.312489),
                    (940.000, 780.000, 36.000000, -7.312489),
                )
            ),
            210: _cues(
                (
                    (155.000, 500.000, 36.000000, 20.160000),
                    (1260.000, 994.000, 36.000000, -20.160000),
                    (330.000, 690.000, 32.350000, 0.000000),
                    (1040.000, 685.000, 41.500000, 0.000000),
                    (430.000, 620.000, 36.000000, 7.312489),
                    (820.000, 790.000, 36.000000, -7.312489),
                )
            ),
            225: _cues(
                (
                    (49.000, 500.000, 36.000000, 20.160000),
                    (1170.000, 1012.000, 36.000000, -20.160000),
                    (210.000, 687.000, 32.350000, 0.000000),
                    (908.000, 686.000, 41.500000, 0.000000),
                    (286.122, 608.302, 36.000000, 7.312489),
                    (666.565, 782.063, 36.000000, -7.312489),
                )
            ),
        }
    )
)


def _marking(
    axis: Literal["x", "y"],
    coordinate_m: float,
    *image_points: tuple[float, float],
) -> HeldOutPitchMarking:
    return HeldOutPitchMarking(
        axis,
        coordinate_m,
        tuple(ImagePoint(u, v) for u, v in image_points),
    )


CLIP_FIT_MARKINGS: Mapping[int, tuple[HeldOutPitchMarking, ...]] = MappingProxyType(
    {
        180: (
            _marking("y", 34.0, (0, 418), (400, 390), (800, 361), (1018, 346)),
            _marking("x", 52.5, (1018, 346), (1200, 405), (1500, 502), (1800, 599)),
            _marking(
                "x", 36.0, (473, 492), (700, 598), (1000, 738), (1300, 878), (1514, 978)
            ),
            _marking("y", 20.16, (473, 492), (800, 480), (1100, 469), (1367, 459)),
            _marking("y", 9.16, (1392, 579), (1500, 567), (1649, 550)),
        ),
        195: (
            _marking(
                "y", 34.0, (0, 382), (250, 374), (500, 366), (750, 358), (968, 351)
            ),
            _marking(
                "x",
                52.5,
                (968, 351),
                (1150, 399),
                (1350, 452),
                (1550, 505),
                (1740, 555),
                (1900, 597),
            ),
            _marking(
                "x",
                36.0,
                (307, 500),
                (500, 590),
                (700, 683),
                (900, 775),
                (1100, 868),
                (1360, 989),
            ),
            _marking(
                "y", 20.16, (307, 500), (600, 493), (900, 486), (1200, 478), (1420, 470)
            ),
            _marking("y", 9.16, (1400, 560), (1500, 559), (1600, 557), (1740, 555)),
        ),
        210: (
            _marking(
                "y", 34.0, (0, 374), (200, 368), (400, 362), (600, 356), (832, 347)
            ),
            _marking(
                "x",
                52.5,
                (832, 347),
                (1000, 393),
                (1200, 447),
                (1400, 501),
                (1600, 555),
                (1800, 609),
            ),
            _marking(
                "x",
                36.0,
                (155, 500),
                (350, 587),
                (550, 676),
                (750, 765),
                (950, 854),
                (1150, 943),
                (1260, 994),
            ),
            _marking(
                "y",
                20.16,
                (155, 500),
                (400, 495),
                (650, 490),
                (900, 484),
                (1150, 479),
                (1300, 475),
            ),
            _marking("y", 9.16, (1250, 560), (1350, 559), (1450, 557), (1600, 555)),
        ),
        225: (
            _marking(
                "y", 34.0, (0, 355), (200, 350), (400, 345), (600, 340), (700, 338)
            ),
            _marking(
                "x",
                52.5,
                (700, 338),
                (900, 403),
                (1100, 468),
                (1300, 533),
                (1500, 598),
                (1700, 663),
                (1900, 728),
            ),
            _marking(
                "x", 36.0, (49, 500), (300, 615), (600, 752), (900, 889), (1170, 1012)
            ),
            _marking(
                "y", 20.16, (49, 500), (300, 493), (600, 484), (900, 476), (1100, 470)
            ),
        ),
    }
)

CLIP_HELD_OUT_MARKINGS: Mapping[int, tuple[HeldOutPitchMarking, ...]] = (
    MappingProxyType(
        {
            0: (
                _marking("x", 0.0, (729, 400), (730, 500), (734, 800), (736, 1000)),
                _marking("y", 34.0, (200, 332), (500, 330), (1000, 327), (1500, 324)),
            ),
            180: (
                _marking("y", 34.0, (200, 404), (600, 376), (900, 354)),
                _marking("x", 52.5, (1150, 389), (1400, 470), (1700, 567)),
                _marking("x", 36.0, (600, 551), (900, 691), (1200, 831)),
                _marking("y", 20.16, (650, 486), (950, 475), (1250, 463)),
                _marking("y", 9.16, (1450, 572), (1580, 558)),
            ),
            195: (
                _marking("y", 34.0, (125, 378), (375, 370), (625, 362), (875, 354)),
                _marking(
                    "x",
                    52.5,
                    (1050, 373),
                    (1250, 426),
                    (1450, 479),
                    (1650, 532),
                    (1850, 584),
                ),
                _marking(
                    "x",
                    36.0,
                    (400, 543),
                    (600, 636),
                    (800, 729),
                    (1000, 822),
                    (1200, 915),
                ),
                _marking("y", 20.16, (450, 497), (750, 490), (1050, 482), (1325, 473)),
                _marking("y", 9.16, (1450, 560), (1550, 558), (1650, 556)),
            ),
            210: (
                _marking("y", 34.0, (100, 371), (300, 365), (500, 359), (700, 352)),
                _marking(
                    "x",
                    52.5,
                    (920, 371),
                    (1100, 420),
                    (1300, 474),
                    (1500, 528),
                    (1700, 582),
                ),
                _marking(
                    "x",
                    36.0,
                    (250, 542),
                    (450, 631),
                    (650, 720),
                    (850, 809),
                    (1050, 898),
                    (1200, 965),
                ),
                _marking(
                    "y",
                    20.16,
                    (275, 498),
                    (525, 493),
                    (775, 487),
                    (1025, 482),
                    (1225, 477),
                ),
                _marking("y", 9.16, (1300, 560), (1400, 558), (1500, 556)),
            ),
            225: (
                _marking("y", 34.0, (100, 353), (300, 348), (500, 343), (650, 339)),
                _marking(
                    "x",
                    52.5,
                    (800, 370),
                    (1000, 435),
                    (1200, 500),
                    (1400, 565),
                    (1600, 630),
                    (1800, 695),
                ),
                _marking("x", 36.0, (150, 546), (450, 683), (750, 820), (1050, 957)),
                _marking("y", 20.16, (150, 497), (450, 489), (750, 480), (1000, 473)),
            ),
        }
    )
)

CLIP_VALIDATION_MAXIMUM_ERROR_PX = 25.0


def create_clip_calibrator(
    anchor_frames: Sequence[int] = tuple(CLIP_ANCHOR_CORRESPONDENCES),
) -> CameraCalibrator:
    """Build the clip-bound calibrator with independent image-space evidence."""
    return FixtureCalibrationValidator(
        CorrespondenceHomographyCalibrator(ransac_reprojection_threshold_px=30.0),
        fit_markings_by_frame={
            i: CLIP_FIT_MARKINGS[i] for i in anchor_frames if i in CLIP_FIT_MARKINGS
        },
        held_out_markings_by_frame={
            i: CLIP_HELD_OUT_MARKINGS[i] for i in anchor_frames
        },
        maximum_error_px=CLIP_VALIDATION_MAXIMUM_ERROR_PX,
        correspondence_reprojection_threshold_px=30.0,
    )


def evaluate_temporal_clip(path: Path) -> dict[str, object]:
    """Evaluate already estimated geometry against reserved fixture markings."""
    records = [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text().splitlines()
    ]
    if [r["frame_index"] for r in records] != list(range(180, 226)):
        raise ValueError(
            "temporal evaluation requires exactly frames 180..225 in order"
        )
    accepted_count = 0
    errors: dict[str, float | None] = {}
    for record in records:
        index = cast(int, record["frame_index"])
        calibration = cast(dict[str, object], record["calibration"])
        accepted = calibration["status"] == "accepted"
        accepted_count += accepted
        if index in (195, 225):
            residuals = (
                [
                    _image_line_error_px(
                        cast(Homography, calibration["homography_img_to_pitch"]),
                        marking,
                        point,
                    )
                    for marking in CLIP_HELD_OUT_MARKINGS[index]
                    for point in marking.image_points
                ]
                if accepted
                else []
            )
            errors[str(index)] = (
                max(cast(list[float], residuals))
                if residuals and all(error is not None for error in residuals)
                else None
            )
    coverage = accepted_count / len(records)
    return {
        "frame_interval": [180, 225],
        "anchor_frames": [180, 210],
        "evaluation_frames": [195, 225],
        "accepted_frame_count": accepted_count,
        "frame_count": len(records),
        "coverage": coverage,
        "minimum_coverage": 0.8,
        "maximum_allowed_line_error_px": 25.0,
        "maximum_line_error_px_by_frame": errors,
        "manual_control_coverage": 4 / 46,
        "passed": coverage >= 0.8
        and all(error is not None and error <= 25.0 for error in errors.values()),
    }
