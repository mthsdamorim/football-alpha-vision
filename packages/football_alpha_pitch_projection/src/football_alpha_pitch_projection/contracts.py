"""Project-owned contracts for provisional image-to-pitch geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from typing import Protocol, cast

from football_alpha_video import DecodedFrame

type Homography = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class CalibrationStatus(StrEnum):
    """Whether a frame has usable image-to-pitch geometry."""

    ACCEPTED = "accepted"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    """Bounded reasons why a calibrator emitted no geometry."""

    NO_FIELD_CUES = "no_field_cues"
    INSUFFICIENT_CORRESPONDENCES = "insufficient_correspondences"
    INSUFFICIENT_INLIERS = "insufficient_inliers"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    NON_FINITE_CALIBRATION = "non_finite_calibration"
    INDEPENDENT_VALIDATION_FAILED = "independent_validation_failed"
    PROVIDER_FAILURE = "provider_failure"
    TEMPORAL_EVIDENCE_LOST = "temporal_evidence_lost"


class ProjectionFailureReason(StrEnum):
    """Bounded reasons why a generic image point has no metric coordinate."""

    CALIBRATION_ABSTAINED = "calibration_abstained"
    NON_FINITE_PROJECTION = "non_finite_projection"


@dataclass(frozen=True, slots=True)
class PitchDimensions:
    """Metric pitch dimensions for a centered canonical minimap.

    The origin is the pitch center. ``+x`` runs from the left goal to the right
    goal and ``+y`` from the lower touchline to the upper touchline.
    """

    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or value <= 0.0
            for value in (self.length_m, self.width_m)
        ):
            raise ValueError("pitch dimensions must be positive finite meters")

    @property
    def x_bounds_m(self) -> tuple[float, float]:
        """Return inclusive left-goal to right-goal bounds."""
        half_length = self.length_m / 2.0
        return (-half_length, half_length)

    @property
    def y_bounds_m(self) -> tuple[float, float]:
        """Return inclusive lower-touchline to upper-touchline bounds."""
        half_width = self.width_m / 2.0
        return (-half_width, half_width)


@dataclass(frozen=True, slots=True)
class ImagePoint:
    """One finite point in continuous source-image pixels."""

    u: float
    v: float

    def __post_init__(self) -> None:
        if not isfinite(self.u) or not isfinite(self.v):
            raise ValueError("image point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class PitchPoint:
    """One finite point in the centered metric pitch convention."""

    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if not isfinite(self.x_m) or not isfinite(self.y_m):
            raise ValueError("pitch point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class ImagePitchCorrespondence:
    """One explicit image point paired with a known metric pitch point."""

    image_point: ImagePoint
    pitch_point: PitchPoint


@dataclass(frozen=True, slots=True)
class FrameCalibrationInput:
    """Explicit geometric evidence and source identity for one frame."""

    frame_index: int
    pts: int
    time_base: Fraction
    image_width: int
    image_height: int
    pitch: PitchDimensions
    correspondences: tuple[ImagePitchCorrespondence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "correspondences", tuple(self.correspondences))
        _validate_frame_identity(
            self.frame_index,
            self.pts,
            self.time_base,
            self.image_width,
            self.image_height,
        )
        if any(
            correspondence.image_point.u < 0.0
            or correspondence.image_point.u > self.image_width
            or correspondence.image_point.v < 0.0
            or correspondence.image_point.v > self.image_height
            for correspondence in self.correspondences
        ):
            raise ValueError("correspondence image point is outside the source image")

    @classmethod
    def from_frame(
        cls,
        frame: DecodedFrame,
        *,
        pitch: PitchDimensions,
        correspondences: Sequence[ImagePitchCorrespondence],
    ) -> FrameCalibrationInput:
        """Capture exact identity and dimensions from a decoded frame."""
        image_height, image_width = frame.image.shape[:2]
        return cls(
            frame_index=frame.index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=image_width,
            image_height=image_height,
            pitch=pitch,
            correspondences=tuple(correspondences),
        )


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Accepted geometry or explicit abstention for one source frame.

    ``quality_score`` is method-local self-verification. It is neither a
    probability that calibration is correct nor an estimate of metric error.
    """

    frame_index: int
    pts: int
    time_base: Fraction
    image_width: int
    image_height: int
    pitch: PitchDimensions
    status: CalibrationStatus
    homography_img_to_pitch: Homography | None
    abstention_reason: AbstentionReason | None
    method: str
    provenance: str
    quality_score: float | None
    quality_threshold: float
    correspondence_count: int
    inlier_count: int
    reprojection_rmse_px: float | None

    def __post_init__(self) -> None:
        _validate_frame_identity(
            self.frame_index,
            self.pts,
            self.time_base,
            self.image_width,
            self.image_height,
        )
        if not self.method.strip() or not self.provenance.strip():
            raise ValueError("calibration method and provenance must be non-empty")
        if not isfinite(self.quality_threshold) or not (
            0.0 <= self.quality_threshold <= 1.0
        ):
            raise ValueError("quality_threshold must be finite and between 0 and 1")
        if self.quality_score is not None and (
            not isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0
        ):
            raise ValueError("quality_score must be finite and between 0 and 1")
        if (
            type(self.correspondence_count) is not int
            or type(self.inlier_count) is not int
            or self.correspondence_count < 0
            or self.inlier_count < 0
            or self.inlier_count > self.correspondence_count
        ):
            raise ValueError("calibration evidence counts are inconsistent")
        if self.reprojection_rmse_px is not None and (
            not isfinite(self.reprojection_rmse_px) or self.reprojection_rmse_px < 0.0
        ):
            raise ValueError("reprojection_rmse_px must be non-negative and finite")

        if self.status is CalibrationStatus.ACCEPTED:
            if self.homography_img_to_pitch is None:
                raise ValueError("accepted calibration requires a homography")
            if self.abstention_reason is not None:
                raise ValueError(
                    "accepted calibration cannot have an abstention reason"
                )
            if (
                self.quality_score is None
                or self.quality_score < self.quality_threshold
            ):
                raise ValueError("accepted calibration must meet its quality threshold")
            object.__setattr__(
                self,
                "homography_img_to_pitch",
                _validated_homography(self.homography_img_to_pitch),
            )
        elif self.status is CalibrationStatus.ABSTAINED:
            if self.homography_img_to_pitch is not None:
                raise ValueError("abstained calibration cannot expose a homography")
            if self.abstention_reason is None:
                raise ValueError("abstained calibration requires an abstention reason")
            if not isinstance(cast(object, self.abstention_reason), AbstentionReason):
                raise ValueError(
                    "abstained calibration requires a bounded abstention reason"
                )
        else:
            raise ValueError("calibration status must be accepted or abstained")


@dataclass(frozen=True, slots=True)
class PointProjection:
    """One generic image point projected for a specific calibrated frame."""

    frame_index: int
    pts: int
    time_base: Fraction
    image_width: int
    image_height: int
    image_point: ImagePoint
    pitch_point: PitchPoint | None
    inside_pitch: bool | None
    failure_reason: ProjectionFailureReason | None

    def __post_init__(self) -> None:
        _validate_frame_identity(
            self.frame_index,
            self.pts,
            self.time_base,
            self.image_width,
            self.image_height,
        )
        if self.pitch_point is None:
            if self.inside_pitch is not None or not isinstance(
                cast(object, self.failure_reason), ProjectionFailureReason
            ):
                raise ValueError(
                    "failed projection requires a bounded reason and no bounds flag"
                )
        elif type(self.inside_pitch) is not bool or self.failure_reason is not None:
            raise ValueError(
                "successful projection requires a bounds flag and no reason"
            )


class CameraCalibrator(Protocol):
    """Calibrate an explicit ordered batch of frame evidence."""

    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        """Return exactly one ordered result for every input frame."""
        ...


def _validate_frame_identity(
    frame_index: int,
    pts: int,
    time_base: Fraction,
    image_width: int,
    image_height: int,
) -> None:
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    if type(pts) is not int:
        raise ValueError("pts must be an integer")
    if type(time_base) is not Fraction or time_base <= 0:
        raise ValueError("time_base must be a positive exact Fraction")
    if (
        type(image_width) is not int
        or type(image_height) is not int
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError("source image dimensions must be positive integers")


def _validated_homography(matrix: Sequence[Sequence[float]]) -> Homography:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("homography must be a finite 3x3 matrix") from error
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("homography must be a finite 3x3 matrix")
    values = tuple(value for row in rows for value in row)
    if any(not isfinite(value) for value in values):
        raise ValueError("homography must be a finite 3x3 matrix")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        raise ValueError("homography must be non-degenerate")
    rows = tuple(tuple(value / scale for value in row) for row in rows)
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if not isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError("homography must be non-degenerate")
    return rows  # type: ignore[return-value]
