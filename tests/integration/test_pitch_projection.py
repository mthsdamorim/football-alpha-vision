"""Behavior tests for abstention-aware image-to-pitch projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import cv2
import numpy as np
import pytest

from football_alpha_pitch_projection import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    CorrespondenceHomographyCalibrator,
    FrameCalibrationInput,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
    PointProjection,
    ProjectionFailureReason,
    calibrate_frames,
    project_image_point,
)
from football_alpha_video import DecodedFrame


def _frame(index: int = 0) -> DecodedFrame:
    return DecodedFrame(
        index=index,
        pts=index * 512,
        time_base=Fraction(1, 15_360),
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )


def test_geometry_contracts_are_immutable_and_snapshot_correspondences() -> None:
    pitch = PitchDimensions(length_m=105.0, width_m=68.0)
    correspondence = ImagePitchCorrespondence(
        image_point=ImagePoint(u=10.5, v=20.25),
        pitch_point=PitchPoint(x_m=-52.5, y_m=-34.0),
    )
    mutable_correspondences = [correspondence]

    calibration_input = FrameCalibrationInput.from_frame(
        _frame(3),
        pitch=pitch,
        correspondences=mutable_correspondences,
    )
    mutable_correspondences.clear()

    assert calibration_input.frame_index == 3
    assert calibration_input.pts == 1_536
    assert calibration_input.time_base == Fraction(1, 15_360)
    assert calibration_input.image_width == 200
    assert calibration_input.image_height == 100
    assert calibration_input.correspondences == (correspondence,)
    assert pitch.x_bounds_m == (-52.5, 52.5)
    assert pitch.y_bounds_m == (-34.0, 34.0)
    with pytest.raises(FrozenInstanceError):
        pitch.length_m = 100.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PitchDimensions(0.0, 68.0), "positive finite"),
        (lambda: PitchDimensions(105.0, float("nan")), "positive finite"),
        (lambda: ImagePoint(float("inf"), 2.0), "finite"),
        (lambda: PitchPoint(1.0, float("nan")), "finite"),
    ],
)
def test_geometry_contracts_reject_invalid_values(
    factory: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_calibration_status_invariants_are_explicit() -> None:
    pitch = PitchDimensions(105.0, 68.0)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    accepted = CameraCalibration(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        pitch=pitch,
        status=CalibrationStatus.ACCEPTED,
        homography_img_to_pitch=identity,
        abstention_reason=None,
        method="synthetic",
        provenance="test-control",
        quality_score=1.0,
        quality_threshold=0.75,
        correspondence_count=4,
        inlier_count=4,
        reprojection_rmse_px=0.0,
    )
    assert accepted.homography_img_to_pitch == identity

    abstained = CameraCalibration(
        frame_index=1,
        pts=1,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        pitch=pitch,
        status=CalibrationStatus.ABSTAINED,
        homography_img_to_pitch=None,
        abstention_reason=AbstentionReason.NO_FIELD_CUES,
        method="synthetic",
        provenance="test-control",
        quality_score=None,
        quality_threshold=0.75,
        correspondence_count=0,
        inlier_count=0,
        reprojection_rmse_px=None,
    )
    assert abstained.homography_img_to_pitch is None

    with pytest.raises(ValueError, match="accepted calibration"):
        replace(accepted, homography_img_to_pitch=None)


class EchoCalibrator:
    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        return [
            CameraCalibration(
                frame_index=item.frame_index,
                pts=item.pts,
                time_base=item.time_base,
                image_width=item.image_width,
                image_height=item.image_height,
                pitch=item.pitch,
                status=CalibrationStatus.ABSTAINED,
                homography_img_to_pitch=None,
                abstention_reason=AbstentionReason.NO_FIELD_CUES,
                method="echo",
                provenance="test-control",
                quality_score=None,
                quality_threshold=0.75,
                correspondence_count=0,
                inlier_count=0,
                reprojection_rmse_px=None,
            )
            for item in inputs
        ]


def _input(index: int) -> FrameCalibrationInput:
    return FrameCalibrationInput.from_frame(
        _frame(index),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=(),
    )


def test_batch_contract_preserves_exact_identity_and_order() -> None:
    inputs = [_input(3), _input(4)]

    results = calibrate_frames(EchoCalibrator(), inputs)

    assert [result.frame_index for result in results] == [3, 4]
    assert [result.pts for result in results] == [1_536, 2_048]
    assert all(result.time_base == Fraction(1, 15_360) for result in results)
    assert all(result.image_width == 200 for result in results)
    assert all(result.image_height == 100 for result in results)


def test_batch_contract_snapshots_input_order_before_provider_call() -> None:
    mutable_inputs = [_input(3), _input(4)]

    class MutatingCalibrator:
        def calibrate(
            self,
            inputs: Sequence[FrameCalibrationInput],
        ) -> list[CameraCalibration]:
            mutable_inputs.reverse()
            return EchoCalibrator().calibrate(inputs)

    results = calibrate_frames(MutatingCalibrator(), mutable_inputs)

    assert [result.frame_index for result in results] == [3, 4]


def test_batch_contract_rejects_duplicate_input_frame_identities() -> None:
    duplicate = _input(3)

    with pytest.raises(ValueError, match="duplicate frame identities"):
        calibrate_frames(EchoCalibrator(), [duplicate, duplicate])


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("non_sequence", "result sequence"),
        ("missing", "result count"),
        ("reordered", "identity or order"),
        ("duplicate", "identity or order"),
        ("mismatched", "identity or order"),
        ("evidence_count", "evidence count"),
        ("malformed", "malformed"),
    ],
)
def test_batch_contract_rejects_malformed_provider_results(
    case: str,
    message: str,
) -> None:
    inputs = [_input(3), _input(4)]
    valid = EchoCalibrator().calibrate(inputs)
    if case == "non_sequence":
        invalid: object = object()
    elif case == "missing":
        invalid: object = valid[:-1]
    elif case == "reordered":
        invalid = [valid[1], valid[0]]
    elif case == "duplicate":
        invalid = [valid[0], valid[0]]
    elif case == "mismatched":
        invalid = [replace(valid[0], pts=999), valid[1]]
    elif case == "evidence_count":
        invalid = [replace(valid[0], correspondence_count=1), valid[1]]
    else:
        invalid = [object(), valid[1]]

    class MalformedCalibrator:
        def calibrate(self, inputs: Sequence[FrameCalibrationInput]) -> object:
            del inputs
            return invalid

    with pytest.raises(ValueError, match=message):
        calibrate_frames(MalformedCalibrator(), inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": ""}, "method and provenance"),
        ({"quality_threshold": 1.1}, "quality_threshold"),
        ({"quality_score": float("nan")}, "quality_score"),
        ({"quality_score": None}, "quality threshold"),
        ({"quality_score": 0.74}, "quality threshold"),
        ({"correspondence_count": -1}, "evidence counts"),
        ({"correspondence_count": 1.5}, "evidence counts"),
        ({"inlier_count": True}, "evidence counts"),
        ({"inlier_count": 5}, "evidence counts"),
        ({"reprojection_rmse_px": -1.0}, "reprojection_rmse_px"),
        (
            {"abstention_reason": AbstentionReason.NO_FIELD_CUES},
            "accepted calibration cannot",
        ),
    ],
)
def test_calibration_contract_rejects_inconsistent_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    accepted = _accepted_calibration()
    with pytest.raises(ValueError, match=message):
        replace(accepted, **changes)  # type: ignore[arg-type]


def test_abstained_calibration_rejects_homography_or_missing_reason() -> None:
    abstained = EchoCalibrator().calibrate([_input(0)])[0]
    with pytest.raises(ValueError, match="cannot expose"):
        replace(
            abstained,
            homography_img_to_pitch=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="requires an abstention reason"):
        replace(abstained, abstention_reason=None)
    with pytest.raises(ValueError, match="bounded abstention reason"):
        replace(abstained, abstention_reason="provider_failure")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="status"):
        replace(abstained, status="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (((1.0, 0.0), (0.0, 1.0)), "3x3"),
        (
            ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)),
            "3x3",
        ),
        (((0.0, 0.0, 0.0),) * 3, "non-degenerate"),
        (
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "non-degenerate",
        ),
    ],
)
def test_accepted_calibration_rejects_malformed_homographies(
    matrix: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            _accepted_calibration(),
            homography_img_to_pitch=matrix,  # type: ignore[arg-type]
        )


def test_accepted_calibration_rejects_unreadable_homography() -> None:
    with pytest.raises(ValueError, match="3x3"):
        replace(
            _accepted_calibration(),
            homography_img_to_pitch=((object(),),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frame_index": -1}, "frame_index"),
        ({"pts": True}, "pts"),
        ({"time_base": Fraction(0, 1)}, "time_base"),
        ({"image_width": 0}, "dimensions"),
        ({"image_width": 1.5}, "dimensions"),
        ({"image_height": True}, "dimensions"),
    ],
)
def test_frame_calibration_input_rejects_malformed_identity(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_input(0), **changes)  # type: ignore[arg-type]


def test_frame_calibration_input_rejects_cues_outside_source_image() -> None:
    outside = ImagePitchCorrespondence(ImagePoint(201.0, 50.0), PitchPoint(0.0, 0.0))
    with pytest.raises(ValueError, match="outside"):
        FrameCalibrationInput.from_frame(
            _frame(),
            pitch=PitchDimensions(105.0, 68.0),
            correspondences=(outside,),
        )


def _accepted_calibration() -> CameraCalibration:
    return CameraCalibration(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        pitch=PitchDimensions(105.0, 68.0),
        status=CalibrationStatus.ACCEPTED,
        homography_img_to_pitch=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        abstention_reason=None,
        method="synthetic",
        provenance="test-control",
        quality_score=1.0,
        quality_threshold=0.75,
        correspondence_count=4,
        inlier_count=4,
        reprojection_rmse_px=0.0,
    )


def _correspondences() -> tuple[ImagePitchCorrespondence, ...]:
    return tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(x_m, y_m))
        for u, v, x_m, y_m in (
            (0.0, 0.0, -52.5, -34.0),
            (200.0, 0.0, 52.5, -34.0),
            (200.0, 100.0, 52.5, 34.0),
            (0.0, 100.0, -52.5, 34.0),
            (100.0, 50.0, 0.0, 0.0),
            (50.0, 25.0, -26.25, -17.0),
        )
    )


def test_correspondence_control_accepts_known_geometry_and_projects_points() -> None:
    accepted_input = FrameCalibrationInput.from_frame(
        _frame(3),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences(),
    )
    no_cue_input = _input(4)

    accepted, abstained = calibrate_frames(
        CorrespondenceHomographyCalibrator(),
        [accepted_input, no_cue_input],
    )
    center = project_image_point(accepted, ImagePoint(100.0, 50.0))
    no_metric_point = project_image_point(abstained, ImagePoint(100.0, 50.0))

    assert accepted.status is CalibrationStatus.ACCEPTED
    assert accepted.quality_score == pytest.approx(1.0)
    assert accepted.quality_threshold == pytest.approx(0.75)
    assert accepted.correspondence_count == 6
    assert accepted.inlier_count == 6
    assert accepted.reprojection_rmse_px == pytest.approx(0.0, abs=1e-5)
    assert "opencv" in accepted.provenance
    assert center.frame_index == 3
    assert center.pts == 1_536
    assert center.time_base == Fraction(1, 15_360)
    assert center.pitch_point is not None
    assert center.pitch_point.x_m == pytest.approx(0.0, abs=1e-10)
    assert center.pitch_point.y_m == pytest.approx(0.0, abs=1e-10)
    assert center.inside_pitch is True
    assert center.failure_reason is None

    assert abstained.status is CalibrationStatus.ABSTAINED
    assert abstained.abstention_reason is AbstentionReason.NO_FIELD_CUES
    assert no_metric_point.pitch_point is None
    assert no_metric_point.inside_pitch is None
    assert (
        no_metric_point.failure_reason is ProjectionFailureReason.CALIBRATION_ABSTAINED
    )


def test_projection_preserves_finite_points_outside_pitch_without_clipping() -> None:
    calibration = CameraCalibration(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        pitch=PitchDimensions(105.0, 68.0),
        status=CalibrationStatus.ACCEPTED,
        homography_img_to_pitch=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        abstention_reason=None,
        method="synthetic",
        provenance="test-control",
        quality_score=1.0,
        quality_threshold=0.75,
        correspondence_count=4,
        inlier_count=4,
        reprojection_rmse_px=0.0,
    )

    result = project_image_point(calibration, ImagePoint(60.0, 10.0))

    assert result.pitch_point == PitchPoint(60.0, 10.0)
    assert result.inside_pitch is False


@pytest.mark.parametrize(
    ("correspondences", "reason"),
    [
        (_correspondences()[:3], AbstentionReason.INSUFFICIENT_CORRESPONDENCES),
        (
            tuple(
                ImagePitchCorrespondence(
                    ImagePoint(float(index * 10), float(index * 10)),
                    PitchPoint(float(index), float(index)),
                )
                for index in range(4)
            ),
            AbstentionReason.DEGENERATE_GEOMETRY,
        ),
    ],
)
def test_correspondence_control_abstains_for_inadequate_geometry(
    correspondences: tuple[ImagePitchCorrespondence, ...],
    reason: AbstentionReason,
) -> None:
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=correspondences,
    )

    result = calibrate_frames(
        CorrespondenceHomographyCalibrator(), [calibration_input]
    )[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is reason
    assert result.homography_img_to_pitch is None


@pytest.mark.parametrize(
    ("matrix", "mask", "reason"),
    [
        (
            np.eye(3, dtype=np.float64),
            np.asarray([[1], [1], [1], [0]], dtype=np.uint8),
            AbstentionReason.INSUFFICIENT_INLIERS,
        ),
        (
            np.eye(3, dtype=np.float64),
            np.asarray([[1], [1], [1], [1], [0], [0]], dtype=np.uint8),
            AbstentionReason.INSUFFICIENT_INLIERS,
        ),
        (
            np.asarray([[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]]),
            np.ones((4, 1), dtype=np.uint8),
            AbstentionReason.NON_FINITE_CALIBRATION,
        ),
    ],
)
def test_correspondence_control_applies_numeric_abstention_gate(
    monkeypatch: pytest.MonkeyPatch,
    matrix: np.ndarray,
    mask: np.ndarray,
    reason: AbstentionReason,
) -> None:
    def fake_find_homography(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        return matrix, mask

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences()[: len(mask)],
    )

    result = calibrate_frames(
        CorrespondenceHomographyCalibrator(), [calibration_input]
    )[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is reason


@pytest.mark.parametrize("zero_set", ["pitch", "image"])
def test_correspondence_control_rejects_underdetermined_inlier_layout(
    monkeypatch: pytest.MonkeyPatch,
    zero_set: str,
) -> None:
    correspondences = tuple(
        ImagePitchCorrespondence(
            ImagePoint(
                0.0 if zero_set == "image" else float(index * 10),
                0.0 if zero_set == "image" else float(index * 10),
            ),
            PitchPoint(
                0.0 if zero_set == "pitch" else float(index),
                0.0 if zero_set == "pitch" else float(index),
            ),
        )
        for index in range(4)
    )

    def fake_find_homography(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return np.eye(3), np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=correspondences,
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.DEGENERATE_GEOMETRY


def test_correspondence_control_preserves_exact_threshold_provenance() -> None:
    threshold = 1.23456789012345
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences(),
    )

    result = CorrespondenceHomographyCalibrator(
        ransac_reprojection_threshold_px=threshold
    ).calibrate([calibration_input])[0]

    assert f"ransac_reprojection_threshold_px:{threshold!r}" in result.provenance


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan")])
def test_correspondence_control_rejects_invalid_ransac_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="ransac_reprojection_threshold_px"):
        CorrespondenceHomographyCalibrator(ransac_reprojection_threshold_px=threshold)


@pytest.mark.parametrize("provider_case", ["none", "wrong_shape"])
def test_correspondence_control_rejects_malformed_provider_geometry(
    monkeypatch: pytest.MonkeyPatch,
    provider_case: str,
) -> None:
    def fake_find_homography(*args: object, **kwargs: object) -> object:
        del args, kwargs
        if provider_case == "none":
            return None, None
        return np.eye(2), np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences()[:4],
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    expected = (
        AbstentionReason.DEGENERATE_GEOMETRY
        if provider_case == "none"
        else AbstentionReason.PROVIDER_FAILURE
    )
    assert result.abstention_reason is expected
    if provider_case == "none":
        assert result.quality_score == 0.0


@pytest.mark.parametrize("inverse_case", ["failure", "non_finite"])
def test_correspondence_control_rejects_unusable_inverse(
    monkeypatch: pytest.MonkeyPatch,
    inverse_case: str,
) -> None:
    original_inverse = np.linalg.inv

    def fake_inverse(matrix: np.ndarray) -> np.ndarray:
        if inverse_case == "failure":
            raise np.linalg.LinAlgError("synthetic")
        result = original_inverse(matrix)
        result[0, 0] = float("inf")
        return result

    monkeypatch.setattr(np.linalg, "inv", fake_inverse)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences(),
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    expected = (
        AbstentionReason.DEGENERATE_GEOMETRY
        if inverse_case == "failure"
        else AbstentionReason.NON_FINITE_CALIBRATION
    )
    assert result.abstention_reason is expected


def test_correspondence_control_abstains_when_inverse_fails_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag([1.0, 1e7, 1e7])
    mask = np.ones((6, 1), dtype=np.uint8)

    def fake_find_homography(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return matrix, mask

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences(),
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.DEGENERATE_GEOMETRY


@pytest.mark.parametrize("rmse_case", ["no_inliers", "zero_denominator", "overflow"])
def test_correspondence_control_handles_unavailable_pixel_rmse(
    monkeypatch: pytest.MonkeyPatch,
    rmse_case: str,
) -> None:
    if rmse_case == "no_inliers":
        matrix = np.eye(3)
        mask = np.zeros((4, 1), dtype=np.uint8)
        correspondences = _correspondences()[:4]
    elif rmse_case == "zero_denominator":
        matrix = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        mask = np.asarray([[0], [0], [0], [0], [1], [0]], dtype=np.uint8)
        correspondences = _correspondences()
    else:
        matrix = np.eye(3)
        mask = np.ones((4, 1), dtype=np.uint8)
        correspondences = tuple(
            ImagePitchCorrespondence(
                ImagePoint(float(index), float(index)),
                PitchPoint(1e308, 1e308 - float(index)),
            )
            for index in range(4)
        )

    def fake_find_homography(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return matrix, mask

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=correspondences,
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    assert result.reprojection_rmse_px is None


def test_correspondence_control_computes_finite_rmse_near_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.asarray([[1e-13, 0.0, 0.0], [0.0, 1e-13, 0.0], [0.0, 0.0, 1e-13]])
    mask = np.ones((6, 1), dtype=np.uint8)

    def fake_find_homography(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return matrix, mask

    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences(),
    )

    result = CorrespondenceHomographyCalibrator().calibrate([calibration_input])[0]

    assert result.reprojection_rmse_px is not None
    assert np.isfinite(result.reprojection_rmse_px)


def test_correspondence_control_encapsulates_opencv_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise cv2.error("synthetic provider failure")

    monkeypatch.setattr(cv2, "findHomography", fail)
    calibration_input = FrameCalibrationInput.from_frame(
        _frame(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=_correspondences()[:4],
    )

    result = calibrate_frames(
        CorrespondenceHomographyCalibrator(), [calibration_input]
    )[0]

    assert result.abstention_reason is AbstentionReason.PROVIDER_FAILURE


def test_projection_abstains_when_homogeneous_result_is_not_finite() -> None:
    calibration = CameraCalibration(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        pitch=PitchDimensions(105.0, 68.0),
        status=CalibrationStatus.ACCEPTED,
        homography_img_to_pitch=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, -10.0),
        ),
        abstention_reason=None,
        method="synthetic",
        provenance="test-control",
        quality_score=1.0,
        quality_threshold=0.75,
        correspondence_count=4,
        inlier_count=4,
        reprojection_rmse_px=0.0,
    )

    result = project_image_point(calibration, ImagePoint(10.0, 5.0))

    assert result.pitch_point is None
    assert result.inside_pitch is None
    assert result.failure_reason is ProjectionFailureReason.NON_FINITE_PROJECTION


def test_projection_preserves_finite_result_near_homography_horizon() -> None:
    calibration = replace(
        _accepted_calibration(),
        homography_img_to_pitch=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, -10.0 + 1e-13),
        ),
    )

    result = project_image_point(calibration, ImagePoint(10.0, 5.0))

    assert result.pitch_point is not None
    assert result.inside_pitch is False
    assert result.failure_reason is None


def test_projection_abstains_when_finite_homogeneous_values_overflow() -> None:
    calibration = replace(
        _accepted_calibration(),
        homography_img_to_pitch=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1e-11),
        ),
    )

    result = project_image_point(calibration, ImagePoint(1e308, 1.0))

    assert result.failure_reason is ProjectionFailureReason.NON_FINITE_PROJECTION


def test_projection_result_contract_rejects_inconsistent_success_or_failure() -> None:
    success = project_image_point(_accepted_calibration(), ImagePoint(10.0, 5.0))
    failure = PointProjection(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=200,
        image_height=100,
        image_point=ImagePoint(10.0, 5.0),
        pitch_point=None,
        inside_pitch=None,
        failure_reason=ProjectionFailureReason.NON_FINITE_PROJECTION,
    )

    with pytest.raises(ValueError, match="failed projection"):
        replace(failure, failure_reason=None)
    with pytest.raises(ValueError, match="bounded reason"):
        replace(failure, failure_reason="non_finite_projection")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="successful projection"):
        replace(success, inside_pitch=None)
    with pytest.raises(ValueError, match="successful projection"):
        replace(success, inside_pitch=1)  # type: ignore[arg-type]
