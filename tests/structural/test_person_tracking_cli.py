"""Behavior tests for the person-tracking diagnostic command."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from hashlib import file_digest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import av
import football_alpha_person_tracking.diagnostic as diagnostic_module
import numpy as np
import pytest
from football_alpha_person_tracking.cli import main
from football_alpha_person_tracking.diagnostic import (
    create_tracking_diagnostic,
    validate_tracking_diagnostic_paths,
)
from football_alpha_person_tracking.fixture_calibration import (
    CLIP_ANCHOR_CORRESPONDENCES,
    CLIP_FIT_MARKINGS,
    CLIP_HELD_OUT_MARKINGS,
    CLIP_PITCH,
    CLIP_SHA256,
    CLIP_VALIDATION_MAXIMUM_ERROR_PX,
    FixtureCalibrationValidator,
    HeldOutPitchMarking,
    create_clip_calibrator,
)

from football_alpha_person_tracking import FramePersonTracks, PersonTrack, PersonTracker
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
from football_alpha_player_detection import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
    PersonDetector,
)
from football_alpha_video import DecodedFrame, decode_video


class DiagnosticDetector:
    """Controlled person detector that records streaming batches."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        self.batches.append([frame.index for frame in frames])
        return [
            FramePersonDetections(
                frame_index=frame.index,
                pts=frame.pts,
                time_base=frame.time_base,
                image_width=frame.image.shape[1],
                image_height=frame.image.shape[0],
                detections=(
                    PersonDetection(
                        bbox=ImageBoundingBox(1.25, 2.5, 10.75, 12.5),
                        confidence=0.83,
                    ),
                ),
            )
            for frame in frames
        ]


def test_fixture_calibration_profile_is_bounded_and_explicitly_assumed() -> None:
    anchor_indices = tuple(CLIP_ANCHOR_CORRESPONDENCES)

    assert CLIP_SHA256 == (
        "0ad6f9a7c5facf49e58bcb9a2afaf579136e8d45153f8860e3ee1b85a947b001"
    )
    assert PitchDimensions(105.0, 68.0) == CLIP_PITCH
    assert len(anchor_indices) == 5
    assert anchor_indices == tuple(sorted(anchor_indices))
    assert anchor_indices[0] < 50
    assert any(100 <= frame_index <= 200 for frame_index in anchor_indices)
    assert anchor_indices[-1] >= 202
    assert all(
        len(correspondences) >= 6
        for correspondences in CLIP_ANCHOR_CORRESPONDENCES.values()
    )
    assert set(CLIP_FIT_MARKINGS) == set(anchor_indices[1:])
    assert set(CLIP_HELD_OUT_MARKINGS) == set(anchor_indices)
    assert CLIP_VALIDATION_MAXIMUM_ERROR_PX == 25.0
    for frame_index in anchor_indices:
        fit_points = {
            point
            for marking in CLIP_FIT_MARKINGS.get(frame_index, ())
            for point in marking.image_points
        }
        held_out_points = {
            point
            for marking in CLIP_HELD_OUT_MARKINGS[frame_index]
            for point in marking.image_points
        }
        assert fit_points.isdisjoint(held_out_points)


def test_fixture_calibration_profile_accepts_exactly_five_real_frames(
    development_clip: Path,
) -> None:
    inputs: list[FrameCalibrationInput] = []
    with av.open(development_clip) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            correspondences = CLIP_ANCHOR_CORRESPONDENCES.get(frame_index)
            if correspondences is not None:
                inputs.append(
                    FrameCalibrationInput(
                        frame_index=frame_index,
                        pts=cast(int, frame.pts),
                        time_base=cast(Fraction, frame.time_base),
                        image_width=frame.width,
                        image_height=frame.height,
                        pitch=CLIP_PITCH,
                        correspondences=correspondences,
                    )
                )

    results = calibrate_frames(create_clip_calibrator(), inputs)

    assert [result.frame_index for result in results] == [0, 180, 195, 210, 225]
    assert all(result.status is CalibrationStatus.ACCEPTED for result in results)
    assert all(result.method.endswith("+fixture_held_out_lines") for result in results)
    assert all(
        "fixture_held_out_maximum_error_px:" in result.provenance for result in results
    )
    for calibration_input, result in zip(inputs, results, strict=True):
        assert result.homography_img_to_pitch is not None
        pitch_to_image = np.linalg.inv(
            np.asarray(result.homography_img_to_pitch, dtype=np.float64)
        )
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
        projected = pitch_points @ pitch_to_image.T
        projected /= projected[:, 2, None]
        image_points = np.asarray(
            [
                (correspondence.image_point.u, correspondence.image_point.v)
                for correspondence in calibration_input.correspondences
            ]
        )
        errors_px = np.linalg.norm(projected[:, :2] - image_points, axis=1)
        inliers = errors_px <= 30.0

        assert result.inlier_count == int(np.count_nonzero(inliers))
        assert result.quality_score == pytest.approx(float(np.mean(inliers)))
        assert result.reprojection_rmse_px == pytest.approx(
            float(np.sqrt(np.mean(errors_px[inliers] ** 2)))
        )


def test_fixture_calibration_rejects_a_self_fit_that_misses_held_out_lines() -> None:
    calibration_input = _fixture_gate_input()
    calibrator = FixtureCalibrationValidator(
        CorrespondenceHomographyCalibrator(),
        held_out_markings_by_frame={
            7: (
                HeldOutPitchMarking(
                    axis="y",
                    coordinate_m=0.0,
                    image_points=(ImagePoint(35.0, 60.0), ImagePoint(65.0, 60.0)),
                ),
            )
        },
        maximum_error_px=1.0,
    )

    result = calibrate_frames(calibrator, (calibration_input,))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED
    assert result.homography_img_to_pitch is None


def _fixture_gate_input(
    *,
    correspondences: tuple[ImagePitchCorrespondence, ...] | None = None,
    image_y_offset: float = 50.0,
) -> FrameCalibrationInput:
    if correspondences is None:
        correspondences = tuple(
            ImagePitchCorrespondence(
                image_point=ImagePoint(u=x_m + 50.0, v=y_m + image_y_offset),
                pitch_point=PitchPoint(x_m=x_m, y_m=y_m),
            )
            for x_m, y_m in (
                (-20.0, -20.0),
                (20.0, -20.0),
                (20.0, 20.0),
                (-20.0, 20.0),
                (0.0, 0.0),
                (10.0, 5.0),
            )
        )
    return FrameCalibrationInput(
        frame_index=7,
        pts=3_584,
        time_base=Fraction(1, 15_360),
        image_width=100,
        image_height=100,
        pitch=PitchDimensions(105.0, 68.0),
        correspondences=correspondences,
    )


def test_fixture_point_line_refinement_repairs_a_locally_biased_fit() -> None:
    fit_markings = (
        HeldOutPitchMarking(
            "x",
            -20.0,
            (ImagePoint(30.0, 30.0), ImagePoint(30.0, 70.0)),
        ),
        HeldOutPitchMarking(
            "x",
            20.0,
            (ImagePoint(70.0, 30.0), ImagePoint(70.0, 70.0)),
        ),
        HeldOutPitchMarking(
            "y",
            -20.0,
            (ImagePoint(30.0, 30.0), ImagePoint(70.0, 30.0)),
        ),
        HeldOutPitchMarking(
            "y",
            20.0,
            (ImagePoint(30.0, 70.0), ImagePoint(70.0, 70.0)),
        ),
    )
    calibrator = FixtureCalibrationValidator(
        CorrespondenceHomographyCalibrator(),
        fit_markings_by_frame={7: fit_markings},
        held_out_markings_by_frame={
            7: (
                HeldOutPitchMarking(
                    "y",
                    0.0,
                    (ImagePoint(35.0, 50.0), ImagePoint(65.0, 50.0)),
                ),
            )
        },
        maximum_error_px=12.0,
        correspondence_reprojection_threshold_px=8.0,
    )

    result = calibrate_frames(
        calibrator,
        (_fixture_gate_input(image_y_offset=60.0),),
    )[0]

    assert result.status is CalibrationStatus.ACCEPTED
    assert result.method.endswith("+fixture_point_line+fixture_held_out_lines")


@pytest.mark.parametrize(
    ("image_point", "homography"),
    [
        (ImagePoint(101.0, 50.0), None),
        (
            ImagePoint(35.0, 50.0),
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, -35.0),
            ),
        ),
    ],
)
def test_fixture_refinement_fails_closed_for_unusable_fit_samples(
    image_point: ImagePoint,
    homography: Homography | None,
) -> None:
    calibrator: CameraCalibrator = (
        CorrespondenceHomographyCalibrator()
        if homography is None
        else FixedHomographyCalibrator(homography)
    )
    validated = FixtureCalibrationValidator(
        calibrator,
        fit_markings_by_frame={7: (HeldOutPitchMarking("y", 0.0, (image_point,)),)},
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


def test_fixture_refinement_fails_closed_when_the_solver_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_svd(_matrix: object) -> object:
        raise np.linalg.LinAlgError("solver failed")

    monkeypatch.setattr(
        np.linalg,
        "svd",
        fail_svd,
    )
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


@pytest.mark.parametrize(
    ("solver_outcome", "parameters"),
    [
        ("raises", np.zeros(8)),
        ("unsuccessful", np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))),
        ("singular", np.zeros(8)),
        ("non-finite", np.asarray((float("nan"), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))),
        ("zero-scale", np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0))),
    ],
)
def test_fixture_refinement_fails_closed_for_nonlinear_solver_failures(
    monkeypatch: pytest.MonkeyPatch,
    solver_outcome: str,
    parameters: np.ndarray,
) -> None:
    def fake_solver(*_args: Any, **_kwargs: Any) -> Any:
        if solver_outcome == "raises":
            raise ValueError("optimizer failed")
        return SimpleNamespace(
            x=parameters,
            success=solver_outcome != "unsuccessful",
        )

    monkeypatch.setattr(
        "football_alpha_person_tracking.fixture_calibration._least_squares",
        fake_solver,
    )
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


@pytest.mark.parametrize(
    ("parameters", "marking"),
    [
        (
            np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0)),
            HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),
        ),
        (
            np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),
        ),
        (
            np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0)),
            HeldOutPitchMarking("y", 1.0, (ImagePoint(35.0, 50.0),)),
        ),
    ],
    ids=("zero-projection-scale", "singular-transform", "degenerate-image-line"),
)
def test_fixture_refinement_objective_penalizes_invalid_transforms(
    monkeypatch: pytest.MonkeyPatch,
    parameters: np.ndarray,
    marking: HeldOutPitchMarking,
) -> None:
    def inspect_objective(*args: Any, **kwargs: Any) -> Any:
        objective = args[0]
        calibration_input, markings = kwargs["args"]
        residuals = objective(parameters, calibration_input, markings)
        assert np.all(residuals == 1e6)
        raise ValueError("stop after inspecting the objective")

    monkeypatch.setattr(
        "football_alpha_person_tracking.fixture_calibration._least_squares",
        inspect_objective,
    )
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={7: (marking,)},
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED


def test_fixture_refinement_fails_closed_for_a_zero_optimizer_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_input = _fixture_gate_input()
    image_scale = np.asarray(((0.02, 0.0, -1.0), (0.0, 0.02, -1.0), (0.0, 0.0, 1.0)))
    pitch_scale = np.asarray(
        ((1.0 / 52.5, 0.0, 0.0), (0.0, 1.0 / 34.0, 0.0), (0.0, 0.0, 1.0))
    )
    target = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0)))
    normalized = pitch_scale @ target @ np.linalg.inv(image_scale)

    def target_svd(_matrix: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right_singular_vectors = np.eye(9)
        right_singular_vectors[-1] = normalized.reshape(-1)
        return np.empty((0, 0)), np.empty(0), right_singular_vectors

    monkeypatch.setattr(np.linalg, "svd", target_svd)
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (calibration_input,))[0]

    assert result.status is CalibrationStatus.ABSTAINED


@pytest.mark.parametrize(
    "correspondences",
    [
        (),
        tuple(
            ImagePitchCorrespondence(ImagePoint(50.0, 50.0), PitchPoint(0.0, 0.0))
            for _ in range(6)
        ),
    ],
    ids=("too-few-equations", "rank-deficient-equations"),
)
def test_fixture_refinement_fails_closed_for_degenerate_equations(
    correspondences: tuple[ImagePitchCorrespondence, ...],
) -> None:
    validated = FixtureCalibrationValidator(
        AlwaysAcceptedHomographyCalibrator(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(
        validated,
        (_fixture_gate_input(correspondences=correspondences),),
    )[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


@pytest.mark.parametrize(
    "solution",
    [
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
    ],
    ids=("rank-deficient-solution", "non-finite-solution", "zero-scale-solution"),
)
def test_fixture_refinement_fails_closed_for_an_unusable_solver_solution(
    monkeypatch: pytest.MonkeyPatch,
    solution: tuple[float, ...],
) -> None:
    def unusable_svd(_matrix: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right_singular_vectors = np.eye(9)
        right_singular_vectors[-1] = np.asarray(solution)
        return np.empty((0, 0)), np.empty(0), right_singular_vectors

    monkeypatch.setattr(np.linalg, "svd", unusable_svd)
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


def test_fixture_calibration_preserves_a_fit_that_matches_reserved_markings() -> None:
    calibrator = FixtureCalibrationValidator(
        CorrespondenceHomographyCalibrator(),
        held_out_markings_by_frame={
            7: (
                HeldOutPitchMarking(
                    axis="x",
                    coordinate_m=0.0,
                    image_points=(ImagePoint(50.0, 35.0), ImagePoint(50.0, 65.0)),
                ),
                HeldOutPitchMarking(
                    axis="y",
                    coordinate_m=0.0,
                    image_points=(ImagePoint(35.0, 50.0), ImagePoint(65.0, 50.0)),
                ),
            )
        },
        maximum_error_px=0.1,
    )

    result = calibrate_frames(calibrator, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ACCEPTED
    assert result.abstention_reason is None
    assert result.method.endswith("+fixture_held_out_lines")
    assert "fixture_held_out_line_sample_count:4" in result.provenance
    assert "fixture_held_out_maximum_error_px:" in result.provenance


def test_fixture_calibration_preserves_the_underlying_no_cue_abstention() -> None:
    calibrator = FixtureCalibrationValidator(
        CorrespondenceHomographyCalibrator(),
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(
        calibrator,
        (_fixture_gate_input(correspondences=()),),
    )[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.NO_FIELD_CUES


@pytest.mark.parametrize(
    "marking",
    [
        pytest.param(
            lambda: HeldOutPitchMarking(cast(Any, "z"), 0.0, (ImagePoint(1.0, 1.0),)),
            id="unknown-axis",
        ),
        pytest.param(
            lambda: HeldOutPitchMarking("x", float("nan"), (ImagePoint(1.0, 1.0),)),
            id="non-finite-coordinate",
        ),
        pytest.param(
            lambda: HeldOutPitchMarking("x", 0.0, ()),
            id="missing-samples",
        ),
    ],
)
def test_held_out_pitch_marking_rejects_malformed_evidence(
    marking: Callable[[], HeldOutPitchMarking],
) -> None:
    with pytest.raises(ValueError, match="held-out pitch marking"):
        marking()


@pytest.mark.parametrize("maximum_error_px", [0.0, float("nan")])
def test_fixture_calibration_rejects_an_invalid_validation_threshold(
    maximum_error_px: float,
) -> None:
    with pytest.raises(ValueError, match="maximum_error_px"):
        FixtureCalibrationValidator(
            CorrespondenceHomographyCalibrator(),
            held_out_markings_by_frame={},
            maximum_error_px=maximum_error_px,
        )


@pytest.mark.parametrize(
    "threshold_px",
    [0.0, float("nan")],
)
def test_fixture_calibration_rejects_an_invalid_correspondence_threshold(
    threshold_px: float,
) -> None:
    with pytest.raises(ValueError, match="correspondence_reprojection_threshold_px"):
        FixtureCalibrationValidator(
            CorrespondenceHomographyCalibrator(),
            held_out_markings_by_frame={},
            maximum_error_px=1.0,
            correspondence_reprojection_threshold_px=threshold_px,
        )


@pytest.mark.parametrize(
    ("homography", "expected_reason"),
    [
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            AbstentionReason.NON_FINITE_CALIBRATION,
        ),
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            AbstentionReason.INSUFFICIENT_INLIERS,
        ),
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.05, 0.0, 1.0)),
            AbstentionReason.NON_FINITE_CALIBRATION,
        ),
        (
            ((1e-200, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            AbstentionReason.NON_FINITE_CALIBRATION,
        ),
    ],
    ids=("singular", "insufficient-inliers", "zero-projection-scale", "overflow"),
)
def test_fixture_final_correspondence_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    homography: Homography,
    expected_reason: AbstentionReason,
) -> None:
    def fixed_homography(
        _calibration_input: FrameCalibrationInput,
        _markings: Sequence[HeldOutPitchMarking],
    ) -> Homography:
        return homography

    monkeypatch.setattr(
        "football_alpha_person_tracking.fixture_calibration._fit_point_line_homography",
        fixed_homography,
    )
    validated = FixtureCalibrationValidator(
        FixedHomographyCalibrator(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        fit_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (ImagePoint(35.0, 50.0),)),)
        },
        held_out_markings_by_frame={},
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is expected_reason
    assert result.homography_img_to_pitch is None


@pytest.mark.parametrize(
    ("image_point", "homography"),
    [
        (ImagePoint(101.0, 50.0), None),
        (
            ImagePoint(35.0, 50.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, -35.0)),
        ),
        (
            ImagePoint(35.0, 50.0),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ),
    ],
    ids=("outside-source", "excessive-error", "degenerate-image-line"),
)
def test_fixture_calibration_fails_closed_for_unusable_reserved_evidence(
    image_point: ImagePoint,
    homography: Homography | None,
) -> None:
    calibrator: CameraCalibrator = (
        CorrespondenceHomographyCalibrator()
        if homography is None
        else FixedHomographyCalibrator(homography)
    )
    validated = FixtureCalibrationValidator(
        calibrator,
        held_out_markings_by_frame={
            7: (HeldOutPitchMarking("y", 0.0, (image_point,)),)
        },
        maximum_error_px=1.0,
    )

    result = calibrate_frames(validated, (_fixture_gate_input(),))[0]

    assert result.status is CalibrationStatus.ABSTAINED
    assert result.abstention_reason is AbstentionReason.INDEPENDENT_VALIDATION_FAILED


class DiagnosticTracker:
    """Controlled session that emits no observation on its first frame."""

    def __init__(self) -> None:
        self.frame_indices: list[int] = []

    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        self.frame_indices.append(frame.frame_index)
        tracks = (
            ()
            if frame.frame_index == 0
            else (
                PersonTrack(
                    track_id=7,
                    bbox=frame.detections[0].bbox,
                    detection_confidence=frame.detections[0].confidence,
                ),
            )
        )
        return FramePersonTracks(
            frame_index=frame.frame_index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=frame.image_width,
            image_height=frame.image_height,
            tracks=tracks,
        )

    def reset(self) -> None:
        self.frame_indices.clear()


class DiagnosticRfDetrPersonDetector(DiagnosticDetector):
    def __init__(self, minimum_confidence: float, *, device: str) -> None:
        super().__init__()
        assert minimum_confidence == 0.10
        assert device == "auto"

    def provenance(self) -> dict[str, str]:
        return {
            "model_name": "RFDETRLarge",
            "checkpoint_name": "test-checkpoint.pth",
            "checkpoint_sha256": "abc123",
            "rfdetr_version": "test-version",
            "device": "cpu",
        }


class DiagnosticByteTrackPersonTracker(DiagnosticTracker):
    def __init__(self, *, source_frame_rate: float) -> None:
        super().__init__()
        assert source_frame_rate == 2.0

    def provenance(self) -> dict[str, object]:
        return {
            "backend": "trackers.ByteTrackTracker",
            "version": "test-version",
            "parameters": {"lost_track_buffer": 30},
        }


class WrongCardinalityDetector:
    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        return []


class WrongIdentityDetector(DiagnosticDetector):
    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        results = super().detect(frames)
        first = results[0]
        results[0] = FramePersonDetections(
            frame_index=first.frame_index + 1,
            pts=first.pts,
            time_base=first.time_base,
            image_width=first.image_width,
            image_height=first.image_height,
            detections=first.detections,
        )
        return results


class WrongIdentityTracker(DiagnosticTracker):
    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        result = super().update(frame)
        return FramePersonTracks(
            frame_index=result.frame_index + 1,
            pts=result.pts,
            time_base=result.time_base,
            image_width=result.image_width,
            image_height=result.image_height,
            tracks=result.tracks,
        )


class GappedTracker(DiagnosticTracker):
    def update(self, frame: FramePersonDetections) -> FramePersonTracks:
        self.frame_indices.append(frame.frame_index)
        tracks = (
            ()
            if frame.frame_index == 1
            else (
                PersonTrack(
                    track_id=7,
                    bbox=frame.detections[0].bbox,
                    detection_confidence=frame.detections[0].confidence,
                ),
            )
        )
        return FramePersonTracks(
            frame_index=frame.frame_index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=frame.image_width,
            image_height=frame.image_height,
            tracks=tracks,
        )


class FixedHomographyCalibrator:
    def __init__(self, homography: Homography) -> None:
        self.homography = homography

    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        baseline = CorrespondenceHomographyCalibrator().calibrate(inputs)
        return [
            replace(result, homography_img_to_pitch=self.homography)
            if result.status is CalibrationStatus.ACCEPTED
            else result
            for result in baseline
        ]


class AlwaysAcceptedHomographyCalibrator:
    """Test double that exposes validator behavior for malformed fit evidence."""

    def __init__(self, homography: Homography) -> None:
        self.homography = homography

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
                status=CalibrationStatus.ACCEPTED,
                homography_img_to_pitch=self.homography,
                abstention_reason=None,
                method="always_accepted_test_double",
                provenance="controlled:test",
                quality_score=1.0,
                quality_threshold=0.75,
                correspondence_count=len(item.correspondences),
                inlier_count=len(item.correspondences),
                reprojection_rmse_px=0.0,
            )
            for item in inputs
        ]


class WrongIdentityCalibrator:
    def calibrate(
        self,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        results = CorrespondenceHomographyCalibrator().calibrate(inputs)
        return [
            replace(results[0], frame_index=results[0].frame_index + 1),
            *results[1:],
        ]


def _calibration_correspondences() -> dict[int, tuple[ImagePitchCorrespondence, ...]]:
    return {
        1: tuple(
            ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(x_m, y_m))
            for u, v, x_m, y_m in (
                (0.0, 0.0, -52.5, 34.0),
                (16.0, 0.0, 52.5, 34.0),
                (16.0, 16.0, 52.5, -34.0),
                (0.0, 16.0, -52.5, -34.0),
                (8.0, 8.0, 0.0, 0.0),
                (4.0, 4.0, -26.25, 17.0),
            )
        )
    }


def test_tracking_preview_preserves_high_resolution_source_timestamps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        for pts in (0, 3000, 6300):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), "rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 90000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    preview = tmp_path / "preview.mp4"
    records_path = tmp_path / "tracks.jsonl"
    create_tracking_diagnostic(
        source,
        preview,
        records_path,
        tmp_path / "summary.json",
        detector=DiagnosticDetector(),
        detector_provenance={},
        tracker_factory=lambda rate: (DiagnosticTracker(), {}),
        calibrator=CorrespondenceHomographyCalibrator(),
        pitch=PitchDimensions(105, 68),
        correspondences_by_frame={},
        batch_size=2,
    )
    assert [f.timestamp for f in decode_video(preview)] == [
        f.timestamp for f in decode_video(source)
    ]
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert [(r["pts"], r["time_base"]) for r in rows] == [
        (0, "1/90000"),
        (3000, "1/90000"),
        (6300, "1/90000"),
    ]


def test_tracking_diagnostic_streams_complete_reconciled_artifacts(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "tracking.mp4"
    jsonl_path = tmp_path / "tracking.jsonl"
    summary_path = tmp_path / "summary.json"
    detector: PersonDetector = DiagnosticDetector()
    tracker: PersonTracker = DiagnosticTracker()
    received_frame_rates: list[float] = []

    def create_tracker(
        source_frame_rate: float,
    ) -> tuple[PersonTracker, dict[str, diagnostic_module.JsonValue]]:
        received_frame_rates.append(source_frame_rate)
        return tracker, {
            "backend": "controlled",
            "version": "test-version",
            "parameters": {"lost_track_buffer": 30},
        }

    summary = create_tracking_diagnostic(
        three_frame_video,
        overlay,
        jsonl_path,
        summary_path,
        detector=detector,
        detector_provenance={
            "model_name": "controlled detector",
            "checkpoint_name": "fixture",
            "checkpoint_sha256": "abc123",
            "rfdetr_version": "test-version",
            "device": "cpu",
        },
        tracker_factory=create_tracker,
        calibrator=CorrespondenceHomographyCalibrator(),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences_by_frame=_calibration_correspondences(),
        batch_size=2,
    )

    source_frames = list(decode_video(three_frame_video))
    overlay_frames = list(decode_video(overlay))
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

    assert detector.batches == [[0, 1], [2]]
    assert tracker.frame_indices == [0, 1, 2]
    assert received_frame_rates == [2.0]
    assert [record["frame_index"] for record in records] == [0, 1, 2]
    assert [record["pts"] for record in records] == [0, 8192, 16384]
    assert [record["time_base"] for record in records] == ["1/16384"] * 3
    assert records[0]["calibration"]["status"] == "abstained"
    assert records[0]["calibration"]["abstention_reason"] == "no_field_cues"
    assert records[0]["tracks"] == []
    assert records[1]["calibration"]["status"] == "accepted"
    projected_track = records[1]["tracks"][0]
    assert projected_track["track_id"] == 7
    assert projected_track["bbox"] == {
        "x_min": 1.25,
        "y_min": 2.5,
        "x_max": 10.75,
        "y_max": 12.5,
    }
    assert projected_track["detection_confidence"] == 0.83
    assert projected_track["support_point_uv"] == {"u": 6.0, "v": 12.5}
    assert projected_track["support_point_method"] == "bbox_bottom_center"
    assert projected_track["pitch_xy"]["x_m"] == pytest.approx(-13.125)
    assert projected_track["pitch_xy"]["y_m"] == pytest.approx(-19.125)
    assert projected_track["inside_pitch"] is True
    assert projected_track["projection_reason"] is None
    assert records[2]["calibration"]["status"] == "abstained"
    assert records[2]["tracks"][0]["pitch_xy"] is None
    assert records[2]["tracks"][0]["inside_pitch"] is None
    assert records[2]["tracks"][0]["projection_reason"] == ("calibration_abstained")
    assert [frame.timestamp for frame in overlay_frames] == [
        frame.timestamp for frame in source_frames
    ]
    assert json.loads(summary_path.read_text()) == summary
    assert summary["frame_count"] == 3
    assert summary["pitch"] == {
        "length_m": 105.0,
        "width_m": 68.0,
        "dimensions_source": "assumed_fixture_template",
    }
    assert summary["accepted_calibration_frame_count"] == 1
    assert summary["abstained_calibration_frame_count"] == 2
    assert summary["calibration_abstention_counts"] == {"no_field_cues": 2}
    assert summary["current_track_observation_count"] == 2
    assert summary["projected_observation_count"] == 1
    assert summary["outside_pitch_observation_count"] == 0
    assert summary["source_frame_rate"] == "2"
    detector_summary = cast(dict[str, object], summary["detector"])
    assert detector_summary["minimum_confidence"] == 0.10
    assert detector_summary["batch_size"] == 2
    assert summary["tracker"] == {
        "backend": "controlled",
        "version": "test-version",
        "parameters": {"lost_track_buffer": 30},
    }
    assert summary["distinct_track_id_count"] == 1
    assert summary["tracklets"] == [
        {
            "track_id": 7,
            "first_observation": {
                "frame_index": 1,
                "pts": 8192,
                "time_base": "1/16384",
            },
            "last_observation": {
                "frame_index": 2,
                "pts": 16384,
                "time_base": "1/16384",
            },
            "observed_frame_count": 2,
            "observed_time_span": "1/2",
            "largest_observed_gap": 0,
        }
    ]


def test_tracking_diagnostic_rejects_calibration_realignment_transactionally(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrator: CameraCalibrator = WrongIdentityCalibrator()

    def bypass_calibration_validation(
        current_calibrator: CameraCalibrator,
        inputs: Sequence[FrameCalibrationInput],
    ) -> list[CameraCalibration]:
        return current_calibrator.calibrate(inputs)

    monkeypatch.setattr(
        diagnostic_module,
        "calibrate_frames",
        bypass_calibration_validation,
    )

    with pytest.raises(ValueError, match=r"tracks and calibration.*different frames"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            calibrator=calibrator,
            pitch=PitchDimensions(105.0, 68.0),
            correspondences_by_frame=_calibration_correspondences(),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []
    assert not (tmp_path / "tracking.mp4").exists()
    assert not (tmp_path / "tracking.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize(
    ("homography", "expected_pitch", "inside_pitch", "reason"),
    [
        (
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, -6.0),
            ),
            None,
            None,
            "non_finite_projection",
        ),
        (
            (
                (1.0, 0.0, 100.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            {"x_m": 106.0, "y_m": 12.5},
            False,
            None,
        ),
    ],
)
def test_tracking_diagnostic_preserves_projection_failure_and_outside_points(
    three_frame_video: Path,
    tmp_path: Path,
    homography: Homography,
    expected_pitch: dict[str, float] | None,
    inside_pitch: bool | None,
    reason: str | None,
) -> None:
    jsonl_path = tmp_path / "tracking.jsonl"
    summary = create_tracking_diagnostic(
        three_frame_video,
        tmp_path / "tracking.mp4",
        jsonl_path,
        tmp_path / "summary.json",
        detector=DiagnosticDetector(),
        detector_provenance={},
        tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
        calibrator=FixedHomographyCalibrator(homography),
        pitch=PitchDimensions(105.0, 68.0),
        correspondences_by_frame=_calibration_correspondences(),
        batch_size=2,
    )
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    track = records[1]["tracks"][0]

    assert track["bbox"] == {
        "x_min": 1.25,
        "y_min": 2.5,
        "x_max": 10.75,
        "y_max": 12.5,
    }
    assert track["detection_confidence"] == 0.83
    assert track["support_point_uv"] == {"u": 6.0, "v": 12.5}
    assert track["pitch_xy"] == expected_pitch
    assert track["inside_pitch"] is inside_pitch
    assert track["projection_reason"] == reason
    assert summary["projected_observation_count"] == (expected_pitch is not None)
    assert summary["outside_pitch_observation_count"] == (inside_pitch is False)


def test_tracking_diagnostic_reports_the_largest_observed_gap(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    summary = create_tracking_diagnostic(
        three_frame_video,
        tmp_path / "tracking.mp4",
        tmp_path / "tracking.jsonl",
        tmp_path / "summary.json",
        detector=DiagnosticDetector(),
        detector_provenance={},
        tracker_factory=lambda _rate: (GappedTracker(), {}),
        batch_size=2,
    )

    assert summary["tracklets"] == [
        {
            "track_id": 7,
            "first_observation": {
                "frame_index": 0,
                "pts": 0,
                "time_base": "1/16384",
            },
            "last_observation": {
                "frame_index": 2,
                "pts": 16384,
                "time_base": "1/16384",
            },
            "observed_frame_count": 2,
            "observed_time_span": "1",
            "largest_observed_gap": 1,
        }
    ]


def test_tracking_diagnostic_preserves_variable_presentation_times(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variable-rate.mp4"
    with av.open(source, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for frame_index, pts in enumerate((0, 1, 3)):
            image = np.full((16, 16, 3), frame_index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = pts
            frame.time_base = Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())

    overlay = tmp_path / "tracking.mp4"
    create_tracking_diagnostic(
        source,
        overlay,
        tmp_path / "tracking.jsonl",
        tmp_path / "summary.json",
        detector=DiagnosticDetector(),
        detector_provenance={},
        tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
        batch_size=2,
    )

    assert [frame.timestamp for frame in decode_video(overlay)] == [
        frame.timestamp for frame in decode_video(source)
    ]


def test_tracking_diagnostic_command_uses_the_fixed_profile(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = tmp_path / "tracking.mp4"
    jsonl_path = tmp_path / "tracking.jsonl"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "football_alpha_person_tracking.cli.RfDetrPersonDetector",
        DiagnosticRfDetrPersonDetector,
    )
    monkeypatch.setattr(
        "football_alpha_person_tracking.cli.ByteTrackPersonTracker",
        DiagnosticByteTrackPersonTracker,
    )

    assert (
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--composite",
                str(overlay),
                "--jsonl",
                str(jsonl_path),
                "--summary",
                str(summary_path),
                "--batch-size",
                "2",
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text())
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["frame_count"] == 3
    assert summary["composite"] == str(overlay)
    assert "overlay" not in summary
    assert summary["detector"]["minimum_confidence"] == 0.10
    assert summary["tracker"]["backend"] == "trackers.ByteTrackTracker"


def test_temporal_diagnostic_preserves_selected_frame_identity_and_tracks(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Use generated camera texture and synthetic labels, never the real holdout.
    import football_alpha_person_tracking.cli as cli
    import football_alpha_person_tracking.fixture_calibration as fixture

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = np.random.default_rng(7).integers(
        50, 220, size=(48, 64), dtype=np.uint8
    )
    frames = [DecodedFrame(i, i * 512, Fraction(1, 15360), image) for i in range(226)]

    def decode_frames(_path: Path) -> Iterator[DecodedFrame]:
        return iter(frames)

    monkeypatch.setattr(diagnostic_module, "decode_video", decode_frames)
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, v / 10))
        for u, v in ((3, 3), (60, 3), (60, 44), (3, 44))
    )
    monkeypatch.setattr(
        fixture,
        "CLIP_HELD_OUT_MARKINGS",
        {
            i: (HeldOutPitchMarking("x", 1.0, (ImagePoint(10, 10),)),)
            for i in (195, 225)
        },
    )

    def anchor_control(indices: Sequence[int]) -> CameraCalibrator:
        assert tuple(indices) == (180, 210)
        return CorrespondenceHomographyCalibrator()

    monkeypatch.setattr(cli, "create_clip_calibrator", anchor_control)
    monkeypatch.setattr(cli, "RfDetrPersonDetector", DiagnosticRfDetrPersonDetector)
    monkeypatch.setattr(cli, "ByteTrackPersonTracker", DiagnosticByteTrackPersonTracker)
    monkeypatch.setattr(
        cli, "CLIP_ANCHOR_CORRESPONDENCES", dict.fromkeys((180, 195, 210, 225), cues)
    )
    with three_frame_video.open("rb") as source:
        monkeypatch.setattr(
            cli, "CLIP_SHA256", file_digest(source, "sha256").hexdigest()
        )
    assert (
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--temporal",
                "--batch-size",
                "4",
                "--composite",
                str(tmp_path / "composite.mp4"),
                "--jsonl",
                str(tmp_path / "tracks.jsonl"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    records = [
        json.loads(line)
        for line in (tmp_path / "tracks.jsonl").read_text().splitlines()
    ]
    assert [r["frame_index"] for r in records] == list(range(180, 226))
    assert summary["frame_count"] == 46
    assert summary["accepted_calibration_frame_count"] == 46
    assert summary["frame_interval"] == [180, 225]
    assert summary["temporal_evaluation"]["passed"] is True
    for record in records:
        assert record["pts"] == record["frame_index"] * 512
        assert record["time_base"] == "1/15360"
        assert record["tracks"][0]["track_id"] == 7
        assert record["tracks"][0]["detection_confidence"] == 0.83
        assert record["calibration"]["homography_img_to_pitch"] is not None
    with av.open(tmp_path / "composite.mp4") as video:
        assert [
            cast(int, f.pts) * cast(Fraction, f.time_base)
            for f in video.decode(video=0)
        ] == [Fraction(r["pts"], 15360) for r in records]
    monkeypatch.setattr(
        fixture,
        "CLIP_HELD_OUT_MARKINGS",
        {
            i: (HeldOutPitchMarking("x", 1.0, (ImagePoint(60, 10),)),)
            for i in (195, 225)
        },
    )
    assert (
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--temporal",
                "--batch-size",
                "7",
                "--composite",
                str(tmp_path / "changed.mp4"),
                "--jsonl",
                str(tmp_path / "changed.jsonl"),
                "--summary",
                str(tmp_path / "changed-summary.json"),
            ]
        )
        == 1
    )
    changed = json.loads(capsys.readouterr().out)
    assert changed["temporal_evaluation"]["passed"] is False
    assert (tmp_path / "changed.jsonl").read_bytes() == (
        tmp_path / "tracks.jsonl"
    ).read_bytes()

    write_summary = diagnostic_module._write_summary  # pyright: ignore[reportPrivateUsage]

    def falsify_evaluation(
        destination: Path, summary: Mapping[str, diagnostic_module.JsonValue]
    ) -> Path:
        path = write_summary(destination, summary)
        contents = json.loads(path.read_text())
        contents["temporal_evaluation"]["passed"] = True
        path.write_text(json.dumps(contents))
        return path

    previous_outputs = {
        name: (tmp_path / name).read_bytes()
        for name in ("changed.mp4", "changed.jsonl", "changed-summary.json")
    }
    monkeypatch.setattr(diagnostic_module, "_write_summary", falsify_evaluation)
    with pytest.raises(ValueError, match="temporal evaluation does not reconcile"):
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--temporal",
                "--composite",
                str(tmp_path / "changed.mp4"),
                "--jsonl",
                str(tmp_path / "changed.jsonl"),
                "--summary",
                str(tmp_path / "changed-summary.json"),
            ]
        )
    assert previous_outputs == {
        name: (tmp_path / name).read_bytes() for name in previous_outputs
    }


@pytest.mark.parametrize(
    ("accepted_count", "offset", "missing_holdout", "passed"),
    [
        (37, 25.0, False, True),
        (36, 0.0, False, False),
        (46, 25.01, False, False),
        (45, 0.0, True, False),
    ],
)
def test_temporal_evaluation_enforces_coverage_and_each_reserved_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_count: int,
    offset: float,
    missing_holdout: bool,
    passed: bool,
) -> None:
    import football_alpha_person_tracking.fixture_calibration as fixture

    monkeypatch.setattr(
        fixture,
        "CLIP_HELD_OUT_MARKINGS",
        {
            i: (HeldOutPitchMarking("x", 10.0, (ImagePoint(10 + offset, 10),)),)
            for i in (195, 225)
        },
    )
    ordered = [195, 225] + [i for i in range(180, 226) if i not in (195, 225)]
    accepted = set(ordered[:accepted_count])
    if missing_holdout:
        accepted.remove(225)
    records = [
        {
            "frame_index": i,
            "calibration": {
                "status": "accepted" if i in accepted else "abstained",
                "homography_img_to_pitch": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
                if i in accepted
                else None,
            },
        }
        for i in range(180, 226)
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    result = fixture.evaluate_temporal_clip(path)
    assert result["passed"] is passed


def test_temporal_diagnostic_rejects_a_different_source_before_publishing(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires the versioned fixture"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "composite.mp4",
            tmp_path / "tracks.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            correspondence_source_sha256="different-source",
            batch_size=4,
            temporal=True,
        )
    assert not (tmp_path / "composite.mp4").exists()
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize("indices", [[180], list(range(225, 179, -1)), [180] * 46])
def test_temporal_evaluation_rejects_incomplete_or_reordered_records(
    tmp_path: Path,
    indices: list[int],
) -> None:
    from football_alpha_person_tracking.fixture_calibration import (
        evaluate_temporal_clip,
    )

    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps({"frame_index": i}) for i in indices))
    with pytest.raises(ValueError, match="requires exactly frames"):
        evaluate_temporal_clip(path)


def test_tracking_diagnostic_command_rejects_non_positive_batch_size(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--composite",
                str(tmp_path / "tracking.mp4"),
                "--jsonl",
                str(tmp_path / "tracking.jsonl"),
                "--summary",
                str(tmp_path / "summary.json"),
                "--batch-size",
                "0",
            ]
        )

    assert error.value.code == 2


def test_tracking_diagnostic_command_validates_paths_before_loading_model(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_detector_creation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("detector must not be created")

    monkeypatch.setattr(
        "football_alpha_person_tracking.cli.RfDetrPersonDetector",
        fail_detector_creation,
    )

    with pytest.raises(ValueError, match="must be distinct"):
        main(
            [
                "diagnose",
                str(three_frame_video),
                "--composite",
                str(three_frame_video),
                "--jsonl",
                str(tmp_path / "tracking.jsonl"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )


def test_tracking_diagnostic_command_validates_source_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_detector_creation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("detector must not be created")

    monkeypatch.setattr(
        "football_alpha_person_tracking.cli.RfDetrPersonDetector",
        fail_detector_creation,
    )

    with pytest.raises(FileNotFoundError):
        main(
            [
                "diagnose",
                str(tmp_path / "missing.mp4"),
                "--composite",
                str(tmp_path / "tracking.mp4"),
                "--jsonl",
                str(tmp_path / "tracking.jsonl"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )


def test_tracking_diagnostic_rejects_non_positive_batch_size_before_staging(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=0,
        )

    assert list(tmp_path.glob(".*")) == []


@pytest.mark.parametrize(
    ("overlay_name", "jsonl_name", "summary_name"),
    [
        ("source", "tracks.jsonl", "summary.json"),
        ("overlay.mp4", "source", "summary.json"),
        ("overlay.mp4", "tracks.jsonl", "source"),
        ("same", "same", "summary.json"),
        ("overlay.mp4", "same", "same"),
    ],
)
def test_tracking_diagnostic_rejects_aliased_paths(
    three_frame_video: Path,
    tmp_path: Path,
    overlay_name: str,
    jsonl_name: str,
    summary_name: str,
) -> None:
    paths = {
        "source": three_frame_video,
        "overlay.mp4": tmp_path / "overlay.mp4",
        "tracks.jsonl": tmp_path / "tracks.jsonl",
        "summary.json": tmp_path / "summary.json",
        "same": tmp_path / "same",
    }

    with pytest.raises(ValueError, match="must be distinct"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            paths[overlay_name],
            paths[jsonl_name],
            paths[summary_name],
        )


def test_tracking_diagnostic_rejects_case_insensitive_path_aliases(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "TRACKING.MP4",
            tmp_path / "summary.json",
        )


def test_tracking_diagnostic_rejects_unicode_normalized_path_aliases(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            tmp_path / "r\u00e9sum\u00e9.jsonl",
            tmp_path / "re\u0301sume\u0301.jsonl",
            tmp_path / "summary.json",
        )


def test_tracking_diagnostic_rejects_filesystem_identity_aliases(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    source_alias = tmp_path / "source-alias.mp4"
    source_alias.hardlink_to(three_frame_video)

    with pytest.raises(ValueError, match="must be distinct"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            source_alias,
            tmp_path / "tracks.jsonl",
            tmp_path / "summary.json",
        )


def test_tracking_diagnostic_rejects_nested_destination_paths(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay.mp4"
    with pytest.raises(ValueError, match="must not contain one another"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            overlay,
            overlay / "tracks.jsonl",
            tmp_path / "summary.json",
        )


@pytest.mark.parametrize("directory_position", ["overlay", "jsonl", "summary"])
def test_tracking_diagnostic_rejects_directory_destinations(
    three_frame_video: Path,
    tmp_path: Path,
    directory_position: str,
) -> None:
    directory = tmp_path / "existing-directory"
    directory.mkdir()
    destinations = {
        "overlay": tmp_path / "overlay.mp4",
        "jsonl": tmp_path / "tracks.jsonl",
        "summary": tmp_path / "summary.json",
    }
    destinations[directory_position] = directory

    with pytest.raises(ValueError, match="must not be directories"):
        validate_tracking_diagnostic_paths(
            three_frame_video,
            destinations["overlay"],
            destinations["jsonl"],
            destinations["summary"],
        )

    assert directory.is_dir()


def test_tracking_diagnostic_provenance_identifies_processed_source_bytes(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with three_frame_video.open("rb") as source_file:
        expected_sha256 = file_digest(source_file, "sha256").hexdigest()

    class SourceMutatingDetector(DiagnosticDetector):
        mutated = False

        def detect(
            self,
            frames: Sequence[DecodedFrame],
        ) -> list[FramePersonDetections]:
            if not self.mutated:
                three_frame_video.write_bytes(b"source changed during diagnostic")
                self.mutated = True
            return super().detect(frames)

    summary = create_tracking_diagnostic(
        three_frame_video,
        tmp_path / "tracking.mp4",
        tmp_path / "tracking.jsonl",
        tmp_path / "summary.json",
        detector=SourceMutatingDetector(),
        detector_provenance={},
        tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
        batch_size=2,
    )

    assert summary["source_sha256"] == expected_sha256
    assert summary["frame_count"] == 3


def test_tracking_diagnostic_restores_artifact_set_when_publication_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = tmp_path / "tracking.mp4"
    jsonl_path = tmp_path / "tracking.jsonl"
    summary_path = tmp_path / "summary.json"
    overlay.write_bytes(b"old overlay")
    jsonl_path.write_text("old jsonl", encoding="utf-8")
    summary_path.write_text("old summary", encoding="utf-8")
    original_replace = Path.replace

    def fail_jsonl_publication(path: Path, target: Path) -> Path:
        if (
            path.name.startswith(".tracking-")
            and path.suffix == ".jsonl"
            and "-backup-" not in path.name
            and target == jsonl_path
        ):
            raise OSError("JSONL publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_jsonl_publication)

    with pytest.raises(OSError, match="JSONL publication failed"):
        create_tracking_diagnostic(
            three_frame_video,
            overlay,
            jsonl_path,
            summary_path,
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert overlay.read_bytes() == b"old overlay"
    assert jsonl_path.read_text(encoding="utf-8") == "old jsonl"
    assert summary_path.read_text(encoding="utf-8") == "old summary"
    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_restores_dangling_symlink_when_publication_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = tmp_path / "tracking.mp4"
    jsonl_path = tmp_path / "tracking.jsonl"
    summary_path = tmp_path / "summary.json"
    overlay.symlink_to("missing-overlay-target.mp4")
    jsonl_path.write_text("old jsonl", encoding="utf-8")
    summary_path.write_text("old summary", encoding="utf-8")
    original_replace = Path.replace

    def fail_jsonl_publication(path: Path, target: Path) -> Path:
        if (
            path.name.startswith(".tracking-")
            and path.suffix == ".jsonl"
            and "-backup-" not in path.name
            and target == jsonl_path
        ):
            raise OSError("JSONL publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_jsonl_publication)

    with pytest.raises(OSError, match="JSONL publication failed"):
        create_tracking_diagnostic(
            three_frame_video,
            overlay,
            jsonl_path,
            summary_path,
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert overlay.is_symlink()
    assert overlay.readlink() == Path("missing-overlay-target.mp4")
    assert jsonl_path.read_text(encoding="utf-8") == "old jsonl"
    assert summary_path.read_text(encoding="utf-8") == "old summary"
    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_rejects_summary_that_does_not_reconcile(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_summary = diagnostic_module._write_summary  # pyright: ignore[reportPrivateUsage]

    def write_inconsistent_summary(
        destination: Path,
        summary: dict[str, diagnostic_module.JsonValue],
    ) -> Path:
        temporary = original_write_summary(destination, summary)
        serialized = cast(dict[str, object], json.loads(temporary.read_text()))
        serialized["frame_count"] = 999
        temporary.write_text(json.dumps(serialized), encoding="utf-8")
        return temporary

    monkeypatch.setattr(
        diagnostic_module,
        "_write_summary",
        write_inconsistent_summary,
    )

    with pytest.raises(ValueError, match="does not reconcile"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert not (tmp_path / "tracking.mp4").exists()
    assert not (tmp_path / "tracking.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()
    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_cleans_staged_files_when_serialization_fails(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="JSON serializable"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={"invalid": object()},  # type: ignore[dict-item]
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


@pytest.mark.parametrize(
    ("detector", "tracker", "message"),
    [
        (WrongCardinalityDetector(), DiagnosticTracker(), "cardinality"),
        (WrongIdentityDetector(), DiagnosticTracker(), "detector.*different frame"),
        (DiagnosticDetector(), WrongIdentityTracker(), "tracker.*different frame"),
    ],
)
def test_tracking_diagnostic_rolls_back_contract_failures(
    three_frame_video: Path,
    tmp_path: Path,
    detector: PersonDetector,
    tracker: PersonTracker,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=detector,
            detector_provenance={},
            tracker_factory=lambda _rate: (tracker, {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_rejects_an_empty_decode_without_artifacts(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decode_no_frames(_source: Path) -> Iterator[DecodedFrame]:
        return iter(())

    monkeypatch.setattr(diagnostic_module, "decode_video", decode_no_frames)

    with pytest.raises(ValueError, match="no decoded frames"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_rolls_back_render_failure(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_render(*_arguments: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr(diagnostic_module, "_render_composite", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_replaces_an_existing_artifact_set(
    three_frame_video: Path,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "tracking.mp4"
    jsonl_path = tmp_path / "tracking.jsonl"
    summary_path = tmp_path / "summary.json"
    overlay.write_bytes(b"old overlay")
    jsonl_path.write_text("old jsonl", encoding="utf-8")
    summary_path.write_text("old summary", encoding="utf-8")

    summary = create_tracking_diagnostic(
        three_frame_video,
        overlay,
        jsonl_path,
        summary_path,
        detector=DiagnosticDetector(),
        detector_provenance={},
        tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
        batch_size=2,
    )

    assert len(list(decode_video(overlay))) == 3
    assert len(jsonl_path.read_text().splitlines()) == 3
    assert json.loads(summary_path.read_text()) == summary
    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_cleans_snapshot_when_artifact_staging_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_staging(_destination: Path) -> Path:
        raise OSError("artifact staging failed")

    monkeypatch.setattr(diagnostic_module, "_temporary_path", fail_staging)

    with pytest.raises(OSError, match="artifact staging failed"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_cleans_partial_snapshot_when_source_copy_fails(
    three_frame_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = cast(Callable[..., Any], Path.open)

    class FailingReader:
        def __init__(self, source_file: Any) -> None:
            self.source_file = source_file
            self.read_once = False

        def __enter__(self) -> FailingReader:
            return self

        def __exit__(self, *_args: object) -> None:
            self.source_file.close()

        def read(self, size: int = -1) -> bytes:
            if self.read_once:
                raise OSError("source copy failed")
            self.read_once = True
            return self.source_file.read(size)

    def fail_second_source_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened = original_open(path, *args, **kwargs)
        if path == three_frame_video and args and args[0] == "rb":
            return FailingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", fail_second_source_read)

    with pytest.raises(OSError, match="source copy failed"):
        create_tracking_diagnostic(
            three_frame_video,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []


def test_tracking_diagnostic_propagates_source_open_failure_without_staging(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        create_tracking_diagnostic(
            missing_source,
            tmp_path / "tracking.mp4",
            tmp_path / "tracking.jsonl",
            tmp_path / "summary.json",
            detector=DiagnosticDetector(),
            detector_provenance={},
            tracker_factory=lambda _rate: (DiagnosticTracker(), {}),
            batch_size=2,
        )

    assert list(tmp_path.glob(".*")) == []
