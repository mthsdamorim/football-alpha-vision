"""Automatic geometry must earn acceptance from current image evidence."""

from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np
import pytest
from football_alpha_pitch_projection.automatic import (
    AutomaticCameraCalibrator,
    AutomaticParameters,
    inspect_anchor,
    inspect_homography,
)
from numpy.typing import NDArray

from football_alpha_pitch_projection import (
    CalibrationStatus,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
    project_image_point,
)
from football_alpha_video import DecodedFrame

PITCH = PitchDimensions(105, 68)
POLYLINES = (
    ((4.0, -4.0), (28.0, -4.0)),
    ((28.0, -4.0), (28.0, -20.0)),
    ((28.0, -20.0), (4.0, -20.0)),
    ((4.0, -20.0), (4.0, -4.0)),
)


def field_frame(index: int = 0) -> DecodedFrame:
    image = np.full((240, 320, 3), (40, 140, 40), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (280, 200), (255, 255, 255), 2)
    return DecodedFrame(index, index * 512, Fraction(1, 15360), image)


def field_cues(offset_u: float = 0.0) -> tuple[ImagePitchCorrespondence, ...]:
    return tuple(
        ImagePitchCorrespondence(
            ImagePoint(u + offset_u, v), PitchPoint(u / 10, -v / 10)
        )
        for u, v in ((40, 40), (280, 40), (280, 200), (40, 200))
    )


def textured_frame(index: int = 0) -> DecodedFrame:
    item = field_frame(index)
    image = item.image.copy()
    image[:, :, 1] = np.minimum(
        image[:, :, 1].astype(np.int16)
        + np.random.default_rng(42).integers(-12, 12, image.shape[:2]),
        255,
    ).astype(np.uint8)
    return replace(item, image=image)


def test_a_finite_homography_with_a_horizon_in_the_field_is_rejected() -> None:
    region = np.full((240, 320), 255, dtype=np.uint8)
    # A solver can fit clustered points exactly while this pole crosses grass.
    matrix = np.asarray(((0.1, 0, 0), (0, 0.1, 0), (0.01, 0, -1.6)))

    result = inspect_homography(matrix, region, PitchDimensions(105, 68))

    assert result.reason == "horizon_in_field"


@pytest.mark.parametrize(
    ("matrix", "empty", "reason"),
    [
        (np.diag([0.1, -0.1, 1.0]), False, None),
        (np.diag([np.nan, 1.0, 1.0]), False, "invalid_homography"),
        (np.zeros((3, 3)), False, "invalid_homography"),
        (np.eye(3), True, "no_field_support"),
        (np.diag([30.0, 30.0, 1.0]), False, "unstable_extrapolation"),
        (
            np.asarray(((0.1, 0, 0), (0, 0.1, 0), (0.01, 0, 0.001))),
            False,
            "unstable_extrapolation",
        ),
    ],
)
def test_projective_region_requires_finite_supported_bounded_geometry(
    matrix: NDArray[np.float64], empty: bool, reason: str | None
) -> None:
    region = np.full((240, 320), 0 if empty else 255, dtype=np.uint8)
    result = inspect_homography(matrix, region, PitchDimensions(105, 68))
    assert result.reason == reason


def test_anchor_needs_image_alignment_even_when_cues_fit_it_exactly() -> None:
    good = np.diag([0.1, -0.1, 1.0])
    assert (
        inspect_anchor(field_frame(), good, PITCH, field_cues(), POLYLINES).reason
        is None
    )
    shifted = good.copy()
    shifted[0, 2] = -2.0

    result = inspect_anchor(field_frame(), shifted, PITCH, field_cues(20), POLYLINES)

    assert result.reason == "marking_alignment"


def test_many_clustered_cues_cannot_qualify_an_anchor() -> None:
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in ((100, 100), (101, 100), (101, 101), (100, 101)) * 5
    )
    result = inspect_anchor(
        field_frame(), np.diag([0.1, -0.1, 1.0]), PITCH, cues, POLYLINES
    )
    assert result.reason == "clustered_cues"


def test_one_bad_cue_does_not_override_distributed_consistent_image_evidence() -> None:
    cues = (
        *field_cues(),
        ImagePitchCorrespondence(ImagePoint(150, 100), PitchPoint(50, 20)),
    )
    result = inspect_anchor(
        field_frame(), np.diag([0.1, -0.1, 1.0]), PITCH, cues, POLYLINES
    )
    assert result.reason is None
    assert result.metrics["cue_inlier_fraction"] == 0.8


@pytest.mark.parametrize("displacement_budget", [64.0, 1.0])
def test_current_stripes_recover_misalignment_beyond_local_cues(
    displacement_budget: float,
) -> None:
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in ((210, 100), (260, 100), (260, 140), (210, 140))
    )
    seed = np.diag([0.1, -0.1, 1.0])
    seed[0, 2] = -2.5
    camera = AutomaticCameraCalibrator(
        PITCH,
        POLYLINES,
        parameters=replace(
            AutomaticParameters(),
            refinement_maximum_displacement_px=displacement_budget,
        ),
    )

    result = camera.update(textured_frame(), seed, cues)

    assert result.raw_gate.reason == "marking_alignment"
    if displacement_budget == 1.0:
        assert result.calibration.status is CalibrationStatus.ABSTAINED
        return
    assert result.calibration.status is CalibrationStatus.ACCEPTED
    assert result.calibration.method == "image_marking_refinement+image_gate"
    point = project_image_point(result.calibration, ImagePoint(80, 80)).pitch_point
    assert point is not None
    assert (point.x_m, point.y_m) == pytest.approx((8, -8), abs=0.2)
    assert result.refinement is not None
    assert result.refinement.gate.reason is None
    assert result.refinement.homography_img_to_pitch is not None
    assert result.refinement.metrics["outside_cue_fraction"] > 0.25
    assert seed[0, 2] == -2.5
    following = camera.update(textured_frame(1))
    assert following.origin == "propagated"
    assert following.anchor_frame_index == 0


def test_equally_plausible_parallel_stripes_do_not_choose_an_anchor() -> None:
    image = np.full((240, 350, 3), (40, 140, 40), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (280, 200), (255, 255, 255), 2)
    # Two equally close versions of each vertical line surround the seed.
    cv2.line(image, (90, 40), (90, 200), (255, 255, 255), 2)
    cv2.line(image, (330, 40), (330, 200), (255, 255, 255), 2)
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in ((210, 100), (260, 100), (260, 140), (210, 140))
    )
    seed = np.diag([0.1, -0.1, 1.0])
    seed[0, 2] = -2.5
    lines = tuple(
        tuple(reversed(line)) if i == 1 else line for i, line in enumerate(POLYLINES)
    )
    result = AutomaticCameraCalibrator(PITCH, lines).update(
        replace(field_frame(), image=image), seed, cues
    )
    assert result.calibration.status is CalibrationStatus.ABSTAINED
    assert result.refinement is not None


def test_curved_markings_supply_support_missing_from_straight_lines() -> None:
    image = np.full((240, 320, 3), (40, 140, 40), dtype=np.uint8)
    cv2.line(image, (40, 40), (280, 40), (255, 255, 255), 2)
    cv2.circle(image, (160, 120), 70, (255, 255, 255), 2)
    angles = np.linspace(0, 2 * np.pi, 121)
    circle = tuple(
        (16 + 7 * float(np.cos(a)), -12 + 7 * float(np.sin(a))) for a in angles
    )
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in ((210, 100), (260, 100), (260, 140), (210, 140))
    )
    seed = np.diag([0.1, -0.1, 1.0])
    seed[0, 2] = -2.5
    result = AutomaticCameraCalibrator(PITCH, (POLYLINES[0], circle)).update(
        replace(field_frame(), image=image), seed, cues
    )
    assert result.calibration.status is CalibrationStatus.ACCEPTED
    point = project_image_point(result.calibration, ImagePoint(90, 120)).pitch_point
    assert point is not None
    assert (point.x_m, point.y_m) == pytest.approx((9, -12), abs=0.3)


@pytest.mark.parametrize("support", ["one_line", "inside_cues", "small_area"])
def test_refinement_requires_new_distributed_image_support(support: str) -> None:
    frame = field_frame()
    lines = POLYLINES
    positions = ((210, 100), (260, 100), (260, 140), (210, 140))
    if support == "one_line":
        image = np.full((240, 320, 3), (40, 140, 40), dtype=np.uint8)
        cv2.line(image, (40, 40), (280, 40), (255, 255, 255), 2)
        frame = replace(frame, image=image)
    elif support == "inside_cues":
        positions = ((10, 10), (310, 10), (310, 230), (10, 230))
    else:
        image = np.full((480, 640, 3), (40, 140, 40), dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (120, 120), (255, 255, 255), 2)
        frame = replace(frame, image=image)
        lines = (
            ((4.0, -4.0), (12.0, -4.0)),
            ((12.0, -4.0), (12.0, -12.0)),
            ((12.0, -12.0), (4.0, -12.0)),
            ((4.0, -12.0), (4.0, -4.0)),
        )
        positions = ((10, 10), (90, 10), (90, 50), (10, 50))
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in positions
    )
    seed = np.diag([0.1, -0.1, 1.0])
    seed[0, 2] = -2.5

    result = AutomaticCameraCalibrator(PITCH, lines).update(frame, seed, cues)

    assert result.raw_gate.reason == "marking_alignment"
    assert result.calibration.status is CalibrationStatus.ABSTAINED
    assert result.refinement is not None
    assert result.refinement.homography_img_to_pitch is None


@pytest.mark.parametrize("failure", ["singular", "horizon"])
def test_unusable_numeric_refit_never_initializes_camera_memory(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    # Fault injection at the numeric solver boundary, as for the flow solver.
    def unusable_solution(
        *args: object, **kwargs: object
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        fitted = np.diag([1.0, 1.0, 0.0])
        if failure == "horizon":
            pole = np.asarray(((0.1, 0, 0), (0, -0.1, 0), (0.01, 0, -1.6)))
            fitted = (
                np.diag([1 / 320, 1 / 320, 1])
                @ np.linalg.inv(pole)
                @ np.diag([50, 50, 1])
            )
        vectors = np.eye(9)
        vectors[-1] = fitted.ravel()
        return np.eye(9), np.ones(9), vectors

    monkeypatch.setattr(np.linalg, "svd", unusable_solution)
    cues = tuple(
        ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
        for u, v in ((210, 100), (260, 100), (260, 140), (210, 140))
    )
    seed = np.diag([0.1, -0.1, 1.0])
    seed[0, 2] = -2.5
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)

    result = camera.update(field_frame(), seed, cues)

    assert result.calibration.status is CalibrationStatus.ABSTAINED
    assert result.refinement is not None
    assert camera.update(field_frame(1)).origin == "abstained"


def test_propagation_tracks_known_camera_motion_from_an_automatic_anchor() -> None:
    initial = textured_frame()
    image = initial.image
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)
    first = camera.update(initial, np.diag([0.1, -0.1, 1.0]), field_cues())
    assert first.calibration.status is CalibrationStatus.ACCEPTED
    motion = np.asarray(((1, 0, 4), (0, 1, 2)), dtype=np.float32)
    moved = np.asarray(cv2.warpAffine(image, motion, (320, 240)), dtype=np.uint8)

    result = camera.update(replace(field_frame(1), image=moved))

    assert result.origin == "propagated"
    assert result.anchor_frame_index == 0
    assert result.anchor_age_seconds == pytest.approx(1 / 30)
    assert result.calibration.pts == 512
    projected = project_image_point(result.calibration, ImagePoint(84, 82)).pitch_point
    assert projected is not None
    assert (projected.x_m, projected.y_m) == pytest.approx((8, -8), abs=0.03)
    assert result.flow["inlier_count"] >= 12


@pytest.mark.parametrize("boundary", ["index", "time", "time_base", "dimensions"])
def test_discontinuity_discards_memory_until_a_new_automatic_anchor(
    boundary: str,
) -> None:
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)
    camera.update(textured_frame(), np.diag([0.1, -0.1, 1.0]), field_cues())
    item = textured_frame(1)
    if boundary == "index":
        item = replace(item, index=2)
    elif boundary == "time":
        item = replace(item, pts=0)
    elif boundary == "time_base":
        item = replace(item, time_base=Fraction(1, 1000))
    else:
        item = replace(item, image=item.image[:200])

    result = camera.update(item)

    assert result.calibration.status is CalibrationStatus.ABSTAINED
    assert result.calibration.homography_img_to_pitch is None
    assert (
        camera.update(textured_frame(3)).calibration.status
        is CalibrationStatus.ABSTAINED
    )
    assert (
        camera.update(
            textured_frame(4), np.diag([0.1, -0.1, 1.0]), field_cues()
        ).calibration.status
        is CalibrationStatus.ACCEPTED
    )


def test_anchor_age_is_bounded_and_current_evidence_reanchors_without_jitter() -> None:
    camera = AutomaticCameraCalibrator(
        PITCH,
        POLYLINES,
        parameters=AutomaticParameters(
            maximum_anchor_age_seconds=0.1, reanchor_interval_seconds=0.05
        ),
    )
    matrix = np.diag([0.1, -0.1, 1.0])
    camera.update(textured_frame(), matrix, field_cues())
    early = camera.update(textured_frame(1), matrix, field_cues())
    assert early.origin == "propagated"
    assert early.anchor_frame_index == 0
    reanchored = camera.update(textured_frame(2), matrix, field_cues())
    assert reanchored.origin == "reanchored"
    assert reanchored.anchor_frame_index == 2
    for index in range(3, 6):
        assert camera.update(textured_frame(index)).origin == "propagated"
    lost = camera.update(textured_frame(6))
    assert lost.origin == "abstained"
    assert lost.output_gate.reason == "anchor_expired"
    assert camera.update(textured_frame(7)).origin == "abstained"
    assert camera.update(textured_frame(8), matrix, field_cues()).origin == "estimated"


def test_foreground_exclusion_preserves_camera_motion() -> None:
    first = textured_frame()
    motion = np.asarray(
        ((1, 0.003, 4), (-0.002, 1, 2), (0.00002, 0.00003, 1)), dtype=np.float64
    )
    second_image = np.asarray(
        cv2.warpPerspective(first.image, motion, (320, 240)), dtype=np.uint8
    )
    first_image = first.image.copy()
    foreground = np.random.default_rng(3).integers(
        0, 256, (110, 150, 3), dtype=np.uint8
    )
    first_image[60:170, 80:230] = foreground
    second_image[45:155, 110:260] = foreground
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)
    camera.update(
        replace(first, image=first_image),
        np.diag([0.1, -0.1, 1.0]),
        field_cues(),
        [(75, 55, 235, 175)],
    )

    result = camera.update(
        replace(field_frame(1), image=second_image),
        excluded_regions=[(105, 40, 265, 160)],
    )

    assert result.origin == "propagated"
    pixel = motion @ np.asarray((45.0, 180.0, 1.0))
    point = project_image_point(
        result.calibration,
        ImagePoint(*(float(value) for value in pixel[:2] / pixel[2])),
    ).pitch_point
    assert point is not None
    assert (point.x_m, point.y_m) == pytest.approx((4.5, -18), abs=0.06)


@pytest.mark.parametrize("loss", ["blank", "excluded", "reset", "long_time_gap"])
def test_loss_and_explicit_reset_require_a_fresh_anchor(loss: str) -> None:
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)
    matrix = np.diag([0.1, -0.1, 1.0])
    camera.update(textured_frame(), matrix, field_cues())
    item = textured_frame(1)
    regions = []
    if loss == "blank":
        item = replace(item, image=np.zeros_like(item.image))
    elif loss == "excluded":
        regions = [(0.0, 0.0, 320.0, 240.0)]
    elif loss == "reset":
        camera.reset()
    else:
        item = replace(item, pts=15360)
    lost = camera.update(item, excluded_regions=regions)
    assert lost.origin == "abstained"
    assert lost.anchor_frame_index is None
    assert camera.update(textured_frame(2)).origin == "abstained"
    assert camera.update(textured_frame(3), matrix, field_cues()).origin == "estimated"


@pytest.mark.parametrize("failure", ["geometry", "few_cues", "wrong_cues", "no_lines"])
def test_anchor_rejects_missing_or_contradictory_evidence(failure: str) -> None:
    matrix = np.diag([0.1, -0.1, 1.0])
    cues = field_cues()
    lines = POLYLINES
    expected = "cue_reprojection"
    if failure == "geometry":
        matrix = np.zeros((3, 3))
        expected = "invalid_homography"
    elif failure == "few_cues":
        cues = cues[:3]
        expected = "insufficient_cues"
    elif failure == "wrong_cues":
        cues = field_cues(80)
    else:
        lines = (((1000.0, 1000.0), (1001.0, 1001.0)),)
        expected = "insufficient_markings"
    assert inspect_anchor(field_frame(), matrix, PITCH, cues, lines).reason == expected


@pytest.mark.parametrize("boundary", ["discontinuity", "expired", "flow_lost"])
def test_fresh_evidence_can_reinitialize_at_the_loss_boundary(boundary: str) -> None:
    camera = AutomaticCameraCalibrator(
        PITCH,
        POLYLINES,
        parameters=AutomaticParameters(maximum_anchor_age_seconds=0.01),
    )
    matrix = np.diag([0.1, -0.1, 1.0])
    camera.update(field_frame(), matrix, field_cues())
    item = field_frame(1)
    if boundary == "discontinuity":
        item = field_frame(2)
    elif boundary == "flow_lost":
        camera.parameters = AutomaticParameters(minimum_flow_count=10000)
    recovered = camera.update(item, matrix, field_cues())
    assert recovered.origin == "estimated"
    assert recovered.anchor_frame_index == item.index
    assert recovered.anchor_age_seconds == 0


@pytest.mark.parametrize("failure", ["no_features", "degenerate", "weak", "horizon"])
def test_motion_failure_invalidates_camera_without_repeating_old_matrix(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    camera = AutomaticCameraCalibrator(PITCH, POLYLINES)
    camera.update(textured_frame(), np.diag([0.1, -0.1, 1.0]), field_cues())
    expected = "no_visual_features"
    if failure == "no_features":

        def missing_features(*args: object, **kwargs: object) -> None:
            return None

        monkeypatch.setattr(cv2, "goodFeaturesToTrack", missing_features)
    elif failure == "degenerate":

        def missing_motion(*args: object) -> tuple[None, None]:
            return None, None

        monkeypatch.setattr(cv2, "findHomography", missing_motion)
        expected = "degenerate_flow"
    elif failure == "weak":
        camera.parameters = AutomaticParameters(minimum_flow_area_fraction=1.0)
        expected = "weak_flow_consensus"
    else:
        # Vendor-shaped RANSAC result with a pole crossing the image. The region
        # gate must still reject it despite a perfect consensus claim.
        def pole(
            xy: NDArray[np.float32], *args: object
        ) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
            return (
                np.asarray(((1.0, 0, 0), (0, 1, 0), (0.01, 0, -1.6))),
                np.ones((len(xy), 1), dtype=np.uint8),
            )

        monkeypatch.setattr(cv2, "findHomography", pole)
        expected = "horizon_in_field"
    lost = camera.update(textured_frame(1))
    assert lost.origin == "abstained"
    assert lost.output_gate.reason == expected
    assert lost.calibration.homography_img_to_pitch is None
    assert camera.update(textured_frame(2)).origin == "abstained"


@pytest.mark.parametrize("refined_anchor", [False, True])
def test_identical_prefix_is_independent_of_future_frames(refined_anchor: bool) -> None:
    first = AutomaticCameraCalibrator(PITCH, POLYLINES)
    second = AutomaticCameraCalibrator(PITCH, POLYLINES)
    for index in range(3):
        matrix = np.diag([0.1, -0.1, 1.0]) if index == 0 else None
        cues = field_cues()
        if refined_anchor:
            cues = tuple(
                ImagePitchCorrespondence(ImagePoint(u, v), PitchPoint(u / 10, -v / 10))
                for u, v in ((210, 100), (260, 100), (260, 140), (210, 140))
            )
            if matrix is not None:
                matrix[0, 2] = -2.5
        a = first.update(textured_frame(index), matrix, cues)
        b = second.update(textured_frame(index), matrix, cues)
        assert a == b
    first.update(replace(field_frame(3), image=np.zeros((240, 320, 3), np.uint8)))
    assert second.update(textured_frame(3)).origin == "propagated"
