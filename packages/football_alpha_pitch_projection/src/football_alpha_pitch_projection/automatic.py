"""Image-supported automatic camera geometry within one continuous shot."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from football_alpha_pitch_projection.contracts import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    Homography,
    ImagePitchCorrespondence,
    PitchDimensions,
)
from football_alpha_video import DecodedFrame

type PitchPolylines = Sequence[Sequence[tuple[float, float]]]
type ExcludedRegion = tuple[float, float, float, float]


@dataclass(frozen=True)
class AutomaticParameters:
    """Explicit engineering thresholds, frozen before reserved evaluation."""

    maximum_anchor_age_seconds: float = 2.0
    reanchor_interval_seconds: float = 1.0
    maximum_frame_gap_seconds: float = 0.25
    minimum_denominator_ratio: float = 0.025
    maximum_pitch_extent: float = 3.0
    alignment_tolerance_px: float = 18.0
    cue_tolerance_px: float = 30.0
    minimum_cue_area_fraction: float = 0.01
    minimum_inlier_fraction: float = 0.75
    minimum_cue_count: int = 4
    minimum_flow_count: int = 12
    minimum_flow_area_fraction: float = 0.04
    flow_ransac_tolerance_px: float = 2.0
    flow_roundtrip_tolerance_px: float = 1.0
    feature_max_count: int = 600
    feature_quality: float = 0.001
    feature_min_distance_px: float = 8.0
    feature_block_size: int = 7
    flow_window_size: int = 31
    flow_max_level: int = 3
    grass_hsv_low: tuple[int, int, int] = (20, 35, 25)
    grass_hsv_high: tuple[int, int, int] = (100, 255, 255)
    stripe_kernel_size: int = 9
    stripe_min_contrast: int = 12
    stripe_max_saturation: int = 150
    marking_samples_per_m: int = 4
    minimum_visible_marking_samples: int = 8
    minimum_visible_markings: int = 2
    marking_error_quantile: float = 0.8
    refinement_search_radius_px: int = 32
    refinement_iterations: int = 6
    refinement_samples_per_line: int = 64
    refinement_match_distance_px: float = 1.5
    refinement_minimum_line_samples: int = 8
    refinement_minimum_area_fraction: float = 0.04
    refinement_minimum_outside_cue_fraction: float = 0.25
    refinement_ambiguity_margin_px: float = 4.0
    refinement_stripe_separation_px: float = 8.0
    refinement_maximum_ambiguous_fraction: float = 0.25
    refinement_maximum_displacement_px: float = 128.0


DEFAULT_PARAMETERS = AutomaticParameters()


@dataclass(frozen=True)
class GateEvidence:
    """Method-local diagnostics, never a probability of correctness."""

    reason: str | None
    metrics: dict[str, float]


def inspect_homography(
    matrix: NDArray[np.float64],
    field_region: NDArray[np.uint8],
    pitch: PitchDimensions,
    *,
    parameters: AutomaticParameters = DEFAULT_PARAMETERS,
) -> GateEvidence:
    """Check numerical stability throughout the observed field region."""
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) < 3
    ):
        return GateEvidence("invalid_homography", {})
    contours, _ = cv2.findContours(
        field_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return GateEvidence("no_field_support", {})
    points = np.concatenate(contours).reshape(-1, 2).astype(np.float64)
    rays = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if float(rays[:, 2].min()) <= 0 <= float(rays[:, 2].max()):
        return GateEvidence("horizon_in_field", {})
    ratio = float(np.min(np.abs(rays[:, 2])) / np.max(np.abs(rays[:, 2])))
    xy = rays[:, :2] / rays[:, 2:3]
    extent = float(np.max(np.abs(xy) / (pitch.length_m, pitch.width_m)))
    metrics = {"denominator_ratio": ratio, "field_extent_in_pitch_units": extent}
    if (
        ratio < parameters.minimum_denominator_ratio
        or extent > parameters.maximum_pitch_extent
    ):
        return GateEvidence("unstable_extrapolation", metrics)
    return GateEvidence(None, metrics)


def inspect_anchor(
    frame: DecodedFrame,
    matrix: NDArray[np.float64],
    pitch: PitchDimensions,
    correspondences: Sequence[ImagePitchCorrespondence],
    polylines: PitchPolylines,
    excluded_regions: Sequence[ExcludedRegion] = (),
    *,
    parameters: AutomaticParameters = DEFAULT_PARAMETERS,
) -> GateEvidence:
    """Qualify a raw hypothesis using current cues and field imagery."""
    region, distance = field_evidence(
        frame.image, excluded_regions, parameters=parameters
    )
    return _inspect_anchor(
        matrix,
        pitch,
        correspondences,
        polylines,
        region,
        distance,
        parameters=parameters,
    )


def _inspect_anchor(
    matrix: NDArray[np.float64],
    pitch: PitchDimensions,
    correspondences: Sequence[ImagePitchCorrespondence],
    polylines: PitchPolylines,
    region: NDArray[np.uint8],
    distance: NDArray[np.float32],
    *,
    parameters: AutomaticParameters,
) -> GateEvidence:
    """Qualify a raw hypothesis from already-measured current frame evidence."""
    geometry = inspect_homography(matrix, region, pitch, parameters=parameters)
    if geometry.reason is not None:
        return geometry
    if len(correspondences) < parameters.minimum_cue_count:
        return GateEvidence("insufficient_cues", geometry.metrics)
    uv = np.asarray(
        [(c.image_point.u, c.image_point.v) for c in correspondences], dtype=np.float64
    )
    metrics = dict(geometry.metrics)
    xy = np.asarray(
        [(c.pitch_point.x_m, c.pitch_point.y_m, 1) for c in correspondences]
    )
    rays = xy @ np.linalg.inv(matrix).T
    errors = np.linalg.norm(rays[:, :2] / rays[:, 2:3] - uv, axis=1)
    inliers = np.isfinite(errors) & (errors <= parameters.cue_tolerance_px)
    count = int(inliers.sum())
    metrics["cue_inlier_count"] = float(count)
    metrics["cue_inlier_fraction"] = count / len(correspondences)
    if (
        count < parameters.minimum_cue_count
        or metrics["cue_inlier_fraction"] < parameters.minimum_inlier_fraction
    ):
        return GateEvidence("cue_reprojection", metrics)
    metrics["cue_reprojection_rmse_px"] = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    area = (
        float(cv2.contourArea(cv2.convexHull(uv[inliers].astype(np.float32))))
        / region.size
    )
    metrics["cue_area_fraction"] = area
    if area < parameters.minimum_cue_area_fraction:
        return GateEvidence("clustered_cues", metrics)
    alignment = inspect_markings(
        matrix, region, distance, polylines, parameters=parameters
    )
    return GateEvidence(alignment.reason, {**metrics, **alignment.metrics})


def field_evidence(
    image: NDArray[np.uint8],
    excluded_regions: Sequence[ExcludedRegion] = (),
    *,
    parameters: AutomaticParameters = DEFAULT_PARAMETERS,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Measure the grass region and distances to visible painted stripes."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    grass = cv2.inRange(hsv, parameters.grass_hsv_low, parameters.grass_hsv_high)
    contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros(image.shape[:2], dtype=np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillConvexPoly(region, hull, 255)
    height, width = region.shape
    for x0, y0, x1, y1 in excluded_regions:
        region[
            max(0, int(np.floor(y0))) : min(height, int(np.ceil(y1))),
            max(0, int(np.floor(x0))) : min(width, int(np.ceil(x1))),
        ] = 0
    contrast = cv2.morphologyEx(
        image[:, :, 2],
        cv2.MORPH_TOPHAT,
        np.ones((parameters.stripe_kernel_size,) * 2, np.uint8),
    )
    white = (
        (contrast > parameters.stripe_min_contrast)
        & (hsv[:, :, 1] < parameters.stripe_max_saturation)
        & (region > 0)
    )
    distance = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 5)
    return region, np.asarray(distance, dtype=np.float32)


def inspect_markings(
    matrix: NDArray[np.float64],
    region: NDArray[np.uint8],
    distance: NDArray[np.float32],
    polylines: PitchPolylines,
    *,
    parameters: AutomaticParameters = DEFAULT_PARAMETERS,
) -> GateEvidence:
    """Compare projected markings with current pixels outside excluded regions."""
    inverse = np.linalg.inv(matrix)
    height, width = region.shape
    errors: list[float] = []
    for line in polylines:
        vertices = np.asarray(line, dtype=np.float64)
        samples = np.concatenate(
            [
                np.linspace(
                    a,
                    b,
                    max(
                        2,
                        int(
                            cast(float, np.linalg.norm(b - a))
                            * parameters.marking_samples_per_m
                        ),
                    ),
                    dtype=np.float64,
                )
                for a, b in pairwise(vertices)
            ]
        )
        rays = np.column_stack((samples, np.ones(len(samples)))) @ inverse.T
        safe = np.abs(rays[:, 2]) > 1e-10
        uv = rays[safe, :2] / rays[safe, 2:3]
        inside = (
            (uv[:, 0] >= 1)
            & (uv[:, 0] < width - 1)
            & (uv[:, 1] >= 1)
            & (uv[:, 1] < height - 1)
        )
        pixels = np.rint(uv[inside]).astype(np.int32)
        visible = region[pixels[:, 1], pixels[:, 0]] > 0
        pixels = pixels[visible]
        if len(pixels) >= parameters.minimum_visible_marking_samples:
            errors.append(
                float(
                    np.quantile(
                        distance[pixels[:, 1], pixels[:, 0]],
                        parameters.marking_error_quantile,
                    )
                )
            )
    metrics = {"visible_marking_count": float(len(errors))}
    if len(errors) < parameters.minimum_visible_markings:
        return GateEvidence("insufficient_markings", metrics)
    metrics["worst_marking_p80_px"] = max(errors)
    reason = (
        "marking_alignment" if max(errors) > parameters.alignment_tolerance_px else None
    )
    return GateEvidence(reason, metrics)


@dataclass(frozen=True)
class RefinementEvidence:
    """A proposed spatial correction is distinct from usable camera geometry."""

    homography_img_to_pitch: Homography | None
    gate: GateEvidence
    metrics: dict[str, float]


def _refine_markings(
    seed: NDArray[np.float64],
    pitch: PitchDimensions,
    region: NDArray[np.uint8],
    distance: NDArray[np.float32],
    polylines: PitchPolylines,
    correspondences: Sequence[ImagePitchCorrespondence],
    parameters: AutomaticParameters,
) -> tuple[NDArray[np.float64] | None, dict[str, float]]:
    """Match nearby stripes and fit point/line constraints without tangent anchors."""
    height, width = region.shape
    uv_cues = np.asarray([(c.image_point.u, c.image_point.v) for c in correspondences])
    xy_cues = np.asarray(
        [(c.pitch_point.x_m, c.pitch_point.y_m) for c in correspondences]
    )
    cue_hull = cv2.convexHull(uv_cues.astype(np.float32))
    offsets = np.arange(
        -parameters.refinement_search_radius_px,
        parameters.refinement_search_radius_px + 1,
    )
    matrix = seed.copy()
    metrics: dict[str, float] = {}
    for iteration in range(parameters.refinement_iterations):
        inverse = np.linalg.inv(matrix)
        worlds: list[NDArray[np.float64]] = []
        images: list[NDArray[np.float64]] = []
        equations: list[NDArray[np.float64]] = []
        for line in polylines:
            world = (
                np.linspace(
                    np.asarray(line[0], dtype=np.float64),
                    np.asarray(line[1], dtype=np.float64),
                    parameters.refinement_samples_per_line,
                )
                if len(line) == 2
                else np.asarray(line, dtype=np.float64)
            )
            rays = np.column_stack((world, np.ones(len(world)))) @ inverse.T
            uv = rays[:, :2] / rays[:, 2:3]
            tangent = np.asarray(np.gradient(uv, axis=0), dtype=np.float64)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            search = uv[:, None, :] + offsets[None, :, None] * normal[:, None, :]
            pixels = np.rint(search).astype(np.int32)
            inside = (
                (pixels[:, :, 0] >= 0)
                & (pixels[:, :, 0] < width)
                & (pixels[:, :, 1] >= 0)
                & (pixels[:, :, 1] < height)
            )
            x = np.clip(pixels[:, :, 0], 0, width - 1)
            y = np.clip(pixels[:, :, 1], 0, height - 1)
            costs = np.where(inside & (region[y, x] > 0), distance[y, x], np.inf)
            # Prefer the nearest stripe, not an arbitrary equally white pixel.
            best = np.argmin(costs + np.abs(offsets)[None, :] * 0.01, axis=1)
            matched = (
                costs[np.arange(len(world)), best]
                <= parameters.refinement_match_distance_px
            )
            alternatives = (
                (costs <= parameters.refinement_match_distance_px)
                & (
                    np.abs(offsets[None, :] - offsets[best, None])
                    >= parameters.refinement_stripe_separation_px
                )
                & (
                    np.abs(np.abs(offsets)[None, :] - np.abs(offsets[best, None]))
                    <= parameters.refinement_ambiguity_margin_px
                )
            )
            ambiguity = float(
                np.count_nonzero(matched & alternatives.any(axis=1))
            ) / max(1, int(np.count_nonzero(matched)))
            if ambiguity > parameters.refinement_maximum_ambiguous_fraction:
                return None, {
                    "ambiguous_sample_fraction": ambiguity,
                    "iteration_count": float(iteration + 1),
                }
            if np.count_nonzero(matched) < parameters.refinement_minimum_line_samples:
                continue
            worlds.append(world[matched])
            found = search[np.arange(len(world)), best][matched]
            images.append(found)
            if len(line) == 2:
                vx, vy, cx, cy = cv2.fitLine(
                    found.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
                ).ravel()
                equation = np.tile(
                    np.asarray((-vy, vx, vy * cx - vx * cy), dtype=np.float64),
                    (len(found), 1),
                )
            else:
                normals = normal[matched]
                equation = np.column_stack((normals, -np.sum(normals * found, axis=1)))
            equations.append(equation)
        metrics = {
            "iteration_count": float(iteration + 1),
            "matched_line_count": float(len(worlds)),
        }
        if len(worlds) < 2:
            return None, metrics
        uv_lines = np.concatenate(images)
        area = (
            float(cv2.contourArea(cv2.convexHull(uv_lines.astype(np.float32))))
            / region.size
        )
        outside = float(
            np.mean(
                [
                    cv2.pointPolygonTest(cue_hull, (float(p[0]), float(p[1])), False)
                    < 0
                    for p in uv_lines
                ]
            )
        )
        metrics.update(
            matched_sample_count=float(len(uv_lines)),
            support_area_fraction=area,
            outside_cue_fraction=outside,
        )
        if (
            area < parameters.refinement_minimum_area_fraction
            or outside < parameters.refinement_minimum_outside_cue_fraction
        ):
            return None, metrics
        # A stripe constrains only the normal, never an invented along-line
        # point identity. Normalize both planes before the linear solve.
        image_scale = float(max(height, width))
        constraints: list[tuple[float, ...]] = []
        for world, equation in zip(worlds, equations, strict=True):
            for (x_m, y_m), (a, b, c) in zip(
                world / 50.0,
                equation / np.asarray((1.0, 1.0, image_scale)),
                strict=True,
            ):
                constraints.append(
                    (a * x_m, a * y_m, a, b * x_m, b * y_m, b, c * x_m, c * y_m, c)
                )
        for (x_m, y_m), (u, v) in zip(
            xy_cues / 50.0, uv_cues / image_scale, strict=True
        ):
            constraints.extend(
                (
                    (x_m, y_m, 1, 0, 0, 0, -u * x_m, -u * y_m, -u),
                    (0, 0, 0, x_m, y_m, 1, -v * x_m, -v * y_m, -v),
                )
            )
        _, _, vectors = np.linalg.svd(np.asarray(constraints), full_matrices=False)
        fitted = (
            np.diag((image_scale, image_scale, 1.0))
            @ vectors[-1].reshape(3, 3)
            @ np.diag((0.02, 0.02, 1.0))
        )
        if not np.isfinite(fitted).all() or np.linalg.matrix_rank(fitted) < 3:
            return None, metrics
        matrix = np.asarray(np.linalg.inv(fitted), dtype=np.float64)
        if (
            inspect_homography(matrix, region, pitch, parameters=parameters).reason
            is not None
        ):
            return matrix, metrics
        probes = np.column_stack((np.concatenate(worlds), np.ones(len(uv_lines))))
        initial_rays = probes @ np.linalg.inv(seed).T
        fitted_rays = probes @ fitted.T
        displacement = float(
            np.max(
                cast(
                    NDArray[np.float64],
                    np.linalg.norm(
                        initial_rays[:, :2] / initial_rays[:, 2:3]
                        - fitted_rays[:, :2] / fitted_rays[:, 2:3],
                        axis=1,
                    ),
                )
            )
        )
        metrics["maximum_displacement_px"] = displacement
    if (
        metrics.get("maximum_displacement_px", 0.0)
        > parameters.refinement_maximum_displacement_px
    ):
        return None, metrics
    return matrix, metrics


@dataclass(frozen=True)
class CameraStep:
    """Accepted geometry plus its separate anchor, image and motion evidence."""

    calibration: CameraCalibration
    origin: Literal["estimated", "reanchored", "propagated", "abstained"]
    anchor_frame_index: int | None
    anchor_age_seconds: float | None
    raw_gate: GateEvidence
    output_gate: GateEvidence
    flow: dict[str, float]
    refinement: RefinementEvidence | None


class AutomaticCameraCalibrator:
    """Qualify automatic anchors without reading evaluation annotations."""

    def __init__(
        self,
        pitch: PitchDimensions,
        polylines: PitchPolylines,
        *,
        parameters: AutomaticParameters = DEFAULT_PARAMETERS,
    ) -> None:
        self.pitch = pitch
        self.polylines = polylines
        self.parameters = parameters
        self._previous: CameraCalibration | None = None
        self._anchor: CameraCalibration | None = None
        self._gray: NDArray[np.uint8] | None = None
        self._region: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        """Invalidate all geometry and image memory at a shot or evidence boundary."""
        self._previous = self._anchor = None
        self._gray = self._region = None

    def update(
        self,
        frame: DecodedFrame,
        hypothesis: NDArray[np.float64] | None = None,
        correspondences: Sequence[ImagePitchCorrespondence] = (),
        excluded_regions: Sequence[ExcludedRegion] = (),
    ) -> CameraStep:
        """Consume only the current frame and its automatic hypothesis."""
        region, distance = field_evidence(
            frame.image, excluded_regions, parameters=self.parameters
        )
        raw_gate = (
            _inspect_anchor(
                hypothesis,
                self.pitch,
                correspondences,
                self.polylines,
                region,
                distance,
                parameters=self.parameters,
            )
            if hypothesis is not None
            else GateEvidence("no_hypothesis", {})
        )
        gray = np.asarray(cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        matrix = hypothesis if raw_gate.reason is None else None
        gate = raw_gate
        refinement = None
        if raw_gate.reason == "marking_alignment":
            refined, metrics = _refine_markings(
                cast(NDArray[np.float64], hypothesis),
                self.pitch,
                region,
                distance,
                self.polylines,
                correspondences,
                self.parameters,
            )
            refined_gate = (
                _inspect_anchor(
                    refined,
                    self.pitch,
                    correspondences,
                    self.polylines,
                    region,
                    distance,
                    parameters=self.parameters,
                )
                if refined is not None
                else GateEvidence("insufficient_refinement_support", {})
            )
            refinement = RefinementEvidence(
                cast(Homography, tuple(tuple(float(v) for v in row) for row in refined))
                if refined is not None
                else None,
                refined_gate,
                metrics,
            )
            if refined_gate.reason is None:
                matrix, gate = refined, refined_gate
        origin: Literal["estimated", "reanchored", "propagated", "abstained"] = (
            "estimated"
        )
        flow: dict[str, float] = {}
        previous = self._previous
        if previous is not None and (
            frame.index != previous.frame_index + 1
            or frame.time_base != previous.time_base
            or frame.pts <= previous.pts
            or float((frame.pts - previous.pts) * frame.time_base)
            > self.parameters.maximum_frame_gap_seconds
            or frame.image.shape[:2] != (previous.image_height, previous.image_width)
        ):
            self.reset()
            if matrix is None:
                gate = GateEvidence("input_discontinuity", {})
        if self._anchor is not None:
            age = float((frame.pts - self._anchor.pts) * frame.time_base)
            if age > self.parameters.maximum_anchor_age_seconds:
                self.reset()
                if matrix is None:
                    gate = GateEvidence("anchor_expired", {})
            else:
                propagated, motion_gate, flow = self._propagate(gray, region, distance)
                if propagated is not None:
                    if (
                        matrix is not None
                        and age >= self.parameters.reanchor_interval_seconds
                    ):
                        origin = "reanchored"
                    else:
                        matrix, gate, origin = propagated, motion_gate, "propagated"
                else:
                    self.reset()
                    if matrix is None:
                        gate = motion_gate
        accepted = matrix is not None and gate.reason is None
        score = (
            flow.get("inlier_fraction", 0.0)
            if origin == "propagated"
            else gate.metrics.get("cue_inlier_fraction", 0.0)
        )
        count = (
            int(flow.get("correspondence_count", 0))
            if origin == "propagated"
            else len(correspondences)
        )
        inliers = (
            int(flow.get("inlier_count", 0))
            if origin == "propagated"
            else int(gate.metrics.get("cue_inlier_count", 0))
        )
        calibration = CameraCalibration(
            frame_index=frame.index,
            pts=frame.pts,
            time_base=frame.time_base,
            image_width=frame.image.shape[1],
            image_height=frame.image.shape[0],
            pitch=self.pitch,
            status=CalibrationStatus.ACCEPTED
            if accepted
            else CalibrationStatus.ABSTAINED,
            homography_img_to_pitch=cast(
                Homography,
                tuple(
                    tuple(float(value) for value in row)
                    for row in cast(NDArray[np.float64], matrix)
                ),
            )
            if accepted
            else None,
            abstention_reason=None
            if accepted
            else AbstentionReason.INDEPENDENT_VALIDATION_FAILED,
            method="opencv_lk_ransac+image_gate"
            if origin == "propagated"
            else (
                "image_marking_refinement+image_gate"
                if refinement is not None and refinement.gate.reason is None
                else "automatic_hypothesis+image_gate"
            ),
            provenance="causal;no_evaluation_labels",
            quality_score=score,
            quality_threshold=self.parameters.minimum_inlier_fraction,
            correspondence_count=count,
            inlier_count=inliers,
            reprojection_rmse_px=gate.metrics.get("cue_reprojection_rmse_px"),
        )
        if accepted:
            if origin != "propagated":
                self._anchor = calibration
            self._previous, self._gray, self._region = calibration, gray, region
        else:
            self.reset()
        anchor = self._anchor
        return CameraStep(
            calibration,
            origin if accepted else "abstained",
            anchor.frame_index if anchor else None,
            float((frame.pts - anchor.pts) * frame.time_base) if anchor else None,
            raw_gate,
            gate,
            flow,
            refinement,
        )

    def _propagate(
        self,
        gray: NDArray[np.uint8],
        region: NDArray[np.uint8],
        distance: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64] | None, GateEvidence, dict[str, float]]:
        previous = cast(CameraCalibration, self._previous)
        old_gray = cast(NDArray[np.uint8], self._gray)
        points = cast(
            NDArray[np.float32] | None,
            cv2.goodFeaturesToTrack(
                old_gray,
                self.parameters.feature_max_count,
                self.parameters.feature_quality,
                self.parameters.feature_min_distance_px,
                mask=self._region,
                blockSize=self.parameters.feature_block_size,
            ),
        )
        metrics: dict[str, float] = {}
        if points is None:
            return None, GateEvidence("no_visual_features", {}), metrics
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            old_gray,
            gray,
            points,
            points.copy(),
            winSize=(self.parameters.flow_window_size,) * 2,
            maxLevel=self.parameters.flow_max_level,
        )
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            old_gray,
            moved,
            moved.copy(),
            winSize=(self.parameters.flow_window_size,) * 2,
            maxLevel=self.parameters.flow_max_level,
        )
        xy = np.asarray(moved, dtype=np.float32).reshape(-1, 2)
        old = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        valid = (
            forward.ravel().astype(bool)
            & backward.ravel().astype(bool)
            & (
                np.linalg.norm(returned.reshape(-1, 2) - old, axis=1)
                <= self.parameters.flow_roundtrip_tolerance_px
            )
        )
        valid &= (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < gray.shape[1] - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < gray.shape[0] - 1)
        )
        xy, old = xy[valid], old[valid]
        inside = (
            region[np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)] > 0
        )
        xy, old = xy[inside], old[inside]
        metrics["correspondence_count"] = float(len(xy))
        if len(xy) < self.parameters.minimum_flow_count:
            return None, GateEvidence("insufficient_flow", {}), metrics
        motion, mask = cast(
            tuple[NDArray[np.float64] | None, NDArray[np.uint8] | None],
            cv2.findHomography(
                xy, old, cv2.RANSAC, self.parameters.flow_ransac_tolerance_px
            ),
        )
        if motion is None or mask is None:
            return None, GateEvidence("degenerate_flow", {}), metrics
        inliers = np.asarray(mask).ravel().astype(bool)
        metrics["inlier_count"] = float(inliers.sum())
        metrics["inlier_fraction"] = float(inliers.mean())
        metrics["support_area_fraction"] = (
            float(cv2.contourArea(cv2.convexHull(xy[inliers]))) / region.size
        )
        if (
            metrics["inlier_count"] < self.parameters.minimum_flow_count
            or metrics["inlier_fraction"] < self.parameters.minimum_inlier_fraction
            or metrics["support_area_fraction"]
            < self.parameters.minimum_flow_area_fraction
        ):
            return None, GateEvidence("weak_flow_consensus", {}), metrics
        matrix = np.asarray(previous.homography_img_to_pitch, dtype=np.float64) @ motion
        gate = inspect_homography(
            matrix, region, self.pitch, parameters=self.parameters
        )
        if gate.reason is None:
            alignment = inspect_markings(
                matrix, region, distance, self.polylines, parameters=self.parameters
            )
            gate = GateEvidence(alignment.reason, {**gate.metrics, **alignment.metrics})
        return matrix if gate.reason is None else None, gate, metrics
