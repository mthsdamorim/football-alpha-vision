"""Sequential camera calibration within an explicitly continuous shot."""

from dataclasses import replace
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from football_alpha_pitch_projection.contracts import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    FrameCalibrationInput,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
)
from football_alpha_pitch_projection.correspondence import (
    CorrespondenceHomographyCalibrator,
)
from football_alpha_video import DecodedFrame


class OpticalFlowCalibrator:
    """Propagate an explicit manual anchor through subsequent video frames."""

    def __init__(self, pitch: PitchDimensions) -> None:
        self.pitch = pitch
        self._control = CorrespondenceHomographyCalibrator()
        self._gray: NDArray[np.uint8] | None = None
        self._points: NDArray[np.float32] | None = None
        self._pitch_points = np.empty((0, 2), dtype=np.float64)
        self._anchor: CameraCalibration | None = None
        self._previous: CameraCalibration | None = None

    def reset(self) -> None:
        """End the current camera session; only an explicit anchor can restart it."""
        self._gray = None
        self._points = None
        self._pitch_points = np.empty((0, 2), dtype=np.float64)
        self._anchor = None
        self._previous = None

    def update(
        self, frame: DecodedFrame, anchor: CameraCalibration | None = None
    ) -> CameraCalibration:
        gray = np.asarray(cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        if anchor is not None:
            self.reset()
            if (
                anchor.frame_index,
                anchor.pts,
                anchor.time_base,
                anchor.image_width,
                anchor.image_height,
                anchor.pitch,
            ) != (
                frame.index,
                frame.pts,
                frame.time_base,
                frame.image.shape[1],
                frame.image.shape[0],
                self.pitch,
            ):
                raise ValueError("anchor and video frame identities differ")
            if anchor.status is CalibrationStatus.ABSTAINED:
                return anchor
            self._anchor = anchor
            self._previous = anchor
            self._gray = gray
            mask = cv2.inRange(
                cv2.cvtColor(frame.image, cv2.COLOR_RGB2HSV),
                (35, 40, 35),
                (95, 255, 255),
            )
            self._points = cast(
                NDArray[np.float32] | None,
                cv2.goodFeaturesToTrack(gray, 300, 0.01, 12, mask=mask, blockSize=7),
            )
            if self._points is not None:
                self._pitch_points = cv2.perspectiveTransform(
                    self._points.astype(np.float64),
                    np.asarray(anchor.homography_img_to_pitch),
                ).reshape(-1, 2)
            return anchor

        previous = self._previous
        if previous is not None and (
            frame.index != previous.frame_index + 1
            or frame.time_base != previous.time_base
            or frame.pts <= previous.pts
            or frame.image.shape[:2] != (previous.image_height, previous.image_width)
            or frame.index - cast(CameraCalibration, self._anchor).frame_index > 30
        ):
            self.reset()
        cues: list[ImagePitchCorrespondence] = []
        if self._points is not None and self._gray is not None:
            moved_raw, forward, _ = cv2.calcOpticalFlowPyrLK(
                self._gray,
                gray,
                self._points,
                self._points.copy(),
                winSize=(21, 21),
                maxLevel=3,
            )
            moved = np.asarray(moved_raw, dtype=np.float32)
            returned, backward, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._gray, moved, moved.copy(), winSize=(21, 21), maxLevel=3
            )
            xy = moved.reshape(-1, 2)
            valid = (
                forward.reshape(-1).astype(bool)
                & backward.reshape(-1).astype(bool)
                & (np.max(np.abs(returned - self._points), axis=(1, 2)) <= 1.0)
                & np.isfinite(xy).all(axis=1)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] <= frame.image.shape[1])
                & (xy[:, 1] >= 0)
                & (xy[:, 1] <= frame.image.shape[0])
            )
            self._points = moved[valid]
            self._pitch_points = self._pitch_points[valid]
            cues = [
                ImagePitchCorrespondence(
                    ImagePoint(float(uv[0]), float(uv[1])),
                    PitchPoint(float(xy[0]), float(xy[1])),
                )
                for uv, xy in zip(
                    self._points.reshape(-1, 2), self._pitch_points, strict=True
                )
            ]
        result = self._control.calibrate(
            [
                FrameCalibrationInput.from_frame(
                    frame, pitch=self.pitch, correspondences=cues
                )
            ]
        )[0]
        self._gray = gray
        if result.status is CalibrationStatus.ACCEPTED:
            anchor = cast(CameraCalibration, self._anchor)
            result = replace(
                result,
                method="opencv_sparse_lk+correspondence_homography",
                provenance=(
                    f"{result.provenance};anchor_frame:{anchor.frame_index};"
                    f"anchor_age_frames:{frame.index - anchor.frame_index};"
                    "pitch_points:anchor_derived;forward_backward_px:1.0"
                ),
            )
            self._previous = result
        else:
            self.reset()
            result = replace(
                result,
                method="opencv_sparse_lk+correspondence_homography",
                abstention_reason=AbstentionReason.TEMPORAL_EVIDENCE_LOST,
            )
        return result
