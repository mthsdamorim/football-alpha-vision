"""Abstention-aware image-to-pitch geometry for football-alpha."""

from football_alpha_pitch_projection.calibration import calibrate_frames
from football_alpha_pitch_projection.contracts import (
    AbstentionReason,
    CalibrationStatus,
    CameraCalibration,
    CameraCalibrator,
    FrameCalibrationInput,
    Homography,
    ImagePitchCorrespondence,
    ImagePoint,
    PitchDimensions,
    PitchPoint,
    PointProjection,
    ProjectionFailureReason,
)
from football_alpha_pitch_projection.correspondence import (
    CorrespondenceHomographyCalibrator,
)
from football_alpha_pitch_projection.projection import project_image_point
from football_alpha_pitch_projection.temporal import OpticalFlowCalibrator

__version__ = "0.1.0"

__all__ = [
    "AbstentionReason",
    "CalibrationStatus",
    "CameraCalibration",
    "CameraCalibrator",
    "CorrespondenceHomographyCalibrator",
    "FrameCalibrationInput",
    "Homography",
    "ImagePitchCorrespondence",
    "ImagePoint",
    "OpticalFlowCalibrator",
    "PitchDimensions",
    "PitchPoint",
    "PointProjection",
    "ProjectionFailureReason",
    "calibrate_frames",
    "project_image_point",
]
