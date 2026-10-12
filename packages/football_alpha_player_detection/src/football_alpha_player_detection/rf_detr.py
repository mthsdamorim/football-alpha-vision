"""RF-DETR adapter for project-owned person detection contracts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from hashlib import file_digest
from importlib import import_module
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from football_alpha_player_detection.contracts import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
)
from football_alpha_video import DecodedFrame


class _RfDetrModel(Protocol):
    def predict(
        self,
        images: list[np.ndarray[tuple[int, ...], np.dtype[np.uint8]]],
        *,
        threshold: float,
        include_source_image: bool,
    ) -> object: ...


class _RfDetrPrediction(Protocol):
    xyxy: object
    confidence: object
    data: object


class _RfDetrFactory(Protocol):
    def __call__(self, *, device: str) -> _RfDetrModel: ...


class _RfDetrModule(Protocol):
    RFDETRLarge: _RfDetrFactory


class _MpsBackend(Protocol):
    def is_available(self) -> bool: ...


class _TorchBackends(Protocol):
    mps: _MpsBackend


class _TorchModule(Protocol):
    backends: _TorchBackends


class _ModelConfig(Protocol):
    pretrain_weights: object
    device: object


class _ModelWithConfig(Protocol):
    model_config: _ModelConfig


class RfDetrPersonDetector:
    """Normalize pretrained RF-DETR-L COCO person predictions."""

    def __init__(
        self,
        minimum_confidence: float = 0.10,
        *,
        device: str = "auto",
        _model: _RfDetrModel | None = None,
    ) -> None:
        if not isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be finite and between 0 and 1")

        self.minimum_confidence = minimum_confidence
        self.device = _resolve_device(device)
        self._model = _model if _model is not None else _load_model(self.device)

    def provenance(self) -> dict[str, str]:
        """Return reproducibility metadata for the loaded pretrained model."""
        if not hasattr(self._model, "model_config"):
            raise RuntimeError("RF-DETR model does not expose its checkpoint identity")
        model_config = cast(_ModelWithConfig, self._model).model_config
        checkpoint_value = model_config.pretrain_weights

        checkpoint = _resolve_checkpoint_path(checkpoint_value)
        with checkpoint.open("rb") as checkpoint_file:
            checkpoint_sha256 = file_digest(checkpoint_file, "sha256").hexdigest()

        actual_device = str(model_config.device)
        return {
            "rfdetr_version": version("rfdetr"),
            "model_name": "RFDETRLarge",
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "device": actual_device,
        }

    def detect(
        self,
        frames: Sequence[DecodedFrame],
    ) -> list[FramePersonDetections]:
        """Return normalized person detections in input-frame order."""
        if not frames:
            return []

        raw_results = self._model.predict(
            [frame.image for frame in frames],
            threshold=self.minimum_confidence,
            include_source_image=False,
        )
        if not isinstance(raw_results, Sequence):
            raise ValueError("RF-DETR did not return a result sequence")
        normalized_results = cast(Sequence[object], raw_results)
        if len(normalized_results) != len(frames):
            raise ValueError(
                "RF-DETR returned a result count different from the frame count"
            )

        return [
            _normalize_frame(frame, raw_result, self.minimum_confidence)
            for frame, raw_result in zip(frames, normalized_results, strict=True)
        ]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    torch = cast(_TorchModule, import_module("torch"))
    mps = torch.backends.mps
    return "mps" if bool(mps.is_available()) else "cpu"


def _load_model(device: str) -> _RfDetrModel:
    rfdetr = cast(_RfDetrModule, import_module("rfdetr"))
    return rfdetr.RFDETRLarge(device=device)


def _resolve_checkpoint_path(checkpoint_value: object) -> Path:
    if not isinstance(checkpoint_value, (str, os.PathLike)):
        raise RuntimeError("RF-DETR checkpoint identity is not path-like")
    checkpoint = Path(cast(str | os.PathLike[str], checkpoint_value)).expanduser()
    if not checkpoint.is_file():
        raise RuntimeError(f"RF-DETR checkpoint does not exist: {checkpoint}")
    return checkpoint


def _normalize_frame(
    frame: DecodedFrame,
    raw_result: object,
    minimum_confidence: float,
) -> FramePersonDetections:
    height, width = frame.image.shape[:2]
    try:
        prediction = cast(_RfDetrPrediction, raw_result)
        xyxy: NDArray[np.float64] = np.asarray(
            cast(ArrayLike, prediction.xyxy), dtype=np.float64
        )
        confidence: NDArray[np.float64] = np.asarray(
            cast(ArrayLike, prediction.confidence), dtype=np.float64
        )
        data = prediction.data
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("RF-DETR returned malformed detection fields") from error

    if not isinstance(data, Mapping) or "class_name" not in data:
        raise ValueError("RF-DETR result has no class_name data")
    try:
        raw_class_names = np.asarray(cast(ArrayLike, data["class_name"]))
    except (TypeError, ValueError) as error:
        raise ValueError("RF-DETR returned malformed class_name data") from error
    if any(
        not isinstance(class_name, (str, np.str_))
        for class_name in raw_class_names.flat
    ):
        raise ValueError("RF-DETR returned malformed class_name data")
    class_names: NDArray[np.str_] = raw_class_names.astype(np.str_, copy=False)

    if confidence.ndim != 1:
        raise ValueError("RF-DETR returned malformed detection fields")
    count = len(confidence)
    if xyxy.shape != (count, 4) or class_names.shape != (count,):
        raise ValueError("RF-DETR detection fields are misaligned")

    detections: list[PersonDetection] = []
    for box, score, class_name in zip(xyxy, confidence, class_names, strict=True):
        if class_name != "person":
            continue

        numeric_score = float(score)
        if not isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError("RF-DETR returned an invalid person confidence")
        if numeric_score < minimum_confidence:
            continue
        if not np.isfinite(box).all():
            raise ValueError("RF-DETR returned non-finite person box coordinates")

        x_min, y_min, x_max, y_max = (
            max(0.0, min(float(box[0]), width)),
            max(0.0, min(float(box[1]), height)),
            max(0.0, min(float(box[2]), width)),
            max(0.0, min(float(box[3]), height)),
        )
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("RF-DETR returned a degenerate person box")

        detections.append(
            PersonDetection(
                bbox=ImageBoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
                confidence=numeric_score,
            )
        )

    return FramePersonDetections(
        frame_index=frame.index,
        pts=frame.pts,
        time_base=frame.time_base,
        image_width=width,
        image_height=height,
        detections=tuple(detections),
    )
