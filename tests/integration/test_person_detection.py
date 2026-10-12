"""Behavior tests for normalized person detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from fractions import Fraction
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from football_alpha_player_detection import (
    FramePersonDetections,
    ImageBoundingBox,
    PersonDetection,
    PersonDetector,
    RfDetrPersonDetector,
)
from football_alpha_video import DecodedFrame


class OrderedDetector:
    """Small contract implementation used to exercise the public protocol."""

    def detect(self, frames: Sequence[DecodedFrame]) -> list[FramePersonDetections]:
        return [
            FramePersonDetections(
                frame_index=frame.index,
                pts=frame.pts,
                time_base=frame.time_base,
                image_width=frame.image.shape[1],
                image_height=frame.image.shape[0],
                detections=(),
            )
            for frame in frames
        ]


class FakeVendorDetections:
    def __init__(
        self,
        *,
        xyxy: list[list[float]],
        confidence: list[float],
        class_name: list[str],
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.data = {"class_name": np.asarray(class_name, dtype=np.str_)}


class FakeRfDetrModel:
    def __init__(self, outputs: object) -> None:
        self.outputs = outputs
        self.images: list[np.ndarray[tuple[int, ...], np.dtype[np.uint8]]] = []
        self.threshold: float | None = None
        self.include_source_image: bool | None = None

    def predict(
        self,
        images: list[np.ndarray[tuple[int, ...], np.dtype[np.uint8]]],
        *,
        threshold: float,
        include_source_image: bool,
    ) -> object:
        self.images = images
        self.threshold = threshold
        self.include_source_image = include_source_image
        return self.outputs


def _frame(index: int = 0) -> DecodedFrame:
    return DecodedFrame(
        index=index,
        pts=index,
        time_base=Fraction(1, 30),
        image=np.zeros((10, 20, 3), dtype=np.uint8),
    )


def test_person_detector_contract_preserves_frame_order_and_timing() -> None:
    frames = [
        DecodedFrame(
            index=index,
            pts=index * 512,
            time_base=Fraction(1, 15_360),
            image=np.zeros((10, 20, 3), dtype=np.uint8),
        )
        for index in (3, 4)
    ]
    detector: PersonDetector = OrderedDetector()

    results = detector.detect(frames)

    assert [result.frame_index for result in results] == [3, 4]
    assert [result.pts for result in results] == [1_536, 2_048]
    assert all(result.time_base == Fraction(1, 15_360) for result in results)
    assert all(result.image_width == 20 for result in results)
    assert all(result.image_height == 10 for result in results)


def test_detection_values_are_immutable() -> None:
    box = ImageBoundingBox(x_min=1.0, y_min=2.0, x_max=10.0, y_max=9.0)
    detection = PersonDetection(bbox=box, confidence=0.75)

    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.5  # type: ignore[misc]

    mutable_detections = [detection]
    frame_detections = FramePersonDetections(
        frame_index=0,
        pts=0,
        time_base=Fraction(1, 30),
        image_width=20,
        image_height=10,
        detections=mutable_detections,  # type: ignore[arg-type]
    )
    mutable_detections.clear()

    assert frame_detections.detections == (detection,)


def test_detection_contract_rejects_values_outside_image_bounds() -> None:
    with pytest.raises(ValueError, match="non-negative finite coordinates"):
        ImageBoundingBox(x_min=-1.0, y_min=2.0, x_max=10.0, y_max=9.0)
    with pytest.raises(ValueError, match="positive width and height"):
        ImageBoundingBox(x_min=2.0, y_min=2.0, x_max=2.0, y_max=9.0)

    box = ImageBoundingBox(x_min=1.0, y_min=2.0, x_max=21.0, y_max=9.0)
    with pytest.raises(ValueError, match="confidence"):
        PersonDetection(bbox=box, confidence=1.1)

    detection = PersonDetection(bbox=box, confidence=0.8)
    with pytest.raises(ValueError, match="outside the source image"):
        FramePersonDetections(
            frame_index=0,
            pts=0,
            time_base=Fraction(1, 30),
            image_width=20,
            image_height=10,
            detections=(detection,),
        )
    with pytest.raises(ValueError, match="positive"):
        FramePersonDetections(
            frame_index=0,
            pts=0,
            time_base=Fraction(1, 30),
            image_width=0,
            image_height=10,
            detections=(),
        )


def test_rf_detr_adapter_normalizes_person_detections_in_frame_order() -> None:
    frames = [_frame(index) for index in (7, 8)]
    model = FakeRfDetrModel(
        [
            FakeVendorDetections(
                xyxy=[[-2.0, 2.0, 25.0, 9.0], [1.0, 1.0, 4.0, 4.0]],
                confidence=[0.8, 0.9],
                class_name=["person", "dog"],
            ),
            FakeVendorDetections(xyxy=[], confidence=[], class_name=[]),
        ]
    )
    detector = RfDetrPersonDetector(
        minimum_confidence=0.10,
        device="cpu",
        _model=model,
    )

    results = detector.detect(frames)

    assert [result.frame_index for result in results] == [7, 8]
    assert len(results[0].detections) == 1
    assert results[0].detections[0].bbox == ImageBoundingBox(
        x_min=0.0,
        y_min=2.0,
        x_max=20.0,
        y_max=9.0,
    )
    assert results[0].detections[0].confidence == pytest.approx(0.8)
    assert results[1].detections == ()
    assert all(
        received is frame.image
        for received, frame in zip(model.images, frames, strict=True)
    )
    assert model.threshold == pytest.approx(0.10)
    assert model.include_source_image is False


def test_rf_detr_adapter_keeps_low_threshold_policy_out_of_tracking() -> None:
    model = FakeRfDetrModel(
        [
            FakeVendorDetections(
                xyxy=[[1.0, 1.0, 4.0, 4.0]],
                confidence=[0.05],
                class_name=["person"],
            )
        ]
    )
    detector = RfDetrPersonDetector(
        minimum_confidence=0.10,
        device="cpu",
        _model=model,
    )

    assert detector.detect([_frame()])[0].detections == ()
    assert detector.detect([]) == []


@pytest.mark.parametrize("minimum_confidence", [-0.1, 1.1, float("nan")])
def test_rf_detr_adapter_rejects_invalid_thresholds(
    minimum_confidence: float,
) -> None:
    with pytest.raises(ValueError, match="minimum_confidence"):
        RfDetrPersonDetector(
            minimum_confidence, device="cpu", _model=FakeRfDetrModel([])
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("non_sequence", "result sequence"),
        ("wrong_count", "result count"),
        ("missing_fields", "malformed detection fields"),
        ("non_mapping_data", "class_name"),
        ("missing_class_name", "class_name"),
        ("malformed_class_name", "malformed class_name"),
        ("unreadable_class_name", "malformed class_name"),
        ("misaligned_box", "misaligned"),
        ("misaligned_class", "misaligned"),
        ("scalar_confidence", "malformed detection fields"),
        ("invalid_confidence", "invalid person confidence"),
        ("non_finite_box", "non-finite"),
        ("degenerate_box", "degenerate"),
    ],
)
def test_rf_detr_adapter_rejects_malformed_vendor_outputs(
    case: str,
    message: str,
) -> None:
    result: object
    if case == "non_sequence":
        outputs: object = object()
    elif case == "wrong_count":
        outputs = []
    elif case == "missing_fields":
        outputs = [object()]
    else:
        result = FakeVendorDetections(
            xyxy=[[1.0, 1.0, 4.0, 4.0]],
            confidence=[0.8],
            class_name=["person"],
        )
        if case == "non_mapping_data":
            result.data = object()  # type: ignore[attr-defined]
        elif case == "missing_class_name":
            result.data = {}  # type: ignore[attr-defined]
        elif case == "malformed_class_name":
            result.data = {"class_name": np.asarray([object()], dtype=object)}  # type: ignore[attr-defined]
        elif case == "unreadable_class_name":

            class UnreadableArray:
                def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
                    raise TypeError("unreadable")

            result.data = {"class_name": UnreadableArray()}  # type: ignore[attr-defined]
        elif case == "misaligned_box":
            result.xyxy = np.asarray([1.0, 2.0])  # type: ignore[attr-defined]
        elif case == "misaligned_class":
            result.data = {"class_name": np.asarray([], dtype=np.str_)}  # type: ignore[attr-defined]
        elif case == "scalar_confidence":
            result.confidence = np.asarray(0.8)  # type: ignore[attr-defined]
        elif case == "invalid_confidence":
            result.confidence = np.asarray([1.1])  # type: ignore[attr-defined]
        elif case == "non_finite_box":
            result.xyxy = np.asarray([[1.0, 1.0, float("nan"), 4.0]])  # type: ignore[attr-defined]
        elif case == "degenerate_box":
            result.xyxy = np.asarray([[30.0, 1.0, 40.0, 4.0]])  # type: ignore[attr-defined]
        outputs = [result]

    detector = RfDetrPersonDetector(
        minimum_confidence=0.10,
        device="cpu",
        _model=FakeRfDetrModel(outputs),
    )
    with pytest.raises(ValueError, match=message):
        detector.detect([_frame()])


@pytest.mark.parametrize(
    ("mps_available", "expected_device"), [(True, "mps"), (False, "cpu")]
)
def test_rf_detr_adapter_resolves_auto_device_and_loads_large_model(
    monkeypatch: pytest.MonkeyPatch,
    mps_available: bool,
    expected_device: str,
) -> None:
    created_devices: list[str] = []

    class MpsBackend:
        def is_available(self) -> bool:
            return mps_available

    class Backends:
        mps = MpsBackend()

    class TorchModule:
        backends = Backends()

    class LargeFactory:
        def __call__(self, *, device: str) -> FakeRfDetrModel:
            created_devices.append(device)
            return FakeRfDetrModel([])

    class RfDetrModule:
        RFDETRLarge = LargeFactory()

    def import_fake_module(name: str) -> object:
        return TorchModule() if name == "torch" else RfDetrModule()

    monkeypatch.setattr(
        "football_alpha_player_detection.rf_detr.import_module",
        import_fake_module,
    )

    detector = RfDetrPersonDetector(device="auto")

    assert detector.device == expected_device
    assert created_devices == [expected_device]


def test_rf_detr_adapter_records_checkpoint_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rf-detr-large.pth"
    checkpoint.write_bytes(b"checkpoint")

    class ModelConfig:
        pretrain_weights: object = checkpoint
        device: object = "cpu"

    model = FakeRfDetrModel([])
    model.model_config = ModelConfig()  # type: ignore[attr-defined]
    detector = RfDetrPersonDetector(device="cpu", _model=model)

    provenance = detector.provenance()

    with checkpoint.open("rb") as checkpoint_file:
        expected_checksum = file_digest(checkpoint_file, "sha256").hexdigest()
    assert provenance == {
        "rfdetr_version": version("rfdetr"),
        "model_name": "RFDETRLarge",
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": expected_checksum,
        "device": "cpu",
    }


@pytest.mark.parametrize("checkpoint", [object(), "missing-checkpoint.pth"])
def test_rf_detr_adapter_rejects_unverifiable_checkpoint_provenance(
    checkpoint: object,
) -> None:
    class ModelConfig:
        pretrain_weights: object = checkpoint
        device: object = "cpu"

    model = FakeRfDetrModel([])
    model.model_config = ModelConfig()  # type: ignore[attr-defined]
    detector = RfDetrPersonDetector(device="cpu", _model=model)

    with pytest.raises(RuntimeError, match="checkpoint"):
        detector.provenance()


def test_rf_detr_adapter_requires_model_config_for_provenance() -> None:
    detector = RfDetrPersonDetector(device="cpu", _model=FakeRfDetrModel([]))

    with pytest.raises(RuntimeError, match="checkpoint identity"):
        detector.provenance()
