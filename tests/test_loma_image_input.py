import numpy as np
import torch

from flatsam2.config import SamplingCfg
from flatsam2.loma_tracker import LoMaTracker


class _FakeDetector:
    def __init__(self) -> None:
        self.keep_aspect_ratio = True
        self.resize = 16
        self.detect_image: torch.Tensor | None = None
        self.forward_image: torch.Tensor | None = None

    def load_image(self, _path: str) -> torch.Tensor:
        raise AssertionError("array path must not call detector.load_image")

    def detect(
        self,
        image: torch.Tensor,
        *,
        num_keypoints: int,
    ) -> dict[str, torch.Tensor]:
        self.detect_image = image
        keypoints = torch.zeros(1, num_keypoints, 2, device=image.device)
        return {"keypoints": keypoints}

    def forward_impl(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_image = image
        _, _, H, W = image.shape
        values = torch.arange(H * W, dtype=torch.float32, device=image.device)
        return values.reshape(1, 1, H, W)


class _FakeDescriptor:
    def __init__(self) -> None:
        self.image: torch.Tensor | None = None
        self.keypoints: torch.Tensor | None = None

    def read_image(self, _path: str) -> torch.Tensor:
        raise AssertionError("array path must not call descriptor.read_image")

    def describe_keypoints(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.image = image
        self.keypoints = keypoints
        descriptions = torch.ones(
            keypoints.shape[0],
            keypoints.shape[1],
            4,
            dtype=image.dtype,
            device=image.device,
        )
        return {"descriptions": descriptions}


class _FakeCfg:
    num_keypoints = 2
    filter_threshold = 0.01


class _FakeModel:
    def __init__(self) -> None:
        self._detector = _FakeDetector()
        self._descriptor = _FakeDescriptor()
        self.cfg = _FakeCfg()
        self._param = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self._param


def test_get_keypoints_from_bgr_array_uses_in_memory_loma_paths() -> None:
    model = _FakeModel()
    tracker = LoMaTracker(model)
    image = np.zeros((8, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    feats = tracker.get_keypoints(image)

    assert feats.h_original == 8
    assert feats.w_original == 4
    assert model._detector.detect_image is not None
    assert model._detector.detect_image.shape == (1, 3, 16, 8)
    assert model._descriptor.image is not None
    assert model._descriptor.image.shape == (1, 3, 784, 784)
    assert torch.isclose(
        model._detector.detect_image[0, 0, 0, 0],
        torch.tensor(30 / 255),
    )
    assert torch.isclose(
        model._detector.detect_image[0, 2, 0, 0],
        torch.tensor(10 / 255),
    )
    assert feats.kpts.shape == (1, 2, 2)
    assert feats.desc.shape == (1, 2, 4)


def test_masked_get_keypoints_from_array_uses_descriptor_tensor() -> None:
    model = _FakeModel()
    tracker = LoMaTracker(model)
    image = np.full((8, 4, 3), 128, dtype=np.uint8)
    mask = np.ones((8, 4), dtype=np.uint8)
    cfg = SamplingCfg(n_object=3, total_budget=3, nms_size=1, subpixel=False)

    feats = tracker.get_keypoints(image, mask=mask, cfg=cfg)

    assert model._detector.forward_image is not None
    assert model._detector.forward_image.shape == (1, 3, 16, 8)
    assert model._descriptor.image is not None
    assert model._descriptor.image.shape == (1, 3, 784, 784)
    assert model._descriptor.keypoints is feats.kpts
    assert feats.h_internal == 16
    assert feats.w_internal == 8
    assert feats.h_original == 8
    assert feats.w_original == 4
    assert feats.is_object is not None
    assert int(feats.is_object.sum()) == 3
