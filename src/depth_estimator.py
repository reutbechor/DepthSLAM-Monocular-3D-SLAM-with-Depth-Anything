"""Reusable wrapper around Depth Anything V2 inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .depth_types import DepthPrediction
from .visualization import colorize_depth

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Estimate relative depth from OpenCV BGR images."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing inference dependency '{exc.name}'. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self.device = self._resolve_device(device, torch)
        self._torch = torch
        self._image_class = Image
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load model '{model_name}'. Check the model ID and internet "
                "connection, or ensure its weights are in the Hugging Face cache."
            ) from exc
        self.model.to(self.device).eval()
        depth_type = getattr(self.model.config, "depth_estimation_type", None)
        if depth_type not in {"relative", "metric"}:
            raise RuntimeError(
                "The loaded model does not declare a supported depth_estimation_type"
            )
        self.depth_type = depth_type
        self.is_metric = depth_type == "metric"
        self.representation = (
            "metric_depth" if self.is_metric else "relative_inverse_depth"
        )

    @staticmethod
    def _resolve_device(requested: str, torch_module: Any) -> str:
        requested = requested.lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a CUDA device. "
                "Use --device cpu or --device auto."
            )
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        return requested

    def predict_result(self, image: np.ndarray) -> DepthPrediction:
        """Return a typed raw model prediction for an OpenCV BGR image."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HxWx3 NumPy array in OpenCV BGR format")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = self._image_class.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self._torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth
            resized = self._torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
        values = resized.squeeze().cpu().numpy().astype(np.float32)
        return DepthPrediction(
            values=values,
            depth_type=self.depth_type,
            is_metric=self.is_metric,
            representation=self.representation,
            model_name=self.model_name,
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return raw model values for backward-compatible Stage 1 callers.

        For the default relative model these values are disparity-like, not
        camera Z. Geometry code must use ``predict_result`` and an explicit
        conversion to ``CameraDepth``.
        """
        return self.predict_result(image).values

    def predict_visualization(self, image: np.ndarray) -> np.ndarray:
        """Return a colorized BGR visualization of relative depth."""
        return colorize_depth(self.predict_result(image).values)
