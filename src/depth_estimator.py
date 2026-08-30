"""Reusable wrapper around Depth Anything V2 inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

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

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an HxW float32 relative-depth map for a BGR image."""
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
        return resized.squeeze().cpu().numpy().astype(np.float32)

    def predict_visualization(self, image: np.ndarray) -> np.ndarray:
        """Return a colorized BGR visualization of relative depth."""
        return colorize_depth(self.predict(image))
