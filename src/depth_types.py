"""Explicit model-output and camera-depth representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraDepth:
    """Positive Z-like values that may be passed to pinhole backprojection."""

    values: np.ndarray
    depth_type: str
    is_metric: bool
    representation: str
    model_name: str
    coordinate_units: str
    conversion: str
    alignment_method: str
    disparity_scale: float | None = None
    disparity_shift: float | None = None
    denominator_epsilon: float | None = None
    minimum_absolute_denominator: float | None = None
    rejected_small_denominator_count: int = 0
    rejected_nonfinite_denominator_count: int = 0
    rejected_invalid_z_count: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("camera-depth values must be a non-empty HxW array")
        if np.isinf(values).any():
            raise ValueError("invalid camera-depth samples must be NaN, not infinity")
        valid = np.isfinite(values)
        if valid.any() and np.any(values[valid] <= 0.0):
            raise ValueError("finite camera-depth values must be positive")


@dataclass(frozen=True)
class DepthPrediction:
    """Raw model prediction with semantics declared by the model configuration."""

    values: np.ndarray
    depth_type: str
    is_metric: bool
    representation: str
    model_name: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("depth prediction values must be a non-empty HxW array")
        if self.depth_type not in {"relative", "metric"}:
            raise ValueError("depth_type must be 'relative' or 'metric'")
        expected = "metric_depth" if self.is_metric else "relative_inverse_depth"
        if self.representation != expected:
            raise ValueError(f"representation must be '{expected}'")

    def to_camera_depth(
        self,
        *,
        disparity_scale: float | None = None,
        disparity_shift: float = 0.0,
        alignment_method: str = "none",
        denominator_epsilon: float = 1e-3,
    ) -> CameraDepth:
        """Create positive camera Z values without changing their stated units.

        Metric predictions already represent Z. Relative predictions are
        disparity-like and affine ambiguous. Their reciprocal is therefore only
        a relative-Z proxy. With no alignment, the median positive prediction is
        used as a nominal scale so a typical valid Z is near one.
        """
        raw = np.asarray(self.values, dtype=np.float64)
        if not np.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
            raise ValueError("denominator_epsilon must be finite and positive")

        if self.is_metric:
            if disparity_scale is not None or disparity_shift != 0.0:
                raise ValueError("disparity alignment cannot be applied to metric depth")
            valid = np.isfinite(raw) & (raw > 0.0)
            camera_z = np.full(raw.shape, np.nan, dtype=np.float64)
            camera_z[valid] = raw[valid]
            return CameraDepth(
                values=camera_z.astype(np.float32),
                depth_type="metric",
                is_metric=True,
                representation="camera_z",
                model_name=self.model_name,
                coordinate_units="metres_model_prediction",
                conversion="direct_metric_depth",
                alignment_method="metric_model",
            )

        valid_raw = np.isfinite(raw) & (raw > 0.0)
        if not valid_raw.any():
            raise ValueError("relative inverse-depth prediction has no usable values")
        if disparity_scale is None:
            scale = float(np.median(raw[valid_raw]))
            conversion = "reciprocal_median_normalized_zero_shift_assumption"
        else:
            scale = float(disparity_scale)
            conversion = "reciprocal_affine_disparity_alignment"
        shift = float(disparity_shift)
        if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(shift):
            raise ValueError("disparity scale must be positive and scale/shift finite")

        denominator = raw - shift
        finite_denominator = np.isfinite(denominator)
        finite_values = denominator[finite_denominator]
        minimum_absolute = (
            float(np.min(np.abs(finite_values))) if finite_values.size else None
        )
        small_denominator = finite_denominator & (
            denominator <= denominator_epsilon
        )
        valid = finite_denominator & (denominator > denominator_epsilon)
        camera_z = np.full(raw.shape, np.nan, dtype=np.float64)
        camera_z[valid] = scale / denominator[valid]
        invalid_z = valid & (~np.isfinite(camera_z) | (camera_z <= 0.0))
        camera_z[invalid_z] = np.nan
        if not np.isfinite(camera_z).any():
            raise ValueError("inverse-depth conversion produced no usable camera Z values")
        return CameraDepth(
            values=camera_z.astype(np.float32),
            depth_type="relative",
            is_metric=False,
            representation="relative_camera_z_proxy",
            model_name=self.model_name,
            coordinate_units="relative_depth_units",
            conversion=conversion,
            alignment_method=alignment_method,
            disparity_scale=scale,
            disparity_shift=shift,
            denominator_epsilon=float(denominator_epsilon),
            minimum_absolute_denominator=minimum_absolute,
            rejected_small_denominator_count=int(np.count_nonzero(small_denominator)),
            rejected_nonfinite_denominator_count=int(
                np.count_nonzero(~finite_denominator)
            ),
            rejected_invalid_z_count=int(np.count_nonzero(invalid_z)),
        )
