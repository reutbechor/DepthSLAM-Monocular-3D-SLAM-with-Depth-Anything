"""Deterministic NumPy-only diagnostics and optional outlier suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATISTIC_NAMES = ("min", "p1", "p5", "median", "p95", "p99", "max")
STATISTIC_PERCENTILES = (0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0)


def distribution_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("at least one finite value is required for statistics")
    percentiles = np.percentile(array, STATISTIC_PERCENTILES)
    return {
        name: float(value) for name, value in zip(STATISTIC_NAMES, percentiles)
    }


@dataclass(frozen=True)
class DepthRangeFilterResult:
    keep_mask: np.ndarray
    statistics_before: dict[str, float]
    input_count: int
    rejected_count: int
    output_count: int
    method: str
    percentile_low: float | None
    percentile_high: float | None
    lower_bound: float | None
    upper_bound: float | None


def filter_depth_range(
    values: np.ndarray,
    *,
    is_metric: bool,
    percentile_low: float | None,
    percentile_high: float | None,
) -> DepthRangeFilterResult:
    """Optionally remove relative-Z percentile tails without clamping values."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("depth filter input must contain finite positive values")
    if (percentile_low is None) != (percentile_high is None):
        raise ValueError("both depth percentiles must be set or both omitted")

    statistics = distribution_statistics(array)
    if percentile_low is None or is_metric:
        method = "not_applied_metric" if is_metric else "none"
        return DepthRangeFilterResult(
            np.ones(array.shape[0], dtype=bool), statistics, array.size, 0,
            array.size, method, None, None, None, None,
        )

    low = float(percentile_low)
    high = float(percentile_high)
    if not np.isfinite([low, high]).all() or not 0.0 <= low < high <= 100.0:
        raise ValueError("depth percentiles must satisfy 0 <= low < high <= 100")
    lower, upper = (float(value) for value in np.percentile(array, (low, high)))
    keep = (array >= lower) & (array <= upper)
    return DepthRangeFilterResult(
        keep_mask=keep,
        statistics_before=statistics,
        input_count=array.size,
        rejected_count=int(np.count_nonzero(~keep)),
        output_count=int(np.count_nonzero(keep)),
        method="relative_z_percentile",
        percentile_low=low,
        percentile_high=high,
        lower_bound=lower,
        upper_bound=upper,
    )


def coordinate_statistics(points: np.ndarray) -> dict[str, dict[str, float]]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.shape[0] == 0:
        raise ValueError("points must be a non-empty Nx3 array")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    return {
        axis: distribution_statistics(array[:, index])
        for index, axis in enumerate(("x", "y", "z"))
    }


@dataclass(frozen=True)
class GlobalOutlierFilterResult:
    points: np.ndarray
    colors: np.ndarray
    input_count: int
    rejected_count: int
    output_count: int
    method: str
    percentile: float | None
    distance_threshold: float | None
    robust_center: np.ndarray
    distance_statistics: dict[str, float]
    diagnostic_robust_radius: float
    points_outside_diagnostic_radius: int
    coordinate_statistics_before: dict[str, dict[str, float]]
    coordinate_statistics_after: dict[str, dict[str, float]]


def filter_global_radius(
    points: np.ndarray,
    colors: np.ndarray,
    percentile: float | None,
) -> GlobalOutlierFilterResult:
    """Filter by distance from coordinate-wise median at a high percentile."""
    point_array = np.asarray(points, dtype=np.float64)
    color_array = np.asarray(colors)
    if point_array.ndim != 2 or point_array.shape[1:] != (3,) or not point_array.size:
        raise ValueError("points must be a non-empty Nx3 array")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    if color_array.shape != point_array.shape or color_array.dtype != np.uint8:
        raise ValueError("colors must be matching Nx3 uint8 RGB values")

    center = np.median(point_array, axis=0)
    distances = np.linalg.norm(point_array - center, axis=1)
    distance_stats = distribution_statistics(distances)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    robust_radius = median_distance + 6.0 * 1.4826 * mad
    outside = int(np.count_nonzero(distances > robust_radius))
    before = coordinate_statistics(point_array)

    if percentile is None:
        keep = np.ones(point_array.shape[0], dtype=bool)
        threshold = None
        method = "none"
        configured = None
    else:
        configured = float(percentile)
        if not np.isfinite(configured) or not 0.0 < configured <= 100.0:
            raise ValueError("global outlier percentile must satisfy 0 < p <= 100")
        threshold = float(np.percentile(distances, configured))
        keep = distances <= threshold
        method = "median_center_distance_percentile"

    filtered_points = point_array[keep]
    filtered_colors = color_array[keep]
    if filtered_points.shape[0] == 0:
        raise ValueError("global outlier filter removed every point")
    return GlobalOutlierFilterResult(
        points=filtered_points,
        colors=filtered_colors,
        input_count=point_array.shape[0],
        rejected_count=int(np.count_nonzero(~keep)),
        output_count=filtered_points.shape[0],
        method=method,
        percentile=configured,
        distance_threshold=threshold,
        robust_center=center,
        distance_statistics=distance_stats,
        diagnostic_robust_radius=robust_radius,
        points_outside_diagnostic_radius=outside,
        coordinate_statistics_before=before,
        coordinate_statistics_after=coordinate_statistics(filtered_points),
    )
