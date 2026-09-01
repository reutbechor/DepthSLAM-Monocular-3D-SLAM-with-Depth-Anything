"""Optional robust rejection of unstable aligned-depth tails.

The stabilizer is deliberately downstream of the existing scale/shift fit.  It
never changes alignment parameters or finite depth values: accepted tail
samples are replaced by NaN only in a copied depth array used for experimental
point-cloud fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .depth_types import CameraDepth, DepthPrediction


@dataclass(frozen=True)
class DepthStabilizationConfig:
    """Conservative relative/non-metric Z-tail rejection settings."""

    enabled: bool = False
    max_z_over_median: float = 12.0
    mad_multiplier: float = 8.0
    minimum_valid_ratio: float = 0.70
    maximum_removed_ratio: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_z_over_median) or self.max_z_over_median <= 1:
            raise ValueError("max_z_over_median must be finite and greater than 1")
        if not np.isfinite(self.mad_multiplier) or self.mad_multiplier <= 0:
            raise ValueError("mad_multiplier must be finite and positive")
        for name, value in (
            ("minimum_valid_ratio", self.minimum_valid_ratio),
            ("maximum_removed_ratio", self.maximum_removed_ratio),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range 0..1")


@dataclass(frozen=True)
class DepthStabilizationDiagnostics:
    """Per-frame evidence from the raw and optional stabilized depth maps."""

    depth_stabilization_attempted: bool
    depth_stabilization_accepted: bool
    depth_stabilization_reason: str
    raw_alignment_a: float | None
    raw_alignment_b: float | None
    denominator_min: float | None
    denominator_p01: float | None
    denominator_p05: float | None
    denominator_median: float | None
    denominator_p95: float | None
    fraction_denominator_below_1pct_median: float | None
    fraction_denominator_below_5pct_median: float | None
    raw_valid_point_count: int
    stabilized_valid_point_count: int
    stabilization_candidate_removed_count: int
    stabilization_candidate_removed_ratio: float
    stabilization_removed_count: int
    stabilization_removed_ratio: float
    raw_z_median: float | None
    raw_z_p95: float | None
    raw_z_p99: float | None
    raw_z_max: float | None
    raw_z_p99_over_median: float | None
    stabilized_z_median: float | None
    stabilized_z_p95: float | None
    stabilized_z_p99: float | None
    stabilized_z_max: float | None
    stabilized_z_p99_over_median: float | None
    median_z_ratio_to_previous: float | None
    p95_z_ratio_to_previous: float | None
    robust_upper_z_limit: float | None
    ratio_upper_z_limit: float | None
    mad_upper_z_limit: float | None


@dataclass(frozen=True)
class DepthStabilizationResult:
    """A copied fusion depth map plus diagnostics; raw input is preserved."""

    stabilized_depth: CameraDepth
    diagnostics: DepthStabilizationDiagnostics


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _z_statistics(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array) & (array > 0.0)]
    if valid.size == 0:
        return {name: None for name in ("median", "p95", "p99", "max", "ratio")}
    median, p95, p99 = np.percentile(valid, (50.0, 95.0, 99.0))
    return {
        "median": float(median),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(np.max(valid)),
        "ratio": _safe_ratio(float(p99), float(median)),
    }


def robust_z_tail_keep_mask(
    values: np.ndarray,
    config: DepthStabilizationConfig,
) -> tuple[np.ndarray, float | None, float | None, float | None]:
    """Return a mask that rejects, rather than clamps, extreme positive Z.

    The larger of the ratio and MAD bounds is used.  A point must exceed both
    conservative estimates before it is considered an unstable tail sample.
    """

    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array) & (array > 0.0)
    keep = valid.copy()
    finite_positive = array[valid]
    if finite_positive.size == 0:
        return keep, None, None, None
    median = float(np.median(finite_positive))
    mad = float(np.median(np.abs(finite_positive - median)))
    ratio_limit = median * config.max_z_over_median
    mad_limit = median + config.mad_multiplier * 1.4826 * mad
    upper_limit = max(ratio_limit, mad_limit)
    keep[valid & (array > upper_limit)] = False
    return keep, float(upper_limit), float(ratio_limit), float(mad_limit)


def _denominator_statistics(
    prediction: DepthPrediction,
    camera_depth: CameraDepth,
    sample_stride: int,
) -> dict[str, float | None]:
    names = (
        "minimum", "p01", "p05", "median", "p95",
        "below_1pct_median", "below_5pct_median",
    )
    empty = {name: None for name in names}
    if prediction.is_metric or camera_depth.disparity_shift is None:
        return empty
    raw = np.asarray(prediction.values, dtype=np.float64)[::sample_stride, ::sample_stride]
    denominator = raw - float(camera_depth.disparity_shift)
    usable = np.isfinite(raw) & (raw > 0.0) & np.isfinite(denominator)
    values = denominator[usable]
    if values.size == 0:
        return empty
    minimum, p01, p05, median, p95 = (
        float(value) for value in np.percentile(values, (0.0, 1.0, 5.0, 50.0, 95.0))
    )
    if median > 0.0:
        below_one = float(np.mean(values < 0.01 * median))
        below_five = float(np.mean(values < 0.05 * median))
    else:
        below_one = None
        below_five = None
    return {
        "minimum": minimum,
        "p01": p01,
        "p05": p05,
        "median": median,
        "p95": p95,
        "below_1pct_median": below_one,
        "below_5pct_median": below_five,
    }


def stabilize_aligned_depth(
    prediction: DepthPrediction,
    camera_depth: CameraDepth,
    config: DepthStabilizationConfig,
    *,
    sample_stride: int = 1,
    previous_median_z: float | None = None,
    previous_p95_z: float | None = None,
) -> DepthStabilizationResult:
    """Analyze raw aligned Z and optionally mask its unstable positive tail."""

    if isinstance(sample_stride, bool) or not isinstance(
        sample_stride, (int, np.integer)
    ) or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    if prediction.values.shape != camera_depth.values.shape:
        raise ValueError("prediction and aligned depth shapes must match")

    raw_full = np.asarray(camera_depth.values)
    raw_sampled = raw_full[::sample_stride, ::sample_stride]
    raw_valid = np.isfinite(raw_sampled) & (raw_sampled > 0.0)
    raw_count = int(np.count_nonzero(raw_valid))
    raw_stats = _z_statistics(raw_sampled)
    denominator = _denominator_statistics(prediction, camera_depth, sample_stride)

    keep_sampled, upper_limit, ratio_limit, mad_limit = robust_z_tail_keep_mask(
        raw_sampled, config
    )
    candidate_removed = raw_valid & ~keep_sampled
    candidate_removed_count = int(np.count_nonzero(candidate_removed))
    candidate_removed_ratio = (
        float(candidate_removed_count / raw_count) if raw_count else 0.0
    )
    candidate_valid_count = raw_count - candidate_removed_count
    candidate_valid_ratio = float(candidate_valid_count / raw_sampled.size)

    attempted = bool(config.enabled and not camera_depth.is_metric)
    accepted = False
    reason = "depth_stabilization_disabled"
    if camera_depth.is_metric and config.enabled:
        reason = "depth_stabilization_not_applicable_metric_depth"
    elif attempted and raw_count == 0:
        reason = "depth_stabilization_no_valid_depth"
    elif attempted and candidate_removed_ratio > config.maximum_removed_ratio:
        reason = "depth_stabilization_fallback_excessive_removal"
    elif attempted and candidate_valid_ratio < config.minimum_valid_ratio:
        reason = "depth_stabilization_fallback_insufficient_valid_ratio"
    elif attempted:
        accepted = True
        reason = (
            "depth_stabilization_accepted_tail_removed"
            if candidate_removed_count
            else "depth_stabilization_accepted_no_tail"
        )

    if accepted:
        keep_full, _, _, _ = robust_z_tail_keep_mask(raw_full, config)
        stabilized_values = raw_full.copy()
        stabilized_values[~keep_full] = np.nan
        stabilized_depth = replace(camera_depth, values=stabilized_values)
        stabilized_sampled = stabilized_values[::sample_stride, ::sample_stride]
        applied_removed_count = candidate_removed_count
        applied_removed_ratio = candidate_removed_ratio
    else:
        stabilized_depth = camera_depth
        stabilized_sampled = raw_sampled
        applied_removed_count = 0
        applied_removed_ratio = 0.0

    stabilized_stats = _z_statistics(stabilized_sampled)
    stabilized_count = int(np.count_nonzero(
        np.isfinite(stabilized_sampled) & (stabilized_sampled > 0.0)
    ))
    diagnostics = DepthStabilizationDiagnostics(
        depth_stabilization_attempted=attempted,
        depth_stabilization_accepted=accepted,
        depth_stabilization_reason=reason,
        raw_alignment_a=camera_depth.disparity_scale,
        raw_alignment_b=camera_depth.disparity_shift,
        denominator_min=denominator["minimum"],
        denominator_p01=denominator["p01"],
        denominator_p05=denominator["p05"],
        denominator_median=denominator["median"],
        denominator_p95=denominator["p95"],
        fraction_denominator_below_1pct_median=denominator["below_1pct_median"],
        fraction_denominator_below_5pct_median=denominator["below_5pct_median"],
        raw_valid_point_count=raw_count,
        stabilized_valid_point_count=stabilized_count,
        stabilization_candidate_removed_count=candidate_removed_count,
        stabilization_candidate_removed_ratio=candidate_removed_ratio,
        stabilization_removed_count=applied_removed_count,
        stabilization_removed_ratio=applied_removed_ratio,
        raw_z_median=raw_stats["median"],
        raw_z_p95=raw_stats["p95"],
        raw_z_p99=raw_stats["p99"],
        raw_z_max=raw_stats["max"],
        raw_z_p99_over_median=raw_stats["ratio"],
        stabilized_z_median=stabilized_stats["median"],
        stabilized_z_p95=stabilized_stats["p95"],
        stabilized_z_p99=stabilized_stats["p99"],
        stabilized_z_max=stabilized_stats["max"],
        stabilized_z_p99_over_median=stabilized_stats["ratio"],
        median_z_ratio_to_previous=_safe_ratio(
            raw_stats["median"], previous_median_z
        ),
        p95_z_ratio_to_previous=_safe_ratio(raw_stats["p95"], previous_p95_z),
        robust_upper_z_limit=upper_limit,
        ratio_upper_z_limit=ratio_limit,
        mad_upper_z_limit=mad_limit,
    )
    return DepthStabilizationResult(stabilized_depth, diagnostics)
