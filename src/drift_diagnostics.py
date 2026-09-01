"""Diagnostic-only summaries and plots for accepted-keyframe drift signals.

Nothing in this module feeds values back into mapping.  It consumes already
recorded frame statistics and writes descriptive relative/non-metric evidence.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DriftDiagnosticRow:
    frame_index: int
    keyframe_sequence_index: int
    depth_alignment_scale_a: float | None
    depth_alignment_shift_b: float | None
    aligned_z_min: float | None
    aligned_z_p01: float | None
    aligned_z_p05: float | None
    aligned_z_median: float | None
    aligned_z_p95: float | None
    aligned_z_p99: float | None
    aligned_z_max: float | None
    valid_depth_ratio: float
    denominator_reject_ratio: float
    relative_translation_x: float | None
    relative_translation_y: float | None
    relative_translation_z: float | None
    relative_translation_magnitude: float | None
    cumulative_position_x: float
    cumulative_position_y: float
    cumulative_position_z: float
    cumulative_distance_from_origin: float
    relative_rotation_deg: float | None
    cumulative_rotation_deg: float | None
    geometric_inlier_ratio: float
    pnp_inlier_ratio: float
    reprojection_rmse: float | None
    depth_alignment_inlier_ratio: float
    refinement_3d_attempted: bool
    refinement_3d_accepted: bool
    baseline_3d_residual_median: float | None
    refined_3d_residual_median: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["3d_refinement_attempted"] = result.pop(
            "refinement_3d_attempted"
        )
        result["3d_refinement_accepted"] = result.pop(
            "refinement_3d_accepted"
        )
        return result


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _field(item: Any, name: str, fallback: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, fallback)
    return getattr(item, name, fallback)


def _z_value(item: Any, statistic: str, fallback_field: str | None = None) -> float | None:
    statistics = _field(item, "z_statistics")
    if isinstance(statistics, dict) and statistic in statistics:
        return _finite_or_none(statistics[statistic])
    if fallback_field is not None:
        return _finite_or_none(_field(item, fallback_field))
    return None


def collect_drift_diagnostics(
    frame_statistics: Iterable[Any],
) -> list[DriftDiagnosticRow]:
    """Create one immutable row per accepted keyframe, in accepted order."""

    rows: list[DriftDiagnosticRow] = []
    for item in frame_statistics:
        if not bool(_field(item, "accepted", False)):
            continue
        sequence = len(rows)
        position = _field(item, "camera_position")
        if position is None:
            if sequence == 0:
                position_array = np.zeros(3, dtype=np.float64)
            else:
                raise ValueError("accepted non-origin frame is missing camera_position")
        else:
            position_array = np.asarray(position, dtype=np.float64).reshape(-1)
            if position_array.shape != (3,) or not np.isfinite(position_array).all():
                raise ValueError("accepted camera_position must contain three finite values")
        relative = _field(item, "relative_translation")
        if relative is None:
            relative_array: np.ndarray | None = (
                np.zeros(3, dtype=np.float64) if sequence == 0 else None
            )
        else:
            relative_array = np.asarray(relative, dtype=np.float64).reshape(-1)
            if relative_array.shape != (3,) or not np.isfinite(relative_array).all():
                relative_array = None

        rows.append(DriftDiagnosticRow(
            frame_index=int(_field(item, "frame_index")),
            keyframe_sequence_index=sequence,
            depth_alignment_scale_a=_finite_or_none(
                _field(item, "disparity_scale")
            ),
            depth_alignment_shift_b=_finite_or_none(
                _field(item, "disparity_shift")
            ),
            aligned_z_min=_z_value(item, "min"),
            aligned_z_p01=_z_value(item, "p1", "aligned_z_p1"),
            aligned_z_p05=_z_value(item, "p5"),
            aligned_z_median=_z_value(
                item, "median", "aligned_z_median"
            ),
            aligned_z_p95=_z_value(item, "p95"),
            aligned_z_p99=_z_value(item, "p99", "aligned_z_p99"),
            aligned_z_max=_z_value(item, "max"),
            valid_depth_ratio=float(_field(item, "valid_aligned_depth_ratio", 0.0)),
            denominator_reject_ratio=float(
                _field(item, "denominator_rejection_ratio", 0.0)
            ),
            relative_translation_x=(
                None if relative_array is None else float(relative_array[0])
            ),
            relative_translation_y=(
                None if relative_array is None else float(relative_array[1])
            ),
            relative_translation_z=(
                None if relative_array is None else float(relative_array[2])
            ),
            relative_translation_magnitude=(
                _finite_or_none(_field(item, "translation_magnitude"))
                if relative_array is None
                else float(np.linalg.norm(relative_array))
            ),
            cumulative_position_x=float(position_array[0]),
            cumulative_position_y=float(position_array[1]),
            cumulative_position_z=float(position_array[2]),
            cumulative_distance_from_origin=float(np.linalg.norm(position_array)),
            relative_rotation_deg=_finite_or_none(
                _field(item, "selected_relative_rotation_deg")
            ),
            cumulative_rotation_deg=_finite_or_none(
                _field(item, "cumulative_rotation_deg")
            ),
            geometric_inlier_ratio=float(
                _field(item, "geometric_inlier_ratio", 0.0)
            ),
            pnp_inlier_ratio=float(_field(item, "pnp_inlier_ratio", 0.0)),
            reprojection_rmse=_finite_or_none(
                _field(item, "reprojection_rmse_pixels")
            ),
            depth_alignment_inlier_ratio=float(
                _field(item, "depth_alignment_inlier_ratio", 0.0)
            ),
            refinement_3d_attempted=bool(
                _field(item, "refinement_3d_attempted", False)
            ),
            refinement_3d_accepted=bool(
                _field(item, "refinement_3d_accepted", False)
            ),
            baseline_3d_residual_median=_finite_or_none(
                _field(item, "baseline_3d_residual_median")
            ),
            refined_3d_residual_median=_finite_or_none(
                _field(item, "refined_3d_residual_median")
            ),
        ))
    return rows


def safe_percent_change(first: float | None, last: float | None) -> float | None:
    """Return percent change, or None for missing/non-finite/zero baseline."""

    first_value = _finite_or_none(first)
    last_value = _finite_or_none(last)
    if first_value is None or last_value is None or abs(first_value) <= np.finfo(float).eps:
        return None
    return (last_value - first_value) / abs(first_value) * 100.0


def _valid_series(rows: list[DriftDiagnosticRow], field: str) -> tuple[np.ndarray, np.ndarray]:
    indices: list[float] = []
    values: list[float] = []
    for row in rows:
        value = _finite_or_none(getattr(row, field))
        if value is not None:
            indices.append(float(row.keyframe_sequence_index))
            values.append(value)
    return np.asarray(indices, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _basic_statistics(values: np.ndarray, *, relative_range: bool = False) -> dict[str, float | None]:
    if values.size == 0:
        result: dict[str, float | None] = {"min": None, "max": None, "median": None}
    else:
        result = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    if relative_range:
        median = result["median"]
        result["relative_range"] = (
            None
            if median is None or abs(median) <= np.finfo(float).eps
            else (float(result["max"]) - float(result["min"])) / abs(median)
        )
    return result


def _first_last_statistics(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "percent_change_first_to_last": None,
        }
    first, last = float(values[0]), float(values[-1])
    return {
        "first": first,
        "last": last,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percent_change_first_to_last": safe_percent_change(first, last),
    }


def _slope(indices: np.ndarray, values: np.ndarray) -> float | None:
    if values.size < 2 or np.ptp(indices) <= 0.0:
        return None
    return float(np.polyfit(indices, values, 1)[0])


def trajectory_path_length(rows: list[DriftDiagnosticRow]) -> float:
    if len(rows) < 2:
        return 0.0
    positions = np.asarray([
        [row.cumulative_position_x, row.cumulative_position_y, row.cumulative_position_z]
        for row in rows
    ])
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def build_drift_summary(rows: list[DriftDiagnosticRow]) -> dict[str, Any]:
    """Build descriptive summaries and explicitly heuristic warning flags."""

    scale_x, scales = _valid_series(rows, "depth_alignment_scale_a")
    shift_x, shifts = _valid_series(rows, "depth_alignment_shift_b")
    median_x, medians = _valid_series(rows, "aligned_z_median")
    p99_x, p99_values = _valid_series(rows, "aligned_z_p99")
    _, translations = _valid_series(rows, "relative_translation_magnitude")
    scale_summary = _basic_statistics(scales, relative_range=True)
    shift_summary = _basic_statistics(shifts)
    median_summary = _first_last_statistics(medians)
    p99_summary = _first_last_statistics(p99_values)

    if rows:
        final_position = [
            rows[-1].cumulative_position_x,
            rows[-1].cumulative_position_y,
            rows[-1].cumulative_position_z,
        ]
        distances = np.asarray(
            [row.cumulative_distance_from_origin for row in rows], dtype=np.float64
        )
    else:
        final_position = None
        distances = np.empty(0, dtype=np.float64)
    monotonic_fraction: float | None = None
    if distances.size >= 2:
        monotonic_fraction = float(np.mean(np.diff(distances) >= -1e-12))

    scale_relative_range = scale_summary.get("relative_range")
    p99_change = p99_summary["percent_change_first_to_last"]
    shift_range_over_scale: float | None = None
    scale_median = scale_summary["median"]
    if shifts.size and scale_median is not None and abs(scale_median) > np.finfo(float).eps:
        shift_range_over_scale = float(np.ptp(shifts) / abs(scale_median))
    flags = {
        "heuristic_only": True,
        "depth_scale_drift_suspected": bool(
            scale_relative_range is not None and scale_relative_range > 0.20
        ),
        "depth_shift_drift_suspected": bool(
            shift_range_over_scale is not None and shift_range_over_scale > 0.20
        ),
        "z_range_growth_suspected": bool(
            p99_change is not None and abs(p99_change) > 50.0
        ),
        "pose_accumulation_growth": bool(
            monotonic_fraction is not None
            and len(rows) >= 3
            and monotonic_fraction >= 0.80
        ),
        "conditions": {
            "depth_scale_relative_range_threshold": 0.20,
            "depth_shift_range_over_median_scale_threshold": 0.20,
            "absolute_p99_percent_change_threshold": 50.0,
            "distance_non_decreasing_fraction_threshold": 0.80,
        },
        "observed": {
            "depth_scale_relative_range": scale_relative_range,
            "depth_shift_range_over_median_scale": shift_range_over_scale,
            "p99_percent_change": p99_change,
            "distance_non_decreasing_fraction": monotonic_fraction,
        },
    }
    return {
        "diagnostic_only": True,
        "coordinate_scale": "relative_non_metric",
        "accepted_keyframe_count": len(rows),
        "first_frame_index": None if not rows else rows[0].frame_index,
        "last_frame_index": None if not rows else rows[-1].frame_index,
        "depth_alignment_scale_a": scale_summary,
        "depth_alignment_shift_b": shift_summary,
        "aligned_median_z": median_summary,
        "aligned_p99_z": p99_summary,
        "relative_translation_magnitude": _basic_statistics(translations),
        "cumulative_trajectory": {
            "final_position": final_position,
            "total_path_length": trajectory_path_length(rows),
        },
        "linear_trend_per_keyframe": {
            "depth_alignment_scale_a_slope": _slope(scale_x, scales),
            "depth_alignment_shift_b_slope": _slope(shift_x, shifts),
            "aligned_median_z_slope": _slope(median_x, medians),
            "aligned_p99_z_slope": _slope(p99_x, p99_values),
            "interpretation": "descriptive slopes only; no statistical significance claimed",
        },
        "heuristic_warning_flags": flags,
    }


_PLOTS = (
    ("depth_alignment_scale_a", "depth_alignment_scale_vs_frame.png", "Depth alignment scale a", "Scale a (relative/non-metric)"),
    ("depth_alignment_shift_b", "depth_alignment_shift_vs_frame.png", "Depth alignment shift b", "Shift b (relative/non-metric)"),
    ("aligned_z_median", "aligned_z_median_vs_frame.png", "Aligned median Z", "Median Z (relative/non-metric)"),
    ("aligned_z_p99", "aligned_z_p99_vs_frame.png", "Aligned p99 Z", "p99 Z (relative/non-metric)"),
    ("relative_translation_magnitude", "translation_magnitude_vs_frame.png", "Relative translation magnitude", "Translation magnitude (relative/non-metric)"),
    ("cumulative_distance_from_origin", "cumulative_distance_vs_frame.png", "Cumulative distance from origin", "Distance (relative/non-metric)"),
    ("reprojection_rmse", "reprojection_rmse_vs_frame.png", "PnP reprojection RMSE", "RMSE (pixels)"),
    ("depth_alignment_inlier_ratio", "depth_alignment_inlier_ratio_vs_frame.png", "Depth alignment inlier ratio", "Inlier ratio"),
)


def _plot_series(axis: plt.Axes, rows: list[DriftDiagnosticRow], field: str) -> None:
    frame_indices: list[int] = []
    values: list[float] = []
    for row in rows:
        value = _finite_or_none(getattr(row, field))
        if value is not None:
            frame_indices.append(row.frame_index)
            values.append(value)
    if values:
        axis.plot(frame_indices, values, "-o", markersize=4)
    axis.grid(True, alpha=0.3)
    axis.set_xlabel("Accepted keyframe source-frame index")


def save_drift_diagnostics(
    relative_map_directory: str | Path,
    rows: list[DriftDiagnosticRow],
    *,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Write the diagnostic directory and return paths plus summary."""

    output = Path(relative_map_directory) / "drift_diagnostics"
    output.mkdir(parents=True, exist_ok=False)
    dictionaries = [row.to_dict() for row in rows]
    csv_path = output / "drift_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        if dictionaries:
            writer = csv.DictWriter(file, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)
        else:
            file.write("")
    json_path = output / "drift_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(dictionaries, file, indent=2)
    summary = build_drift_summary(rows)
    summary_path = output / "drift_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_paths: list[Path] = []
    if generate_plots:
        for field, filename, title, ylabel in _PLOTS:
            path = output / filename
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            _plot_series(axis, rows, field)
            axis.set_title(f"{title} — diagnostic only")
            axis.set_ylabel(ylabel)
            figure.savefig(path, dpi=150)
            plt.close(figure)
            plot_paths.append(path)

        overview_path = output / "drift_overview.png"
        figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
        for axis, (field, _, title, ylabel) in zip(axes.flat, _PLOTS[:6]):
            _plot_series(axis, rows, field)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        figure.suptitle(
            "DepthSLAM drift overview — heuristic diagnostics, relative/non-metric"
        )
        figure.savefig(overview_path, dpi=150)
        plt.close(figure)
        plot_paths.append(overview_path)

    return {
        "directory": output,
        "csv": csv_path,
        "json": json_path,
        "summary_path": summary_path,
        "plots": plot_paths,
        "summary": summary,
    }
