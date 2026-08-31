"""Transparent, position-only refinement of saved relative trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCALED_MAD_FACTOR = 1.4826
TRAJECTORY_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    "x",
    "y",
    "z",
    "trajectory_units",
    "trajectory_type",
)


@dataclass(frozen=True)
class TrajectoryData:
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float | None, ...]
    positions: np.ndarray
    units: str


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    step_magnitudes: np.ndarray
    direction_change_degrees: np.ndarray
    second_differences: np.ndarray


@dataclass(frozen=True)
class JumpDetectionResult:
    median_step_magnitude: float | None
    mad_step_magnitude: float | None
    scaled_mad_step_magnitude: float | None
    threshold: float | None
    mad_multiplier: float
    suspicious_step_indices: tuple[int, ...]
    suspicious_pose_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrajectoryMetrics:
    pose_count: int
    step_count: int
    mean_step_magnitude: float | None
    median_step_magnitude: float | None
    maximum_step_magnitude: float | None
    standard_deviation_step_magnitude: float | None
    mean_second_difference: float | None
    median_second_difference: float | None
    maximum_second_difference: float | None
    total_path_length_relative_units: float
    suspicious_jump_count: int
    trajectory_units: str


@dataclass(frozen=True)
class TrajectoryRefinementResult:
    raw: TrajectoryData
    refined_positions: np.ndarray
    mode: str
    weights: tuple[float, float, float]
    raw_diagnostics: TrajectoryDiagnostics
    refined_diagnostics: TrajectoryDiagnostics
    raw_detection: JumpDetectionResult
    refined_detection: JumpDetectionResult
    raw_metrics: TrajectoryMetrics
    refined_metrics: TrajectoryMetrics
    modified_pose_indices: tuple[int, ...]


def _positions_array(positions: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("trajectory positions must have shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("trajectory positions must contain only finite values")
    return values


def validate_moving_average_weights(
    weights: Iterable[float],
) -> tuple[float, float, float]:
    values = np.asarray(tuple(weights), dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("moving-average weights must contain three finite values")
    if np.any(values < 0.0):
        raise ValueError("moving-average weights cannot be negative")
    if not np.isclose(float(values.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("moving-average weights must sum to 1.0")
    return tuple(float(value) for value in values)


def trajectory_diagnostics(
    positions: np.ndarray | Iterable[Iterable[float]],
) -> TrajectoryDiagnostics:
    values = _positions_array(positions)
    deltas = np.diff(values, axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    second = (
        np.linalg.norm(values[2:] - 2.0 * values[1:-1] + values[:-2], axis=1)
        if len(values) >= 3
        else np.empty(0, dtype=np.float64)
    )
    if not np.isfinite(deltas).all() or not np.isfinite(steps).all():
        raise ValueError("trajectory differences must remain finite")
    if not np.isfinite(second).all():
        raise ValueError("trajectory second differences must remain finite")

    direction_changes = np.full(max(len(values) - 2, 0), np.nan, dtype=np.float64)
    if len(values) >= 3:
        previous_norms = steps[:-1]
        next_norms = steps[1:]
        valid = (previous_norms > 0.0) & (next_norms > 0.0)
        if np.any(valid):
            cosine = np.sum(deltas[:-1][valid] * deltas[1:][valid], axis=1)
            cosine /= previous_norms[valid] * next_norms[valid]
            direction_changes[valid] = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )
    return TrajectoryDiagnostics(steps, direction_changes, second)


def detect_suspicious_jumps(
    positions: np.ndarray | Iterable[Iterable[float]],
    mad_multiplier: float = 4.0,
) -> JumpDetectionResult:
    multiplier = float(mad_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("mad_multiplier must be finite and non-negative")
    steps = trajectory_diagnostics(positions).step_magnitudes
    if steps.size == 0:
        return JumpDetectionResult(
            None, None, None, None, multiplier, (), ()
        )
    median = float(np.median(steps))
    mad = float(np.median(np.abs(steps - median)))
    scaled_mad = SCALED_MAD_FACTOR * mad
    threshold = median + multiplier * scaled_mad
    if not np.isfinite(threshold):
        raise ValueError("MAD jump threshold must remain finite")
    suspicious_steps = tuple(
        int(index) for index in np.flatnonzero(steps > threshold)
    )
    return JumpDetectionResult(
        median_step_magnitude=median,
        mad_step_magnitude=mad,
        scaled_mad_step_magnitude=scaled_mad,
        threshold=threshold,
        mad_multiplier=multiplier,
        suspicious_step_indices=suspicious_steps,
        suspicious_pose_indices=tuple(index + 1 for index in suspicious_steps),
    )


def _optional_distribution(values: np.ndarray) -> tuple[float | None, ...]:
    if values.size == 0:
        return None, None, None, None
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.max(values)),
        float(np.std(values)),
    )


def trajectory_metrics(
    positions: np.ndarray | Iterable[Iterable[float]],
    *,
    units: str = "relative_depth_units",
    mad_multiplier: float = 4.0,
) -> TrajectoryMetrics:
    values = _positions_array(positions)
    diagnostics = trajectory_diagnostics(values)
    detection = detect_suspicious_jumps(values, mad_multiplier)
    step_mean, step_median, step_maximum, step_std = _optional_distribution(
        diagnostics.step_magnitudes
    )
    second_mean, second_median, second_maximum, _ = _optional_distribution(
        diagnostics.second_differences
    )
    return TrajectoryMetrics(
        pose_count=len(values),
        step_count=int(diagnostics.step_magnitudes.size),
        mean_step_magnitude=step_mean,
        median_step_magnitude=step_median,
        maximum_step_magnitude=step_maximum,
        standard_deviation_step_magnitude=step_std,
        mean_second_difference=second_mean,
        median_second_difference=second_median,
        maximum_second_difference=second_maximum,
        total_path_length_relative_units=float(
            np.sum(diagnostics.step_magnitudes)
        ),
        suspicious_jump_count=len(detection.suspicious_pose_indices),
        trajectory_units=units,
    )


def refine_trajectory(
    trajectory: TrajectoryData,
    *,
    mode: str = "jump_aware",
    mad_multiplier: float = 4.0,
    weights: Iterable[float] = (0.25, 0.50, 0.25),
) -> TrajectoryRefinementResult:
    if mode not in {"jump_aware", "moving_average"}:
        raise ValueError("mode must be 'jump_aware' or 'moving_average'")
    values = _positions_array(trajectory.positions)
    if len(trajectory.frame_indices) != len(values):
        raise ValueError("frame index count must match trajectory pose count")
    if len(trajectory.timestamps_seconds) != len(values):
        raise ValueError("timestamp count must match trajectory pose count")
    smoothing_weights = validate_moving_average_weights(weights)
    raw_diagnostics = trajectory_diagnostics(values)
    raw_detection = detect_suspicious_jumps(values, mad_multiplier)

    if mode == "jump_aware":
        targets = raw_detection.suspicious_pose_indices
    else:
        targets = tuple(range(1, max(len(values) - 1, 1)))

    refined = values.copy()
    for index in targets:
        if index <= 0 or index >= len(values) - 1:
            continue
        refined[index] = (
            smoothing_weights[0] * values[index - 1]
            + smoothing_weights[1] * values[index]
            + smoothing_weights[2] * values[index + 1]
        )
    changed = np.flatnonzero(np.any(refined != values, axis=1))
    refined_diagnostics = trajectory_diagnostics(refined)
    refined_detection = detect_suspicious_jumps(refined, mad_multiplier)
    return TrajectoryRefinementResult(
        raw=trajectory,
        refined_positions=refined,
        mode=mode,
        weights=smoothing_weights,
        raw_diagnostics=raw_diagnostics,
        refined_diagnostics=refined_diagnostics,
        raw_detection=raw_detection,
        refined_detection=refined_detection,
        raw_metrics=trajectory_metrics(
            values, units=trajectory.units, mad_multiplier=mad_multiplier
        ),
        refined_metrics=trajectory_metrics(
            refined, units=trajectory.units, mad_multiplier=mad_multiplier
        ),
        modified_pose_indices=tuple(int(index) for index in changed),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required trajectory artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _frame_timestamps(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Required trajectory artifact is missing: {path}")
    timestamps: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "accepted_keyframe" or row.get("accepted"):
                timestamps[int(row["frame_index"])] = float(
                    row["timestamp_seconds"]
                )
    return timestamps


def load_map_trajectory(run_directory: Path) -> TrajectoryData:
    run_path = Path(run_directory).resolve()
    metadata = _read_json(run_path / "metadata.json")
    timestamps_by_frame = _frame_timestamps(run_path / "frame_stats.jsonl")
    csv_path = run_path / "trajectory_relative.csv"
    npy_path = run_path / "trajectory_relative.npy"
    if not csv_path.is_file() or not npy_path.is_file():
        missing = csv_path if not csv_path.is_file() else npy_path
        raise FileNotFoundError(f"Required trajectory artifact is missing: {missing}")

    frame_indices: list[int] = []
    timestamps: list[float | None] = []
    csv_positions: list[tuple[float, float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            frame_index = int(row["frame_index"])
            frame_indices.append(frame_index)
            timestamp_text = row.get("timestamp_seconds")
            timestamps.append(
                float(timestamp_text)
                if timestamp_text not in (None, "")
                else timestamps_by_frame.get(frame_index)
            )
            csv_positions.append(
                (float(row["x"]), float(row["y"]), float(row["z"]))
            )
    if not frame_indices:
        raise ValueError("trajectory_relative.csv contains no poses")
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("trajectory frame indices must be unique")

    positions = _positions_array(np.load(npy_path, allow_pickle=False))
    csv_values = _positions_array(csv_positions)
    if positions.shape != csv_values.shape or not np.allclose(
        positions, csv_values, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("trajectory CSV and NumPy artifacts do not agree")
    units = str(metadata.get("translation_units", "relative_depth_units"))
    return TrajectoryData(
        tuple(frame_indices), tuple(timestamps), positions.copy(), units
    )


def _write_trajectory_csv(
    path: Path,
    trajectory: TrajectoryData,
    positions: np.ndarray,
    trajectory_type: str,
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        for frame_index, timestamp, position in zip(
            trajectory.frame_indices,
            trajectory.timestamps_seconds,
            positions,
            strict=True,
        ):
            writer.writerow({
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "trajectory_units": trajectory.units,
                "trajectory_type": trajectory_type,
            })
    return path


def _value_at(values: np.ndarray, index: int, offset: int) -> float | None:
    array_index = index - offset
    if array_index < 0 or array_index >= len(values):
        return None
    value = float(values[array_index])
    return value if np.isfinite(value) else None


def write_diagnostics_csv(
    path: Path, result: TrajectoryRefinementResult
) -> Path:
    columns = (
        "pose_index", "frame_index", "raw_step_magnitude",
        "refined_step_magnitude", "raw_direction_change_degrees",
        "refined_direction_change_degrees", "raw_second_difference",
        "refined_second_difference", "suspicious_jump",
    )
    suspicious = set(result.raw_detection.suspicious_pose_indices)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, frame_index in enumerate(result.raw.frame_indices):
            writer.writerow({
                "pose_index": index,
                "frame_index": frame_index,
                "raw_step_magnitude": _value_at(
                    result.raw_diagnostics.step_magnitudes, index, 1
                ),
                "refined_step_magnitude": _value_at(
                    result.refined_diagnostics.step_magnitudes, index, 1
                ),
                "raw_direction_change_degrees": _value_at(
                    result.raw_diagnostics.direction_change_degrees, index, 1
                ),
                "refined_direction_change_degrees": _value_at(
                    result.refined_diagnostics.direction_change_degrees, index, 1
                ),
                "raw_second_difference": _value_at(
                    result.raw_diagnostics.second_differences, index, 1
                ),
                "refined_second_difference": _value_at(
                    result.refined_diagnostics.second_differences, index, 1
                ),
                "suspicious_jump": index in suspicious,
            })
    return path


def generate_refinement_plots(
    output_directory: Path, result: TrajectoryRefinementResult
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install requirements.txt."
        ) from exc

    created: list[Path] = []
    raw = result.raw.positions
    refined = result.refined_positions
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(raw[:, 0], raw[:, 2], marker="o", label="Raw")
    axis.plot(refined[:, 0], refined[:, 2], marker="x", label="Refined")
    axis.set(
        title="Raw vs refined trajectory (X-Z projection)",
        xlabel="X (relative units)",
        ylabel="Z (relative units)",
    )
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    comparison_path = output_directory / "trajectory_raw_vs_refined_xz.png"
    figure.savefig(comparison_path, dpi=140)
    plt.close(figure)
    created.append(comparison_path)

    if result.raw_diagnostics.step_magnitudes.size:
        step_frames = np.asarray(result.raw.frame_indices[1:], dtype=np.int64)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(
            step_frames,
            result.raw_diagnostics.step_magnitudes,
            marker="o",
            label="Raw step",
        )
        axis.plot(
            step_frames,
            result.refined_diagnostics.step_magnitudes,
            marker="x",
            linestyle="--",
            label="Refined step",
        )
        suspicious = result.raw_detection.suspicious_step_indices
        if suspicious:
            indices = np.asarray(suspicious, dtype=np.int64)
            axis.scatter(
                step_frames[indices],
                result.raw_diagnostics.step_magnitudes[indices],
                color="red",
                marker="x",
                s=70,
                label="Suspicious raw jump",
                zorder=3,
            )
        axis.set(
            title="Accepted-pose translation step magnitude",
            xlabel="Accepted frame index",
            ylabel="Step magnitude (relative units)",
        )
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        step_path = output_directory / "trajectory_step_magnitude.png"
        figure.savefig(step_path, dpi=140)
        plt.close(figure)
        created.append(step_path)
    return created


def write_refinement_outputs(
    source_run_directory: Path,
    result: TrajectoryRefinementResult,
    output_directory: Path,
    *,
    plots: bool = False,
) -> dict[str, Any]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    raw_csv = _write_trajectory_csv(
        directory / "trajectory_raw.csv", result.raw,
        result.raw.positions, "raw",
    )
    raw_npy = directory / "trajectory_raw.npy"
    np.save(raw_npy, result.raw.positions, allow_pickle=False)
    refined_csv = _write_trajectory_csv(
        directory / "trajectory_refined.csv", result.raw,
        result.refined_positions, "refined",
    )
    refined_npy = directory / "trajectory_refined.npy"
    np.save(refined_npy, result.refined_positions, allow_pickle=False)
    diagnostics_csv = write_diagnostics_csv(
        directory / "trajectory_diagnostics.csv", result
    )
    plot_paths = generate_refinement_plots(directory, result) if plots else []

    summary = {
        "source_run_directory": str(Path(source_run_directory).resolve()),
        "trajectory_refinement": {
            "enabled": True,
            "method": result.mode,
            "mad_multiplier": result.raw_detection.mad_multiplier,
            "moving_average_weights": list(result.weights),
            "suspicious_jump_count": len(
                result.raw_detection.suspicious_pose_indices
            ),
            "suspicious_pose_indices": list(
                result.raw_detection.suspicious_pose_indices
            ),
            "suspicious_frame_indices": [
                result.raw.frame_indices[index]
                for index in result.raw_detection.suspicious_pose_indices
            ],
            "modified_pose_count": len(result.modified_pose_indices),
            "modified_pose_indices": list(result.modified_pose_indices),
            "modified_frame_indices": [
                result.raw.frame_indices[index]
                for index in result.modified_pose_indices
            ],
            "jump_detection": asdict(result.raw_detection),
            "raw_metrics": asdict(result.raw_metrics),
            "refined_metrics": asdict(result.refined_metrics),
        },
        "trajectory_artifacts": {
            "raw": {
                "trajectory_type": "raw",
                "csv": raw_csv.name,
                "npy": raw_npy.name,
                "coordinate_units": result.raw.units,
            },
            "refined": {
                "trajectory_type": "refined",
                "csv": refined_csv.name,
                "npy": refined_npy.name,
                "coordinate_units": result.raw.units,
            },
        },
        "map_geometry_note": (
            "The refined trajectory is an analysis/refinement artifact. The "
            "existing fused map was generated from raw accepted poses."
        ),
        "scientific_limitations": [
            "Smoothing does not recover ground truth or guarantee greater accuracy.",
            "Refinement does not resolve scale ambiguity or accumulated drift.",
            "Refinement is not loop closure, bundle adjustment, or pose-graph optimization.",
            "Coordinates remain relative and non-metric.",
            "Rotations are unchanged; rotation matrices are not averaged.",
        ],
        "generated_plots": [path.name for path in plot_paths],
    }
    summary_path = directory / "refinement_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return {
        "directory": directory,
        "trajectory_raw_csv": raw_csv,
        "trajectory_raw_npy": raw_npy,
        "trajectory_refined_csv": refined_csv,
        "trajectory_refined_npy": refined_npy,
        "diagnostics": diagnostics_csv,
        "summary": summary_path,
        "plots": plot_paths,
        "summary_data": summary,
    }
