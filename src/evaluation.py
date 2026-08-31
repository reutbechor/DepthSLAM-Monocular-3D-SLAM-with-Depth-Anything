"""Artifact-based numerical and visual evaluation for relative DepthSLAM runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FRAME_METRIC_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    "status",
    "good_matches",
    "geometric_inliers",
    "geometric_inlier_ratio",
    "median_feature_displacement_px",
    "rotation_deg",
    "pnp_inliers",
    "pnp_inlier_ratio",
    "reprojection_rmse_pixels",
    "reprojection_median_pixels",
    "translation_magnitude",
    "translation_units",
    "depth_alignment_inliers",
    "depth_alignment_inlier_ratio",
    "denominator_rejection_ratio",
    "valid_aligned_depth_ratio",
    "aligned_z_median",
    "aligned_z_p99",
    "relative_z_p99_over_median",
    "cloud_points",
    "keyframe_reason",
    "skip_reason",
    "rejection_reason",
    "depth_inference_executed",
)


@dataclass(frozen=True)
class MetricDistribution:
    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    standard_deviation: float | None


@dataclass
class EvaluationResult:
    source_run_directory: Path
    frame_metrics: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]


def metric_distribution(values: Iterable[Any]) -> MetricDistribution:
    cleaned = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size == 0:
        return MetricDistribution(0, None, None, None, None, None)
    return MetricDistribution(
        count=int(cleaned.size),
        mean=float(np.mean(cleaned)),
        median=float(np.median(cleaned)),
        minimum=float(np.min(cleaned)),
        maximum=float(np.max(cleaned)),
        standard_deviation=float(np.std(cleaned)),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_frame_statistics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid frame statistics at line {line_number}")
            rows.append(value)
    if not rows:
        raise ValueError("frame_stats.jsonl contains no candidate frames")
    return rows


def normalize_frame_metric(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize absent pipeline stages to empty CSV/JSON values."""
    initial = row.get("keyframe_reason") == "initial_frame"
    feature_reached = not initial and row.get("rejection_reason") != "feature_matching"
    pose_reached = row.get("scale_estimation_method") == "depth_pnp"
    alignment_reached = int(
        row.get("depth_alignment_input_correspondences", 0)
    ) > 0
    depth_quality_reached = (
        int(row.get("total_depth_candidates", 0)) > 0
        or row.get("aligned_z_median") is not None
    )
    accepted = row.get("status") == "accepted_keyframe"

    normalized = {
        "frame_index": int(row["frame_index"]),
        "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
        "status": row.get("status", "rejected"),
        "good_matches": row.get("good_matches") if feature_reached else None,
        "geometric_inliers": (
            row.get("geometric_inliers") if feature_reached else None
        ),
        "geometric_inlier_ratio": (
            row.get("geometric_inlier_ratio") if feature_reached else None
        ),
        "median_feature_displacement_px": row.get(
            "median_feature_displacement_px"
        ),
        "rotation_deg": row.get("rotation_deg"),
        "pnp_inliers": row.get("pnp_inliers") if pose_reached else None,
        "pnp_inlier_ratio": row.get("pnp_inlier_ratio") if pose_reached else None,
        "reprojection_rmse_pixels": row.get("reprojection_rmse_pixels"),
        "reprojection_median_pixels": row.get("reprojection_median_pixels"),
        "translation_magnitude": row.get("translation_magnitude"),
        "translation_units": row.get("translation_units"),
        "depth_alignment_inliers": (
            row.get("depth_alignment_inliers") if alignment_reached else None
        ),
        "depth_alignment_inlier_ratio": (
            row.get("depth_alignment_inlier_ratio")
            if alignment_reached else None
        ),
        "denominator_rejection_ratio": (
            row.get("denominator_rejection_ratio")
            if depth_quality_reached else None
        ),
        "valid_aligned_depth_ratio": (
            row.get("valid_aligned_depth_ratio") if depth_quality_reached else None
        ),
        "aligned_z_median": (
            row.get("aligned_z_median") if depth_quality_reached else None
        ),
        "aligned_z_p99": (
            row.get("aligned_z_p99") if depth_quality_reached else None
        ),
        "relative_z_p99_over_median": (
            row.get("relative_z_p99_over_median")
            if depth_quality_reached else None
        ),
        "cloud_points": row.get("cloud_points") if accepted else None,
        "keyframe_reason": row.get("keyframe_reason"),
        "skip_reason": row.get("skip_reason"),
        "rejection_reason": row.get("rejection_reason"),
        "depth_inference_executed": bool(
            row.get("depth_inference_executed", False)
        ),
    }
    return normalized


def _load_trajectory(
    path: Path, frame_metrics: list[dict[str, Any]], units: str
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation artifact is missing: {path}")
    timestamps = {
        row["frame_index"]: row["timestamp_seconds"]
        for row in frame_metrics
        if row["status"] == "accepted_keyframe"
    }
    trajectory: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            frame_index = int(row["frame_index"])
            if frame_index not in timestamps:
                raise ValueError(
                    f"Trajectory frame {frame_index} is not an accepted keyframe"
                )
            trajectory.append({
                "frame_index": frame_index,
                "timestamp_seconds": timestamps[frame_index],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "trajectory_units": units,
            })
    if len(trajectory) != len(timestamps):
        raise ValueError("Trajectory pose count does not match accepted keyframes")
    return trajectory


def _runtime_summary(
    metadata: dict[str, Any],
    *,
    total_runtime_seconds: float | None,
    candidate_count: int,
    accepted_count: int,
    depth_count: int,
) -> dict[str, Any]:
    recorded = dict(metadata.get("runtime_metrics", {}))
    total = (
        float(total_runtime_seconds)
        if total_runtime_seconds is not None
        else recorded.get("total_pipeline_runtime_seconds")
    )
    if total is not None:
        total = float(total)
    recorded_depth_total = recorded.get("depth_inference_seconds")
    depth_total = (
        float(recorded_depth_total) if recorded_depth_total is not None else None
    )
    values = [value for value in recorded.values() if value is not None]
    if (total is not None and total < 0.0) or any(float(value) < 0.0 for value in values):
        raise ValueError("runtime metrics cannot be negative")
    return {
        "total_pipeline_runtime_seconds": total,
        "average_runtime_per_candidate_seconds": (
            total / candidate_count if total is not None and candidate_count else None
        ),
        "average_runtime_per_accepted_keyframe_seconds": (
            total / accepted_count if total is not None and accepted_count else None
        ),
        "feature_motion_seconds": recorded.get("feature_motion_seconds"),
        "depth_inference_seconds": depth_total,
        "average_depth_inference_seconds": (
            depth_total / depth_count
            if depth_total is not None and depth_count
            else None
        ),
        "pnp_depth_alignment_seconds": recorded.get(
            "pnp_depth_alignment_seconds"
        ),
        "point_cloud_fusion_seconds": recorded.get(
            "point_cloud_fusion_seconds"
        ),
        "timing_note": (
            "Approximate wall-clock instrumentation; hardware, CPU load, model/cache "
            "state, file I/O, and subprocess overhead affect results."
        ),
    }


def _trajectory_refinement_summary(
    run_path: Path, refinement_directory: Path | None
) -> dict[str, Any]:
    if refinement_directory is None:
        candidate = run_path / "refinement_summary.json"
        if not candidate.is_file():
            return {"enabled": False}
        summary_path = candidate
    else:
        summary_path = Path(refinement_directory).resolve() / "refinement_summary.json"
    summary = _load_json(summary_path)
    source = summary.get("source_run_directory")
    if source is None or Path(source).resolve() != run_path:
        raise ValueError(
            "refinement summary source does not match the evaluated map run"
        )
    refinement = summary.get("trajectory_refinement")
    if not isinstance(refinement, dict) or not refinement.get("enabled"):
        raise ValueError("refinement_summary.json has no enabled refinement result")
    return {
        "enabled": True,
        "method": refinement.get("method"),
        "suspicious_jump_count": refinement.get("suspicious_jump_count"),
        "modified_pose_count": refinement.get("modified_pose_count"),
        "raw_metrics": refinement.get("raw_metrics"),
        "refined_metrics": refinement.get("refined_metrics"),
        "source_directory": str(summary_path.parent),
        "map_geometry_note": summary.get("map_geometry_note"),
    }


def evaluate_run_directory(
    run_directory: Path,
    *,
    total_runtime_seconds: float | None = None,
    intrinsics_source: str | None = None,
    intrinsics_approximate: bool | None = None,
    refinement_directory: Path | None = None,
) -> EvaluationResult:
    run_path = Path(run_directory).resolve()
    metadata = _load_json(run_path / "metadata.json")
    frame_metrics = [
        normalize_frame_metric(row)
        for row in _load_frame_statistics(run_path / "frame_stats.jsonl")
    ]
    units = str(metadata.get("translation_units", "relative_depth_units"))
    trajectory = _load_trajectory(
        run_path / "trajectory_relative.csv", frame_metrics, units
    )

    status_counts = Counter(row["status"] for row in frame_metrics)
    rejection_counts = Counter(
        row["rejection_reason"]
        for row in frame_metrics
        if row["status"] == "rejected" and row["rejection_reason"]
    )
    candidate_count = len(frame_metrics)
    accepted_count = status_counts["accepted_keyframe"]
    depth_count = sum(row["depth_inference_executed"] for row in frame_metrics)

    visual = {
        "good_matches": asdict(metric_distribution(
            row["good_matches"] for row in frame_metrics
        )),
        "geometric_inlier_ratio": asdict(metric_distribution(
            row["geometric_inlier_ratio"] for row in frame_metrics
        )),
    }
    pose = {
        "pnp_inlier_ratio": asdict(metric_distribution(
            row["pnp_inlier_ratio"] for row in frame_metrics
        )),
        "reprojection_rmse_pixels": asdict(metric_distribution(
            row["reprojection_rmse_pixels"]
            for row in frame_metrics
            if row["rejection_reason"] != "pnp"
        )),
        "translation_magnitude_relative_units": asdict(metric_distribution(
            row["translation_magnitude"]
            for row in frame_metrics
            if row["status"] == "accepted_keyframe"
            and row["keyframe_reason"] != "initial_frame"
        )),
        "translation_units": units,
        "reprojection_note": (
            "Reprojection RMSE measures consistency with matched image observations; "
            "it is not absolute trajectory accuracy."
        ),
    }
    depth = {
        "depth_alignment_inlier_ratio": asdict(metric_distribution(
            row["depth_alignment_inlier_ratio"] for row in frame_metrics
        )),
        "denominator_rejection_ratio": asdict(metric_distribution(
            row["denominator_rejection_ratio"] for row in frame_metrics
        )),
        "valid_aligned_depth_ratio": asdict(metric_distribution(
            row["valid_aligned_depth_ratio"] for row in frame_metrics
        )),
    }
    map_summary = {
        "raw_fused_points": int(metadata.get("raw_fused_point_count", 0)),
        "voxelized_points": int(metadata.get("voxel_downsampled_point_count", 0)),
        "global_outliers_removed": int(
            metadata.get("global_outlier_filter", {}).get("points_rejected", 0)
        ),
        "final_map_points": int(metadata.get("final_map_point_count", 0)),
        "coordinate_units": metadata.get("coordinate_units", units),
    }
    configuration = {
        "source": metadata.get("source"),
        "depth_model": metadata.get("model"),
        "device": metadata.get("device"),
        "scale_mode": metadata.get("scale_mode"),
        "sample_every": metadata.get("sample_every"),
        "max_candidate_frames": metadata.get("max_mapping_frames"),
        "point_cloud_stride": metadata.get("point_cloud_stride"),
        "camera_intrinsics": metadata.get("camera_intrinsics"),
        "camera_intrinsics_source": intrinsics_source or "not_recorded",
        "camera_intrinsics_approximate": intrinsics_approximate,
        "keyframe_settings": metadata.get("keyframe_selection"),
        "quality_thresholds": {
            "visual_motion": metadata.get("motion_quality_thresholds"),
            "pnp": metadata.get("pnp_quality_thresholds"),
            "depth": metadata.get("depth_quality_thresholds"),
        },
    }
    summary = {
        "source_run_directory": str(run_path),
        "run_configuration": configuration,
        "counts": {
            "total_candidates": candidate_count,
            "accepted_keyframes": accepted_count,
            "skipped_non_keyframes": status_counts["skipped_non_keyframe"],
            "rejected_frames": status_counts["rejected"],
            "depth_inference_count": depth_count,
            "trajectory_pose_count": len(trajectory),
        },
        "visual_geometry": visual,
        "pose": pose,
        "depth_alignment": depth,
        "map": map_summary,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "runtime": _runtime_summary(
            metadata,
            total_runtime_seconds=total_runtime_seconds,
            candidate_count=candidate_count,
            accepted_count=accepted_count,
            depth_count=depth_count,
        ),
        "trajectory_refinement": _trajectory_refinement_summary(
            run_path, refinement_directory
        ),
        "ground_truth": {
            "available": False,
            "ate_computed": False,
            "rpe_computed": False,
            "note": (
                "No external ground-truth trajectory was supplied; absolute and "
                "relative trajectory errors are not computed."
            ),
        },
        "scientific_limitations": [
            "Trajectory and map coordinates are relative and non-metric.",
            "Internal matching and reprojection metrics are not ground-truth accuracy.",
            "No ATE or RPE is available without an external ground-truth trajectory.",
            "Evaluation does not correct drift, scale ambiguity, or depth inconsistency.",
            "Timing values are approximate and environment-dependent.",
        ],
    }
    return EvaluationResult(run_path, frame_metrics, trajectory, summary)


def write_frame_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in FRAME_METRIC_COLUMNS})
    return path


def write_trajectory_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns = ("frame_index", "timestamp_seconds", "x", "y", "z", "trajectory_units")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_evaluation_plots(
    output_directory: Path,
    frame_metrics: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
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

    def line_plot(metric: str, filename: str, title: str, ylabel: str) -> None:
        points = [
            (row["frame_index"], row[metric])
            for row in frame_metrics if row.get(metric) is not None
        ]
        if not points:
            return
        x, y = zip(*points)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(x, y, marker="o", linewidth=1.4)
        axis.set(title=title, xlabel="Frame index", ylabel=ylabel)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=140)
        plt.close(figure)
        created.append(path)

    line_plot(
        "geometric_inlier_ratio", "geometric_inlier_ratio.png",
        "Geometric inlier ratio", "Inlier ratio",
    )
    line_plot(
        "pnp_inlier_ratio", "pnp_inlier_ratio.png",
        "PnP inlier ratio", "Inlier ratio",
    )
    line_plot(
        "reprojection_rmse_pixels", "reprojection_rmse.png",
        "PnP reprojection RMSE", "RMSE (pixels)",
    )
    line_plot(
        "depth_alignment_inlier_ratio", "depth_alignment_inlier_ratio.png",
        "Depth alignment inlier ratio", "Inlier ratio",
    )
    line_plot(
        "denominator_rejection_ratio", "denominator_rejection_ratio.png",
        "Aligned-depth denominator rejection ratio", "Rejection ratio",
    )

    counts = Counter(row["status"] for row in frame_metrics)
    labels = ("accepted_keyframe", "skipped_non_keyframe", "rejected")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(labels, [counts[label] for label in labels])
    axis.set(title="Candidate frame status", ylabel="Frame count")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    status_path = output_directory / "frame_status.png"
    figure.savefig(status_path, dpi=140)
    plt.close(figure)
    created.append(status_path)

    if trajectory:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot(
            [row["x"] for row in trajectory],
            [row["z"] for row in trajectory],
            marker="o",
        )
        axis.set(
            title="Relative camera trajectory (X-Z projection)",
            xlabel="X (relative depth units)",
            ylabel="Z (relative depth units)",
        )
        axis.axis("equal")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        trajectory_path = output_directory / "trajectory_xz.png"
        figure.savefig(trajectory_path, dpi=140)
        plt.close(figure)
        created.append(trajectory_path)
    return created


def write_evaluation_report(path: Path, summary: dict[str, Any]) -> Path:
    counts = summary["counts"]
    visual = summary["visual_geometry"]
    pose = summary["pose"]
    depth = summary["depth_alignment"]
    map_summary = summary["map"]
    runtime = summary["runtime"]
    rejection_counts = summary["rejection_reason_counts"]
    refinement = summary["trajectory_refinement"]
    lines = [
        "DepthSLAM Stage 7 Evaluation Report",
        "===================================",
        "",
        f"Source run: {summary['source_run_directory']}",
        "",
        "Counts",
        f"- Total candidates: {counts['total_candidates']}",
        f"- Accepted keyframes: {counts['accepted_keyframes']}",
        f"- Skipped non-keyframes: {counts['skipped_non_keyframes']}",
        f"- Rejected frames: {counts['rejected_frames']}",
        f"- Depth inference count: {counts['depth_inference_count']}",
        f"- Trajectory poses: {counts['trajectory_pose_count']}",
        "",
        "Internal consistency summaries",
        f"- Geometric inlier ratio mean/median: "
        f"{visual['geometric_inlier_ratio']['mean']} / "
        f"{visual['geometric_inlier_ratio']['median']}",
        f"- PnP inlier ratio mean/median: {pose['pnp_inlier_ratio']['mean']} / "
        f"{pose['pnp_inlier_ratio']['median']}",
        f"- Reprojection RMSE mean/median (pixels): "
        f"{pose['reprojection_rmse_pixels']['mean']} / "
        f"{pose['reprojection_rmse_pixels']['median']}",
        f"- Depth alignment inlier ratio mean/median: "
        f"{depth['depth_alignment_inlier_ratio']['mean']} / "
        f"{depth['depth_alignment_inlier_ratio']['median']}",
        f"- Maximum denominator rejection ratio: "
        f"{depth['denominator_rejection_ratio']['maximum']}",
        "",
        "Map",
        f"- Raw fused points: {map_summary['raw_fused_points']}",
        f"- Voxelized points: {map_summary['voxelized_points']}",
        f"- Global outliers removed: {map_summary['global_outliers_removed']}",
        f"- Final map points: {map_summary['final_map_points']}",
        f"- Units: {map_summary['coordinate_units']}",
        "",
        "Runtime (approximate)",
        f"- Total pipeline seconds: {runtime['total_pipeline_runtime_seconds']}",
        f"- Depth inference seconds: {runtime['depth_inference_seconds']}",
        f"- Average depth inference seconds: "
        f"{runtime['average_depth_inference_seconds']}",
        "",
        "Trajectory refinement",
        f"- Enabled: {refinement['enabled']}",
    ]
    if refinement["enabled"]:
        lines.extend([
            f"- Method: {refinement['method']}",
            f"- Suspicious jumps: {refinement['suspicious_jump_count']}",
            f"- Modified poses: {refinement['modified_pose_count']}",
            f"- Raw path length (relative units): "
            f"{refinement['raw_metrics']['total_path_length_relative_units']}",
            f"- Refined path length (relative units): "
            f"{refinement['refined_metrics']['total_path_length_relative_units']}",
            "- The fused map remains based on raw accepted poses.",
        ])
    lines.extend([
        "",
        "Rejection reasons",
    ])
    lines.extend(
        f"- {reason}: {count}" for reason, count in rejection_counts.items()
    )
    if not rejection_counts:
        lines.append("- None")
    lines.extend([
        "",
        "Scientific limitations",
        "- No ground-truth trajectory was supplied; ATE and RPE were not computed.",
        "- Reprojection RMSE is image-observation consistency, not trajectory accuracy.",
        "- Trajectory and map coordinates use relative depth units, not metres.",
        "- The evaluation does not correct drift, scale, or depth inconsistency.",
        "- Timings depend on hardware, system load, model/cache state, and I/O.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_evaluation_outputs(
    result: EvaluationResult,
    output_directory: Path,
    *,
    plots: bool = True,
) -> dict[str, Any]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_csv = write_frame_metrics_csv(
        directory / "frame_metrics.csv", result.frame_metrics
    )
    trajectory_csv = write_trajectory_csv(
        directory / "trajectory.csv", result.trajectory
    )
    plot_paths = (
        generate_evaluation_plots(directory, result.frame_metrics, result.trajectory)
        if plots else []
    )
    result.summary["generated_plots"] = [path.name for path in plot_paths]
    summary_path = directory / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(result.summary, file, indent=2, allow_nan=False)
    report_path = write_evaluation_report(
        directory / "evaluation_report.txt", result.summary
    )
    return {
        "directory": directory,
        "frame_metrics": frame_csv,
        "summary": summary_path,
        "trajectory": trajectory_csv,
        "report": report_path,
        "plots": plot_paths,
    }
