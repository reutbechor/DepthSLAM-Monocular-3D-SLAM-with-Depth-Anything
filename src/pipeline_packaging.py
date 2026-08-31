"""Final Stage 9 summaries, manifests, reports, and artifact validation."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VideoInformation:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class PipelineTimings:
    mapping_seconds: float
    evaluation_seconds: float
    trajectory_refinement_seconds: float | None
    packaging_seconds: float
    end_to_end_seconds: float


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required pipeline artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def package_versions() -> dict[str, str | None]:
    packages = {
        "numpy": "numpy",
        "opencv": "opencv-python",
        "torch": "torch",
        "transformers": "transformers",
        "matplotlib": "matplotlib",
    }
    versions: dict[str, str | None] = {}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def relative_path(path: Path, root: Path) -> str:
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Final artifact must be inside {resolved_root}: {resolved_path}"
        ) from exc


def build_artifact_index(
    final_directory: Path,
    mapping_run: Path,
    evaluation_paths: dict[str, Any],
    refinement_paths: dict[str, Any] | None,
) -> dict[str, Any]:
    root = Path(final_directory).resolve()
    mapping = Path(mapping_run).resolve()
    index: dict[str, Any] = {
        "mapping": {
            "global_relative_map_ply": relative_path(
                mapping / "global_relative_map.ply", root
            ),
            "trajectory_relative_csv": relative_path(
                mapping / "trajectory_relative.csv", root
            ),
            "trajectory_relative_npy": relative_path(
                mapping / "trajectory_relative.npy", root
            ),
            "frame_statistics": relative_path(mapping / "frame_stats.jsonl", root),
            "metadata": relative_path(mapping / "metadata.json", root),
        },
        "evaluation": {
            "frame_metrics_csv": relative_path(
                Path(evaluation_paths["frame_metrics"]), root
            ),
            "summary_json": relative_path(Path(evaluation_paths["summary"]), root),
            "trajectory_csv": relative_path(
                Path(evaluation_paths["trajectory"]), root
            ),
            "evaluation_report": relative_path(
                Path(evaluation_paths["report"]), root
            ),
            "plots": [
                relative_path(Path(path), root)
                for path in evaluation_paths.get("plots", [])
            ],
        },
        "final": {
            "final_summary": "final_summary.json",
            "run_manifest": "run_manifest.json",
            "final_report": "FINAL_REPORT.md",
            "artifact_index": "artifacts.json",
        },
    }
    if refinement_paths is not None:
        index["trajectory_refinement"] = {
            "trajectory_raw_csv": relative_path(
                Path(refinement_paths["trajectory_raw_csv"]), root
            ),
            "trajectory_raw_npy": relative_path(
                Path(refinement_paths["trajectory_raw_npy"]), root
            ),
            "trajectory_refined_csv": relative_path(
                Path(refinement_paths["trajectory_refined_csv"]), root
            ),
            "trajectory_refined_npy": relative_path(
                Path(refinement_paths["trajectory_refined_npy"]), root
            ),
            "diagnostics_csv": relative_path(
                Path(refinement_paths["diagnostics"]), root
            ),
            "summary_json": relative_path(Path(refinement_paths["summary"]), root),
            "plots": [
                relative_path(Path(path), root)
                for path in refinement_paths.get("plots", [])
            ],
        }
    return index


def _artifact_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _artifact_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_strings(child)


def validate_artifact_index(
    artifact_index: dict[str, Any], final_directory: Path
) -> list[Path]:
    root = Path(final_directory).resolve()
    checked: list[Path] = []
    for value in _artifact_strings(artifact_index):
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes final directory: {value}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Referenced artifact does not exist: {candidate}")
        checked.append(candidate)
    return checked


def refinement_record(
    *,
    enabled: bool,
    status: str,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "disabled", "failed"}:
        raise ValueError("invalid trajectory refinement status")
    record: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": status,
        "method": None,
        "suspicious_jump_count": None,
        "modified_pose_count": None,
        "error": error,
    }
    if summary is not None:
        values = summary.get("trajectory_refinement", {})
        record.update({
            "method": values.get("method"),
            "suspicious_jump_count": values.get("suspicious_jump_count"),
            "modified_pose_count": values.get("modified_pose_count"),
            "raw_metrics": values.get("raw_metrics"),
            "refined_metrics": values.get("refined_metrics"),
        })
    return record


def build_final_summary(
    *,
    source_video: Path,
    video: VideoInformation,
    mapping_metadata: dict[str, Any],
    evaluation_summary: dict[str, Any],
    refinement: dict[str, Any],
    artifacts: dict[str, Any],
    timings: PipelineTimings,
) -> dict[str, Any]:
    counts = evaluation_summary["counts"]
    run_config = evaluation_summary["run_configuration"]
    trajectory_artifacts = artifacts.get("trajectory_refinement", {})
    raw_path = artifacts["mapping"]["trajectory_relative_csv"]
    limitations = list(evaluation_summary.get("scientific_limitations", []))
    limitations.extend([
        "Camera intrinsics were manually supplied and are uncalibrated; "
        "3D geometry may be warped.",
        "The monocular trajectory and map are relative and non-metric.",
        "No ground truth is available, so no absolute accuracy claim is made.",
        "The pipeline has no loop closure, bundle adjustment, or pose-graph optimization.",
        "A refined trajectory, when present, does not alter the map fused from raw poses.",
    ])
    limitations = list(dict.fromkeys(limitations))
    return {
        "input": {
            "source_video": str(Path(source_video).resolve()),
            "image_width": video.width,
            "image_height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
            "camera_intrinsics": mapping_metadata.get("camera_intrinsics"),
            "camera_intrinsics_source": "manual_command_line",
            "camera_intrinsics_approximate": True,
            "depth_model": mapping_metadata.get("model"),
            "device": mapping_metadata.get("device"),
            "input_type": "monocular_rgb_video",
        },
        "pipeline": {
            "sample_every": mapping_metadata.get("sample_every"),
            "candidate_frame_count": counts["total_candidates"],
            "keyframe_settings": mapping_metadata.get("keyframe_selection"),
            "quality_thresholds": run_config.get("quality_thresholds"),
            "scale_mode": mapping_metadata.get("scale_mode"),
            "trajectory_refinement": refinement,
        },
        "results": {
            "accepted_keyframes": counts["accepted_keyframes"],
            "skipped_frames": counts["skipped_non_keyframes"],
            "rejected_frames": counts["rejected_frames"],
            "depth_inference_count": counts["depth_inference_count"],
            "trajectory_pose_count": counts["trajectory_pose_count"],
            "final_map_point_count": evaluation_summary["map"]["final_map_points"],
            "pnp_inlier_ratio": evaluation_summary["pose"]["pnp_inlier_ratio"],
            "reprojection_rmse_pixels": evaluation_summary["pose"][
                "reprojection_rmse_pixels"
            ],
            "depth_alignment_inlier_ratio": evaluation_summary[
                "depth_alignment"
            ]["depth_alignment_inlier_ratio"],
            "rejection_reason_counts": evaluation_summary[
                "rejection_reason_counts"
            ],
        },
        "trajectory": {
            "units": mapping_metadata.get(
                "translation_units", "relative_depth_units"
            ),
            "is_metric": False,
            "raw_trajectory_csv": raw_path,
            "raw_trajectory_npy": artifacts["mapping"][
                "trajectory_relative_npy"
            ],
            "preserved_raw_copy_csv": trajectory_artifacts.get(
                "trajectory_raw_csv"
            ),
            "refined_trajectory_csv": trajectory_artifacts.get(
                "trajectory_refined_csv"
            ),
            "refined_trajectory_npy": trajectory_artifacts.get(
                "trajectory_refined_npy"
            ),
            "suspicious_jump_count": refinement.get("suspicious_jump_count"),
            "modified_pose_count": refinement.get("modified_pose_count"),
        },
        "map": {
            "ply_path": artifacts["mapping"]["global_relative_map_ply"],
            "coordinate_units": mapping_metadata.get(
                "coordinate_units", "relative_depth_units"
            ),
            "is_metric": False,
            "metric_status": "relative_non_metric",
        },
        "ground_truth": {
            "available": False,
            "ate_computed": False,
            "rpe_computed": False,
        },
        "runtime": {
            **asdict(timings),
            "internal_mapping_timings": mapping_metadata.get("runtime_metrics"),
            "note": (
                "Wall-clock timings are environment dependent and are not "
                "formal benchmarks."
            ),
        },
        "limitations": limitations,
        "artifact_index": "artifacts.json",
    }


def build_run_manifest(
    *,
    timestamp: str,
    cli_arguments: list[str],
    parsed_arguments: dict[str, Any],
    mapping_command: list[str],
    config_path: Path,
    mapping_metadata: dict[str, Any],
    video: VideoInformation,
    refinement: dict[str, Any],
) -> dict[str, Any]:
    resolved_configuration = {
        "camera_intrinsics": mapping_metadata.get("camera_intrinsics"),
        "camera_intrinsics_source": "manual_command_line",
        "camera_intrinsics_approximate": True,
        "sample_every": mapping_metadata.get("sample_every"),
        "max_candidate_frames": mapping_metadata.get("max_mapping_frames"),
        "point_cloud_stride": mapping_metadata.get("point_cloud_stride"),
        "scale_mode": mapping_metadata.get("scale_mode"),
        "keyframe_settings": mapping_metadata.get("keyframe_selection"),
        "motion_quality_thresholds": mapping_metadata.get(
            "motion_quality_thresholds"
        ),
        "pnp_quality_thresholds": mapping_metadata.get(
            "pnp_quality_thresholds"
        ),
        "depth_quality_thresholds": mapping_metadata.get(
            "depth_quality_thresholds"
        ),
        "trajectory_refinement": refinement,
    }
    return {
        "timestamp": timestamp,
        "cli_arguments": cli_arguments,
        "parsed_arguments": parsed_arguments,
        "mapping_subprocess_command": mapping_command,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "package_versions": package_versions(),
        "depth_model": mapping_metadata.get("model"),
        "device": mapping_metadata.get("device"),
        "configuration_file": str(Path(config_path).resolve()),
        "resolved_configuration": resolved_configuration,
        "input_video": mapping_metadata.get("source"),
        "video": asdict(video),
        "random_seeds": {
            "explicit_seeds_configured": False,
            "values": {},
        },
        "determinism_note": (
            "Perfect determinism is not claimed. OpenCV robust estimation, "
            "PyTorch/model execution, hardware, and library versions can affect results."
        ),
    }


def final_report_markdown(
    summary: dict[str, Any], artifacts: dict[str, Any]
) -> str:
    input_data = summary["input"]
    pipeline = summary["pipeline"]
    results = summary["results"]
    trajectory = summary["trajectory"]
    runtime = summary["runtime"]
    refinement = pipeline["trajectory_refinement"]
    lines = [
        "# DepthSLAM Final Run",
        "",
        "## Input",
        "",
        f"- Source: `{input_data['source_video']}`",
        f"- Monocular RGB video: {input_data['image_width']} x "
        f"{input_data['image_height']}, {input_data['fps']} FPS, "
        f"{input_data['frame_count']} frames",
        "- Camera intrinsics source: `manual_command_line` (approximate and uncalibrated)",
        "- Warning: 3D geometry may be warped by inaccurate intrinsics.",
        "",
        "## Pipeline",
        "",
        f"- Depth model: `{input_data['depth_model']}` on `{input_data['device']}`",
        f"- Candidate sampling interval: {pipeline['sample_every']}",
        f"- Scale mode: `{pipeline['scale_mode']}`",
        f"- Trajectory refinement: `{refinement['status']}` "
        f"(`{refinement.get('method')}`)",
        "",
        "## Mapping Result",
        "",
        f"- Accepted / skipped / rejected: {results['accepted_keyframes']} / "
        f"{results['skipped_frames']} / {results['rejected_frames']}",
        f"- Final map points: {results['final_map_point_count']}",
        "- Map coordinates are relative and non-metric, not metres.",
        "",
        "## Trajectory",
        "",
        f"- Poses: {results['trajectory_pose_count']}",
        f"- Units: `{trajectory['units']}` (relative, non-metric)",
        f"- Suspicious jumps: {trajectory['suspicious_jump_count']}",
        f"- Modified poses: {trajectory['modified_pose_count']}",
        "- The raw accepted-pose trajectory remains authoritative.",
        "- Any refined trajectory is analysis-only; the fused map uses raw poses.",
        "",
        "## Evaluation",
        "",
        f"- Mean/median PnP inlier ratio: "
        f"{results['pnp_inlier_ratio']['mean']} / "
        f"{results['pnp_inlier_ratio']['median']}",
        f"- Mean/median reprojection RMSE (pixels): "
        f"{results['reprojection_rmse_pixels']['mean']} / "
        f"{results['reprojection_rmse_pixels']['median']}",
        f"- Mean/median depth-alignment inlier ratio: "
        f"{results['depth_alignment_inlier_ratio']['mean']} / "
        f"{results['depth_alignment_inlier_ratio']['median']}",
        "- These are internal-consistency metrics, not absolute accuracy.",
        "",
        "## Runtime",
        "",
        f"- Mapping: {runtime['mapping_seconds']} seconds",
        f"- Evaluation: {runtime['evaluation_seconds']} seconds",
        f"- Trajectory refinement: {runtime['trajectory_refinement_seconds']} seconds",
        f"- Packaging: {runtime['packaging_seconds']} seconds",
        f"- End-to-end: {runtime['end_to_end_seconds']} seconds",
        "- Timings are environment dependent and are not formal benchmarks.",
        "",
        "## Limitations",
        "",
        "- The input is monocular and depth, trajectory, and map scale remain "
        "relative/non-metric.",
        "- Camera intrinsics were manually estimated and are not calibrated.",
        "- No ground-truth trajectory is available; ATE and RPE were not computed.",
        "- No loop closure, bundle adjustment, or pose-graph optimization is implemented.",
        "- No absolute trajectory or map accuracy is claimed.",
        "- Smoothing does not recover ground truth or correct accumulated drift.",
        "",
        "## Generated Artifacts",
        "",
    ]
    for path in _artifact_strings(artifacts):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, value: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
    return path
