"""CLI for depth-scaled short-video incremental relative mapping."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.depth_estimator import DEFAULT_MODEL, DepthEstimator
from src.drift_diagnostics import (
    collect_drift_diagnostics,
    save_drift_diagnostics,
)
from src.depth_pose_estimator import DepthPoseEstimator
from src.depth_stabilization import DepthStabilizationConfig
from src.keyframe_selector import KeyframeSelector, KeyframeThresholds
from src.map_builder import MappingFrame, RelativeMapBuilder, RelativeMapResult
from src.ply_io import write_ascii_ply
from src.pose_chain_diagnostics import (
    PoseChainDiagnosticConfig,
    PoseChainDiagnosticRow,
    ReferenceDirectPoseEstimator,
    analyze_pose_chain,
    save_pose_chain_diagnostics,
)
from src.pose_refinement_3d import (
    PoseRefinement3DConfig,
    RobustPoseRefiner3D,
)
from src.robust_filtering import coordinate_statistics
from src.temporal_depth_normalization import TemporalDepthNormalizationConfig
from src.video_loader import VideoLoader
from src.visual_outputs import (
    CloudVisualArtifacts,
    display_cleaning_metadata,
    save_cloud_visual_artifacts,
    save_depth_stabilization_comparison,
    save_temporal_depth_normalization_comparison,
    save_map_overview_panel,
    save_point_cloud_preview,
    save_trajectory_previews,
)
from src.visualization import colorize_depth
from tools.run_depth_geometry import build_motion_estimator, build_tracker
from tools.run_motion import camera_matrix_from_args, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a depth-scaled short multi-frame map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Calibrated focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Calibrated focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Calibrated principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Calibrated principal point y (pixels)")
    parser.add_argument("--sample-every", type=int, metavar="N")
    parser.add_argument(
        "--max-mapping-frames", "--max-candidate-frames",
        dest="max_mapping_frames", type=int,
        help="Maximum sampled candidate frames to consider",
    )
    parser.add_argument(
        "--keyframes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Stage 6 visual keyframe selection",
    )
    parser.add_argument(
        "--kf-min-displacement", type=float,
        help="Minimum median geometric-inlier displacement in pixels",
    )
    parser.add_argument(
        "--kf-min-rotation-deg", type=float,
        help="Minimum recovered rotation magnitude in degrees",
    )
    parser.add_argument(
        "--kf-max-gap", type=int,
        help="Force a keyframe attempt after this many candidates",
    )
    parser.add_argument(
        "--scale-mode", choices=("depth-pnp", "fixed-step"),
        help="depth-pnp is reconstruction mode; fixed-step is arbitrary debug mode",
    )
    parser.add_argument(
        "--translation-step", type=float,
        help="Arbitrary magnitude used only with --scale-mode fixed-step",
    )
    parser.add_argument("--point-cloud-stride", type=int)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--disparity-denominator-epsilon", type=float)
    parser.add_argument("--depth-percentile-low", type=float)
    parser.add_argument("--depth-percentile-high", type=float)
    parser.add_argument(
        "--global-outlier-percentile", type=float,
        help="Keep map points at or below this center-distance percentile",
    )
    parser.add_argument(
        "--min-valid-depth-ratio", type=float,
        help="Minimum full-frame finite positive aligned-depth fraction",
    )
    parser.add_argument(
        "--max-denominator-reject-ratio", type=float,
        help="Maximum unsafe aligned-disparity denominator fraction",
    )
    parser.add_argument(
        "--min-depth-alignment-inliers", type=int,
        help="Minimum robust affine depth-alignment inlier count",
    )
    parser.add_argument(
        "--min-depth-alignment-inlier-ratio", type=float,
        help="Minimum robust affine depth-alignment inlier fraction",
    )
    parser.add_argument(
        "--max-relative-z-p99-over-median", type=float,
        help="Maximum allowed aligned relative-Z p99/median ratio",
    )
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--pose-refinement-3d",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable optional robust matched-point 3D-to-3D pose refinement",
    )
    parser.add_argument(
        "--depth-stabilization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable experimental post-alignment relative-Z tail rejection",
    )
    parser.add_argument(
        "--temporal-depth-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable experimental pairwise aligned-depth scale normalization",
    )
    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save map, trajectory, and overview PNG previews",
    )
    parser.add_argument(
        "--save-display-clean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save a conservatively cleaned display-only map PLY",
    )
    parser.add_argument(
        "--save-drift-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save diagnostic-only accepted-keyframe drift tables and plots",
    )
    parser.add_argument(
        "--pose-chain-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Estimate diagnostic-only direct reference-to-keyframe poses",
    )
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--minimum-matches", type=int)
    parser.add_argument("--ransac-probability", type=float)
    parser.add_argument("--ransac-threshold", type=float, metavar="PIXELS")
    parser.add_argument("--minimum-inliers", type=int)
    parser.add_argument("--minimum-inlier-ratio", type=float)
    parser.add_argument("--pnp-reprojection-error", type=float, metavar="PIXELS")
    parser.add_argument("--minimum-pnp-inliers", type=int)
    parser.add_argument("--minimum-pnp-inlier-ratio", type=float)
    parser.add_argument("--output-dir", type=Path, help="Root for generated outputs")
    return parser.parse_args()


def setting(cli_value: Any, config: dict[str, Any], name: str, fallback: Any) -> Any:
    return cli_value if cli_value is not None else config.get(name, fallback)


def select_mapping_frames(
    video: Path, sample_every: int, maximum: int
) -> tuple[list[MappingFrame], float]:
    frames: list[MappingFrame] = []
    source_fps = 0.0
    for frame in VideoLoader(video, sample_every):
        source_fps = frame.source_fps
        frames.append(MappingFrame(
            image=frame.image,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
        ))
        if len(frames) >= maximum:
            break
    if not frames:
        raise RuntimeError(f"No mapping frames could be read from video: {video}")
    return frames, source_fps


def create_output_directory(root: Path, video: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / f"relative_map_{video.stem}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_trajectory(directory: Path, result: RelativeMapResult) -> None:
    np.save(directory / "trajectory_relative.npy", result.trajectory_positions)
    with (directory / "trajectory_relative.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "x", "y", "z", "accepted"))
        for frame_index, position in zip(
            result.trajectory_frame_indices, result.trajectory_positions
        ):
            writer.writerow((
                int(frame_index), float(position[0]), float(position[1]),
                float(position[2]), "true"
            ))


def save_frame_statistics(directory: Path, result: RelativeMapResult) -> None:
    with (directory / "frame_stats.jsonl").open("w", encoding="utf-8") as file:
        for statistics in result.frame_statistics:
            file.write(json.dumps(statistics.to_dict()) + "\n")


def save_pair_alignment_artifacts(
    directory: Path,
    result: RelativeMapResult,
    *,
    save_previews: bool,
    preview_max_points: int,
) -> dict[str, str] | None:
    """Save the first accepted refined pair under baseline and selected poses."""

    pair = result.pair_alignment
    if pair is None:
        return None
    before_path = write_ascii_ply(
        directory / "pair_alignment_before.ply",
        pair.clouds.before_points,
        pair.clouds.colors,
    )
    after_path = write_ascii_ply(
        directory / "pair_alignment_after.ply",
        pair.clouds.after_points,
        pair.clouds.colors,
    )
    metrics_path = directory / "pair_alignment_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(pair.metrics, file, indent=2)
    paths = {
        "before_ply": before_path.name,
        "after_ply": after_path.name,
        "metrics": metrics_path.name,
    }
    if save_previews:
        before_preview = save_point_cloud_preview(
            directory / "pair_alignment_before_oblique.png",
            pair.clouds.before_points,
            pair.clouds.colors,
            view="oblique",
            title="Pair alignment before 3D refinement (PnP, relative/non-metric)",
            max_points=preview_max_points,
        )
        after_preview = save_point_cloud_preview(
            directory / "pair_alignment_after_oblique.png",
            pair.clouds.after_points,
            pair.clouds.colors,
            view="oblique",
            title="Pair alignment after gated 3D refinement (relative/non-metric)",
            max_points=preview_max_points,
        )
        paths.update({
            "before_preview": before_preview.name,
            "after_preview": after_preview.name,
        })
    return paths


def save_optional_drift_diagnostics(
    directory: Path,
    result: RelativeMapResult,
    metadata: dict[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    """Write read-only drift diagnostics, or explicitly record they are off."""

    if not enabled:
        metadata["drift_diagnostics"] = {
            "enabled": False,
            "diagnostic_only": True,
        }
        return None
    drift = save_drift_diagnostics(
        directory,
        collect_drift_diagnostics(result.frame_statistics),
        generate_plots=True,
    )
    metadata["drift_diagnostics"] = {
        "enabled": True,
        "diagnostic_only": True,
        "directory": Path(drift["directory"]).name,
        "csv": Path(drift["csv"]).name,
        "json": Path(drift["json"]).name,
        "summary": Path(drift["summary_path"]).name,
        "plots": [Path(path).name for path in drift["plots"]],
        "results": drift["summary"],
    }
    return drift


def save_optional_pose_chain_diagnostics(
    directory: Path,
    metadata: dict[str, Any],
    rows: list[PoseChainDiagnosticRow] | None,
    config: PoseChainDiagnosticConfig,
) -> dict[str, Any] | None:
    """Write isolated direct-pose diagnostics without changing mapping data."""

    if rows is None:
        metadata["pose_chain_diagnostics"] = {"enabled": False}
        return None
    result = save_pose_chain_diagnostics(directory, rows, config)
    metadata["pose_chain_diagnostics"] = {
        "enabled": True,
        "diagnostic_only": True,
        "directory": Path(result["directory"]).name,
        "csv": Path(result["csv"]).name,
        "json": Path(result["json"]).name,
        "summary": Path(result["summary_path"]).name,
        "plots": [Path(path).name for path in result["plots"]],
        "quality_thresholds": {
            "minimum_geometric_inlier_ratio": (
                config.minimum_geometric_inlier_ratio
            ),
            "minimum_pnp_inlier_ratio": config.minimum_pnp_inlier_ratio,
            "maximum_reprojection_rmse_pixels": (
                config.maximum_reprojection_rmse_pixels
            ),
        },
        "results": result["summary"],
    }
    return result


def temporal_map_statistics(points: np.ndarray) -> dict[str, Any]:
    """Return relative/non-metric axis summaries for a comparison map."""

    statistics = coordinate_statistics(points)
    z = statistics["z"]
    median = z["median"]
    return {
        "coordinate_scale": "relative_non_metric",
        "axis_percentiles": statistics,
        "z_median": median,
        "z_p95": z["p95"],
        "z_p99": z["p99"],
        "z_p99_over_median": (
            None if median == 0.0 else float(z["p99"] / abs(median))
        ),
    }


def save_outputs(
    directory: Path,
    result: RelativeMapResult,
    metadata: dict[str, Any],
    visual_config: dict[str, Any],
    *,
    save_previews: bool,
    save_display_clean: bool,
    save_drift: bool,
    pose_chain_rows: list[PoseChainDiagnosticRow] | None,
    pose_chain_config: PoseChainDiagnosticConfig,
) -> tuple[Path, CloudVisualArtifacts]:
    np.save(directory / "global_points_relative.npy", result.fused_cloud.points)
    np.save(directory / "global_colors_rgb.npy", result.fused_cloud.colors)
    z_low = visual_config.get("display_z_percentile_min", 1.0)
    z_high = visual_config.get("display_z_percentile_max", 99.0)
    if (z_low is None) != (z_high is None):
        raise ValueError("both display Z percentiles must be set or null")
    z_percentiles = (
        None if z_low is None else (float(z_low), float(z_high))
    )
    center_percentile = visual_config.get(
        "display_center_distance_percentile", 99.5
    )
    artifacts = save_cloud_visual_artifacts(
        directory,
        result.fused_cloud.points,
        result.fused_cloud.colors,
        raw_filenames=(
            "global_relative_map.ply",
            "global_relative_map_raw.ply",
        ),
        display_filename="global_relative_map_display.ply",
        preview_prefix="global_map_preview",
        title="Global relative map (display only, non-metric)",
        save_display_clean=save_display_clean,
        save_previews=save_previews,
        z_percentiles=z_percentiles,
        center_distance_percentile=(
            None if center_percentile is None else float(center_percentile)
        ),
        center_mad_multiplier=float(
            visual_config.get("display_center_mad_multiplier", 6.0)
        ),
        preview_max_points=int(visual_config.get("preview_max_points", 40_000)),
    )
    save_trajectory(directory, result)
    save_frame_statistics(directory, result)
    pair_artifacts = save_pair_alignment_artifacts(
        directory,
        result,
        save_previews=save_previews,
        preview_max_points=int(visual_config.get("preview_max_points", 40_000)),
    )
    if save_previews:
        save_trajectory_previews(directory, result.trajectory_positions)
        save_map_overview_panel(
            directory / "map_overview_panel.png",
            result.preview_image_bgr,
            colorize_depth(result.preview_depth_values),
            result.trajectory_positions,
            artifacts.cleaning.points,
            artifacts.cleaning.colors,
            max_points=int(visual_config.get("overview_max_points", 20_000)),
        )

    metadata["visual_output"] = display_cleaning_metadata(
        artifacts.cleaning,
        raw_artifact="global_relative_map_raw.ply",
        compatibility_artifact="global_relative_map.ply",
        display_artifact=(
            "global_relative_map_display.ply" if save_display_clean else None
        ),
    )
    metadata["visual_output"].update({
        "previews_saved": save_previews,
        "display_clean_saved": save_display_clean,
    })
    metadata["pair_alignment_artifacts"] = pair_artifacts
    if result.depth_stabilization_enabled:
        if result.stabilized_fused_cloud is None:
            raise RuntimeError("enabled depth stabilization produced no comparison map")
        comparison = save_depth_stabilization_comparison(
            directory,
            result.fused_cloud.points,
            result.fused_cloud.colors,
            result.stabilized_fused_cloud.points,
            result.stabilized_fused_cloud.colors,
            preview_max_points=int(visual_config.get("preview_max_points", 40_000)),
        )
        metadata["depth_stabilization"]["comparison_artifacts"] = {
            name: path.name for name, path in comparison.items()
        }
        metadata["depth_stabilization"].update({
            "unstabilized_map_point_count": result.fused_cloud.output_point_count,
            "stabilized_map_point_count": (
                result.stabilized_fused_cloud.output_point_count
            ),
            "comparison_display_filtering": "none",
        })
    else:
        metadata["depth_stabilization"]["comparison_artifacts"] = None
    if result.temporal_depth_normalization_enabled:
        if result.temporal_normalized_fused_cloud is None:
            raise RuntimeError(
                "enabled temporal normalization produced no comparison map"
            )
        temporal_artifacts = save_temporal_depth_normalization_comparison(
            directory,
            result.fused_cloud.points,
            result.fused_cloud.colors,
            result.temporal_normalized_fused_cloud.points,
            result.temporal_normalized_fused_cloud.colors,
            preview_max_points=int(visual_config.get("preview_max_points", 40_000)),
        )
        metadata["temporal_depth_normalization"].update({
            "comparison_artifacts": {
                name: path.name for name, path in temporal_artifacts.items()
            },
            "same_accepted_keyframes": True,
            "same_poses": True,
            "same_fusion_and_voxel_settings": True,
            "comparison_display_filtering": "none",
            "baseline": {
                "fused_input_point_count": result.raw_fused_point_count,
                "final_point_count": result.fused_cloud.output_point_count,
                **temporal_map_statistics(result.fused_cloud.points),
            },
            "temporal_normalized": {
                "fused_input_point_count": (
                    result.temporal_normalized_raw_fused_point_count
                ),
                "final_point_count": (
                    result.temporal_normalized_fused_cloud.output_point_count
                ),
                **temporal_map_statistics(
                    result.temporal_normalized_fused_cloud.points
                ),
            },
        })
    else:
        metadata["temporal_depth_normalization"]["comparison_artifacts"] = None
    save_optional_drift_diagnostics(
        directory, result, metadata, enabled=save_drift
    )
    save_optional_pose_chain_diagnostics(
        directory, metadata, pose_chain_rows, pose_chain_config
    )
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return artifacts.raw_paths[0], artifacts


def scientific_metadata(
    result: RelativeMapResult, scale_mode: str, translation_step: float
) -> dict[str, Any]:
    """Return the scientific labels shared by saved metadata and tests."""
    return {
        "is_metric": result.is_metric,
        "depth_type": result.depth_type,
        "depth_representation": result.depth_representation,
        "geometry_depth_representation": (
            "camera_z" if result.is_metric else "relative_camera_z_proxy"
        ),
        "scale_mode": scale_mode,
        "scale_estimation_method": (
            "depth_pnp" if scale_mode == "depth-pnp" else "fixed_step_debug"
        ),
        "translation_scale": (
            "depth_assisted_pnp" if scale_mode == "depth-pnp"
            else "arbitrary_fixed_step_debug"
        ),
        "translation_step": translation_step if scale_mode == "fixed-step" else None,
        "translation_units": result.translation_units,
        "depth_alignment_method": (
            "metric_model" if result.is_metric
            else "scale_and_shift_per_accepted_pair"
            if scale_mode == "depth-pnp"
            else "none"
        ),
        "coordinate_units": result.translation_units,
        "voxel_units": result.translation_units,
    }


def robustness_metadata(result: RelativeMapResult) -> dict[str, Any]:
    """Return explicit pre/post counts and global geometry diagnostics."""
    filtered = result.global_filter
    return {
        "raw_fused_point_count": result.raw_fused_point_count,
        "voxel_downsampled_point_count": result.voxel_downsampled_point_count,
        "global_outlier_filter": {
            "method": filtered.method,
            "percentile": filtered.percentile,
            "distance_threshold": filtered.distance_threshold,
            "points_before": filtered.input_count,
            "points_rejected": filtered.rejected_count,
            "points_after": filtered.output_count,
        },
        "global_geometry_diagnostics": {
            "coordinate_statistics_before": filtered.coordinate_statistics_before,
            "coordinate_statistics_after": filtered.coordinate_statistics_after,
            "robust_center": [float(value) for value in filtered.robust_center],
            "center_distance_statistics": filtered.distance_statistics,
            "diagnostic_robust_radius_median_plus_6_scaled_mad": (
                filtered.diagnostic_robust_radius
            ),
            "points_outside_diagnostic_robust_radius": (
                filtered.points_outside_diagnostic_radius
            ),
        },
    }


def main() -> int:
    args = parse_args()
    pipeline_started = perf_counter()
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {args.video}")
        config = load_config(args.config)
        model_config = config.get("model", {})
        motion_config = config.get("motion", {})
        keyframe_config = config.get("keyframes", {})
        map_config = config.get("map", {})
        pose_refinement_config = config.get("pose_refinement_3d", {})
        depth_stabilization_config = config.get("depth_stabilization", {})
        pose_chain_config = config.get("pose_chain_diagnostics", {})
        temporal_normalization_config = config.get(
            "temporal_depth_normalization", {}
        )
        visual_config = config.get("visual_output", {})
        diagnostics_config = config.get("diagnostics", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config,
            motion_config,
            keyframe_config,
            map_config,
            pose_refinement_config,
            depth_stabilization_config,
            pose_chain_config,
            temporal_normalization_config,
            visual_config,
            diagnostics_config,
            output_config,
        )):
            raise ValueError(
                "model, motion, keyframes, map, pose_refinement_3d, "
                "depth_stabilization, temporal_depth_normalization, "
                "pose_chain_diagnostics, visual_output, diagnostics, and "
                "output config sections must be mappings"
            )
        camera_matrix = camera_matrix_from_args(args, config)

        sample_every = int(setting(args.sample_every, map_config, "sample_every", 10))
        maximum = int(setting(
            args.max_mapping_frames, map_config, "max_mapping_frames", 5
        ))
        translation_step = float(setting(
            args.translation_step, map_config, "translation_step", 1.0
        ))
        scale_mode = setting(args.scale_mode, map_config, "scale_mode", "depth-pnp")
        cloud_stride = int(setting(
            args.point_cloud_stride, map_config, "point_cloud_stride", 6
        ))
        voxel_size = float(setting(args.voxel_size, map_config, "voxel_size", 0.05))
        denominator_epsilon = float(setting(
            args.disparity_denominator_epsilon,
            map_config,
            "disparity_denominator_epsilon",
            1e-3,
        ))
        depth_percentile_low = setting(
            args.depth_percentile_low, map_config, "depth_percentile_low", 1.0
        )
        depth_percentile_high = setting(
            args.depth_percentile_high, map_config, "depth_percentile_high", 99.0
        )
        global_outlier_percentile = setting(
            args.global_outlier_percentile,
            map_config,
            "global_outlier_percentile",
            99.5,
        )
        min_valid_depth_ratio = setting(
            args.min_valid_depth_ratio, map_config, "min_valid_depth_ratio", 0.60
        )
        max_denominator_reject_ratio = setting(
            args.max_denominator_reject_ratio,
            map_config,
            "max_denominator_reject_ratio",
            0.30,
        )
        min_depth_alignment_inliers = setting(
            args.min_depth_alignment_inliers,
            map_config,
            "min_depth_alignment_inliers",
            500,
        )
        min_depth_alignment_inlier_ratio = setting(
            args.min_depth_alignment_inlier_ratio,
            map_config,
            "min_depth_alignment_inlier_ratio",
            0.30,
        )
        max_relative_z_p99_over_median = setting(
            args.max_relative_z_p99_over_median,
            map_config,
            "max_relative_z_p99_over_median",
            50.0,
        )
        keyframes_enabled = (
            args.keyframes
            if args.keyframes is not None
            else keyframe_config.get("enabled", True)
        )
        save_previews = bool(setting(
            args.save_previews, visual_config, "save_previews", True
        ))
        save_display_clean = bool(setting(
            args.save_display_clean, visual_config, "save_display_clean", True
        ))
        save_drift = bool(setting(
            args.save_drift_diagnostics,
            diagnostics_config,
            "save_drift_diagnostics",
            True,
        ))
        refinement_3d_settings = PoseRefinement3DConfig(
            enabled=bool(setting(
                args.pose_refinement_3d,
                pose_refinement_config,
                "enabled",
                False,
            )),
            minimum_correspondences=int(pose_refinement_config.get(
                "minimum_correspondences", 100
            )),
            minimum_inliers=int(pose_refinement_config.get(
                "minimum_inliers", 80
            )),
            minimum_inlier_ratio=float(pose_refinement_config.get(
                "minimum_inlier_ratio", 0.40
            )),
            minimum_relative_improvement=float(pose_refinement_config.get(
                "minimum_relative_improvement", 0.10
            )),
            random_seed=int(pose_refinement_config.get("random_seed", 0)),
            ransac_iterations=int(pose_refinement_config.get(
                "ransac_iterations", 512
            )),
            residual_threshold_fraction=float(pose_refinement_config.get(
                "residual_threshold_fraction", 0.05
            )),
            maximum_translation_change_ratio=float(pose_refinement_config.get(
                "maximum_translation_change_ratio", 2.0
            )),
            maximum_rotation_change_degrees=float(pose_refinement_config.get(
                "maximum_rotation_change_degrees", 10.0
            )),
        )
        depth_stabilization_settings = DepthStabilizationConfig(
            enabled=bool(setting(
                args.depth_stabilization,
                depth_stabilization_config,
                "enabled",
                False,
            )),
            max_z_over_median=float(depth_stabilization_config.get(
                "max_z_over_median", 12.0
            )),
            mad_multiplier=float(depth_stabilization_config.get(
                "mad_multiplier", 8.0
            )),
            minimum_valid_ratio=float(depth_stabilization_config.get(
                "minimum_valid_ratio", 0.70
            )),
            maximum_removed_ratio=float(depth_stabilization_config.get(
                "maximum_removed_ratio", 0.20
            )),
        )
        pose_chain_settings = PoseChainDiagnosticConfig(
            enabled=bool(setting(
                args.pose_chain_diagnostics,
                pose_chain_config,
                "enabled",
                False,
            )),
            minimum_geometric_inlier_ratio=float(pose_chain_config.get(
                "minimum_geometric_inlier_ratio",
                setting(
                    args.minimum_inlier_ratio,
                    motion_config,
                    "minimum_inlier_ratio",
                    0.25,
                ),
            )),
            minimum_pnp_inlier_ratio=float(pose_chain_config.get(
                "minimum_pnp_inlier_ratio",
                setting(
                    args.minimum_pnp_inlier_ratio,
                    map_config,
                    "minimum_pnp_inlier_ratio",
                    0.25,
                ),
            )),
            maximum_reprojection_rmse_pixels=float(pose_chain_config.get(
                "maximum_reprojection_rmse_pixels",
                setting(
                    args.pnp_reprojection_error,
                    map_config,
                    "pnp_reprojection_error_pixels",
                    3.0,
                ),
            )),
            translation_relative_difference_threshold=float(
                pose_chain_config.get(
                    "translation_relative_difference_threshold", 0.25
                )
            ),
            rotation_difference_threshold_deg=float(pose_chain_config.get(
                "rotation_difference_threshold_deg", 5.0
            )),
            increasing_fraction_threshold=float(pose_chain_config.get(
                "increasing_fraction_threshold", 0.70
            )),
        )
        temporal_normalization_settings = TemporalDepthNormalizationConfig(
            enabled=bool(setting(
                args.temporal_depth_normalization,
                temporal_normalization_config,
                "enabled",
                False,
            )),
            minimum_correspondences=int(temporal_normalization_config.get(
                "minimum_correspondences", 200
            )),
            minimum_inliers=int(temporal_normalization_config.get(
                "minimum_inliers", 150
            )),
            minimum_inlier_ratio=float(temporal_normalization_config.get(
                "minimum_inlier_ratio", 0.50
            )),
            minimum_scale=float(temporal_normalization_config.get(
                "minimum_scale", 0.70
            )),
            maximum_scale=float(temporal_normalization_config.get(
                "maximum_scale", 1.30
            )),
            maximum_log_mad=float(temporal_normalization_config.get(
                "maximum_log_mad", 0.25
            )),
            minimum_cumulative_scale=float(temporal_normalization_config.get(
                "minimum_cumulative_scale", 0.50
            )),
            maximum_cumulative_scale=float(temporal_normalization_config.get(
                "maximum_cumulative_scale", 2.0
            )),
            log_mad_outlier_multiplier=float(temporal_normalization_config.get(
                "log_mad_outlier_multiplier", 3.5
            )),
        )
        keyframe_thresholds = KeyframeThresholds(
            enabled=keyframes_enabled,
            min_good_matches=keyframe_config.get("min_good_matches", 100),
            min_geometric_inliers=keyframe_config.get(
                "min_geometric_inliers", 80
            ),
            min_geometric_inlier_ratio=keyframe_config.get(
                "min_geometric_inlier_ratio", 0.40
            ),
            min_median_feature_displacement_px=setting(
                args.kf_min_displacement,
                keyframe_config,
                "min_median_feature_displacement_px",
                8.0,
            ),
            min_rotation_deg=setting(
                args.kf_min_rotation_deg,
                keyframe_config,
                "min_rotation_deg",
                1.0,
            ),
            max_frames_without_keyframe=setting(
                args.kf_max_gap,
                keyframe_config,
                "max_frames_without_keyframe",
                30,
            ),
        )
        if sample_every < 1 or maximum < 1 or cloud_stride < 1:
            raise ValueError("sampling, maximum frame count, and cloud stride must be positive")

        selected_frames, source_fps = select_mapping_frames(
            args.video, sample_every, maximum
        )
        print(
            f"Selected {len(selected_frames)} mapping frames: "
            + ", ".join(str(frame.frame_index) for frame in selected_frames)
        )

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        print(f"Loading {model_name} on {device}...")
        depth_estimator = DepthEstimator(model_name=model_name, device=device)
        builder = RelativeMapBuilder(
            depth_estimator=depth_estimator,
            feature_tracker=build_tracker(args, motion_config),
            motion_estimator=build_motion_estimator(args, motion_config),
            camera_matrix=camera_matrix,
            scale_mode=scale_mode,
            translation_step=translation_step,
            point_cloud_stride=cloud_stride,
            voxel_size=voxel_size,
            disparity_denominator_epsilon=denominator_epsilon,
            depth_percentile_low=depth_percentile_low,
            depth_percentile_high=depth_percentile_high,
            global_outlier_percentile=global_outlier_percentile,
            min_valid_depth_ratio=min_valid_depth_ratio,
            max_denominator_reject_ratio=max_denominator_reject_ratio,
            min_depth_alignment_inliers=min_depth_alignment_inliers,
            min_depth_alignment_inlier_ratio=min_depth_alignment_inlier_ratio,
            max_relative_z_p99_over_median=max_relative_z_p99_over_median,
            keyframe_selector=KeyframeSelector(keyframe_thresholds),
            depth_pose_estimator=DepthPoseEstimator(
                minimum_correspondences=int(map_config.get(
                    "minimum_pnp_correspondences", 6
                )),
                minimum_inliers=int(setting(
                    args.minimum_pnp_inliers, map_config,
                    "minimum_pnp_inliers", 6,
                )),
                minimum_inlier_ratio=float(setting(
                    args.minimum_pnp_inlier_ratio, map_config,
                    "minimum_pnp_inlier_ratio", 0.25,
                )),
                reprojection_error_pixels=float(setting(
                    args.pnp_reprojection_error, map_config,
                    "pnp_reprojection_error_pixels", 3.0,
                )),
            ),
            pose_refiner_3d=RobustPoseRefiner3D(refinement_3d_settings),
            depth_stabilization=depth_stabilization_settings,
            capture_pose_chain_diagnostics=pose_chain_settings.enabled,
            temporal_depth_normalization=temporal_normalization_settings,
        )
        result = builder.build(selected_frames)
        pose_chain_rows: list[PoseChainDiagnosticRow] | None = None
        pose_chain_runtime_seconds = 0.0
        if pose_chain_settings.enabled:
            if result.pose_chain_frames is None:
                raise RuntimeError("pose-chain diagnostic inputs were not captured")
            direct_started = perf_counter()
            direct_estimator = ReferenceDirectPoseEstimator(
                build_tracker(args, motion_config),
                build_motion_estimator(args, motion_config),
                DepthPoseEstimator(
                    minimum_correspondences=int(map_config.get(
                        "minimum_pnp_correspondences", 6
                    )),
                    minimum_inliers=int(setting(
                        args.minimum_pnp_inliers,
                        map_config,
                        "minimum_pnp_inliers",
                        6,
                    )),
                    minimum_inlier_ratio=float(setting(
                        args.minimum_pnp_inlier_ratio,
                        map_config,
                        "minimum_pnp_inlier_ratio",
                        0.25,
                    )),
                    reprojection_error_pixels=float(setting(
                        args.pnp_reprojection_error,
                        map_config,
                        "pnp_reprojection_error_pixels",
                        3.0,
                    )),
                ),
                camera_matrix,
            )
            pose_chain_rows = analyze_pose_chain(
                result.pose_chain_frames,
                direct_estimator,
                pose_chain_settings,
            )
            pose_chain_runtime_seconds = perf_counter() - direct_started
        total_pipeline_runtime = perf_counter() - pipeline_started

        output_root = args.output_dir or Path(output_config.get("directory", "outputs"))
        run_directory = create_output_directory(output_root / "relative_map", args.video)
        metadata = {
            "map_type": (
                "relative_keyframe_map"
                if keyframe_thresholds.enabled
                else "relative_fixed_sample_map"
            ),
            **scientific_metadata(result, scale_mode, translation_step),
            "source": str(args.video.resolve()),
            "source_fps": source_fps,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "camera_intrinsics": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
            "model": depth_estimator.model_name,
            "device": depth_estimator.device,
            "sample_every": sample_every,
            "max_mapping_frames": maximum,
            "point_cloud_stride": cloud_stride,
            "voxel_size": voxel_size,
            "keyframe_selection": {
                "enabled": keyframe_thresholds.enabled,
                "min_good_matches": keyframe_thresholds.min_good_matches,
                "min_geometric_inliers": keyframe_thresholds.min_geometric_inliers,
                "min_geometric_inlier_ratio": (
                    keyframe_thresholds.min_geometric_inlier_ratio
                ),
                "min_median_feature_displacement_px": (
                    keyframe_thresholds.min_median_feature_displacement_px
                ),
                "min_rotation_deg": keyframe_thresholds.min_rotation_deg,
                "max_frames_without_keyframe": (
                    keyframe_thresholds.max_frames_without_keyframe
                ),
            },
            "motion_quality_thresholds": {
                "ratio_threshold": builder.feature_tracker.ratio_threshold,
                "minimum_matches": builder.feature_tracker.minimum_matches,
                "ransac_probability": builder.motion_estimator.ransac_probability,
                "ransac_threshold_pixels": builder.motion_estimator.ransac_threshold_pixels,
                "minimum_inliers": builder.motion_estimator.minimum_inliers,
                "minimum_inlier_ratio": builder.motion_estimator.minimum_inlier_ratio,
            },
            "pnp_quality_thresholds": {
                "minimum_correspondences": (
                    builder.depth_pose_estimator.minimum_correspondences
                ),
                "minimum_inliers": builder.depth_pose_estimator.minimum_inliers,
                "minimum_inlier_ratio": (
                    builder.depth_pose_estimator.minimum_inlier_ratio
                ),
                "reprojection_error_pixels": (
                    builder.depth_pose_estimator.reprojection_error_pixels
                ),
            },
            "pose_refinement_3d": {
                "enabled": refinement_3d_settings.enabled,
                "transform_convention": "current_from_previous",
                "scale_estimation_applied": False,
                "minimum_correspondences": (
                    refinement_3d_settings.minimum_correspondences
                ),
                "minimum_inliers": refinement_3d_settings.minimum_inliers,
                "minimum_inlier_ratio": (
                    refinement_3d_settings.minimum_inlier_ratio
                ),
                "minimum_relative_improvement": (
                    refinement_3d_settings.minimum_relative_improvement
                ),
                "random_seed": refinement_3d_settings.random_seed,
                "ransac_iterations": refinement_3d_settings.ransac_iterations,
                "residual_threshold_fraction": (
                    refinement_3d_settings.residual_threshold_fraction
                ),
                "maximum_translation_change_ratio": (
                    refinement_3d_settings.maximum_translation_change_ratio
                ),
                "maximum_rotation_change_degrees": (
                    refinement_3d_settings.maximum_rotation_change_degrees
                ),
                "attempted_pairs": sum(
                    item.refinement_3d_attempted
                    for item in result.frame_statistics
                ),
                "accepted_pairs": sum(
                    item.refinement_3d_accepted
                    for item in result.frame_statistics
                ),
                "reason_counts": {
                    reason: sum(
                        item.refinement_3d_reason == reason
                        for item in result.frame_statistics
                    )
                    for reason in sorted({
                        item.refinement_3d_reason
                        for item in result.frame_statistics
                        if item.refinement_3d_reason is not None
                    })
                },
                "first_pair_result": (
                    None
                    if result.pair_alignment is None
                    else result.pair_alignment.metrics
                ),
            },
            "depth_stabilization": {
                "enabled": depth_stabilization_settings.enabled,
                "experimental": True,
                "method": "post_alignment_robust_z_tail_rejection",
                "changes_alignment_scale_or_shift": False,
                "clamps_depth_values": False,
                "affects_pose_estimation": False,
                "coordinate_scale": "relative_non_metric",
                "max_z_over_median": (
                    depth_stabilization_settings.max_z_over_median
                ),
                "mad_multiplier": depth_stabilization_settings.mad_multiplier,
                "minimum_valid_ratio": (
                    depth_stabilization_settings.minimum_valid_ratio
                ),
                "maximum_removed_ratio": (
                    depth_stabilization_settings.maximum_removed_ratio
                ),
                "attempted_frames": sum(
                    item.depth_stabilization_attempted
                    for item in result.frame_statistics
                ),
                "accepted_frames": sum(
                    item.depth_stabilization_accepted
                    for item in result.frame_statistics
                ),
                "reason_counts": {
                    reason: sum(
                        item.depth_stabilization_reason == reason
                        for item in result.frame_statistics
                    )
                    for reason in sorted({
                        item.depth_stabilization_reason
                        for item in result.frame_statistics
                        if item.depth_stabilization_reason is not None
                    })
                },
                "stabilized_raw_fused_point_count": (
                    result.stabilized_raw_fused_point_count
                ),
                "stabilized_voxel_downsampled_point_count": (
                    result.stabilized_voxel_downsampled_point_count
                ),
                "scientific_note": (
                    "Relative monocular depth robustness heuristic; it does not "
                    "make depth metric, correct pose drift, or prove accuracy."
                ),
            },
            "temporal_depth_normalization": {
                "enabled": temporal_normalization_settings.enabled,
                "experimental": True,
                "method": "pairwise_robust_log_depth_ratio",
                "aligned_depth_original_preserved": True,
                "normalized_depth_label": (
                    "aligned_depth_temporally_normalized"
                ),
                "changes_pose": False,
                "coordinate_scale": "relative_non_metric",
                "minimum_correspondences": (
                    temporal_normalization_settings.minimum_correspondences
                ),
                "minimum_inliers": temporal_normalization_settings.minimum_inliers,
                "minimum_inlier_ratio": (
                    temporal_normalization_settings.minimum_inlier_ratio
                ),
                "minimum_scale": temporal_normalization_settings.minimum_scale,
                "maximum_scale": temporal_normalization_settings.maximum_scale,
                "maximum_log_mad": (
                    temporal_normalization_settings.maximum_log_mad
                ),
                "minimum_cumulative_scale": (
                    temporal_normalization_settings.minimum_cumulative_scale
                ),
                "maximum_cumulative_scale": (
                    temporal_normalization_settings.maximum_cumulative_scale
                ),
                "attempted_frames": sum(
                    item.temporal_depth_normalization_attempted
                    for item in result.frame_statistics
                ),
                "accepted_frames": sum(
                    item.temporal_depth_normalization_accepted
                    for item in result.frame_statistics
                ),
                "reason_counts": {
                    reason: sum(
                        item.temporal_depth_normalization_reason == reason
                        for item in result.frame_statistics
                    )
                    for reason in sorted({
                        item.temporal_depth_normalization_reason
                        for item in result.frame_statistics
                        if item.temporal_depth_normalization_reason is not None
                    })
                },
                "scientific_note": (
                    "Pairwise relative-depth heuristic only; it does not recover "
                    "metric scale, correct pose error, or prove accuracy."
                ),
            },
            "depth_quality_thresholds": {
                "min_valid_depth_ratio": min_valid_depth_ratio,
                "max_denominator_reject_ratio": max_denominator_reject_ratio,
                "min_depth_alignment_inliers": min_depth_alignment_inliers,
                "min_depth_alignment_inlier_ratio": (
                    min_depth_alignment_inlier_ratio
                ),
                "max_relative_z_p99_over_median": (
                    max_relative_z_p99_over_median
                ),
            },
            "sampled_frames": result.sampled_frame_count,
            "total_candidate_frames": result.sampled_frame_count,
            "accepted_frames": result.accepted_frame_count,
            "accepted_keyframes": result.accepted_frame_count,
            "skipped_non_keyframes": result.skipped_non_keyframe_count,
            "rejected_frames": result.rejected_frame_count,
            "depth_inference_count": result.depth_inference_count,
            "runtime_metrics": {
                "total_pipeline_runtime_seconds": total_pipeline_runtime,
                "pose_chain_diagnostics_seconds": pose_chain_runtime_seconds,
                **result.stage_timings,
            },
            "rejection_reason_counts": {
                reason: sum(
                    item.rejection_reason == reason
                    for item in result.frame_statistics
                )
                for reason in sorted({
                    item.rejection_reason
                    for item in result.frame_statistics
                    if item.rejection_reason is not None
                })
            },
            "trajectory_pose_count": result.trajectory_positions.shape[0],
            "initial_map_point_count": result.raw_fused_point_count,
            "final_map_point_count": result.fused_cloud.output_point_count,
            **robustness_metadata(result),
            "trajectory_format": (
                "accepted keyframes only; skipped and rejected candidates are in "
                "frame_stats.jsonl"
            ),
            "note": (
                "Relative-mode map uses propagated reciprocal-disparity units, not metres."
                if not result.is_metric
                else "Metric-mode values remain subject to model and calibration error."
            ),
        }
        ply_path, visual_artifacts = save_outputs(
            run_directory,
            result,
            metadata,
            visual_config,
            save_previews=save_previews,
            save_display_clean=save_display_clean,
            save_drift=save_drift,
            pose_chain_rows=pose_chain_rows,
            pose_chain_config=pose_chain_settings,
        )

        print("Relative multi-frame map completed")
        print(f"Sampled frames: {result.sampled_frame_count}")
        print(f"Accepted keyframes: {result.accepted_frame_count}")
        print(f"Skipped non-keyframes: {result.skipped_non_keyframe_count}")
        print(f"Rejected frames: {result.rejected_frame_count}")
        print(f"Depth inference count: {result.depth_inference_count}")
        print(f"Raw fused points: {result.raw_fused_point_count}")
        print(f"After voxel downsampling: {result.voxel_downsampled_point_count}")
        print(
            f"Global outliers rejected: {result.global_filter.rejected_count} "
            f"({result.global_filter.method})"
        )
        print(f"Final map points: {result.fused_cloud.output_point_count}")
        if result.depth_stabilization_enabled:
            assert result.stabilized_fused_cloud is not None
            print(
                "Experimental stabilized map points: "
                f"{result.stabilized_fused_cloud.output_point_count}"
            )
        if result.temporal_depth_normalization_enabled:
            assert result.temporal_normalized_fused_cloud is not None
            print(
                "Experimental temporal-normalized map points: "
                f"{result.temporal_normalized_fused_cloud.output_point_count}"
            )
        print(f"Relative trajectory poses: {result.trajectory_positions.shape[0]}")
        for item in result.frame_statistics:
            if item.status == "rejected":
                decision_reason = item.rejection_reason or item.reason
            elif item.status == "skipped_non_keyframe":
                decision_reason = item.skip_reason or item.reason
            else:
                decision_reason = item.keyframe_reason or item.reason
            print(
                f"Frame {item.frame_index}: status={item.status}, "
                f"matches={item.good_matches}, geometric inliers="
                f"{item.geometric_inliers}, geometric ratio="
                f"{item.geometric_inlier_ratio:.6f}, median displacement="
                f"{item.median_feature_displacement_px}, rotation="
                f"{item.rotation_deg}, reason={decision_reason}, "
                f"depth inference={item.depth_inference_executed}, "
                f"PnP inliers={item.pnp_inliers}, RMSE={item.reprojection_rmse_pixels}, "
                f"translation={item.translation_magnitude}, "
                f"denominator reject ratio={item.denominator_rejection_ratio:.6f}, "
                f"valid depth ratio={item.valid_aligned_depth_ratio:.6f}, "
                f"alignment inliers={item.depth_alignment_inliers}, "
                f"alignment inlier ratio={item.depth_alignment_inlier_ratio:.6f}, "
                f"Z median={item.aligned_z_median}, Z p99={item.aligned_z_p99}, "
                f"p99/median={item.relative_z_p99_over_median}"
            )
            if item.accepted and result.depth_stabilization_enabled:
                print(
                    "  Depth stabilization: "
                    f"accepted={item.depth_stabilization_accepted}, "
                    f"reason={item.depth_stabilization_reason}, "
                    f"raw median/p95/p99={item.raw_z_median}/"
                    f"{item.raw_z_p95}/{item.raw_z_p99}, "
                    f"stabilized median/p95/p99={item.stabilized_z_median}/"
                    f"{item.stabilized_z_p95}/{item.stabilized_z_p99}, "
                    f"removed={item.stabilization_removed_count} "
                    f"({item.stabilization_removed_ratio:.6f}), "
                    f"denominator p01/p05/median={item.denominator_p01}/"
                    f"{item.denominator_p05}/{item.denominator_median}"
                )
            if item.accepted and result.temporal_depth_normalization_enabled:
                print(
                    "  Temporal depth normalization: "
                    f"accepted={item.temporal_depth_normalization_accepted}, "
                    f"reason={item.temporal_depth_normalization_reason}, "
                    f"correspondences={item.temporal_depth_correspondence_count}, "
                    f"inliers={item.temporal_depth_inlier_count}, "
                    f"inlier_ratio={item.temporal_depth_inlier_ratio:.6f}, "
                    f"pairwise_scale={item.temporal_depth_scale_pairwise}, "
                    f"cumulative_scale={item.temporal_depth_scale_cumulative}, "
                    f"log_mad={item.temporal_depth_log_ratio_mad}, "
                    f"original median/p95/p99={item.original_z_median}/"
                    f"{item.original_z_p95}/{item.original_z_p99}, "
                    f"normalized median/p95/p99={item.normalized_z_median}/"
                    f"{item.normalized_z_p95}/{item.normalized_z_p99}, "
                    f"residual median before/after="
                    f"{item.temporal_residual_before_median}/"
                    f"{item.temporal_residual_after_median}, "
                    f"RMSE before/after={item.temporal_residual_before_rmse}/"
                    f"{item.temporal_residual_after_rmse}"
                )
            if item.refinement_3d_attempted:
                print(
                    "  3D refinement: "
                    f"correspondences={item.correspondence_3d_count}, "
                    f"inliers={item.refinement_3d_inliers}, "
                    f"inlier_ratio={item.refinement_3d_inlier_ratio:.6f}, "
                    f"baseline_median={item.baseline_3d_residual_median}, "
                    f"baseline_rmse={item.baseline_3d_residual_rmse}, "
                    f"refined_median={item.refined_3d_residual_median}, "
                    f"refined_rmse={item.refined_3d_residual_rmse}, "
                    f"improvement={item.refinement_3d_relative_improvement}, "
                    f"accepted={item.refinement_3d_accepted}, "
                    f"reason={item.refinement_3d_reason}"
                )
            if item.z_statistics is None:
                continue
            print(
                f"  sampled pre-filter Z (min/p1/p5/median/p95/p99/max): "
                + ", ".join(
                    f"{item.z_statistics[name]:.6f}"
                    for name in ("min", "p1", "p5", "median", "p95", "p99", "max")
                )
            )
        for axis, statistics in (
            result.global_filter.coordinate_statistics_after.items()
        ):
            print(
                f"Global {axis.upper()} (min/p1/p5/median/p95/p99/max): "
                + ", ".join(
                    f"{statistics[name]:.6f}"
                    for name in ("min", "p1", "p5", "median", "p95", "p99", "max")
                )
            )
        print(
            "Robust-center distance diagnostic: "
            f"radius={result.global_filter.diagnostic_robust_radius:.6f}, "
            f"outside={result.global_filter.points_outside_diagnostic_radius}"
        )
        if scale_mode == "fixed-step":
            print("WARNING: fixed-step is an arbitrary DEBUG mode, not reconstruction.")
        if not result.is_metric:
            print(
                "WARNING: Translation and map coordinates use relative depth units, "
                "not metres."
            )
        print(f"Final map: {ply_path.resolve()}")
        if result.pair_alignment is not None:
            print(
                "Pair alignment before: "
                f"{(run_directory / 'pair_alignment_before.ply').resolve()}"
            )
            print(
                "Pair alignment after: "
                f"{(run_directory / 'pair_alignment_after.ply').resolve()}"
            )
            print(
                "Pair alignment metrics: "
                f"{(run_directory / 'pair_alignment_metrics.json').resolve()}"
            )
        print(
            "Display cleaning (presentation only): "
            f"{visual_artifacts.cleaning.raw_count} raw -> "
            f"{visual_artifacts.cleaning.display_count} displayed "
            f"({visual_artifacts.cleaning.removed_count} removed)"
        )
        if save_drift:
            drift_summary = metadata["drift_diagnostics"]["results"]
            print(
                "Drift diagnostics (heuristic only): "
                f"{drift_summary['heuristic_warning_flags']}"
            )
            print(
                "Drift diagnostic directory: "
                f"{(run_directory / 'drift_diagnostics').resolve()}"
            )
        if pose_chain_rows is not None:
            pose_summary = metadata["pose_chain_diagnostics"]["results"]
            print(
                "Pose-chain diagnostics (heuristic only): "
                f"{pose_summary['heuristic_warning_flags']}"
            )
            print(
                "Pose-chain diagnostic directory: "
                f"{(run_directory / 'pose_chain_diagnostics').resolve()}"
            )
        print(f"Outputs: {run_directory.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Relative map failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
