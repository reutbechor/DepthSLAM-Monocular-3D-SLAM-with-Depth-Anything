"""CLI for depth-scaled short-video incremental relative mapping."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
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
from src.depth_pose_estimator import DepthPoseEstimator
from src.map_builder import MappingFrame, RelativeMapBuilder, RelativeMapResult
from src.ply_io import write_ascii_ply
from src.video_loader import VideoLoader
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
    parser.add_argument("--max-mapping-frames", type=int)
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


def save_outputs(
    directory: Path, result: RelativeMapResult, metadata: dict[str, Any]
) -> Path:
    np.save(directory / "global_points_relative.npy", result.fused_cloud.points)
    np.save(directory / "global_colors_rgb.npy", result.fused_cloud.colors)
    ply_path = write_ascii_ply(
        directory / "global_relative_map.ply",
        result.fused_cloud.points,
        result.fused_cloud.colors,
    )
    save_trajectory(directory, result)
    save_frame_statistics(directory, result)
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return ply_path


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
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {args.video}")
        config = load_config(args.config)
        model_config = config.get("model", {})
        motion_config = config.get("motion", {})
        map_config = config.get("map", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config, motion_config, map_config, output_config
        )):
            raise ValueError("model, motion, map, and output config sections must be mappings")
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
        )
        result = builder.build(selected_frames)

        output_root = args.output_dir or Path(output_config.get("directory", "outputs"))
        run_directory = create_output_directory(output_root / "relative_map", args.video)
        metadata = {
            "map_type": "relative_multi_frame",
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
            "accepted_frames": result.accepted_frame_count,
            "rejected_frames": result.rejected_frame_count,
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
            "trajectory_format": "accepted frames only; rejected frames are in frame_stats.jsonl",
            "note": (
                "Relative-mode map uses propagated reciprocal-disparity units, not metres."
                if not result.is_metric
                else "Metric-mode values remain subject to model and calibration error."
            ),
        }
        ply_path = save_outputs(run_directory, result, metadata)

        print("Relative multi-frame map completed")
        print(f"Sampled frames: {result.sampled_frame_count}")
        print(f"Accepted mapping frames: {result.accepted_frame_count}")
        print(f"Rejected frames: {result.rejected_frame_count}")
        print(f"Raw fused points: {result.raw_fused_point_count}")
        print(f"After voxel downsampling: {result.voxel_downsampled_point_count}")
        print(
            f"Global outliers rejected: {result.global_filter.rejected_count} "
            f"({result.global_filter.method})"
        )
        print(f"Final map points: {result.fused_cloud.output_point_count}")
        print(f"Relative trajectory poses: {result.trajectory_positions.shape[0]}")
        for item in result.frame_statistics[1:]:
            if item.accepted:
                if item.scale_estimation_method == "depth_pnp":
                    print(
                        f"Frame {item.frame_index}: t={item.translation_magnitude:.6f} "
                        f"{item.translation_units}, PnP inliers={item.pnp_inliers}, "
                        f"RMSE={item.reprojection_rmse_pixels:.3f}px, "
                        f"depth alignment={item.depth_alignment_method}"
                    )
                else:
                    print(
                        f"Frame {item.frame_index}: debug fixed step="
                        f"{item.translation_magnitude:.6f}"
                    )
            else:
                print(f"Frame {item.frame_index}: rejected ({item.reason})")
        for item in result.frame_statistics:
            status = "accepted" if item.accepted else "rejected"
            rejection = item.rejection_reason or "none"
            print(
                f"Frame {item.frame_index} quality: {status}, "
                f"rejection={rejection}, PnP inliers={item.pnp_inliers}, "
                f"RMSE={item.reprojection_rmse_pixels}, "
                f"denominator reject ratio={item.denominator_rejection_ratio:.6f}, "
                f"valid depth ratio={item.valid_aligned_depth_ratio:.6f}, "
                f"alignment inliers={item.depth_alignment_inliers}, "
                f"alignment inlier ratio={item.depth_alignment_inlier_ratio:.6f}, "
                f"Z median={item.aligned_z_median}, Z p99={item.aligned_z_p99}, "
                f"p99/median={item.relative_z_p99_over_median}"
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
        print(f"Outputs: {run_directory.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Relative map failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
