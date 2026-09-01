"""CLI for a single-frame colored point cloud in relative depth units."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.depth_estimator import DEFAULT_MODEL, DepthEstimator
from src.depth_types import CameraDepth, DepthPrediction
from src.point_cloud import PointCloudResult, generate_colored_point_cloud
from src.visual_outputs import (
    CloudVisualArtifacts,
    display_cleaning_metadata,
    save_cloud_visual_artifacts,
    save_rgb_depth_side_by_side,
)
from src.visualization import colorize_depth
from tools.run_motion import camera_matrix_from_args, load_config, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a colored single-frame point cloud in relative depth units",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Input RGB frame")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Calibrated focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Calibrated focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Calibrated principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Calibrated principal point y (pixels)")
    parser.add_argument("--stride", type=int, help="Sample every Nth pixel in x and y")
    parser.add_argument(
        "--disparity-denominator-epsilon", type=float,
        help="Reject relative-disparity denominators at or below this value",
    )
    parser.add_argument(
        "--depth-percentile-low", type=float,
        help="Lower relative-Z percentile kept for point-cloud export",
    )
    parser.add_argument(
        "--depth-percentile-high", type=float,
        help="Upper relative-Z percentile kept for point-cloud export",
    )
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save front, oblique, and top PNG previews",
    )
    parser.add_argument(
        "--save-display-clean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save a conservatively cleaned display-only PLY",
    )
    parser.add_argument("--output-dir", type=Path, help="Root for generated outputs")
    return parser.parse_args()


def create_output_directory(root: Path, image_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / f"point_cloud_{image_path.stem}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV failed to write: {path}")


def save_outputs(
    directory: Path,
    prediction: DepthPrediction,
    camera_depth: CameraDepth,
    cloud: PointCloudResult,
    metadata: dict[str, Any],
    image_bgr: np.ndarray,
    visual_config: dict[str, Any],
    *,
    save_previews: bool,
    save_display_clean: bool,
) -> tuple[Path, CloudVisualArtifacts]:
    np.save(directory / "depth_raw.npy", prediction.values.astype(np.float32, copy=False))
    np.save(
        directory / "camera_z.npy",
        camera_depth.values.astype(np.float32, copy=False),
    )
    np.save(directory / "points_3d_relative.npy", cloud.points)
    np.save(directory / "colors_rgb.npy", cloud.colors)
    write_image(directory / "rgb_input.png", image_bgr)
    depth_visualization = colorize_depth(prediction.values)
    write_image(directory / "depth_vis.png", depth_visualization)
    save_rgb_depth_side_by_side(
        directory / "rgb_depth_side_by_side.png", image_bgr, depth_visualization
    )
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
        cloud.points,
        cloud.colors,
        raw_filenames=("cloud_relative.ply", "cloud_relative_raw.ply"),
        display_filename="cloud_relative_display.ply",
        preview_prefix="point_cloud_preview",
        title="Single-frame relative point cloud (display only, non-metric)",
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
    metadata["visual_output"] = display_cleaning_metadata(
        artifacts.cleaning,
        raw_artifact="cloud_relative_raw.ply",
        compatibility_artifact="cloud_relative.ply",
        display_artifact=(
            "cloud_relative_display.ply" if save_display_clean else None
        ),
    )
    metadata["visual_output"].update({
        "previews_saved": save_previews,
        "display_clean_saved": save_display_clean,
    })
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return artifacts.raw_paths[0], artifacts


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_config = config.get("model", {})
        cloud_config = config.get("point_cloud", {})
        visual_config = config.get("visual_output", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config, cloud_config, visual_config, output_config
        )):
            raise ValueError(
                "model, point_cloud, visual_output, and output config sections "
                "must be mappings"
            )

        camera_matrix = camera_matrix_from_args(args, config)
        image_bgr = load_image(args.image, "input image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stride = args.stride if args.stride is not None else cloud_config.get("stride", 4)
        denominator_epsilon = (
            args.disparity_denominator_epsilon
            if args.disparity_denominator_epsilon is not None
            else cloud_config.get("disparity_denominator_epsilon", 1e-3)
        )
        depth_percentile_low = (
            args.depth_percentile_low
            if args.depth_percentile_low is not None
            else cloud_config.get("depth_percentile_low", 1.0)
        )
        depth_percentile_high = (
            args.depth_percentile_high
            if args.depth_percentile_high is not None
            else cloud_config.get("depth_percentile_high", 99.0)
        )
        save_previews = bool(
            args.save_previews
            if args.save_previews is not None
            else visual_config.get("save_previews", True)
        )
        save_display_clean = bool(
            args.save_display_clean
            if args.save_display_clean is not None
            else visual_config.get("save_display_clean", True)
        )

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        print(f"Loading {model_name} on {device}...")
        estimator = DepthEstimator(model_name=model_name, device=device)
        prediction = estimator.predict_result(image_bgr)
        camera_depth = prediction.to_camera_depth(
            alignment_method="metric_model" if prediction.is_metric else "none",
            denominator_epsilon=float(denominator_epsilon),
        )
        cloud = generate_colored_point_cloud(
            image_rgb,
            camera_depth,
            camera_matrix,
            stride=stride,
            depth_percentile_low=depth_percentile_low,
            depth_percentile_high=depth_percentile_high,
        )

        output_root = args.output_dir or Path(output_config.get("directory", "outputs"))
        run_directory = create_output_directory(
            output_root / "point_cloud", args.image
        )
        height, width = image_bgr.shape[:2]
        metadata = {
            "success": cloud.valid_point_count > 0,
            "input_image_path": str(args.image.resolve()),
            "model": estimator.model_name,
            "device": estimator.device,
            "camera_intrinsics": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
            "image_width": width,
            "image_height": height,
            "stride": cloud.stride,
            "sampled_pixel_count": cloud.sampled_pixel_count,
            "valid_point_count": cloud.valid_point_count,
            "depth_type": prediction.depth_type,
            "raw_depth_representation": prediction.representation,
            "geometry_depth_representation": camera_depth.representation,
            "depth_conversion": camera_depth.conversion,
            "depth_alignment_method": camera_depth.alignment_method,
            "disparity_alignment": {
                "scale_a": camera_depth.disparity_scale,
                "shift_b": camera_depth.disparity_shift,
                "denominator_epsilon": camera_depth.denominator_epsilon,
                "minimum_absolute_denominator": (
                    camera_depth.minimum_absolute_denominator
                ),
                "rejected_small_denominator_count": (
                    camera_depth.rejected_small_denominator_count
                ),
                "rejected_nonfinite_denominator_count": (
                    camera_depth.rejected_nonfinite_denominator_count
                ),
                "rejected_invalid_z_count": camera_depth.rejected_invalid_z_count,
            },
            "depth_outlier_filter": {
                "method": cloud.depth_filter_method,
                "percentile_low": cloud.depth_percentile_low,
                "percentile_high": cloud.depth_percentile_high,
                "lower_bound": cloud.depth_lower_bound,
                "upper_bound": cloud.depth_upper_bound,
                "points_before": cloud.valid_depth_count_before_filter,
                "points_rejected": cloud.depth_outlier_rejected_count,
                "points_after": cloud.valid_point_count,
            },
            "coordinate_frame": cloud.coordinate_frame,
            "coordinate_units": cloud.coordinate_units,
            "is_metric": prediction.is_metric,
            "z_statistics": cloud.z_statistics,
            "note": (
                "Relative coordinates are reciprocal disparity proxies, not metres."
                if not prediction.is_metric
                else "Metric values are model predictions and remain subject to model error."
            ),
        }
        ply_path, visual_artifacts = save_outputs(
            run_directory,
            prediction,
            camera_depth,
            cloud,
            metadata,
            image_bgr,
            visual_config,
            save_previews=save_previews,
            save_display_clean=save_display_clean,
        )

        if cloud.valid_point_count == 0:
            print("Point cloud generation failed: no valid relative-depth points")
        else:
            print("Relative colored point cloud generated successfully")
        print(f"Image dimensions: {width}x{height}")
        print(f"Stride: {cloud.stride}")
        print(f"Sampled pixels: {cloud.sampled_pixel_count}")
        print(f"Valid points: {cloud.valid_point_count}")
        print(f"Points shape: {cloud.points.shape}")
        print(f"RGB colors shape: {cloud.colors.shape}")
        assert cloud.z_statistics is not None
        print(
            "Usable sampled Z (min/p1/p5/median/p95/p99/max): "
            + ", ".join(
                f"{cloud.z_statistics[name]:.6f}"
                for name in ("min", "p1", "p5", "median", "p95", "p99", "max")
            )
        )
        print(
            f"Depth outlier filter: {cloud.depth_filter_method}; removed "
            f"{cloud.depth_outlier_rejected_count} of "
            f"{cloud.valid_depth_count_before_filter} sampled valid depths"
        )
        print(
            "Disparity denominator: "
            f"epsilon={camera_depth.denominator_epsilon}, "
            f"min_abs={camera_depth.minimum_absolute_denominator}, "
            f"rejected_small={camera_depth.rejected_small_denominator_count}, "
            f"rejected_nonfinite={camera_depth.rejected_nonfinite_denominator_count}"
        )
        if not prediction.is_metric:
            print(
                "WARNING: raw relative output is disparity-like; camera Z is a "
                "reciprocal proxy in relative units, NOT metres."
            )
        print(f"PLY: {ply_path.resolve()}")
        print(
            "Display cleaning (presentation only): "
            f"{visual_artifacts.cleaning.raw_count} raw -> "
            f"{visual_artifacts.cleaning.display_count} displayed "
            f"({visual_artifacts.cleaning.removed_count} removed)"
        )
        print(f"Outputs: {run_directory.resolve()}")
        return 0 if cloud.valid_point_count > 0 else 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Point cloud generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
