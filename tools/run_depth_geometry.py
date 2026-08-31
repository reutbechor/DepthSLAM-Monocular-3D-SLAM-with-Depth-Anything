"""Run relative depth, two-view motion, and relative 3D backprojection."""

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
from src.depth_geometry import DepthGeometryProcessor
from src.feature_tracker import FeatureTracker, FeatureTrackingError
from src.motion_estimator import MotionEstimator
from src.visualization import colorize_depth, draw_feature_matches
from tools.run_motion import (
    camera_matrix_from_args,
    configured_value,
    load_config,
    load_image,
    print_match_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create relative Frame 1 camera geometry from two RGB frames and "
            "Depth Anything V2"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("frame1", type=Path, help="Frame used for relative depth and 3D")
    parser.add_argument("frame2", type=Path, help="Second frame used for relative pose")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Calibrated focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Calibrated focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Calibrated principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Calibrated principal point y (pixels)")
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--sampling", choices=("bilinear", "nearest"))
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--minimum-matches", type=int)
    parser.add_argument("--ransac-probability", type=float)
    parser.add_argument("--ransac-threshold", type=float, metavar="PIXELS")
    parser.add_argument("--minimum-inliers", type=int)
    parser.add_argument("--minimum-inlier-ratio", type=float)
    parser.add_argument("--output-dir", type=Path, help="Root for generated outputs")
    return parser.parse_args()


def build_tracker(args: argparse.Namespace, motion: dict[str, Any]) -> FeatureTracker:
    max_features = configured_value(args.max_features, motion, "max_features", None)
    if max_features is not None:
        max_features = int(max_features)
    return FeatureTracker(
        ratio_threshold=float(configured_value(
            args.ratio_threshold, motion, "ratio_threshold", 0.75
        )),
        max_features=max_features,
        minimum_matches=int(configured_value(
            args.minimum_matches, motion, "minimum_matches", 8
        )),
    )


def build_motion_estimator(
    args: argparse.Namespace, motion: dict[str, Any]
) -> MotionEstimator:
    return MotionEstimator(
        ransac_probability=float(configured_value(
            args.ransac_probability, motion, "ransac_probability", 0.999
        )),
        ransac_threshold_pixels=float(configured_value(
            args.ransac_threshold, motion, "ransac_threshold_pixels", 1.0
        )),
        minimum_correspondences=int(configured_value(
            args.minimum_matches, motion, "minimum_matches", 8
        )),
        minimum_inliers=int(configured_value(
            args.minimum_inliers, motion, "minimum_inliers", 8
        )),
        minimum_inlier_ratio=float(configured_value(
            args.minimum_inlier_ratio, motion, "minimum_inlier_ratio", 0.25
        )),
    )


def create_output_directory(root: Path, frame1: Path, frame2: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"depth_geometry_{frame1.stem}_{frame2.stem}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV failed to write: {path}")


def save_outputs(
    directory: Path,
    raw_prediction: np.ndarray,
    camera_z: np.ndarray,
    match_image: np.ndarray,
    geometry: Any,
    metadata: dict[str, Any],
) -> None:
    np.save(directory / "depth_raw.npy", raw_prediction.astype(np.float32, copy=False))
    np.save(directory / "camera_z.npy", camera_z.astype(np.float32, copy=False))
    np.save(directory / "feature_points_2d.npy", geometry.valid_pixel_coordinates)
    np.save(directory / "feature_depths.npy", geometry.sampled_camera_depths)
    np.save(directory / "points_3d_relative.npy", geometry.points_3d_relative)
    write_image(directory / "depth_vis.png", colorize_depth(raw_prediction))
    write_image(directory / "matches.png", match_image)
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_config = config.get("model", {})
        motion_config = config.get("motion", {})
        geometry_config = config.get("depth_geometry", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config, motion_config, geometry_config, output_config
        )):
            raise ValueError("model, motion, depth_geometry, and output must be mappings")

        camera_matrix = camera_matrix_from_args(args, config)
        image1 = load_image(args.frame1, "frame 1")
        image2 = load_image(args.frame2, "frame 2")
        matches = build_tracker(args, motion_config).match(image1, image2)
        pose = build_motion_estimator(args, motion_config).estimate(
            matches.points1, matches.points2, camera_matrix
        )
        if not pose.success:
            raise RuntimeError(pose.message)

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        print(f"Loading {model_name} on {device} for Frame 1 depth prediction...")
        depth_estimator = DepthEstimator(model_name=model_name, device=device)
        prediction = depth_estimator.predict_result(image1)
        camera_depth = prediction.to_camera_depth(
            alignment_method="metric_model" if prediction.is_metric else "none"
        )
        sampling = args.sampling or geometry_config.get("sampling_method", "bilinear")
        geometry = DepthGeometryProcessor(sampling).process(
            matches.points1, pose.inlier_mask, camera_depth, camera_matrix
        )

        output_root = args.output_dir or Path(output_config.get("directory", "outputs"))
        run_directory = create_output_directory(
            output_root / "depth_geometry", args.frame1, args.frame2
        )
        match_image = draw_feature_matches(
            image1,
            image2,
            matches.keypoints1,
            matches.keypoints2,
            matches.good_matches,
            pose.inlier_mask,
        )
        metadata = {
            "success": geometry.valid_depth_sample_count > 0,
            "frame1_path": str(args.frame1.resolve()),
            "frame2_path": str(args.frame2.resolve()),
            "model": depth_estimator.model_name,
            "device": depth_estimator.device,
            "camera_intrinsics": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
            "good_matches": matches.statistics.good_matches,
            "pose_inliers": pose.num_inliers,
            "pose_inlier_ratio": pose.inlier_ratio,
            "valid_depth_samples": geometry.valid_depth_sample_count,
            "relative_3d_points": geometry.points_3d_relative.shape[0],
            "depth_sampling": geometry.sampling_method,
            "depth_type": prediction.depth_type,
            "raw_depth_representation": prediction.representation,
            "geometry_depth_representation": camera_depth.representation,
            "depth_conversion": camera_depth.conversion,
            "is_metric": prediction.is_metric,
            "coordinate_frame": "frame1_camera",
            "coordinate_units": camera_depth.coordinate_units,
            "translation_scale": "unknown",
            "note": (
                "3D coordinates use a reciprocal relative-depth proxy, not metres."
                if not prediction.is_metric
                else "3D coordinates use model-predicted metric depth."
            ),
            "rotation": pose.rotation.tolist(),
            "translation_direction": pose.translation_direction.reshape(-1).tolist(),
        }
        save_outputs(
            run_directory,
            prediction.values,
            camera_depth.values,
            match_image,
            geometry,
            metadata,
        )

        if geometry.valid_depth_sample_count == 0:
            print("Depth-assisted relative geometry failed: no valid depth samples")
        else:
            print("Depth-assisted relative geometry successful")
        print_match_report(matches.statistics)
        print(f"Pose inliers: {pose.num_inliers}")
        print(f"Inlier ratio: {pose.inlier_ratio:.3f}")
        print(f"Valid camera-Z samples: {geometry.valid_depth_sample_count}")
        print(f"Relative 3D points: {geometry.points_3d_relative.shape[0]}")
        print("\nRotation R:")
        print(np.array2string(pose.rotation, precision=6, suppress_small=True))
        print("\nTranslation direction t (unknown scale):")
        print(np.array2string(
            pose.translation_direction, precision=6, suppress_small=True
        ))
        if not prediction.is_metric:
            print(
                "\nWARNING: raw output is disparity-like. 3D coordinates use a "
                "reciprocal relative-Z proxy and are NOT metric."
            )
        print(f"Outputs: {run_directory.resolve()}")
        return 0 if geometry.valid_depth_sample_count > 0 else 1
    except (FeatureTrackingError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Depth geometry failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
