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
from src.ply_io import write_ascii_ply
from src.point_cloud import PointCloudResult, generate_colored_point_cloud
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
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
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
    depth: np.ndarray,
    cloud: PointCloudResult,
    metadata: dict[str, Any],
) -> Path:
    np.save(directory / "depth_raw.npy", depth.astype(np.float32, copy=False))
    np.save(directory / "points_3d_relative.npy", cloud.points)
    np.save(directory / "colors_rgb.npy", cloud.colors)
    write_image(directory / "depth_vis.png", colorize_depth(depth))
    ply_path = write_ascii_ply(
        directory / "cloud_relative.ply", cloud.points, cloud.colors
    )
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return ply_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_config = config.get("model", {})
        cloud_config = config.get("point_cloud", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config, cloud_config, output_config
        )):
            raise ValueError("model, point_cloud, and output config sections must be mappings")

        camera_matrix = camera_matrix_from_args(args, config)
        image_bgr = load_image(args.image, "input image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stride = args.stride if args.stride is not None else cloud_config.get("stride", 4)

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        print(f"Loading {model_name} on {device} for relative depth...")
        estimator = DepthEstimator(model_name=model_name, device=device)
        relative_depth = estimator.predict(image_bgr)
        cloud = generate_colored_point_cloud(
            image_rgb, relative_depth, camera_matrix, stride=stride
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
            "depth_type": "relative",
            "coordinate_frame": cloud.coordinate_frame,
            "coordinate_units": cloud.coordinate_units,
            "is_metric": False,
            "note": "Point coordinates use relative depth units and are not metres.",
        }
        ply_path = save_outputs(run_directory, relative_depth, cloud, metadata)

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
        print("WARNING: Point coordinates use relative Depth Anything units and are NOT metric.")
        print(f"PLY: {ply_path.resolve()}")
        print(f"Outputs: {run_directory.resolve()}")
        return 0 if cloud.valid_point_count > 0 else 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Point cloud generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
