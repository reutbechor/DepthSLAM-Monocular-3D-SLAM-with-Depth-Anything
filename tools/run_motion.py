"""CLI for calibrated relative motion estimation between two RGB frames."""

from __future__ import annotations

import argparse
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
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.feature_tracker import FeatureTracker, FeatureTrackingError
from src.motion_estimator import MotionEstimator
from src.visualization import draw_feature_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate relative rotation and translation direction between two RGB frames"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("frame1", type=Path, help="Path to the first RGB frame")
    parser.add_argument("frame2", type=Path, help="Path to the second RGB frame")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Calibrated focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Calibrated focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Calibrated principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Calibrated principal point y (pixels)")
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--minimum-matches", type=int)
    parser.add_argument("--ransac-probability", type=float)
    parser.add_argument("--ransac-threshold", type=float, metavar="PIXELS")
    parser.add_argument("--minimum-inliers", type=int)
    parser.add_argument("--minimum-inlier-ratio", type=float)
    parser.add_argument("--output-dir", type=Path, help="Root for the motion run output")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def camera_matrix_from_args(
    args: argparse.Namespace, config: dict[str, Any]
) -> np.ndarray:
    camera = config.get("camera", {})
    if not isinstance(camera, dict):
        raise ValueError("The config camera section must be a mapping")
    values = {
        name: getattr(args, name) if getattr(args, name) is not None else camera.get(name)
        for name in ("fx", "fy", "cx", "cy")
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "Camera intrinsics are required. Supply "
            + ", ".join(f"--{name}" for name in missing)
            + " or set them in config/default.yaml"
        )
    try:
        numeric = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera intrinsics must be numeric") from exc
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("Camera intrinsics must be finite")
    if numeric["fx"] <= 0 or numeric["fy"] <= 0:
        raise ValueError("fx and fy must be positive")
    return np.array(
        [
            [numeric["fx"], 0.0, numeric["cx"]],
            [0.0, numeric["fy"], numeric["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def configured_value(
    cli_value: Any, config: dict[str, Any], name: str, fallback: Any
) -> Any:
    return cli_value if cli_value is not None else config.get(name, fallback)


def load_image(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode {label}: {path}")
    return image


def create_output_directory(root: Path, frame1: Path, frame2: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / f"motion_{frame1.stem}_{frame2.stem}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def print_match_report(statistics: Any) -> None:
    print(f"Keypoints frame 1: {statistics.keypoints_image1}")
    print(f"Keypoints frame 2: {statistics.keypoints_image2}")
    print(f"Lowe-ratio good matches: {statistics.good_matches}")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        motion = config.get("motion", {})
        output = config.get("output", {})
        if not isinstance(motion, dict) or not isinstance(output, dict):
            raise ValueError("The motion and output config sections must be mappings")
        camera_matrix = camera_matrix_from_args(args, config)
        image1 = load_image(args.frame1, "frame 1")
        image2 = load_image(args.frame2, "frame 2")

        max_features = configured_value(args.max_features, motion, "max_features", None)
        if max_features is not None:
            max_features = int(max_features)
        tracker = FeatureTracker(
            ratio_threshold=float(configured_value(
                args.ratio_threshold, motion, "ratio_threshold", 0.75
            )),
            max_features=max_features,
            minimum_matches=int(configured_value(
                args.minimum_matches, motion, "minimum_matches", 8
            )),
        )
        matches = tracker.match(image1, image2)
        estimator = MotionEstimator(
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
        estimate = estimator.estimate(matches.points1, matches.points2, camera_matrix)

        output_root = args.output_dir or Path(output.get("directory", "outputs"))
        run_directory = create_output_directory(
            output_root / "motion", args.frame1, args.frame2
        )
        visualization = draw_feature_matches(
            image1,
            image2,
            matches.keypoints1,
            matches.keypoints2,
            matches.good_matches,
            estimate.inlier_mask,
        )
        visualization_path = run_directory / "matches.png"
        if not cv2.imwrite(str(visualization_path), visualization):
            raise RuntimeError(f"OpenCV failed to write: {visualization_path}")

        print(estimate.message)
        print_match_report(matches.statistics)
        print(f"Inliers: {estimate.num_inliers}")
        print(f"Inlier ratio: {estimate.inlier_ratio:.3f}")
        if estimate.rotation is not None and estimate.translation_direction is not None:
            print("\nRotation R:")
            print(np.array2string(estimate.rotation, precision=6, suppress_small=True))
            print("\nTranslation direction t (unknown scale):")
            print(np.array2string(
                estimate.translation_direction, precision=6, suppress_small=True
            ))
        print(f"\nMatch visualization: {visualization_path.resolve()}")
        return 0 if estimate.success else 1
    except (FeatureTrackingError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Motion estimation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
