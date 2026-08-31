"""Diagnose a depth-assisted scaled pose between two RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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

from src.depth_alignment import align_prediction_to_pose
from src.depth_estimator import DEFAULT_MODEL, DepthEstimator
from src.depth_pose_estimator import DepthPoseEstimator
from src.feature_tracker import FeatureTrackingError
from src.visualization import draw_feature_matches
from tools.run_depth_geometry import build_motion_estimator, build_tracker
from tools.run_motion import camera_matrix_from_args, load_config, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a depth-scaled relative pose with solvePnPRansac",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("frame1", type=Path)
    parser.add_argument("frame2", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    for name in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--minimum-matches", type=int)
    parser.add_argument("--ransac-probability", type=float)
    parser.add_argument("--ransac-threshold", type=float)
    parser.add_argument("--minimum-inliers", type=int)
    parser.add_argument("--minimum-inlier-ratio", type=float)
    parser.add_argument("--pnp-reprojection-error", type=float)
    parser.add_argument("--minimum-pnp-inliers", type=int)
    parser.add_argument("--minimum-pnp-inlier-ratio", type=float)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def configured(cli, config, name, fallback):
    return cli if cli is not None else config.get(name, fallback)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        model_config = config.get("model", {})
        motion_config = config.get("motion", {})
        map_config = config.get("map", {})
        output_config = config.get("output", {})
        camera_matrix = camera_matrix_from_args(args, config)
        image1 = load_image(args.frame1, "frame 1")
        image2 = load_image(args.frame2, "frame 2")
        matches = build_tracker(args, motion_config).match(image1, image2)
        geometry_pose = build_motion_estimator(args, motion_config).estimate(
            matches.points1, matches.points2, camera_matrix
        )
        if not geometry_pose.success:
            raise RuntimeError(geometry_pose.message)

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        estimator = DepthEstimator(model_name, device)
        prediction1 = estimator.predict_result(image1)
        camera_depth1 = prediction1.to_camera_depth(
            alignment_method="metric_model" if prediction1.is_metric else "none"
        )
        pnp = DepthPoseEstimator(
            minimum_correspondences=int(map_config.get(
                "minimum_pnp_correspondences", 6
            )),
            minimum_inliers=int(configured(
                args.minimum_pnp_inliers, map_config, "minimum_pnp_inliers", 6
            )),
            minimum_inlier_ratio=float(configured(
                args.minimum_pnp_inlier_ratio,
                map_config,
                "minimum_pnp_inlier_ratio",
                0.25,
            )),
            reprojection_error_pixels=float(configured(
                args.pnp_reprojection_error,
                map_config,
                "pnp_reprojection_error_pixels",
                3.0,
            )),
        ).estimate(
            matches.points1,
            matches.points2,
            geometry_pose.inlier_mask,
            camera_depth1,
            camera_matrix,
        )
        if not pnp.success:
            raise RuntimeError(pnp.message)
        prediction2 = estimator.predict_result(image2)
        alignment = align_prediction_to_pose(prediction2, matches.points2, pnp)

        root = args.output_dir or Path(output_config.get("directory", "outputs"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = root / "depth_pose" / f"depth_pose_{args.frame1.stem}_{args.frame2.stem}_{stamp}"
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "frame1_raw_prediction.npy", prediction1.values)
        np.save(directory / "frame1_camera_z.npy", camera_depth1.values)
        debug = draw_feature_matches(
            image1, image2, matches.keypoints1, matches.keypoints2,
            matches.good_matches, pnp.inlier_mask,
        )
        if not cv2.imwrite(str(directory / "pnp_inliers.png"), debug):
            raise RuntimeError("OpenCV could not save PnP visualization")
        metadata = {
            "success": True,
            "frame1": str(args.frame1.resolve()),
            "frame2": str(args.frame2.resolve()),
            "model": estimator.model_name,
            "device": estimator.device,
            "is_metric": prediction1.is_metric,
            "depth_type": prediction1.depth_type,
            "depth_representation": prediction1.representation,
            "geometry_depth_representation": camera_depth1.representation,
            "depth_conversion": camera_depth1.conversion,
            "scale_estimation_method": "depth_pnp",
            "translation_units": pnp.translation_units,
            "depth_alignment_method": alignment.method,
            "depth_alignment_success": alignment.success,
            "depth_alignment_inliers": alignment.inliers,
            "feature_matches": matches.statistics.good_matches,
            "geometric_inliers": geometry_pose.num_inliers,
            "valid_depth_correspondences": pnp.valid_depth_correspondences,
            "pnp_inliers": pnp.pnp_inliers,
            "pnp_inlier_ratio": pnp.pnp_inlier_ratio,
            "reprojection_rmse_pixels": pnp.reprojection_rmse_pixels,
            "reprojection_median_pixels": pnp.reprojection_median_pixels,
            "rotation": pnp.rotation.tolist(),
            "translation": pnp.translation.tolist(),
            "translation_magnitude": pnp.translation_magnitude,
            "note": (
                "Translation is expressed in relative depth units, not metres."
                if not prediction1.is_metric
                else "Translation uses model-predicted metric-depth units."
            ),
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print(f"Feature matches: {matches.statistics.good_matches}")
        print(f"Geometric inliers: {geometry_pose.num_inliers}")
        print(f"Usable depth correspondences: {pnp.valid_depth_correspondences}")
        print(f"PnP inliers: {pnp.pnp_inliers} ({pnp.pnp_inlier_ratio:.3f})")
        print(f"Reprojection RMSE: {pnp.reprojection_rmse_pixels:.6f} pixels")
        print(f"Reprojection median: {pnp.reprojection_median_pixels:.6f} pixels")
        print("Rotation R:")
        print(np.array2string(pnp.rotation, precision=8, suppress_small=True))
        print("Translation t:")
        print(np.array2string(pnp.translation, precision=8, suppress_small=True))
        print(
            f"Translation magnitude: {pnp.translation_magnitude:.8f} "
            f"{pnp.translation_units}"
        )
        print(
            f"Depth alignment: {alignment.method}, success={alignment.success}, "
            f"inliers={alignment.inliers}"
        )
        if not prediction1.is_metric:
            print("Translation is expressed in relative depth units, not metres.")
        print(f"Outputs: {directory.resolve()}")
        return 0
    except (
        FeatureTrackingError, FileNotFoundError, RuntimeError, TypeError, ValueError
    ) as exc:
        print(f"Depth pose failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
