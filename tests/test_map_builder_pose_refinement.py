from types import SimpleNamespace

import numpy as np

from src.depth_alignment import DepthAlignmentResult
from src.depth_types import CameraDepth, DepthPrediction
from src.keyframe_selector import KeyframeSelector, KeyframeThresholds
from src.map_builder import MappingFrame, RelativeMapBuilder
from src.pose_refinement_3d import (
    PoseRefinement3DConfig,
    RobustPoseRefiner3D,
)
from tools.run_relative_map import save_pair_alignment_artifacts


class GridDepthEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def predict_result(self, image: np.ndarray) -> DepthPrediction:
        self.calls += 1
        return DepthPrediction(
            np.ones(image.shape[:2], dtype=np.float32),
            "relative",
            False,
            "relative_inverse_depth",
            "synthetic",
        )


class GridFeatureTracker:
    def __init__(self) -> None:
        grid = np.asarray([
            [float(x), float(y)] for y in range(2, 10) for x in range(2, 10)
        ])
        self.previous = grid
        self.current = grid + [1.0, 0.0]

    def match(self, first: np.ndarray, second: np.ndarray) -> SimpleNamespace:
        del first, second
        return SimpleNamespace(
            points1=self.previous.copy(),
            points2=self.current.copy(),
            statistics=SimpleNamespace(good_matches=self.previous.shape[0]),
        )


class AcceptedGeometry:
    def estimate(self, points1, points2, camera_matrix) -> SimpleNamespace:
        del points2, camera_matrix
        count = points1.shape[0]
        return SimpleNamespace(
            success=True,
            message="accepted",
            rotation=np.eye(3),
            translation_direction=np.asarray([1.0, 0.0, 0.0]),
            inlier_mask=np.ones(count, dtype=bool),
            num_inliers=count,
            inlier_ratio=1.0,
        )


class BiasedPnpPose:
    def estimate(self, points1, points2, *args) -> SimpleNamespace:
        del points2, args
        count = points1.shape[0]
        return SimpleNamespace(
            success=True,
            message="synthetic biased PnP",
            valid_depth_correspondences=count,
            pnp_inliers=count,
            pnp_inlier_ratio=1.0,
            reprojection_rmse_pixels=1.0,
            reprojection_median_pixels=0.9,
            translation_magnitude=0.2,
            translation_units="relative_depth_units",
            rotation=np.eye(3),
            translation=np.asarray([0.2, 0.0, 0.0]),
            inlier_mask=np.ones(count, dtype=bool),
        )


class UnitDepthAligner:
    def __call__(self, prediction, *args, **kwargs) -> DepthAlignmentResult:
        del args, kwargs
        depth = CameraDepth(
            np.ones(prediction.values.shape, dtype=np.float32),
            "relative",
            False,
            "relative_camera_z_proxy",
            "synthetic",
            "relative_depth_units",
            "reciprocal_affine_disparity_alignment",
            "scale_and_shift",
            disparity_scale=1.0,
            disparity_shift=0.0,
            denominator_epsilon=0.001,
        )
        return DepthAlignmentResult(
            True,
            "synthetic alignment",
            depth,
            "scale_and_shift",
            1000,
            1000,
            1.0,
            0.0,
            0.0,
        )


def build_two_frames(enabled: bool):
    refiner = RobustPoseRefiner3D(PoseRefinement3DConfig(
        enabled=enabled,
        minimum_correspondences=20,
        minimum_inliers=20,
        minimum_inlier_ratio=0.4,
        minimum_relative_improvement=0.1,
        random_seed=0,
        ransac_iterations=100,
        residual_threshold_fraction=0.05,
    ))
    builder = RelativeMapBuilder(
        depth_estimator=GridDepthEstimator(),
        feature_tracker=GridFeatureTracker(),
        motion_estimator=AcceptedGeometry(),
        depth_pose_estimator=BiasedPnpPose(),
        depth_aligner=UnitDepthAligner(),
        pose_refiner_3d=refiner,
        camera_matrix=np.asarray([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        keyframe_selector=KeyframeSelector(KeyframeThresholds(enabled=False)),
        point_cloud_stride=1,
        voxel_size=0.01,
        min_depth_alignment_inliers=20,
    )
    frames = [
        MappingFrame(np.zeros((12, 12, 3), dtype=np.uint8), 0),
        MappingFrame(np.zeros((12, 12, 3), dtype=np.uint8), 15),
    ]
    return builder.build(frames)


def test_enabled_refinement_selects_better_pose_and_records_pair_artifacts() -> None:
    result = build_two_frames(enabled=True)

    statistics = result.frame_statistics[1]
    assert statistics.refinement_3d_attempted
    assert statistics.refinement_3d_accepted
    assert statistics.correspondence_3d_count == 64
    assert statistics.refinement_3d_inliers == 64
    assert statistics.refinement_3d_reason == "3d_refinement_accepted"
    assert np.isclose(result.trajectory_positions[1, 0], -0.1, atol=1e-8)
    assert result.pair_alignment is not None
    assert result.pair_alignment.previous_frame_index == 0
    assert result.pair_alignment.current_frame_index == 15
    assert not np.array_equal(
        result.pair_alignment.clouds.before_points,
        result.pair_alignment.clouds.after_points,
    )
    saved = statistics.to_dict()
    assert saved["3d_refinement_attempted"] is True
    assert saved["3d_refinement_accepted"] is True


def test_disabled_refinement_reproduces_pnp_pose_and_creates_no_pair_experiment() -> None:
    result = build_two_frames(enabled=False)

    statistics = result.frame_statistics[1]
    assert not statistics.refinement_3d_attempted
    assert not statistics.refinement_3d_accepted
    assert statistics.refinement_3d_reason == "3d_refinement_disabled"
    assert np.isclose(result.trajectory_positions[1, 0], -0.2, atol=1e-8)
    assert result.pair_alignment is None


def test_pair_before_after_ply_json_and_previews_are_saved(tmp_path) -> None:
    result = build_two_frames(enabled=True)

    paths = save_pair_alignment_artifacts(
        tmp_path,
        result,
        save_previews=True,
        preview_max_points=1000,
    )

    assert paths is not None
    expected = {
        "pair_alignment_before.ply",
        "pair_alignment_after.ply",
        "pair_alignment_metrics.json",
        "pair_alignment_before_oblique.png",
        "pair_alignment_after_oblique.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
