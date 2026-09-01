from pathlib import Path

import numpy as np

from src.depth_types import CameraDepth
from src.temporal_depth_normalization import (
    TemporalDepthNormalizationConfig,
    estimate_temporal_scale,
    matched_world_residual_statistics,
    normalize_temporal_depth,
)
from src.visual_outputs import save_temporal_depth_normalization_comparison


def config(**overrides) -> TemporalDepthNormalizationConfig:
    values = {
        "enabled": True,
        "minimum_correspondences": 4,
        "minimum_inliers": 3,
        "minimum_inlier_ratio": 0.5,
        "minimum_scale": 0.5,
        "maximum_scale": 2.5,
        "maximum_log_mad": 0.25,
        "minimum_cumulative_scale": 0.5,
        "maximum_cumulative_scale": 2.0,
    }
    values.update(overrides)
    return TemporalDepthNormalizationConfig(**values)


def depth(values: np.ndarray) -> CameraDepth:
    return CameraDepth(
        np.asarray(values, dtype=np.float32),
        "relative",
        False,
        "relative_camera_z_proxy",
        "synthetic",
        "relative_depth_units",
        "reciprocal_affine_disparity_alignment",
        "scale_and_shift",
        disparity_scale=2.0,
        disparity_shift=0.25,
    )


def grid_points() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_identical_depth_scales_recover_one() -> None:
    values = np.linspace(1.0, 2.0, 10)
    result = estimate_temporal_scale(values, values.copy(), config())
    assert result.accepted
    assert np.isclose(result.temporal_scale_pairwise, 1.0)


def test_known_multiplicative_mismatch_is_recovered() -> None:
    previous = np.linspace(2.0, 4.0, 10)
    current = previous / 2.0
    result = estimate_temporal_scale(previous, current, config())
    assert result.accepted
    assert np.isclose(result.temporal_scale_pairwise, 2.0)


def test_gross_ratio_outlier_is_robustly_rejected() -> None:
    previous = np.ones(10)
    current = np.ones(10)
    previous[-1] = 1000.0
    result = estimate_temporal_scale(previous, current, config())
    assert result.valid_ratio_count == 10
    assert result.inlier_count == 9
    assert np.isclose(result.temporal_scale_pairwise, 1.0)


def test_invalid_and_nonpositive_depth_samples_are_excluded() -> None:
    previous = np.asarray([1.0, 2.0, np.nan, -1.0, 0.0, 3.0])
    current = np.asarray([1.0, 2.0, 1.0, 1.0, 1.0, np.inf])
    result = estimate_temporal_scale(
        previous, current, config(minimum_correspondences=2, minimum_inliers=2)
    )
    assert result.valid_ratio_count == 2
    assert result.inlier_count == 2


def test_insufficient_correspondences_falls_back_gracefully() -> None:
    result = estimate_temporal_scale(
        np.ones(3), np.ones(3), config(minimum_correspondences=4)
    )
    assert not result.accepted
    assert result.reason.endswith("insufficient_correspondences")
    assert result.temporal_scale_cumulative == 1.0


def test_scale_outside_conservative_range_is_rejected() -> None:
    result = estimate_temporal_scale(
        np.full(10, 2.0),
        np.ones(10),
        config(minimum_scale=0.7, maximum_scale=1.3),
    )
    assert not result.accepted
    assert result.reason.endswith("scale_out_of_range")


def test_excessive_log_ratio_mad_is_rejected() -> None:
    ratios = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2] * 2)
    result = estimate_temporal_scale(
        ratios, np.ones(10), config(maximum_log_mad=0.01)
    )
    assert not result.accepted
    assert result.reason.endswith("excessive_log_mad")


def test_disabled_mode_returns_original_depth_object() -> None:
    original = depth(np.ones((2, 2)))
    result = normalize_temporal_depth(
        grid_points(), grid_points(), np.ones(4, dtype=bool),
        original, original, TemporalDepthNormalizationConfig(enabled=False),
    )
    assert result.normalized_depth is original
    assert not result.diagnostics.temporal_depth_normalization_attempted


def test_original_depth_arrays_are_not_mutated() -> None:
    previous = depth(np.full((2, 2), 2.0))
    current = depth(np.ones((2, 2)))
    previous_before = previous.values.copy()
    current_before = current.values.copy()
    result = normalize_temporal_depth(
        grid_points(), grid_points(), np.ones(4, dtype=bool),
        previous, current, config(),
    )
    np.testing.assert_array_equal(previous.values, previous_before)
    np.testing.assert_array_equal(current.values, current_before)
    assert result.normalized_depth is not current


def test_fixed_pose_residual_diagnostic_does_not_mutate_poses() -> None:
    points = grid_points()
    pose_previous = np.eye(4)
    pose_current = np.eye(4)
    pose_previous_before = pose_previous.copy()
    pose_current_before = pose_current.copy()
    stats = matched_world_residual_statistics(
        points, points, np.ones(4, dtype=bool),
        depth(np.ones((2, 2))), depth(np.ones((2, 2))),
        np.eye(3), pose_previous, pose_current,
    )
    assert stats["median"] == 0.0
    np.testing.assert_array_equal(pose_previous, pose_previous_before)
    np.testing.assert_array_equal(pose_current, pose_current_before)


def test_before_after_artifacts_use_two_explicit_point_sets(tmp_path: Path) -> None:
    baseline = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    normalized = baseline[:1]
    colors = np.full((2, 3), 120, dtype=np.uint8)
    paths = save_temporal_depth_normalization_comparison(
        tmp_path, baseline, colors, normalized, colors[:1], preview_max_points=10
    )
    assert len(paths) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_cumulative_scale_guard_rejects_unbounded_product() -> None:
    result = estimate_temporal_scale(
        np.full(10, 1.2), np.ones(10), config(),
        previous_cumulative_scale=1.9,
    )
    assert not result.accepted
    assert result.reason.endswith("cumulative_scale_guard")
    assert np.isclose(result.temporal_scale_cumulative, 1.9)
    assert np.isclose(result.temporal_scale_cumulative_candidate, 2.28)


def test_normalized_depth_preserves_relative_nonmetric_semantics() -> None:
    previous = depth(np.full((2, 2), 1.1))
    current = depth(np.ones((2, 2)))
    result = normalize_temporal_depth(
        grid_points(), grid_points(), np.ones(4, dtype=bool),
        previous, current, config(),
    )
    assert result.normalized_depth.depth_type == "relative"
    assert result.normalized_depth.is_metric is False
    assert result.normalized_depth.coordinate_units == "relative_depth_units"
    assert result.normalized_depth.representation == "relative_camera_z_proxy"
    assert result.diagnostics.temporal_original_alignment_a == 2.0
    assert result.diagnostics.temporal_original_alignment_b == 0.25
