from __future__ import annotations

import numpy as np

from src.depth_types import CameraDepth
from src.pose_refinement_3d import (
    Correspondences3D,
    PoseRefinement3DConfig,
    RobustPoseRefiner3D,
    build_3d_correspondences,
    build_pair_alignment_clouds,
    correspondence_residuals,
    kabsch_rigid_transform,
    robust_rigid_alignment,
)


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def synthetic_points(count: int = 200, seed: int = 4) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(count, 3)) * [1.0, 0.7, 0.4] + [0.2, -0.1, 3.0]


def correspondences(
    previous: np.ndarray, current: np.ndarray
) -> Correspondences3D:
    return Correspondences3D(
        previous_points=previous,
        current_points=current,
        match_indices=np.arange(previous.shape[0]),
        supported_match_count=previous.shape[0],
        coordinate_units="relative_depth_units",
        is_metric=False,
    )


def camera_depth(values: np.ndarray) -> CameraDepth:
    return CameraDepth(
        np.asarray(values, dtype=np.float32),
        "relative",
        False,
        "relative_camera_z_proxy",
        "synthetic",
        "relative_depth_units",
        "synthetic",
        "scale_and_shift",
    )


def permissive_config(**overrides: object) -> PoseRefinement3DConfig:
    values = {
        "enabled": True,
        "minimum_correspondences": 20,
        "minimum_inliers": 15,
        "minimum_inlier_ratio": 0.4,
        "minimum_relative_improvement": 0.1,
        "random_seed": 9,
        "ransac_iterations": 300,
        "residual_threshold_fraction": 0.04,
        "maximum_translation_change_ratio": 5.0,
        "maximum_rotation_change_degrees": 20.0,
    }
    values.update(overrides)
    return PoseRefinement3DConfig(**values)


def test_kabsch_recovers_known_current_from_previous_transform() -> None:
    previous = synthetic_points(80)
    expected_rotation = rotation_z(7.0)
    expected_translation = np.asarray([0.25, -0.08, 0.12])
    current = (expected_rotation @ previous.T).T + expected_translation

    rotation, translation = kabsch_rigid_transform(previous, current)

    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-10)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-10)
    np.testing.assert_allclose(
        (rotation @ previous.T).T + translation, current, atol=1e-10
    )


def test_kabsch_reflection_correction_returns_proper_rotation() -> None:
    previous = synthetic_points(60)
    reflected = previous.copy()
    reflected[:, 0] *= -1.0

    rotation, _ = kabsch_rigid_transform(previous, reflected)

    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


def test_kabsch_recovers_transform_with_small_gaussian_noise() -> None:
    generator = np.random.default_rng(12)
    previous = synthetic_points(300)
    expected_rotation = rotation_z(-4.0)
    expected_translation = np.asarray([-0.16, 0.05, 0.09])
    current = (
        (expected_rotation @ previous.T).T
        + expected_translation
        + generator.normal(scale=0.002, size=previous.shape)
    )

    rotation, translation = kabsch_rigid_transform(previous, current)

    np.testing.assert_allclose(rotation, expected_rotation, atol=5e-4)
    np.testing.assert_allclose(translation, expected_translation, atol=2e-3)


def test_robust_estimator_rejects_injected_3d_outliers_deterministically() -> None:
    generator = np.random.default_rng(22)
    previous = synthetic_points(250)
    expected_rotation = rotation_z(5.0)
    expected_translation = np.asarray([0.2, -0.04, 0.07])
    current = (expected_rotation @ previous.T).T + expected_translation
    current += generator.normal(scale=0.002, size=current.shape)
    outliers = generator.choice(previous.shape[0], size=70, replace=False)
    current[outliers] += generator.normal(scale=3.0, size=(outliers.size, 3))

    first = robust_rigid_alignment(
        previous, current, random_seed=3, iterations=400, threshold_fraction=0.04
    )
    second = robust_rigid_alignment(
        previous, current, random_seed=3, iterations=400, threshold_fraction=0.04
    )

    assert first.success
    assert first.inlier_count >= 175
    assert first.inlier_count <= 185
    np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)
    np.testing.assert_allclose(first.rotation, expected_rotation, atol=1e-3)
    np.testing.assert_allclose(first.translation, expected_translation, atol=3e-3)


def test_invalid_depth_correspondences_are_removed_on_either_side() -> None:
    previous_depth = np.ones((5, 5), dtype=np.float32)
    current_depth = np.ones((5, 5), dtype=np.float32)
    previous_depth[1, 1] = np.nan
    current_depth[2, 2] = np.nan
    previous_pixels = np.asarray([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]
    ])
    current_pixels = previous_pixels.copy()
    support = np.asarray([True, True, True, False, True])

    result = build_3d_correspondences(
        previous_pixels,
        current_pixels,
        support,
        camera_depth(previous_depth),
        camera_depth(current_depth),
        np.eye(3),
    )

    np.testing.assert_array_equal(result.match_indices, [0, 4])
    assert result.count == 2
    assert np.isfinite(result.previous_points).all()
    assert np.isfinite(result.current_points).all()


def test_insufficient_correspondences_fall_back_to_pnp() -> None:
    previous = synthetic_points(8)
    baseline_rotation = np.eye(3)
    baseline_translation = np.asarray([0.1, 0.0, 0.0])
    current = previous + baseline_translation
    refiner = RobustPoseRefiner3D(permissive_config(minimum_correspondences=20))

    result = refiner.refine(
        correspondences(previous, current), baseline_rotation, baseline_translation
    )

    assert not result.accepted
    assert result.reason == "3d_refinement_insufficient_correspondences"
    np.testing.assert_array_equal(result.selected_rotation, baseline_rotation)
    np.testing.assert_array_equal(result.selected_translation, baseline_translation)


def test_refinement_is_rejected_when_baseline_already_fits() -> None:
    previous = synthetic_points()
    baseline_rotation = rotation_z(3.0)
    baseline_translation = np.asarray([0.15, -0.03, 0.02])
    current = (baseline_rotation @ previous.T).T + baseline_translation
    refiner = RobustPoseRefiner3D(permissive_config())

    result = refiner.refine(
        correspondences(previous, current), baseline_rotation, baseline_translation
    )

    assert not result.accepted
    assert result.reason == "3d_refinement_not_improved"
    np.testing.assert_array_equal(result.selected_translation, baseline_translation)


def test_refinement_is_accepted_when_baseline_residual_is_clearly_worse() -> None:
    previous = synthetic_points()
    true_rotation = rotation_z(4.0)
    true_translation = np.asarray([0.22, -0.05, 0.06])
    current = (true_rotation @ previous.T).T + true_translation
    baseline_rotation = rotation_z(1.0)
    baseline_translation = np.asarray([0.32, 0.02, 0.01])
    refiner = RobustPoseRefiner3D(permissive_config())

    result = refiner.refine(
        correspondences(previous, current), baseline_rotation, baseline_translation
    )

    assert result.accepted
    assert result.reason == "3d_refinement_accepted"
    assert result.relative_improvement is not None
    assert result.relative_improvement > 0.9
    np.testing.assert_allclose(result.selected_rotation, true_rotation, atol=1e-8)
    np.testing.assert_allclose(result.selected_translation, true_translation, atol=1e-8)


def test_disabled_refinement_preserves_existing_pnp_pose() -> None:
    previous = synthetic_points(30)
    baseline_rotation = rotation_z(2.0)
    baseline_translation = np.asarray([0.1, 0.02, -0.01])
    current = previous.copy()
    refiner = RobustPoseRefiner3D(PoseRefinement3DConfig(enabled=False))

    result = refiner.refine(
        correspondences(previous, current), baseline_rotation, baseline_translation
    )

    assert not result.attempted
    assert not result.accepted
    assert result.reason == "3d_refinement_disabled"
    np.testing.assert_array_equal(result.selected_rotation, baseline_rotation)
    np.testing.assert_array_equal(result.selected_translation, baseline_translation)


def test_transform_direction_is_current_from_previous() -> None:
    previous = synthetic_points(50)
    rotation_current_previous = rotation_z(6.0)
    translation_current_previous = np.asarray([0.18, -0.07, 0.03])
    current = (
        rotation_current_previous @ previous.T
    ).T + translation_current_previous

    rotation, translation = kabsch_rigid_transform(previous, current)
    forward = correspondence_residuals(
        previous, current, rotation, translation
    )
    reverse = correspondence_residuals(
        current, previous, rotation, translation
    )

    assert np.max(forward) < 1e-10
    assert np.median(reverse) > 0.1


def test_kabsch_does_not_estimate_or_apply_scale() -> None:
    previous = synthetic_points(80)
    current = 1.7 * previous + np.asarray([0.1, -0.2, 0.3])

    rotation, translation = kabsch_rigid_transform(previous, current)
    transformed = (rotation @ previous.T).T + translation

    source_distance = np.linalg.norm(previous[0] - previous[1])
    transformed_distance = np.linalg.norm(transformed[0] - transformed[1])
    target_distance = np.linalg.norm(current[0] - current[1])
    assert np.isclose(transformed_distance, source_distance)
    assert not np.isclose(transformed_distance, target_distance)


def test_pair_alignment_artifacts_do_not_modify_raw_camera_clouds() -> None:
    previous = synthetic_points(10)
    current = synthetic_points(12, seed=10)
    previous_colors = np.full((10, 3), 50, dtype=np.uint8)
    current_colors = np.full((12, 3), 200, dtype=np.uint8)
    previous_copy = previous.copy()
    current_copy = current.copy()
    previous_pose = np.eye(4)
    baseline_pose = np.eye(4)
    baseline_pose[0, 3] = -0.2
    selected_pose = np.eye(4)
    selected_pose[0, 3] = -0.1

    pair = build_pair_alignment_clouds(
        previous,
        previous_colors,
        current,
        current_colors,
        previous_pose,
        baseline_pose,
        selected_pose,
    )

    np.testing.assert_array_equal(previous, previous_copy)
    np.testing.assert_array_equal(current, current_copy)
    np.testing.assert_array_equal(pair.before_points[:10], previous)
    np.testing.assert_array_equal(pair.after_points[:10], previous)
    np.testing.assert_array_equal(pair.colors[:10], previous_colors)
    np.testing.assert_array_equal(pair.colors[10:], current_colors)
    assert pair.before_points.shape == pair.after_points.shape == (22, 3)
