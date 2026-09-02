from pathlib import Path

import cv2
import numpy as np

from src.backprojection import backproject_pixels
from src.depth_pose_estimator import DepthPoseEstimator
from src.depth_types import CameraDepth
from src.map_fusion import RelativeMapFusion
from src.pose_manager import PoseManager
from src.transform_audit import (
    camera_center_world_from_camera_from_world,
    camera_center_world_from_world_from_camera,
    evaluate_two_frame_transform_consistency,
    make_transform_trace_entry,
    save_transform_trace,
)
from src.transforms import invert_transform, make_transform, transform_points


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def project(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = (camera_matrix @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def test_identity_and_known_translation_follow_column_transform_equation() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    identity = transform_points(points, np.eye(3), np.zeros(3))
    translated = transform_points(points, np.eye(3), [10.0, -2.0, 0.5])

    np.testing.assert_array_equal(identity, points)
    np.testing.assert_allclose(translated, points + [10.0, -2.0, 0.5])


def test_known_90_degree_rotation_and_row_vector_equivalence() -> None:
    rotation = rotation_z(90.0)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    transformed = transform_points(points, rotation, np.zeros(3))

    np.testing.assert_allclose(transformed, [[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(transformed, points @ rotation.T, atol=1e-12)


def test_transform_then_inverse_returns_original_points() -> None:
    transform = make_transform(rotation_z(37.0), [1.2, -0.4, 2.5])
    inverse = invert_transform(transform)
    points = np.array([[0.2, 1.0, 3.0], [-2.0, 0.3, 4.5]])

    moved = transform_points(points, transform[:3, :3], transform[:3, 3])
    restored = transform_points(moved, inverse[:3, :3], inverse[:3, 3])

    np.testing.assert_allclose(restored, points, atol=1e-12)


def test_transform_composition_maps_c_to_a_in_correct_order() -> None:
    a_from_b = make_transform(rotation_z(25.0), [1.0, 2.0, -0.5])
    b_from_c = make_transform(rotation_z(-10.0), [-0.2, 0.4, 1.5])
    a_from_c = a_from_b @ b_from_c
    points_c = np.array([[0.5, -1.0, 4.0], [2.0, 0.25, 3.0]])

    points_b = transform_points(points_c, b_from_c[:3, :3], b_from_c[:3, 3])
    sequential = transform_points(points_b, a_from_b[:3, :3], a_from_b[:3, 3])
    composed = transform_points(points_c, a_from_c[:3, :3], a_from_c[:3, 3])

    np.testing.assert_allclose(composed, sequential, atol=1e-12)


def test_three_frame_pose_accumulation_matches_analytic_composition() -> None:
    local_transforms = [
        make_transform(rotation_z(10.0), [0.4, -0.1, 0.05]),
        make_transform(rotation_z(-6.0), [0.2, 0.3, -0.02]),
        make_transform(rotation_z(3.0), [-0.1, 0.25, 0.04]),
    ]
    manager = PoseManager()
    for local in local_transforms:
        manager.add_scaled_relative_pose(local[:3, :3], local[:3, 3])

    expected = np.eye(4)
    for current_from_previous in local_transforms:
        expected = expected @ invert_transform(current_from_previous)

    np.testing.assert_allclose(manager.current_pose(), expected, atol=1e-12)


def test_three_camera_static_landmark_fuses_at_one_world_coordinate() -> None:
    manager = PoseManager()
    world_poses = [manager.current_pose()]
    for rotation, translation in (
        (rotation_z(8.0), np.array([0.3, 0.0, 0.05])),
        (rotation_z(-5.0), np.array([0.2, 0.1, -0.02])),
    ):
        world_poses.append(manager.add_scaled_relative_pose(rotation, translation))

    landmark_world = np.array([[1.5, -0.4, 6.0]])
    fusion = RelativeMapFusion(voxel_size=1e-6)
    for pose in world_poses:
        camera_from_world = invert_transform(pose)
        landmark_camera = transform_points(
            landmark_world,
            camera_from_world[:3, :3],
            camera_from_world[:3, 3],
        )
        reconstructed_world = transform_points(
            landmark_camera, pose[:3, :3], pose[:3, 3]
        )
        fusion.add(reconstructed_world, np.array([[10, 20, 30]], dtype=np.uint8))

    fused = fusion.finalize()
    assert fused.output_point_count == 1
    np.testing.assert_allclose(fused.points, landmark_world, atol=1e-12)


def test_camera_center_formulas_agree_for_inverse_pose_forms() -> None:
    world_from_camera = make_transform(rotation_z(31.0), [2.0, -1.0, 0.5])
    camera_from_world = invert_transform(world_from_camera)

    direct = camera_center_world_from_world_from_camera(world_from_camera)
    opencv_form = camera_center_world_from_camera_from_world(camera_from_world)

    np.testing.assert_allclose(direct, world_from_camera[:3, 3], atol=1e-12)
    np.testing.assert_allclose(opencv_form, direct, atol=1e-12)


def test_depth_pose_estimator_returns_current_from_previous() -> None:
    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    u = np.array([100, 160, 220, 280, 340, 400, 460, 520] * 2, dtype=float)
    v = np.array([120] * 8 + [340] * 8, dtype=float)
    previous_pixels = np.column_stack((u, v))
    depths = np.array([
        3.0, 4.2, 5.1, 3.7, 6.0, 4.8, 5.5, 3.4,
        5.8, 3.3, 4.5, 6.2, 3.9, 5.3, 4.0, 6.5,
    ])
    previous_points = backproject_pixels(previous_pixels, depths, camera_matrix)
    rotation, _ = cv2.Rodrigues(np.array([0.025, -0.035, 0.015]))
    translation = np.array([0.35, -0.12, 0.22])
    current_points = transform_points(previous_points, rotation, translation)
    current_pixels = project(current_points, camera_matrix)
    values = np.full((480, 640), np.nan, dtype=np.float32)
    values[v.astype(int), u.astype(int)] = depths
    depth = CameraDepth(
        values, "relative", False, "relative_camera_z_proxy", "synthetic",
        "relative_depth_units", "synthetic", "none",
    )

    result = DepthPoseEstimator(
        sampling_method="nearest", minimum_inliers=8,
        reprojection_error_pixels=1.0,
    ).estimate(
        previous_pixels,
        current_pixels,
        np.ones(previous_pixels.shape[0], dtype=bool),
        depth,
        camera_matrix,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.rotation, rotation, atol=1e-5)
    np.testing.assert_allclose(result.translation, translation, atol=1e-5)
    np.testing.assert_allclose(
        transform_points(previous_points, result.rotation, result.translation),
        current_points,
        atol=1e-5,
    )


def test_recover_pose_point_order_returns_current_from_previous() -> None:
    generator = np.random.default_rng(11)
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    previous_points = np.column_stack((
        generator.uniform(-1.5, 1.5, 120),
        generator.uniform(-1.0, 1.0, 120),
        generator.uniform(4.0, 9.0, 120),
    ))
    rotation, _ = cv2.Rodrigues(np.array([0.02, -0.04, 0.01]))
    translation = np.array([0.35, 0.04, 0.08])
    current_points = transform_points(previous_points, rotation, translation)
    points1 = project(previous_points, camera_matrix)
    points2 = project(current_points, camera_matrix)
    tx = np.array([
        [0.0, -translation[2], translation[1]],
        [translation[2], 0.0, -translation[0]],
        [-translation[1], translation[0], 0.0],
    ])
    essential = tx @ rotation

    _, recovered_rotation, recovered_translation, _ = cv2.recoverPose(
        essential, points1, points2, camera_matrix
    )

    expected_direction = translation / np.linalg.norm(translation)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-7)
    np.testing.assert_allclose(
        recovered_translation.reshape(3), expected_direction, atol=1e-7
    )


def test_local_and_world_pair_discrepancies_are_identical() -> None:
    previous = np.array([
        [0.2, -0.1, 3.0], [1.0, 0.4, 5.0], [-0.7, 0.2, 4.0]
    ])
    rotation = rotation_z(12.0)
    translation = np.array([0.3, -0.05, 0.1])
    current = transform_points(previous, rotation, translation)
    current[1] += [0.01, -0.02, 0.03]
    world_from_previous = make_transform(rotation_z(-20.0), [2.0, 1.0, -0.5])
    current_from_previous = make_transform(rotation, translation)
    world_from_current = world_from_previous @ invert_transform(current_from_previous)

    result = evaluate_two_frame_transform_consistency(
        previous,
        current,
        rotation,
        translation,
        world_from_previous,
        world_from_current,
        previous_frame_index=0,
        current_frame_index=15,
        coordinate_units="relative_depth_units",
        is_metric=False,
    )

    assert result.correspondence_count == 3
    np.testing.assert_allclose(
        result.local_median_discrepancy, result.world_median_discrepancy, atol=1e-12
    )
    np.testing.assert_allclose(
        result.local_rmse_discrepancy, result.world_rmse_discrepancy, atol=1e-12
    )
    assert result.maximum_local_world_residual_difference < 1e-12


def test_transform_trace_serializes_explicit_4x4_matrices(tmp_path: Path) -> None:
    local = make_transform(rotation_z(7.0), [0.2, -0.1, 0.05])
    world = invert_transform(local)
    entry = make_transform_trace_entry(
        15, local[:3, :3], local[:3, 3], world
    )

    destination = save_transform_trace(tmp_path / "transform_trace.json", [entry], [])
    text = destination.read_text(encoding="utf-8")

    assert '"T_current_from_previous"' in text
    assert '"T_previous_from_current"' in text
    assert '"T_world_from_camera"' in text
    assert '"camera_center_world"' in text
