import cv2
import numpy as np

from src.depth_types import CameraDepth
from src.sliding_window_pose_optimization import (
    PoseWindowFrame,
    ReprojectionEdge,
    SlidingWindowPoseOptimizationConfig,
    SlidingWindowPoseOptimizer,
    SolverResult,
    build_reprojection_edge,
)
from src.transforms import invert_transform, make_transform, transform_points


K = np.array(
    [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (640, 480)


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])


def project_camera(points: np.ndarray) -> np.ndarray:
    return np.column_stack((
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
    ))


def camera_points(world_points: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    camera_from_world = invert_transform(world_from_camera)
    return transform_points(
        world_points,
        camera_from_world[:3, :3],
        camera_from_world[:3, 3],
    )


def synthetic_problem(
    *, perturb: bool, outliers: bool = False, include_direct: bool = True
):
    generator = np.random.default_rng(13)
    world_points = np.column_stack((
        generator.uniform(-1.2, 1.2, 120),
        generator.uniform(-0.8, 0.8, 120),
        generator.uniform(4.0, 8.0, 120),
    ))
    true_poses = [
        np.eye(4),
        make_transform(rotation_y(1.0), [0.20, 0.00, 0.02]),
        make_transform(rotation_y(2.0), [0.41, 0.01, 0.04]),
    ]
    baseline = [pose.copy() for pose in true_poses]
    if perturb:
        baseline[1] = make_transform(rotation_y(1.35), [0.218, -0.006, 0.026])
        baseline[2] = make_transform(rotation_y(2.55), [0.443, 0.018, 0.052])
    camera = [camera_points(world_points, pose) for pose in true_poses]
    pixels = [project_camera(points) for points in camera]
    pairs = [(0, 1), (1, 2)] + ([(0, 2)] if include_direct else [])
    edges = []
    for source, target in pairs:
        observations = pixels[target].copy()
        if outliers:
            observations[:12] += np.column_stack((
                np.linspace(60.0, 120.0, 12),
                np.linspace(-100.0, 80.0, 12),
            ))
        edges.append(ReprojectionEdge(
            source,
            target,
            camera[source].copy(),
            observations,
        ))
    frames = [PoseWindowFrame(index, pose) for index, pose in enumerate(baseline)]
    return frames, edges, true_poses


def permissive_config(**overrides) -> SlidingWindowPoseOptimizationConfig:
    values = dict(
        enabled=True,
        window_size=3,
        minimum_observations=20,
        minimum_relative_median_improvement=0.0,
        maximum_rotation_change_deg=10.0,
        maximum_translation_change_relative=10.0,
        robust_loss="huber",
        f_scale_px=2.0,
        max_nfev=220,
    )
    values.update(overrides)
    return SlidingWindowPoseOptimizationConfig(**values)


def pose_error(pose: np.ndarray, truth: np.ndarray) -> float:
    rotation = pose[:3, :3] @ truth[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    translation = np.linalg.norm(pose[:3, 3] - truth[:3, 3])
    return float(angle + translation)


def test_zero_error_synthetic_poses_remain_unchanged() -> None:
    frames, edges, _ = synthetic_problem(perturb=False)

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    for before, selected in zip(frames, result.selected_world_from_camera):
        np.testing.assert_array_equal(selected, before.world_from_camera)


def test_perturbed_poses_move_toward_known_correct_poses() -> None:
    frames, edges, truth = synthetic_problem(perturb=True)

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    assert result.accepted, result.reason
    for index in (1, 2):
        assert pose_error(result.proposed_world_from_camera[index], truth[index]) < pose_error(
            frames[index].world_from_camera, truth[index]
        )


def test_first_pose_in_window_remains_bitwise_fixed() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)
    original = frames[0].world_from_camera.copy()

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    np.testing.assert_array_equal(result.proposed_world_from_camera[0], original)
    np.testing.assert_array_equal(result.selected_world_from_camera[0], original)


def test_multiple_edges_constrain_shared_pose() -> None:
    frames, edges, truth = synthetic_problem(perturb=True, include_direct=True)

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    assert set(result.edges_used) == {(0, 1), (1, 2), (0, 2)}
    assert result.observations_per_edge["0->2"] == 120
    assert pose_error(result.proposed_world_from_camera[2], truth[2]) < 1e-4


def test_huber_loss_tolerates_injected_pixel_outliers() -> None:
    frames, edges, truth = synthetic_problem(perturb=True, outliers=True)

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    assert result.solver_success
    assert pose_error(result.proposed_world_from_camera[2], truth[2]) < 0.01
    assert result.metrics_after.median < result.metrics_before.median


def test_invalid_pixels_depths_and_behind_camera_points_are_rejected() -> None:
    frames, edges, _ = synthetic_problem(perturb=False, include_direct=False)
    edge = edges[0]
    source = np.vstack((edge.source_points_camera[:10], [np.nan, 0, 1], [0, 0, -1]))
    target = np.vstack((edge.target_pixels[:10], [10, 10], [20, 20]))
    target[0] = [-1, 20]

    result = SlidingWindowPoseOptimizer(permissive_config(
        minimum_observations=1
    )).optimize(
        frames[:2], [ReprojectionEdge(0, 1, source, target)], K, IMAGE_SIZE
    )

    assert result.total_observation_count == 9


def test_optimizer_failure_falls_back_to_baseline() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)

    def failing_solver(residual, initial, config):
        del residual, config
        return SolverResult(False, "synthetic failure", initial.copy(), 1.0, 1)

    result = SlidingWindowPoseOptimizer(
        permissive_config(), solver=failing_solver
    ).optimize(frames, edges, K, IMAGE_SIZE)

    assert not result.accepted
    assert result.reason == "sliding_window_solver_failed"
    for frame, selected in zip(frames, result.selected_world_from_camera):
        np.testing.assert_array_equal(selected, frame.world_from_camera)


def test_insufficient_observations_fall_back() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)

    result = SlidingWindowPoseOptimizer(permissive_config(
        minimum_observations=10_000
    )).optimize(frames, edges, K, IMAGE_SIZE)

    assert not result.accepted
    assert result.reason == "sliding_window_insufficient_observations"


def test_improvement_below_threshold_is_rejected() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)

    result = SlidingWindowPoseOptimizer(permissive_config(
        minimum_relative_median_improvement=1.0
    )).optimize(frames, edges, K, IMAGE_SIZE)

    assert not result.accepted
    assert result.reason == "sliding_window_improvement_below_threshold"
    assert result.relative_median_improvement is not None


def test_excessive_pose_change_is_rejected() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)

    result = SlidingWindowPoseOptimizer(permissive_config(
        maximum_translation_change_relative=1e-5
    )).optimize(frames, edges, K, IMAGE_SIZE)

    assert not result.accepted
    assert result.reason == "sliding_window_excessive_translation_change"


def test_accepted_optimization_changes_only_later_pose_matrices() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)
    source_copies = [edge.source_points_camera.copy() for edge in edges]
    pixel_copies = [edge.target_pixels.copy() for edge in edges]

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    assert result.accepted
    assert result.window_frame_indices == (0, 1, 2)
    assert result.fixed_frame_index == 0
    assert result.optimized_frame_indices == (1, 2)
    np.testing.assert_array_equal(
        result.selected_world_from_camera[0], frames[0].world_from_camera
    )
    for edge, source, pixels in zip(edges, source_copies, pixel_copies):
        np.testing.assert_array_equal(edge.source_points_camera, source)
        np.testing.assert_array_equal(edge.target_pixels, pixels)


def test_building_edge_does_not_mutate_depth_array() -> None:
    values = np.full((8, 8), 2.0, dtype=np.float32)
    original = values.copy()
    depth = CameraDepth(
        values, "relative", False, "relative_camera_z_proxy", "synthetic",
        "relative_depth_units", "synthetic", "none",
    )
    source = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = source + [0.5, 0.0]

    edge = build_reprojection_edge(
        0, 1, source, target, np.ones(3, dtype=bool), depth,
        np.array([[4.0, 0, 4.0], [0, 4.0, 4.0], [0, 0, 1.0]]),
    )

    assert edge.observation_count == 3
    np.testing.assert_array_equal(depth.values, original)


def test_transform_convention_projects_static_world_points_consistently() -> None:
    frames, edges, _ = synthetic_problem(perturb=False)

    result = SlidingWindowPoseOptimizer(permissive_config()).optimize(
        frames, edges, K, IMAGE_SIZE
    )

    assert result.metrics_before.rmse is not None
    assert result.metrics_before.rmse < 1e-10


def test_disabled_optimizer_reproduces_input_pose_sequence() -> None:
    frames, edges, _ = synthetic_problem(perturb=True)

    result = SlidingWindowPoseOptimizer(
        SlidingWindowPoseOptimizationConfig(enabled=False)
    ).optimize(frames, edges, K, IMAGE_SIZE)

    assert not result.attempted
    assert result.reason == "sliding_window_pose_optimization_disabled"
    for frame, selected in zip(frames, result.selected_world_from_camera):
        np.testing.assert_array_equal(selected, frame.world_from_camera)
