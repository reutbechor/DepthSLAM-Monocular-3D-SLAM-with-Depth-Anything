from pathlib import Path

import numpy as np

from src.depth_types import CameraDepth
from src.pose_chain_diagnostics import (
    DirectPoseEstimate,
    PoseChainDiagnosticConfig,
    PoseChainFrameInput,
    analyze_pose_chain,
    build_pose_chain_summary,
    compare_world_poses,
    direct_world_pose_from_reference_to_current,
    rotation_difference_degrees,
    save_pose_chain_diagnostics,
)
from tools.run_relative_map import save_optional_pose_chain_diagnostics


def world_pose(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: float = 0.0,
) -> np.ndarray:
    angle = np.radians(rotation_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = position
    return pose


def camera_depth() -> CameraDepth:
    return CameraDepth(
        np.ones((2, 2), dtype=np.float32),
        "relative",
        False,
        "relative_camera_z_proxy",
        "synthetic",
        "relative_depth_units",
        "reciprocal_median_normalized_zero_shift_assumption",
        "none",
        disparity_scale=1.0,
        disparity_shift=0.0,
    )


def frame(index: int, pose: np.ndarray) -> PoseChainFrameInput:
    return PoseChainFrameInput(
        index,
        np.full((2, 2, 3), index, dtype=np.uint8),
        camera_depth(),
        pose,
    )


class FixedDirectEstimator:
    def __init__(self, estimates: list[DirectPoseEstimate]) -> None:
        self.estimates = iter(estimates)

    def estimate(self, reference, current) -> DirectPoseEstimate:
        del reference, current
        return next(self.estimates)


def successful_direct(pose: np.ndarray) -> DirectPoseEstimate:
    return DirectPoseEstimate(
        True,
        "accepted",
        pose,
        feature_matches=200,
        geometric_inliers=160,
        geometric_inlier_ratio=0.8,
        pnp_inliers=120,
        pnp_inlier_ratio=0.75,
        reprojection_rmse=1.0,
    )


def test_identical_chained_and_direct_poses_have_zero_error() -> None:
    pose = world_pose((1.0, -2.0, 3.0), 7.0)

    comparison = compare_world_poses(pose, pose.copy())

    assert comparison["translation_difference"] == 0.0
    assert comparison["relative_translation_difference"] == 0.0
    assert comparison["rotation_difference_deg"] == 0.0


def test_known_translation_offset_has_expected_difference() -> None:
    chained = world_pose((1.5, 0.0, 0.0))
    direct = world_pose((1.0, 0.0, 0.0))

    comparison = compare_world_poses(chained, direct)

    assert comparison["translation_difference"] == 0.5
    assert comparison["relative_translation_difference"] == 0.5


def test_known_rotation_offset_has_expected_angle() -> None:
    assert np.isclose(
        rotation_difference_degrees(world_pose(), world_pose(rotation_deg=12.0)),
        12.0,
    )


def test_current_from_reference_is_inverted_into_world_from_current() -> None:
    reference = world_pose()

    direct = direct_world_pose_from_reference_to_current(
        reference, np.eye(3), np.array([1.0, 2.0, 3.0])
    )

    np.testing.assert_allclose(direct[:3, 3], [-1.0, -2.0, -3.0])


def test_direct_pose_failure_is_recorded_without_comparison() -> None:
    frames = [frame(0, world_pose()), frame(15, world_pose((1.0, 0.0, 0.0)))]
    estimator = FixedDirectEstimator([
        DirectPoseEstimate(False, "direct_depth_pnp_rejected", None)
    ])

    rows = analyze_pose_chain(frames, estimator, PoseChainDiagnosticConfig())

    assert rows[1].direct_pose_available is False
    assert rows[1].comparison_confidence == "unavailable"
    assert rows[1].translation_difference is None


def test_unreliable_direct_pose_is_explicitly_low_confidence() -> None:
    poor = DirectPoseEstimate(
        True,
        "accepted_but_poor",
        world_pose((0.9, 0.0, 0.0)),
        feature_matches=20,
        geometric_inliers=5,
        geometric_inlier_ratio=0.10,
        pnp_inliers=4,
        pnp_inlier_ratio=0.20,
        reprojection_rmse=5.0,
    )

    rows = analyze_pose_chain(
        [frame(0, world_pose()), frame(15, world_pose((1.0, 0.0, 0.0)))],
        FixedDirectEstimator([poor]),
        PoseChainDiagnosticConfig(),
    )
    summary = build_pose_chain_summary(rows, PoseChainDiagnosticConfig())

    assert rows[1].comparison_confidence == "low"
    assert rows[1].comparison_low_confidence is True
    assert summary["high_confidence_comparison_count"] == 0
    assert summary["heuristic_warning_flags"]["direct_pose_quality_insufficient"]


def test_diagnostics_do_not_mutate_captured_mapper_poses_or_images() -> None:
    frames = [frame(0, world_pose()), frame(15, world_pose((1.0, 0.0, 0.0)))]
    poses_before = [item.world_from_camera.copy() for item in frames]
    images_before = [item.image_bgr.copy() for item in frames]

    analyze_pose_chain(
        frames,
        FixedDirectEstimator([successful_direct(world_pose((0.9, 0.0, 0.0)))]),
        PoseChainDiagnosticConfig(),
    )

    for item, pose_before, image_before in zip(frames, poses_before, images_before):
        np.testing.assert_array_equal(item.world_from_camera, pose_before)
        np.testing.assert_array_equal(item.image_bgr, image_before)


def test_disabled_diagnostics_create_no_directory(tmp_path: Path) -> None:
    metadata: dict[str, object] = {}

    result = save_optional_pose_chain_diagnostics(
        tmp_path, metadata, None, PoseChainDiagnosticConfig(enabled=False)
    )

    assert result is None
    assert not (tmp_path / "pose_chain_diagnostics").exists()
    assert metadata["pose_chain_diagnostics"] == {"enabled": False}


def test_all_required_plots_and_tables_are_generated(tmp_path: Path) -> None:
    rows = analyze_pose_chain(
        [frame(0, world_pose()), frame(15, world_pose((1.0, 0.0, 0.0)))],
        FixedDirectEstimator([successful_direct(world_pose((0.9, 0.0, 0.0)))]),
        PoseChainDiagnosticConfig(enabled=True),
    )

    result = save_pose_chain_diagnostics(
        tmp_path, rows, PoseChainDiagnosticConfig(enabled=True)
    )

    assert Path(result["csv"]).is_file()
    assert Path(result["json"]).is_file()
    assert Path(result["summary_path"]).is_file()
    assert {Path(path).name for path in result["plots"]} == {
        "translation_difference_vs_frame.png",
        "relative_translation_difference_vs_frame.png",
        "rotation_difference_vs_frame.png",
        "chained_vs_direct_distance.png",
        "chained_vs_direct_xz.png",
        "chained_vs_direct_xy.png",
        "pose_chain_overview.png",
    }
    assert all(Path(path).stat().st_size > 0 for path in result["plots"])


def test_summary_preserves_relative_nonmetric_metadata() -> None:
    rows = analyze_pose_chain(
        [frame(0, world_pose()), frame(15, world_pose((1.0, 0.0, 0.0)))],
        FixedDirectEstimator([successful_direct(world_pose((0.9, 0.0, 0.0)))]),
        PoseChainDiagnosticConfig(),
    )

    summary = build_pose_chain_summary(rows, PoseChainDiagnosticConfig())

    assert summary["diagnostic_only"] is True
    assert summary["coordinate_scale"] == "relative_non_metric"
    assert summary["transform_conventions"] == {
        "pnp_relative": "current_from_reference",
        "stored_pose": "world_from_camera",
        "rotation_comparison": "R_direct^T @ R_chained",
    }
