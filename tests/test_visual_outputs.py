from pathlib import Path

import numpy as np

from src.visual_outputs import (
    clean_point_cloud_for_display,
    display_cleaning_metadata,
    save_cloud_visual_artifacts,
    save_map_overview_panel,
    save_pose_optimization_comparison,
    save_pose_trajectory_comparison,
    save_rgb_depth_side_by_side,
    save_trajectory_previews,
)


def synthetic_cloud(count: int = 100) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, count)
    points = np.column_stack((x, 0.25 * x, np.linspace(1.0, 4.0, count)))
    colors = np.column_stack((
        np.linspace(0, 255, count),
        np.full(count, 120),
        np.linspace(255, 0, count),
    )).astype(np.uint8)
    return points, colors


def ply_vertex_count(path: Path) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("element vertex "):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError("PLY vertex count was not found")


def test_display_cleaning_removes_nonfinite_points_with_matching_colors() -> None:
    points, colors = synthetic_cloud(10)
    points[2, 0] = np.nan
    points[7, 2] = np.inf

    result = clean_point_cloud_for_display(
        points, colors, z_percentiles=None, center_distance_percentile=100.0
    )

    assert result.nonfinite_removed == 2
    assert result.points.shape == result.colors.shape
    assert np.isfinite(result.points).all()


def test_display_cleaning_is_deterministic_and_removes_extreme_tails() -> None:
    points, colors = synthetic_cloud(200)
    points[-1] = [1000.0, -1000.0, 5000.0]

    first = clean_point_cloud_for_display(points, colors)
    second = clean_point_cloud_for_display(points, colors)

    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.colors, second.colors)
    assert first.removed_count > 0
    assert first.display_count < first.raw_count


def test_optional_display_filters_can_be_disabled() -> None:
    points, colors = synthetic_cloud(20)

    result = clean_point_cloud_for_display(
        points,
        colors,
        z_percentiles=None,
        center_distance_percentile=None,
    )

    np.testing.assert_array_equal(result.points, points)
    np.testing.assert_array_equal(result.colors, colors)
    assert result.removed_count == 0


def test_raw_and_display_ply_outputs_are_separate_and_compatibility_is_preserved(
    tmp_path: Path,
) -> None:
    points, colors = synthetic_cloud()

    artifacts = save_cloud_visual_artifacts(
        tmp_path,
        points,
        colors,
        raw_filenames=("cloud_relative.ply", "cloud_relative_raw.ply"),
        display_filename="cloud_relative_display.ply",
        preview_prefix="preview",
        title="Relative, non-metric",
        save_previews=False,
    )

    compatibility, explicit_raw = artifacts.raw_paths
    assert compatibility.read_bytes() == explicit_raw.read_bytes()
    assert ply_vertex_count(explicit_raw) == points.shape[0]
    assert artifacts.display_path is not None
    assert ply_vertex_count(artifacts.display_path) == artifacts.cleaning.display_count
    assert artifacts.cleaning.display_count < points.shape[0]


def test_front_oblique_and_top_point_cloud_previews_are_written(tmp_path: Path) -> None:
    points, colors = synthetic_cloud(40)

    artifacts = save_cloud_visual_artifacts(
        tmp_path,
        points,
        colors,
        raw_filenames=("raw.ply",),
        display_filename="display.ply",
        preview_prefix="point_cloud_preview",
        title="Relative, non-metric",
        preview_max_points=30,
    )

    assert [path.name for path in artifacts.preview_paths] == [
        "point_cloud_preview_front.png",
        "point_cloud_preview_oblique.png",
        "point_cloud_preview_top.png",
    ]
    assert all(path.stat().st_size > 0 for path in artifacts.preview_paths)


def test_rgb_depth_trajectory_and_overview_pngs_are_written(tmp_path: Path) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    depth = np.zeros_like(image)
    depth[:, :, 0] = np.arange(32, dtype=np.uint8)
    trajectory = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.1, 1.0]])
    points, colors = synthetic_cloud(40)

    comparison = save_rgb_depth_side_by_side(
        tmp_path / "rgb_depth_side_by_side.png", image, depth
    )
    trajectories = save_trajectory_previews(tmp_path, trajectory)
    overview = save_map_overview_panel(
        tmp_path / "map_overview_panel.png",
        image,
        depth,
        trajectory,
        points,
        colors,
        max_points=30,
    )

    assert comparison.stat().st_size > 0
    assert [path.name for path in trajectories] == [
        "trajectory_xz.png", "trajectory_xy.png", "trajectory_3d.png"
    ]
    assert all(path.stat().st_size > 0 for path in trajectories)
    assert overview.stat().st_size > 0


def test_visual_metadata_reports_raw_display_and_removed_counts() -> None:
    points, colors = synthetic_cloud()
    cleaning = clean_point_cloud_for_display(points, colors)

    metadata = display_cleaning_metadata(
        cleaning,
        raw_artifact="raw.ply",
        compatibility_artifact="legacy.ply",
        display_artifact="display.ply",
    )

    assert metadata["presentation_only"] is True
    assert metadata["raw_point_count"] == points.shape[0]
    assert metadata["display_filtered_point_count"] == cleaning.display_count
    assert metadata["removed_for_display_count"] == cleaning.removed_count
    assert metadata["display_point_count"] == cleaning.display_count
    assert metadata["removed_point_count"] == (
        metadata["raw_point_count"] - metadata["display_point_count"]
    )
    assert metadata["coordinate_scale"] == "relative_non_metric"


def test_pose_optimization_map_and_trajectory_artifacts_use_required_names(
    tmp_path: Path,
) -> None:
    points, colors = synthetic_cloud(40)
    optimized = points + [0.01, -0.02, 0.0]
    baseline_trajectory = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    optimized_trajectory = np.array([[0.0, 0.0, 0.0], [0.09, 0.01, 0.0]])

    maps = save_pose_optimization_comparison(
        tmp_path, points, colors, optimized, colors, preview_max_points=30
    )
    trajectories = save_pose_trajectory_comparison(
        tmp_path, baseline_trajectory, optimized_trajectory
    )

    expected = {
        "global_relative_map_pose_baseline.ply",
        "global_relative_map_pose_optimized.ply",
        "global_map_pose_baseline_front.png",
        "global_map_pose_optimized_front.png",
        "global_map_pose_baseline_oblique.png",
        "global_map_pose_optimized_oblique.png",
        "global_map_pose_baseline_top.png",
        "global_map_pose_optimized_top.png",
        "trajectory_pose_baseline.png",
        "trajectory_pose_optimized.png",
    }
    paths = [*maps.values(), *trajectories.values()]
    assert {path.name for path in paths} == expected
    assert all(path.stat().st_size > 0 for path in paths)
