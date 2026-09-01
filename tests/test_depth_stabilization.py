from pathlib import Path

import numpy as np

from src.depth_stabilization import (
    DepthStabilizationConfig,
    robust_z_tail_keep_mask,
    stabilize_aligned_depth,
)
from src.depth_types import CameraDepth, DepthPrediction
from src.visual_outputs import save_depth_stabilization_comparison


def aligned_pair(
    z_values: np.ndarray,
    *,
    scale: float = 2.0,
    shift: float = 0.5,
) -> tuple[DepthPrediction, CameraDepth]:
    z = np.asarray(z_values, dtype=np.float32)
    disparity = np.full(z.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0.0)
    disparity[valid] = scale / z[valid] + shift
    prediction = DepthPrediction(
        disparity,
        "relative",
        False,
        "relative_inverse_depth",
        "synthetic",
    )
    depth = CameraDepth(
        z.copy(),
        "relative",
        False,
        "relative_camera_z_proxy",
        "synthetic",
        "relative_depth_units",
        "reciprocal_affine_disparity_alignment",
        "scale_and_shift",
        disparity_scale=scale,
        disparity_shift=shift,
        denominator_epsilon=1e-3,
    )
    return prediction, depth


def tail_distribution(tail_fraction: float = 0.01) -> np.ndarray:
    count = 100
    tail_count = int(count * tail_fraction)
    values = np.ones(count, dtype=np.float32)
    if tail_count:
        values[-tail_count:] = 1000.0
    return values.reshape(10, 10)


def ply_vertex_count(path: Path) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("element vertex "):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError("PLY vertex count not found")


def test_extreme_positive_z_tail_points_are_removed() -> None:
    prediction, depth = aligned_pair(tail_distribution())

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    assert result.diagnostics.depth_stabilization_accepted is True
    assert result.diagnostics.stabilization_removed_count == 1
    assert np.isnan(result.stabilized_depth.values[-1, -1])


def test_central_distribution_is_unchanged() -> None:
    prediction, depth = aligned_pair(tail_distribution())

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    np.testing.assert_array_equal(
        result.stabilized_depth.values[:-1], depth.values[:-1]
    )
    np.testing.assert_array_equal(
        result.stabilized_depth.values[-1, :-1], depth.values[-1, :-1]
    )


def test_rejected_values_are_invalid_not_clamped_to_a_fake_surface() -> None:
    prediction, depth = aligned_pair(tail_distribution())

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    assert np.isnan(result.stabilized_depth.values[-1, -1])
    assert not np.any(
        result.stabilized_depth.values == result.diagnostics.robust_upper_z_limit
    )


def test_compact_distribution_loses_no_points() -> None:
    prediction, depth = aligned_pair(np.linspace(1.0, 2.0, 100).reshape(10, 10))

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    assert result.diagnostics.depth_stabilization_accepted is True
    assert result.diagnostics.stabilization_removed_count == 0
    np.testing.assert_array_equal(result.stabilized_depth.values, depth.values)


def test_pathological_removal_ratio_triggers_exact_raw_fallback() -> None:
    prediction, depth = aligned_pair(tail_distribution(0.30))

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    diagnostic = result.diagnostics
    assert diagnostic.depth_stabilization_accepted is False
    assert diagnostic.depth_stabilization_reason.endswith("excessive_removal")
    assert diagnostic.stabilization_candidate_removed_ratio == 0.30
    assert diagnostic.stabilization_removed_ratio == 0.0
    assert result.stabilized_depth is depth


def test_disabled_stabilization_preserves_existing_depth_output() -> None:
    prediction, depth = aligned_pair(tail_distribution())

    result = stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=False)
    )

    assert result.diagnostics.depth_stabilization_attempted is False
    assert result.stabilized_depth is depth
    np.testing.assert_array_equal(result.stabilized_depth.values, depth.values)


def test_nonfinite_and_negative_depths_are_safely_excluded() -> None:
    values = np.asarray([1.0, np.nan, np.inf, -2.0, 1000.0])

    keep, upper, _, _ = robust_z_tail_keep_mask(
        values, DepthStabilizationConfig(enabled=True)
    )

    np.testing.assert_array_equal(keep, [True, False, False, False, True])
    assert upper is not None


def test_raw_aligned_depth_and_prediction_arrays_are_not_mutated() -> None:
    prediction, depth = aligned_pair(tail_distribution())
    raw_before = depth.values.copy()
    prediction_before = prediction.values.copy()

    stabilize_aligned_depth(
        prediction, depth, DepthStabilizationConfig(enabled=True)
    )

    np.testing.assert_array_equal(depth.values, raw_before)
    np.testing.assert_array_equal(prediction.values, prediction_before)


def test_relative_nonmetric_semantics_are_preserved() -> None:
    prediction, depth = aligned_pair(tail_distribution())

    result = stabilize_aligned_depth(
        prediction,
        depth,
        DepthStabilizationConfig(enabled=True),
        previous_median_z=0.5,
        previous_p95_z=0.8,
    )

    assert result.stabilized_depth.is_metric is False
    assert result.stabilized_depth.depth_type == "relative"
    assert result.stabilized_depth.coordinate_units == "relative_depth_units"
    assert result.stabilized_depth.representation == "relative_camera_z_proxy"
    assert result.diagnostics.median_z_ratio_to_previous == 2.0
    assert result.diagnostics.raw_alignment_a == 2.0
    assert result.diagnostics.raw_alignment_b == 0.5
    assert result.diagnostics.denominator_median is not None


def test_map_comparison_artifacts_use_unfiltered_input_counts(tmp_path: Path) -> None:
    raw_points = np.column_stack((
        np.arange(6, dtype=np.float64),
        np.zeros(6),
        np.ones(6),
    ))
    stabilized_points = raw_points[:-2]
    raw_colors = np.full((6, 3), 128, dtype=np.uint8)
    stabilized_colors = raw_colors[:-2]

    paths = save_depth_stabilization_comparison(
        tmp_path,
        raw_points,
        raw_colors,
        stabilized_points,
        stabilized_colors,
        preview_max_points=20,
    )

    assert set(paths) == {
        "unstabilized_ply",
        "stabilized_ply",
        "unstabilized_oblique",
        "stabilized_oblique",
        "unstabilized_top",
        "stabilized_top",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert ply_vertex_count(paths["unstabilized_ply"]) == 6
    assert ply_vertex_count(paths["stabilized_ply"]) == 4
