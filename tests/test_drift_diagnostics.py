from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from src.drift_diagnostics import (
    build_drift_summary,
    collect_drift_diagnostics,
    safe_percent_change,
    save_drift_diagnostics,
    trajectory_path_length,
)
from tools.run_relative_map import save_optional_drift_diagnostics


def accepted_frame(
    frame_index: int,
    position: tuple[float, float, float],
    *,
    scale: float,
    shift: float,
    median_z: float,
    p99_z: float,
    translation: tuple[float, float, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_index=frame_index,
        accepted=True,
        disparity_scale=scale,
        disparity_shift=shift,
        z_statistics={
            "min": median_z * 0.5,
            "p1": median_z * 0.6,
            "p5": median_z * 0.7,
            "median": median_z,
            "p95": p99_z * 0.8,
            "p99": p99_z,
            "max": p99_z * 1.2,
        },
        valid_aligned_depth_ratio=0.95,
        denominator_rejection_ratio=0.05,
        relative_translation=translation,
        translation_magnitude=float(np.linalg.norm(translation)),
        camera_position=position,
        selected_relative_rotation_deg=1.0,
        cumulative_rotation_deg=frame_index * 0.1,
        geometric_inlier_ratio=0.8,
        pnp_inlier_ratio=0.7,
        reprojection_rmse_pixels=1.5,
        depth_alignment_inlier_ratio=0.9,
    )


def sample_statistics() -> list[SimpleNamespace]:
    return [
        accepted_frame(
            0, (0.0, 0.0, 0.0), scale=1.0, shift=0.0,
            median_z=1.0, p99_z=2.0, translation=(0.0, 0.0, 0.0),
        ),
        SimpleNamespace(frame_index=5, accepted=False),
        accepted_frame(
            10, (3.0, 0.0, 0.0), scale=1.2, shift=0.1,
            median_z=1.5, p99_z=3.0, translation=(3.0, 0.0, 0.0),
        ),
        accepted_frame(
            20, (3.0, 4.0, 0.0), scale=1.3, shift=0.2,
            median_z=2.0, p99_z=4.0, translation=(0.0, 4.0, 0.0),
        ),
    ]


def test_rows_and_serialized_tables_include_only_accepted_keyframes(tmp_path) -> None:
    rows = collect_drift_diagnostics(sample_statistics())
    result = save_drift_diagnostics(tmp_path, rows, generate_plots=False)

    assert [row.frame_index for row in rows] == [0, 10, 20]
    assert [row.keyframe_sequence_index for row in rows] == [0, 1, 2]
    with result["csv"].open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    json_rows = json.loads(result["json"].read_text(encoding="utf-8"))
    assert len(csv_rows) == len(json_rows) == 3
    assert [int(row["frame_index"]) for row in csv_rows] == [0, 10, 20]


def test_relative_scale_range_and_trend_are_calculated_correctly() -> None:
    summary = build_drift_summary(collect_drift_diagnostics(sample_statistics()))

    scale = summary["depth_alignment_scale_a"]
    assert scale["min"] == 1.0
    assert scale["max"] == 1.3
    assert scale["median"] == 1.2
    assert np.isclose(scale["relative_range"], 0.25)
    assert np.isclose(
        summary["linear_trend_per_keyframe"][
            "depth_alignment_scale_a_slope"
        ],
        0.15,
    )


def test_percent_change_handles_zero_missing_and_nonfinite_safely() -> None:
    assert safe_percent_change(0.0, 2.0) is None
    assert safe_percent_change(None, 2.0) is None
    assert safe_percent_change(np.nan, 2.0) is None
    assert safe_percent_change(2.0, 3.0) == 50.0


def test_cumulative_trajectory_path_length_and_final_position() -> None:
    rows = collect_drift_diagnostics(sample_statistics())
    summary = build_drift_summary(rows)

    assert trajectory_path_length(rows) == 7.0
    assert summary["cumulative_trajectory"]["total_path_length"] == 7.0
    assert summary["cumulative_trajectory"]["final_position"] == [3.0, 4.0, 0.0]


def test_expected_plots_and_overview_are_generated(tmp_path) -> None:
    result = save_drift_diagnostics(
        tmp_path,
        collect_drift_diagnostics(sample_statistics()),
        generate_plots=True,
    )

    names = {path.name for path in result["plots"]}
    assert names == {
        "depth_alignment_scale_vs_frame.png",
        "depth_alignment_shift_vs_frame.png",
        "aligned_z_median_vs_frame.png",
        "aligned_z_p99_vs_frame.png",
        "translation_magnitude_vs_frame.png",
        "cumulative_distance_vs_frame.png",
        "reprojection_rmse_vs_frame.png",
        "depth_alignment_inlier_ratio_vs_frame.png",
        "drift_overview.png",
    }
    assert all(path.stat().st_size > 0 for path in result["plots"])


def test_diagnostic_collection_does_not_mutate_recorded_arrays_or_statistics() -> None:
    statistics = sample_statistics()
    original_position = np.asarray(statistics[0].camera_position).copy()
    original_z = dict(statistics[0].z_statistics)

    collect_drift_diagnostics(statistics)

    np.testing.assert_array_equal(statistics[0].camera_position, original_position)
    assert statistics[0].z_statistics == original_z


def test_disabled_diagnostics_create_no_directory_and_preserve_old_outputs(tmp_path) -> None:
    metadata: dict[str, object] = {}
    result = SimpleNamespace(frame_statistics=sample_statistics())

    written = save_optional_drift_diagnostics(
        tmp_path, result, metadata, enabled=False
    )

    assert written is None
    assert not (tmp_path / "drift_diagnostics").exists()
    assert metadata["drift_diagnostics"] == {
        "enabled": False,
        "diagnostic_only": True,
    }


def test_missing_optional_3d_refinement_fields_are_graceful() -> None:
    rows = collect_drift_diagnostics(sample_statistics())

    assert all(not row.refinement_3d_attempted for row in rows)
    assert all(not row.refinement_3d_accepted for row in rows)
    assert all(row.baseline_3d_residual_median is None for row in rows)
    assert all(row.refined_3d_residual_median is None for row in rows)


def test_heuristic_flags_are_explicit_and_descriptive_only() -> None:
    summary = build_drift_summary(collect_drift_diagnostics(sample_statistics()))
    flags = summary["heuristic_warning_flags"]

    assert flags["heuristic_only"] is True
    assert flags["depth_scale_drift_suspected"] is True
    assert flags["z_range_growth_suspected"] is True
    assert flags["pose_accumulation_growth"] is True
