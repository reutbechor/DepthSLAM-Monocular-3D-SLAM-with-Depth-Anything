import csv
import json
from pathlib import Path

import pytest

from src.evaluation import (
    FRAME_METRIC_COLUMNS,
    evaluate_run_directory,
    write_evaluation_outputs,
)


def _frame(
    frame_index: int,
    status: str,
    *,
    timestamp: float,
    rejection_reason: str | None = None,
    skip_reason: str | None = None,
    **values: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp,
        "status": status,
        "accepted": status == "accepted_keyframe",
        "keyframe_reason": "motion_threshold" if status == "accepted_keyframe" else None,
        "skip_reason": skip_reason,
        "rejection_reason": rejection_reason,
        "depth_inference_executed": False,
        "good_matches": 0,
        "geometric_inliers": 0,
        "geometric_inlier_ratio": 0.0,
        "pnp_inliers": 0,
        "pnp_inlier_ratio": 0.0,
        "reprojection_rmse_pixels": None,
        "reprojection_median_pixels": None,
        "translation_magnitude": None,
        "translation_units": None,
        "depth_alignment_input_correspondences": 0,
        "depth_alignment_inliers": 0,
        "depth_alignment_inlier_ratio": 0.0,
        "denominator_rejection_ratio": 0.0,
        "valid_aligned_depth_ratio": 0.0,
        "aligned_z_median": None,
        "aligned_z_p99": None,
        "relative_z_p99_over_median": None,
        "cloud_points": 0,
    }
    row.update(values)
    return row


@pytest.fixture
def artifact_run(tmp_path: Path) -> Path:
    run = tmp_path / "relative_map_fixture"
    run.mkdir()
    metadata = {
        "source": "fixture.mp4",
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "device": "cpu",
        "scale_mode": "depth-pnp",
        "translation_units": "relative_depth_units",
        "coordinate_units": "relative_depth_units",
        "sample_every": 5,
        "max_mapping_frames": 6,
        "point_cloud_stride": 8,
        "camera_intrinsics": {"fx": 800, "fy": 800, "cx": 636, "cy": 321},
        "keyframe_selection": {"enabled": True},
        "depth_quality_thresholds": {"min_valid_depth_ratio": 0.6},
        "raw_fused_point_count": 180,
        "voxel_downsampled_point_count": 140,
        "final_map_point_count": 130,
        "global_outlier_filter": {"points_rejected": 10},
        "runtime_metrics": {
            "total_pipeline_runtime_seconds": 12.0,
            "feature_motion_seconds": 1.0,
            "depth_inference_seconds": 6.0,
            "pnp_depth_alignment_seconds": 2.0,
            "point_cloud_fusion_seconds": 1.5,
        },
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = [
        _frame(
            0, "accepted_keyframe", timestamp=0.0,
            keyframe_reason="initial_frame", depth_inference_executed=True,
            translation_magnitude=0.0, translation_units="relative_depth_units",
            total_depth_candidates=100,
            denominator_rejection_ratio=0.0, valid_aligned_depth_ratio=1.0,
            aligned_z_median=1.0, aligned_z_p99=5.0,
            relative_z_p99_over_median=5.0, cloud_points=100,
        ),
        _frame(
            5, "skipped_non_keyframe", timestamp=0.2,
            skip_reason="insufficient_motion", good_matches=100,
            geometric_inliers=60, geometric_inlier_ratio=0.6,
            median_feature_displacement_px=3.0, rotation_deg=0.2,
        ),
        _frame(
            10, "rejected", timestamp=0.4,
            rejection_reason="geometric_filtering", good_matches=80,
            geometric_inliers=16, geometric_inlier_ratio=0.2,
        ),
        _frame(
            15, "rejected", timestamp=0.6, rejection_reason="pnp",
            good_matches=120, geometric_inliers=84, geometric_inlier_ratio=0.7,
            scale_estimation_method="depth_pnp", pnp_inliers=4,
            pnp_inlier_ratio=0.2, reprojection_rmse_pixels=99.0,
            translation_magnitude=9.0, translation_units="relative_depth_units",
        ),
        _frame(
            20, "accepted_keyframe", timestamp=0.8,
            good_matches=140, geometric_inliers=112, geometric_inlier_ratio=0.8,
            scale_estimation_method="depth_pnp", pnp_inliers=75,
            pnp_inlier_ratio=0.75, reprojection_rmse_pixels=1.5,
            reprojection_median_pixels=1.2, translation_magnitude=2.0,
            translation_units="relative_depth_units", depth_inference_executed=True,
            total_depth_candidates=100,
            depth_alignment_input_correspondences=100,
            depth_alignment_inliers=90, depth_alignment_inlier_ratio=0.9,
            denominator_rejection_ratio=0.1, valid_aligned_depth_ratio=0.95,
            aligned_z_median=1.2, aligned_z_p99=6.0,
            relative_z_p99_over_median=5.0, cloud_points=80,
        ),
        _frame(
            25, "rejected", timestamp=1.0, rejection_reason="depth_z_distribution",
            good_matches=130, geometric_inliers=65, geometric_inlier_ratio=0.5,
            scale_estimation_method="depth_pnp", pnp_inliers=70,
            pnp_inlier_ratio=0.7, reprojection_rmse_pixels=2.5,
            reprojection_median_pixels=2.0, translation_magnitude=3.0,
            translation_units="relative_depth_units", depth_inference_executed=True,
            total_depth_candidates=100,
            depth_alignment_input_correspondences=100,
            depth_alignment_inliers=85, depth_alignment_inlier_ratio=0.85,
            denominator_rejection_ratio=0.4, valid_aligned_depth_ratio=0.6,
            aligned_z_median=1.0, aligned_z_p99=80.0,
            relative_z_p99_over_median=80.0,
        ),
    ]
    with (run / "frame_stats.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    with (run / "trajectory_relative.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "x", "y", "z"))
        writer.writerow((0, 0, 0, 0))
        writer.writerow((20, 1, 2, 3))
    return run


def test_run_counts_statuses_and_rejection_reasons(artifact_run: Path) -> None:
    summary = evaluate_run_directory(artifact_run).summary
    assert summary["counts"] == {
        "total_candidates": 6,
        "accepted_keyframes": 2,
        "skipped_non_keyframes": 1,
        "rejected_frames": 3,
        "depth_inference_count": 3,
        "trajectory_pose_count": 2,
    }
    assert summary["rejection_reason_counts"] == {
        "depth_z_distribution": 1,
        "geometric_filtering": 1,
        "pnp": 1,
    }


def test_metric_summaries_ignore_unavailable_stage_defaults(artifact_run: Path) -> None:
    summary = evaluate_run_directory(artifact_run).summary
    geometric = summary["visual_geometry"]["geometric_inlier_ratio"]
    assert geometric["count"] == 5
    assert geometric["mean"] == pytest.approx(0.56)
    assert geometric["median"] == pytest.approx(0.6)
    rmse = summary["pose"]["reprojection_rmse_pixels"]
    assert rmse["count"] == 2
    assert rmse["mean"] == pytest.approx(2.0)
    assert rmse["median"] == pytest.approx(2.0)
    assert summary["depth_alignment"]["denominator_rejection_ratio"]["maximum"] == 0.4


def test_trajectory_alignment_and_relative_units(artifact_run: Path) -> None:
    result = evaluate_run_directory(artifact_run)
    assert [row["frame_index"] for row in result.trajectory] == [0, 20]
    assert [row["timestamp_seconds"] for row in result.trajectory] == [0.0, 0.8]
    assert result.trajectory[1]["x"] == 1.0
    assert {row["trajectory_units"] for row in result.trajectory} == {
        "relative_depth_units"
    }
    assert result.summary["pose"]["translation_units"] == "relative_depth_units"


def test_output_schema_json_and_no_ground_truth_metrics(
    artifact_run: Path, tmp_path: Path
) -> None:
    result = evaluate_run_directory(artifact_run)
    paths = write_evaluation_outputs(result, tmp_path / "evaluation", plots=False)
    with Path(paths["frame_metrics"]).open(encoding="utf-8", newline="") as file:
        assert tuple(next(csv.reader(file))) == FRAME_METRIC_COLUMNS
    summary_text = Path(paths["summary"]).read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["ground_truth"] == {
        "available": False,
        "ate_computed": False,
        "rpe_computed": False,
        "note": (
            "No external ground-truth trajectory was supplied; absolute and "
            "relative trajectory errors are not computed."
        ),
    }
    assert "NaN" not in summary_text
    assert "ate" not in summary and "rpe" not in summary


def test_sparse_plot_generation_skips_missing_line_metrics(
    artifact_run: Path, tmp_path: Path
) -> None:
    result = evaluate_run_directory(artifact_run)
    result.frame_metrics = [result.frame_metrics[0]]
    result.trajectory = [result.trajectory[0]]
    paths = write_evaluation_outputs(result, tmp_path / "plots", plots=True)
    names = {path.name for path in paths["plots"]}
    assert names == {
        "denominator_rejection_ratio.png",
        "frame_status.png",
        "trajectory_xz.png",
    }
    assert all(path.stat().st_size > 0 for path in paths["plots"])


def test_runtime_fields_are_non_negative_and_negative_values_fail(
    artifact_run: Path
) -> None:
    runtime = evaluate_run_directory(artifact_run).summary["runtime"]
    numeric = [
        value for key, value in runtime.items()
        if key != "timing_note" and value is not None
    ]
    assert all(value >= 0 for value in numeric)

    metadata_path = artifact_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["runtime_metrics"]["depth_inference_seconds"] = -1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_run_directory(artifact_run)
