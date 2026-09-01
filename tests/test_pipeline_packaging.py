import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline_packaging import validate_artifact_index
from tools import run_pipeline


def _write_mapping_run(root: Path, source: Path, *, complete: bool = True) -> Path:
    run = root / "relative_map" / "relative_map_fixture"
    run.mkdir(parents=True)
    metadata = {
        "source": str(source.resolve()),
        "source_fps": 30.0,
        "image_width": 640,
        "image_height": 480,
        "camera_intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "device": "cpu",
        "sample_every": 5,
        "max_mapping_frames": 2,
        "point_cloud_stride": 8,
        "scale_mode": "depth-pnp",
        "translation_units": "relative_depth_units",
        "coordinate_units": "relative_depth_units",
        "is_metric": False,
        "keyframe_selection": {"enabled": True, "min_good_matches": 100},
        "motion_quality_thresholds": {"minimum_inliers": 8},
        "pnp_quality_thresholds": {"minimum_inliers": 6},
        "depth_quality_thresholds": {"min_valid_depth_ratio": 0.6},
        "total_candidate_frames": 2,
        "accepted_keyframes": 2,
        "skipped_non_keyframes": 0,
        "rejected_frames": 0,
        "depth_inference_count": 2,
        "trajectory_pose_count": 2,
        "raw_fused_point_count": 20,
        "voxel_downsampled_point_count": 15,
        "final_map_point_count": 14,
        "global_outlier_filter": {"points_rejected": 1},
        "runtime_metrics": {
            "total_pipeline_runtime_seconds": 2.0,
            "depth_inference_seconds": 1.0,
        },
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    frames = [
        {
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "status": "accepted_keyframe",
            "accepted": True,
            "keyframe_reason": "initial_frame",
            "depth_inference_executed": True,
            "translation_magnitude": 0.0,
            "translation_units": "relative_depth_units",
            "total_depth_candidates": 100,
            "denominator_rejection_ratio": 0.0,
            "valid_aligned_depth_ratio": 1.0,
            "aligned_z_median": 1.0,
            "aligned_z_p99": 2.0,
            "cloud_points": 10,
        },
        {
            "frame_index": 5,
            "timestamp_seconds": 0.2,
            "status": "accepted_keyframe",
            "accepted": True,
            "keyframe_reason": "sufficient_feature_displacement",
            "depth_inference_executed": True,
            "good_matches": 100,
            "geometric_inliers": 80,
            "geometric_inlier_ratio": 0.8,
            "scale_estimation_method": "depth_pnp",
            "pnp_inliers": 60,
            "pnp_inlier_ratio": 0.75,
            "reprojection_rmse_pixels": 1.5,
            "reprojection_median_pixels": 1.2,
            "translation_magnitude": 1.0,
            "translation_units": "relative_depth_units",
            "depth_alignment_input_correspondences": 80,
            "depth_alignment_inliers": 72,
            "depth_alignment_inlier_ratio": 0.9,
            "total_depth_candidates": 100,
            "denominator_rejection_ratio": 0.1,
            "valid_aligned_depth_ratio": 0.9,
            "aligned_z_median": 1.0,
            "aligned_z_p99": 3.0,
            "cloud_points": 10,
        },
    ]
    with (run / "frame_stats.jsonl").open("w", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(frame) + "\n")
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with (run / "trajectory_relative.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "x", "y", "z", "accepted"))
        writer.writerow((0, 0.0, 0.0, 0.0, "true"))
        writer.writerow((5, 1.0, 0.0, 0.0, "true"))
    np.save(run / "trajectory_relative.npy", positions, allow_pickle=False)
    if complete:
        (run / "global_relative_map.ply").write_text(
            "ply\nformat ascii 1.0\nend_header\n", encoding="ascii"
        )
        for name in (
            "global_relative_map_raw.ply",
            "global_relative_map_display.ply",
            "global_map_preview_front.png",
            "global_map_preview_oblique.png",
            "global_map_preview_top.png",
            "trajectory_xz.png",
            "trajectory_xy.png",
            "trajectory_3d.png",
            "map_overview_panel.png",
            "pair_alignment_before.ply",
            "pair_alignment_after.ply",
            "pair_alignment_metrics.json",
            "pair_alignment_before_oblique.png",
            "pair_alignment_after_oblique.png",
        ):
            (run / name).write_bytes(b"fixture")
        drift = run / "drift_diagnostics"
        drift.mkdir()
        for name in (
            "drift_diagnostics.csv",
            "drift_diagnostics.json",
            "drift_summary.json",
            "drift_overview.png",
            "depth_alignment_scale_vs_frame.png",
            "depth_alignment_shift_vs_frame.png",
            "aligned_z_median_vs_frame.png",
            "aligned_z_p99_vs_frame.png",
            "translation_magnitude_vs_frame.png",
            "cumulative_distance_vs_frame.png",
            "reprojection_rmse_vs_frame.png",
            "depth_alignment_inlier_ratio_vs_frame.png",
        ):
            (drift / name).write_bytes(b"fixture")
    return run


def _arguments(tmp_path: Path, *, refine: bool) -> tuple[object, Path]:
    config = tmp_path / "default.yaml"
    config.write_text(
        "trajectory_refinement:\n"
        "  enabled: false\n"
        "  mode: jump_aware\n"
        "  mad_multiplier: 4.0\n"
        "  moving_average_weights: [0.25, 0.50, 0.25]\n",
        encoding="utf-8",
    )
    video = tmp_path / "video.mp4"
    arguments = run_pipeline.parse_args([
        str(video), "--fx", "800", "--fy", "800", "--cx", "320",
        "--cy", "240", "--config", str(config), "--device", "cpu",
        "--sample-every", "5", "--max-candidate-frames", "2",
        "--point-cloud-stride", "8", "--output-dir", str(tmp_path / "outputs"),
        "--refine-trajectory" if refine else "--no-refine-trajectory",
    ])
    return arguments, video


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    refine: bool,
    complete_mapping: bool = True,
) -> tuple[dict[str, object], list[list[str]], object]:
    args, video = _arguments(tmp_path, refine=refine)
    monkeypatch.setattr(
        run_pipeline,
        "probe_video",
        lambda _: run_pipeline.VideoInformation(640, 480, 30.0, 60),
    )
    commands: list[list[str]] = []

    def fake_mapping(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        output_root = Path(command[command.index("--output-dir") + 1])
        _write_mapping_run(output_root, video, complete=complete_mapping)
        return SimpleNamespace(returncode=0)

    result = run_pipeline.execute_pipeline(
        args,
        cli_arguments=["run_pipeline.py", str(video)],
        mapping_executor=fake_mapping,
        generate_plots=False,
    )
    return result, commands, args


def test_final_summary_manifest_and_relative_scientific_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, monkeypatch, refine=False)
    summary = result["summary"]
    manifest = result["manifest"]
    assert summary["results"]["accepted_keyframes"] == 2
    assert summary["input"]["camera_intrinsics_source"] == "manual_command_line"
    assert summary["input"]["camera_intrinsics_approximate"] is True
    assert summary["trajectory"]["units"] == "relative_depth_units"
    assert summary["map"]["is_metric"] is False
    assert summary["ground_truth"] == {
        "available": False,
        "ate_computed": False,
        "rpe_computed": False,
    }
    assert manifest["python"]["version"]
    assert set(manifest["package_versions"]) == {
        "numpy", "opencv", "torch", "transformers", "matplotlib"
    }
    assert manifest["resolved_configuration"][
        "camera_intrinsics_approximate"
    ] is True
    assert manifest["resolved_configuration"]["trajectory_refinement"][
        "status"
    ] == "disabled"
    assert manifest["random_seeds"]["explicit_seeds_configured"] is False


def test_artifact_references_exist_and_evaluation_paths_are_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, monkeypatch, refine=False)
    checked = validate_artifact_index(
        result["artifact_index"], result["directory"]
    )
    assert len(checked) == result["validated_artifact_count"]
    evaluation = result["artifact_index"]["evaluation"]
    assert evaluation["frame_metrics_csv"] == "evaluation/frame_metrics.csv"
    assert evaluation["summary_json"] == "evaluation/summary.json"
    assert evaluation["evaluation_report"] == "evaluation/evaluation_report.txt"
    mapping = result["artifact_index"]["mapping"]
    assert mapping["global_relative_map_raw_ply"].endswith(
        "global_relative_map_raw.ply"
    )
    assert mapping["global_relative_map_display_ply"].endswith(
        "global_relative_map_display.ply"
    )
    assert mapping["map_overview_panel"].endswith("map_overview_panel.png")
    assert mapping["pair_alignment_metrics"].endswith(
        "pair_alignment_metrics.json"
    )
    assert mapping["drift_summary"].endswith(
        "drift_diagnostics/drift_summary.json"
    )


def test_missing_required_mapping_artifact_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="global_relative_map.ply"):
        _run(tmp_path, monkeypatch, refine=False, complete_mapping=False)


def test_refinement_disabled_workflow_has_only_authoritative_raw_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, monkeypatch, refine=False)
    refinement = result["summary"]["pipeline"]["trajectory_refinement"]
    trajectory = result["summary"]["trajectory"]
    assert refinement["status"] == "disabled"
    assert trajectory["raw_trajectory_csv"].endswith("trajectory_relative.csv")
    assert trajectory["refined_trajectory_csv"] is None
    assert "trajectory_refinement" not in result["artifact_index"]


def test_refinement_enabled_records_raw_and_refined_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, monkeypatch, refine=True)
    refinement = result["summary"]["pipeline"]["trajectory_refinement"]
    trajectory = result["summary"]["trajectory"]
    assert refinement["status"] == "completed"
    assert trajectory["preserved_raw_copy_csv"].endswith("trajectory_raw.csv")
    assert trajectory["refined_trajectory_csv"].endswith(
        "trajectory_refined.csv"
    )
    assert trajectory["units"] == "relative_depth_units"


def test_optional_refinement_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, video = _arguments(tmp_path, refine=True)
    args.refinement_weights = [0.2, 0.2, 0.2]
    monkeypatch.setattr(
        run_pipeline,
        "probe_video",
        lambda _: run_pipeline.VideoInformation(640, 480, 30.0, 60),
    )

    def fake_mapping(command: list[str], **_: object) -> SimpleNamespace:
        output_root = Path(command[command.index("--output-dir") + 1])
        _write_mapping_run(output_root, video)
        return SimpleNamespace(returncode=0)

    result = run_pipeline.execute_pipeline(
        args,
        cli_arguments=["run_pipeline.py", str(video)],
        mapping_executor=fake_mapping,
        generate_plots=False,
    )
    refinement = result["summary"]["pipeline"]["trajectory_refinement"]
    assert refinement["status"] == "failed"
    assert "sum to 1.0" in refinement["error"]
    assert result["summary"]["results"]["final_map_point_count"] == 14


def test_final_report_contains_required_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, monkeypatch, refine=True)
    report = (Path(result["directory"]) / "FINAL_REPORT.md").read_text(
        encoding="utf-8"
    )
    for statement in (
        "Monocular RGB video",
        "relative and non-metric",
        "manually estimated",
        "No ground-truth trajectory",
        "ATE and RPE were not computed",
        "No loop closure, bundle adjustment, or pose-graph optimization",
        "No absolute trajectory or map accuracy is claimed",
    ):
        assert statement in report


def test_orchestration_runs_mapping_once_and_does_not_duplicate_depth_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, commands, _ = _run(tmp_path, monkeypatch, refine=True)
    assert len(commands) == 1
    assert commands[0][1].endswith("run_relative_map.py")
    assert all("run_depth.py" not in part for part in commands[0])
    assert result["summary"]["results"]["depth_inference_count"] == 2


def test_artifact_validator_rejects_missing_reference(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_artifact_index({"final": {"missing": "missing.json"}}, tmp_path)
