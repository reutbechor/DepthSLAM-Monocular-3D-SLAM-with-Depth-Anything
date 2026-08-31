import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.trajectory_refinement import (
    TrajectoryData,
    detect_suspicious_jumps,
    load_map_trajectory,
    refine_trajectory,
    trajectory_metrics,
    validate_moving_average_weights,
    write_refinement_outputs,
)


def _trajectory(positions: list[list[float]]) -> TrajectoryData:
    count = len(positions)
    return TrajectoryData(
        frame_indices=tuple(index * 5 for index in range(count)),
        timestamps_seconds=tuple(index * 0.2 for index in range(count)),
        positions=np.asarray(positions, dtype=np.float64),
        units="relative_depth_units",
    )


def _map_run(tmp_path: Path) -> Path:
    run = tmp_path / "relative_map_fixture"
    run.mkdir()
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    (run / "metadata.json").write_text(
        json.dumps({"translation_units": "relative_depth_units"}),
        encoding="utf-8",
    )
    with (run / "frame_stats.jsonl").open("w", encoding="utf-8") as file:
        for index in range(3):
            file.write(json.dumps({
                "frame_index": index * 5,
                "timestamp_seconds": index * 0.2,
                "status": "accepted_keyframe",
                "accepted": True,
            }) + "\n")
    with (run / "trajectory_relative.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "x", "y", "z", "accepted"))
        for index, position in enumerate(positions):
            writer.writerow((index * 5, *position, "true"))
    np.save(run / "trajectory_relative.npy", positions, allow_pickle=False)
    return run


def test_constant_step_trajectory_has_no_suspicious_jumps() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    detection = detect_suspicious_jumps(positions, mad_multiplier=4.0)
    assert detection.median_step_magnitude == 1.0
    assert detection.mad_step_magnitude == 0.0
    assert detection.threshold == 1.0
    assert detection.suspicious_pose_indices == ()


def test_isolated_large_step_is_detected() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]]
    detection = detect_suspicious_jumps(positions)
    assert detection.suspicious_step_indices == (2,)
    assert detection.suspicious_pose_indices == (3,)


def test_jump_aware_changes_only_detected_interior_pose() -> None:
    trajectory = _trajectory(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]]
    )
    result = refine_trajectory(trajectory, mode="jump_aware")
    assert result.modified_pose_indices == (3,)
    assert result.refined_positions[3, 0] == pytest.approx(8.25)
    np.testing.assert_array_equal(
        result.refined_positions[[0, 1, 2, 4]],
        trajectory.positions[[0, 1, 2, 4]],
    )


def test_moving_average_keeps_endpoints_and_pose_order() -> None:
    trajectory = _trajectory([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]])
    result = refine_trajectory(trajectory, mode="moving_average")
    assert result.refined_positions.shape == trajectory.positions.shape
    np.testing.assert_array_equal(result.refined_positions[0], trajectory.positions[0])
    np.testing.assert_array_equal(result.refined_positions[-1], trajectory.positions[-1])
    assert result.raw.frame_indices == trajectory.frame_indices


def test_moving_average_weights_are_validated() -> None:
    assert validate_moving_average_weights((0.25, 0.5, 0.25)) == (
        0.25, 0.5, 0.25
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_moving_average_weights((0.2, 0.5, 0.2))
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_moving_average_weights((-0.1, 0.6, 0.5))


def test_nonfinite_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        detect_suspicious_jumps([[0, 0, 0], [np.nan, 0, 0]])


@pytest.mark.parametrize("positions", [[[1, 2, 3]], [[1, 2, 3], [2, 2, 3]]])
def test_short_trajectories_are_safe(positions: list[list[float]]) -> None:
    trajectory = _trajectory(positions)
    result = refine_trajectory(trajectory, mode="moving_average")
    np.testing.assert_array_equal(result.refined_positions, trajectory.positions)
    assert result.modified_pose_indices == ()
    assert result.raw_metrics.pose_count == len(positions)
    assert result.raw_metrics.total_path_length_relative_units >= 0.0


def test_smoothness_metrics_are_computed_in_relative_units() -> None:
    metrics = trajectory_metrics(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        units="relative_depth_units",
    )
    assert metrics.pose_count == 3
    assert metrics.mean_step_magnitude == 1.0
    assert metrics.median_step_magnitude == 1.0
    assert metrics.maximum_step_magnitude == 1.0
    assert metrics.standard_deviation_step_magnitude == 0.0
    assert metrics.mean_second_difference == 0.0
    assert metrics.total_path_length_relative_units == 2.0
    assert metrics.trajectory_units == "relative_depth_units"


def test_outputs_preserve_source_raw_trajectory_and_units(tmp_path: Path) -> None:
    run = _map_run(tmp_path)
    source_csv_before = (run / "trajectory_relative.csv").read_bytes()
    source_npy_before = (run / "trajectory_relative.npy").read_bytes()
    trajectory = load_map_trajectory(run)
    result = refine_trajectory(trajectory, mode="moving_average")
    paths = write_refinement_outputs(
        run, result, tmp_path / "refinement", plots=False
    )

    assert (run / "trajectory_relative.csv").read_bytes() == source_csv_before
    assert (run / "trajectory_relative.npy").read_bytes() == source_npy_before
    np.testing.assert_array_equal(
        np.load(paths["trajectory_raw_npy"]), trajectory.positions
    )
    with Path(paths["trajectory_raw_csv"]).open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert [int(row["frame_index"]) for row in rows] == [0, 5, 10]
    assert {row["trajectory_units"] for row in rows} == {"relative_depth_units"}
    assert {row["trajectory_type"] for row in rows} == {"raw"}

    summary_text = Path(paths["summary"]).read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["trajectory_artifacts"]["raw"]["trajectory_type"] == "raw"
    assert summary["trajectory_artifacts"]["refined"]["trajectory_type"] == "refined"
    assert "\"ate\"" not in summary_text.lower()
    assert "\"rpe\"" not in summary_text.lower()


def test_refinement_plots_are_generated(tmp_path: Path) -> None:
    trajectory = _trajectory([[0, 0, 0], [1, 0, 1], [5, 0, 2], [6, 0, 3]])
    result = refine_trajectory(trajectory, mode="moving_average")
    paths = write_refinement_outputs(
        tmp_path, result, tmp_path / "plots", plots=True
    )
    assert {path.name for path in paths["plots"]} == {
        "trajectory_raw_vs_refined_xz.png",
        "trajectory_step_magnitude.png",
    }
    assert all(path.stat().st_size > 0 for path in paths["plots"])
