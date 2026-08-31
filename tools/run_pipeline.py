"""Final one-pass orchestration and packaging for the DepthSLAM project."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.evaluation import evaluate_run_directory, write_evaluation_outputs
from src.pipeline_packaging import (
    PipelineTimings,
    VideoInformation,
    build_artifact_index,
    build_final_summary,
    build_run_manifest,
    final_report_markdown,
    read_json_object,
    refinement_record,
    validate_artifact_index,
    write_json,
)
from src.trajectory_refinement import (
    load_map_trajectory,
    refine_trajectory,
    write_refinement_outputs,
)


MappingExecutor = Callable[..., Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mapping once, evaluate it, optionally refine its trajectory, "
            "and package the run"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Monocular RGB input video")
    parser.add_argument("--fx", type=float, required=True, help="Manual focal length x")
    parser.add_argument("--fy", type=float, required=True, help="Manual focal length y")
    parser.add_argument("--cx", type=float, required=True, help="Manual principal point x")
    parser.add_argument("--cy", type=float, required=True, help="Manual principal point y")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--sample-every", type=int, metavar="N")
    parser.add_argument("--max-candidate-frames", type=int)
    parser.add_argument("--point-cloud-stride", type=int)
    parser.add_argument(
        "--keyframes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Stage 6 visual keyframe selection",
    )
    parser.add_argument(
        "--refine-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optional Stage 8 position-only refinement",
    )
    parser.add_argument(
        "--refinement-mode",
        choices=("jump_aware", "moving_average"),
        help="Stage 8 refinement mode",
    )
    parser.add_argument("--mad-multiplier", type=float)
    parser.add_argument(
        "--refinement-weights",
        type=float,
        nargs=3,
        metavar=("PREV", "CENTER", "NEXT"),
    )
    parser.add_argument(
        "--require-refinement",
        action="store_true",
        help="Treat optional trajectory-refinement failure as a pipeline failure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_pipeline"),
        help="Root for timestamped final runs",
    )
    return parser.parse_args(argv)


def probe_video(path: Path) -> VideoInformation:
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Video cannot be opened: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0 or not math.isfinite(fps):
        raise RuntimeError(f"Video metadata is invalid: {path}")
    return VideoInformation(width, height, fps, frame_count)


def _load_configuration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def _final_directory(root: Path, video: Path, timestamp: str) -> Path:
    return root / f"final_{video.stem}_{timestamp}"


def _append_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def build_mapping_command(
    args: argparse.Namespace, final_directory: Path
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_relative_map.py"),
        str(args.video.resolve()),
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str((final_directory / "mapping").resolve()),
        "--fx",
        str(args.fx),
        "--fy",
        str(args.fy),
        "--cx",
        str(args.cx),
        "--cy",
        str(args.cy),
        "--keyframes" if args.keyframes else "--no-keyframes",
    ]
    for name, value in (
        ("--device", args.device),
        ("--model", args.model),
        ("--sample-every", args.sample_every),
        ("--max-candidate-frames", args.max_candidate_frames),
        ("--point-cloud-stride", args.point_cloud_stride),
    ):
        _append_option(command, name, value)
    return command


def _mapping_run_directory(final_directory: Path) -> Path:
    parent = final_directory / "mapping" / "relative_map"
    runs = sorted(path for path in parent.glob("relative_map_*") if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(
            f"Expected exactly one mapping run in {parent}; found {len(runs)}"
        )
    return runs[0]


def _validate_mapping_artifacts(run_directory: Path) -> None:
    for name in (
        "metadata.json",
        "frame_stats.jsonl",
        "trajectory_relative.csv",
        "trajectory_relative.npy",
        "global_relative_map.ply",
    ):
        path = run_directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Required mapping artifact is missing: {path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def execute_pipeline(
    args: argparse.Namespace,
    *,
    cli_arguments: list[str],
    mapping_executor: MappingExecutor = subprocess.run,
    generate_plots: bool = True,
) -> dict[str, Any]:
    pipeline_started = perf_counter()
    numeric_intrinsics = (args.fx, args.fy, args.cx, args.cy)
    if not all(math.isfinite(value) for value in numeric_intrinsics):
        raise ValueError("Camera intrinsics must be finite")
    if args.fx <= 0.0 or args.fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    config = _load_configuration(args.config)
    video = probe_video(args.video)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp_iso = datetime.now().astimezone().isoformat()
    final_directory = _final_directory(
        args.output_dir.resolve(), args.video, timestamp
    )
    final_directory.mkdir(parents=True, exist_ok=False)

    print(
        "WARNING: Camera intrinsics are manually supplied and uncalibrated.",
        flush=True,
    )
    print("WARNING: 3D geometry may be warped.", flush=True)
    print("camera_intrinsics_source = manual_command_line", flush=True)
    print("camera_intrinsics_approximate = true", flush=True)

    mapping_command = build_mapping_command(args, final_directory)
    mapping_started = perf_counter()
    completed = mapping_executor(mapping_command, cwd=ROOT, check=False)
    mapping_seconds = perf_counter() - mapping_started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Mapping failed with subprocess exit code {completed.returncode}"
        )
    mapping_run = _mapping_run_directory(final_directory)
    _validate_mapping_artifacts(mapping_run)
    mapping_metadata = read_json_object(mapping_run / "metadata.json")

    evaluation_started = perf_counter()
    evaluation = evaluate_run_directory(
        mapping_run,
        total_runtime_seconds=mapping_seconds,
        intrinsics_source="manual_command_line",
        intrinsics_approximate=True,
    )
    evaluation_paths = write_evaluation_outputs(
        evaluation,
        final_directory / "evaluation",
        plots=generate_plots,
    )
    evaluation_seconds = perf_counter() - evaluation_started

    refinement_paths: dict[str, Any] | None = None
    refinement_seconds: float | None = None
    if args.refine_trajectory:
        refinement_started = perf_counter()
        try:
            refinement_config = config.get("trajectory_refinement", {})
            if not isinstance(refinement_config, dict):
                raise ValueError(
                    "trajectory_refinement config section must be a mapping"
                )
            mode = args.refinement_mode or str(
                refinement_config.get("mode", "jump_aware")
            )
            mad_multiplier = (
                args.mad_multiplier
                if args.mad_multiplier is not None
                else float(refinement_config.get("mad_multiplier", 4.0))
            )
            weights = args.refinement_weights or refinement_config.get(
                "moving_average_weights", [0.25, 0.50, 0.25]
            )
            trajectory = load_map_trajectory(mapping_run)
            refined = refine_trajectory(
                trajectory,
                mode=mode,
                mad_multiplier=mad_multiplier,
                weights=weights,
            )
            refinement_paths = write_refinement_outputs(
                mapping_run,
                refined,
                final_directory / "trajectory_refinement",
                plots=generate_plots,
            )
            refinement = refinement_record(
                enabled=True,
                status="completed",
                summary=refinement_paths["summary_data"],
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            refinement = refinement_record(
                enabled=True, status="failed", error=str(exc)
            )
            print(f"WARNING: Optional trajectory refinement failed: {exc}")
            if args.require_refinement:
                raise RuntimeError(
                    f"Required trajectory refinement failed: {exc}"
                ) from exc
        finally:
            refinement_seconds = perf_counter() - refinement_started
    else:
        refinement = refinement_record(enabled=False, status="disabled")

    packaging_started = perf_counter()
    artifact_index = build_artifact_index(
        final_directory, mapping_run, evaluation_paths, refinement_paths
    )
    provisional_timings = PipelineTimings(
        mapping_seconds=mapping_seconds,
        evaluation_seconds=evaluation_seconds,
        trajectory_refinement_seconds=refinement_seconds,
        packaging_seconds=0.0,
        end_to_end_seconds=perf_counter() - pipeline_started,
    )
    final_summary = build_final_summary(
        source_video=args.video,
        video=video,
        mapping_metadata=mapping_metadata,
        evaluation_summary=evaluation.summary,
        refinement=refinement,
        artifacts=artifact_index,
        timings=provisional_timings,
    )
    manifest = build_run_manifest(
        timestamp=timestamp_iso,
        cli_arguments=cli_arguments,
        parsed_arguments=_jsonable(vars(args)),
        mapping_command=mapping_command,
        config_path=args.config,
        mapping_metadata=mapping_metadata,
        video=video,
        refinement=refinement,
    )
    write_json(final_directory / "final_summary.json", final_summary)
    write_json(final_directory / "run_manifest.json", manifest)
    write_json(final_directory / "artifacts.json", artifact_index)
    (final_directory / "FINAL_REPORT.md").write_text(
        final_report_markdown(final_summary, artifact_index), encoding="utf-8"
    )
    validate_artifact_index(artifact_index, final_directory)

    final_timings = PipelineTimings(
        mapping_seconds=mapping_seconds,
        evaluation_seconds=evaluation_seconds,
        trajectory_refinement_seconds=refinement_seconds,
        packaging_seconds=perf_counter() - packaging_started,
        end_to_end_seconds=perf_counter() - pipeline_started,
    )
    final_summary = build_final_summary(
        source_video=args.video,
        video=video,
        mapping_metadata=mapping_metadata,
        evaluation_summary=evaluation.summary,
        refinement=refinement,
        artifacts=artifact_index,
        timings=final_timings,
    )
    write_json(final_directory / "final_summary.json", final_summary)
    (final_directory / "FINAL_REPORT.md").write_text(
        final_report_markdown(final_summary, artifact_index), encoding="utf-8"
    )
    checked = validate_artifact_index(artifact_index, final_directory)
    return {
        "directory": final_directory,
        "mapping_run": mapping_run,
        "summary": final_summary,
        "manifest": manifest,
        "artifact_index": artifact_index,
        "validated_artifact_count": len(checked),
    }


def _print_final_result(result: dict[str, Any]) -> None:
    summary = result["summary"]
    results = summary["results"]
    refinement = summary["pipeline"]["trajectory_refinement"]
    runtime = summary["runtime"]
    print("DepthSLAM final pipeline completed")
    print(
        "Candidates / accepted / skipped / rejected: "
        f"{summary['pipeline']['candidate_frame_count']} / "
        f"{results['accepted_keyframes']} / {results['skipped_frames']} / "
        f"{results['rejected_frames']}"
    )
    print(f"Trajectory poses: {results['trajectory_pose_count']}")
    print(f"Final map points: {results['final_map_point_count']}")
    print(
        "PnP inlier ratio mean / median: "
        f"{results['pnp_inlier_ratio']['mean']} / "
        f"{results['pnp_inlier_ratio']['median']}"
    )
    print(
        "Reprojection RMSE mean / median: "
        f"{results['reprojection_rmse_pixels']['mean']} / "
        f"{results['reprojection_rmse_pixels']['median']} pixels"
    )
    print(
        "Depth alignment ratio mean / median: "
        f"{results['depth_alignment_inlier_ratio']['mean']} / "
        f"{results['depth_alignment_inlier_ratio']['median']}"
    )
    print(f"Rejection reasons: {results['rejection_reason_counts']}")
    print(
        "Trajectory refinement: "
        f"status={refinement['status']}, method={refinement['method']}, "
        f"jumps={refinement['suspicious_jump_count']}, "
        f"modified={refinement['modified_pose_count']}"
    )
    print(
        "Runtime seconds (mapping / evaluation / refinement / total): "
        f"{runtime['mapping_seconds']} / {runtime['evaluation_seconds']} / "
        f"{runtime['trajectory_refinement_seconds']} / "
        f"{runtime['end_to_end_seconds']}"
    )
    print(f"Validated artifacts: {result['validated_artifact_count']}")
    print(f"Final output: {Path(result['directory']).resolve()}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else ["run_pipeline.py", *argv])
    args = parse_args(argv)
    try:
        result = execute_pipeline(args, cli_arguments=arguments)
        _print_final_result(result)
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Final pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
