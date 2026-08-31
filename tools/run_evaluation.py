"""Run Stage 7 evaluation from a video or an existing relative-map run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_run_directory, write_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved relative-map run, or run mapping and evaluation "
            "for an input video"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Input video or saved map-run directory")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Principal point y (pixels)")
    parser.add_argument("--sample-every", type=int, metavar="N")
    parser.add_argument("--max-candidate-frames", type=int)
    parser.add_argument("--point-cloud-stride", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument(
        "--keyframes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Stage 6 keyframe selection during a new mapping run",
    )
    parser.add_argument(
        "--intrinsics-approximate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label manually supplied intrinsics as approximate",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate optional matplotlib plots",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/evaluation"),
        help="Root for timestamped evaluation directories",
    )
    parser.add_argument(
        "--refinement-dir", type=Path,
        help="Optional matching Stage 8 refinement output directory",
    )
    return parser.parse_args()


def _timestamped_directory(root: Path, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return root / f"evaluation_{stem}_{timestamp}"


def _append_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def _run_mapping(args: argparse.Namespace, evaluation_directory: Path) -> tuple[Path, float]:
    intrinsics = (args.fx, args.fy, args.cx, args.cy)
    if any(value is None for value in intrinsics):
        raise ValueError("Video evaluation requires --fx, --fy, --cx, and --cy")

    mapping_root = evaluation_directory / "mapping"
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_relative_map.py"),
        str(args.source.resolve()),
        "--config",
        str(args.config),
        "--output-dir",
        str(mapping_root.resolve()),
    ]
    for name, value in (
        ("--fx", args.fx),
        ("--fy", args.fy),
        ("--cx", args.cx),
        ("--cy", args.cy),
        ("--sample-every", args.sample_every),
        ("--max-candidate-frames", args.max_candidate_frames),
        ("--point-cloud-stride", args.point_cloud_stride),
        ("--device", args.device),
        ("--model", args.model),
    ):
        _append_option(command, name, value)
    if args.keyframes is not None:
        command.append("--keyframes" if args.keyframes else "--no-keyframes")

    started = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    wall_seconds = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Mapping subprocess failed with exit code {completed.returncode}"
        )
    runs = sorted((mapping_root / "relative_map").glob("relative_map_*"))
    if len(runs) != 1 or not runs[0].is_dir():
        raise RuntimeError("Expected exactly one generated relative-map run directory")
    return runs[0], wall_seconds


def _print_summary(summary: dict[str, object], paths: dict[str, object]) -> None:
    counts = summary["counts"]
    visual = summary["visual_geometry"]
    pose = summary["pose"]
    depth = summary["depth_alignment"]
    print("Stage 7 evaluation completed")
    print(
        "Candidates / accepted / skipped / rejected: "
        f"{counts['total_candidates']} / {counts['accepted_keyframes']} / "
        f"{counts['skipped_non_keyframes']} / {counts['rejected_frames']}"
    )
    print(
        "Geometric inlier ratio mean / median: "
        f"{visual['geometric_inlier_ratio']['mean']} / "
        f"{visual['geometric_inlier_ratio']['median']}"
    )
    print(
        "PnP inlier ratio mean / median: "
        f"{pose['pnp_inlier_ratio']['mean']} / {pose['pnp_inlier_ratio']['median']}"
    )
    print(
        "Reprojection RMSE mean / median (pixels): "
        f"{pose['reprojection_rmse_pixels']['mean']} / "
        f"{pose['reprojection_rmse_pixels']['median']}"
    )
    print(
        "Depth alignment inlier ratio mean / median: "
        f"{depth['depth_alignment_inlier_ratio']['mean']} / "
        f"{depth['depth_alignment_inlier_ratio']['median']}"
    )
    print(
        "Maximum denominator rejection ratio: "
        f"{depth['denominator_rejection_ratio']['maximum']}"
    )
    print(f"Final map points: {summary['map']['final_map_points']}")
    print(f"Total runtime seconds: {summary['runtime']['total_pipeline_runtime_seconds']}")
    print(f"Rejection reasons: {summary['rejection_reason_counts']}")
    for label in ("frame_metrics", "summary", "trajectory", "report"):
        print(f"{label}: {Path(paths[label]).resolve()}")
    for plot in paths["plots"]:
        print(f"plot: {Path(plot).resolve()}")


def main() -> int:
    args = parse_args()
    try:
        if not args.source.exists():
            raise FileNotFoundError(f"Evaluation source does not exist: {args.source}")
        output_root = args.output_dir.resolve()
        evaluation_directory = _timestamped_directory(output_root, args.source.stem)
        if args.source.is_dir():
            map_run = args.source
            runtime_override = None
            intrinsics_source = "saved_run_metadata"
            intrinsics_approximate = None
        else:
            evaluation_directory.mkdir(parents=True, exist_ok=False)
            map_run, runtime_override = _run_mapping(args, evaluation_directory)
            intrinsics_source = "manual_command_line"
            intrinsics_approximate = args.intrinsics_approximate

        result = evaluate_run_directory(
            map_run,
            total_runtime_seconds=runtime_override,
            intrinsics_source=intrinsics_source,
            intrinsics_approximate=intrinsics_approximate,
            refinement_directory=args.refinement_dir,
        )
        paths = write_evaluation_outputs(
            result, evaluation_directory, plots=args.plots
        )
        _print_summary(result.summary, paths)
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
