"""CLI for optional Stage 8 refinement of an existing saved trajectory."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'yaml'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.trajectory_refinement import (
    load_map_trajectory,
    refine_trajectory,
    write_refinement_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine positions in an existing relative trajectory without "
            "rerunning depth inference or mapping"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_directory", type=Path, help="Saved relative-map run")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--mode", choices=("jump_aware", "moving_average"),
        help="Position-only refinement method",
    )
    parser.add_argument(
        "--mad-multiplier", type=float,
        help="Scaled-MAD multiplier for suspicious-step detection",
    )
    parser.add_argument(
        "--weights", type=float, nargs=3, metavar=("PREV", "CENTER", "NEXT"),
        help="Three non-negative smoothing weights that sum to one",
    )
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=False,
        help="Generate raw/refined trajectory and step-magnitude plots",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/trajectory_refinement"),
        help="Root for timestamped refinement outputs",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    section = config.get("trajectory_refinement", {})
    if not isinstance(section, dict):
        raise ValueError("trajectory_refinement config section must be a mapping")
    return section


def _output_directory(root: Path, source: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return root / f"trajectory_refinement_{source.name}_{timestamp}"


def _print_metrics(label: str, metrics: dict[str, object]) -> None:
    print(f"{label} trajectory metrics:")
    for name in (
        "pose_count",
        "mean_step_magnitude",
        "median_step_magnitude",
        "maximum_step_magnitude",
        "standard_deviation_step_magnitude",
        "mean_second_difference",
        "median_second_difference",
        "maximum_second_difference",
        "total_path_length_relative_units",
        "suspicious_jump_count",
    ):
        print(f"  {name}: {metrics[name]}")


def main() -> int:
    args = parse_args()
    try:
        if not args.run_directory.is_dir():
            raise FileNotFoundError(
                f"Map run directory does not exist: {args.run_directory}"
            )
        config = _load_config(args.config)
        mode = args.mode or str(config.get("mode", "jump_aware"))
        mad_multiplier = (
            args.mad_multiplier
            if args.mad_multiplier is not None
            else float(config.get("mad_multiplier", 4.0))
        )
        weights = args.weights or config.get(
            "moving_average_weights", [0.25, 0.50, 0.25]
        )
        trajectory = load_map_trajectory(args.run_directory)
        result = refine_trajectory(
            trajectory,
            mode=mode,
            mad_multiplier=mad_multiplier,
            weights=weights,
        )
        output_directory = _output_directory(
            args.output_dir.resolve(), args.run_directory.resolve()
        )
        paths = write_refinement_outputs(
            args.run_directory,
            result,
            output_directory,
            plots=args.plots,
        )
        refinement = paths["summary_data"]["trajectory_refinement"]
        print("Stage 8 trajectory refinement completed")
        print(f"Method: {refinement['method']}")
        print(f"Units: {trajectory.units} (relative, non-metric)")
        print(
            "Suspicious pose indices: "
            f"{refinement['suspicious_pose_indices']}"
        )
        print(
            "Suspicious frame indices: "
            f"{refinement['suspicious_frame_indices']}"
        )
        print(f"Modified poses: {refinement['modified_pose_count']}")
        _print_metrics("Raw", refinement["raw_metrics"])
        _print_metrics("Refined", refinement["refined_metrics"])
        for name in (
            "trajectory_raw_csv",
            "trajectory_raw_npy",
            "trajectory_refined_csv",
            "trajectory_refined_npy",
            "diagnostics",
            "summary",
        ):
            print(f"{name}: {Path(paths[name]).resolve()}")
        for plot in paths["plots"]:
            print(f"plot: {Path(plot).resolve()}")
        print(
            "The fused map remains based on raw accepted poses; smoothing is "
            "not ground-truth correction."
        )
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Trajectory refinement failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
