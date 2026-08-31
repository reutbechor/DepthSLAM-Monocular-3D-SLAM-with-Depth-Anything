"""CLI for a short-video incremental map in arbitrary relative units."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: py -m pip install -r requirements.txt"
    ) from exc

from src.depth_estimator import DEFAULT_MODEL, DepthEstimator
from src.map_builder import MappingFrame, RelativeMapBuilder, RelativeMapResult
from src.ply_io import write_ascii_ply
from src.video_loader import VideoLoader
from tools.run_depth_geometry import build_motion_estimator, build_tracker
from tools.run_motion import camera_matrix_from_args, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a short multi-frame map in arbitrary relative units",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--fx", type=float, help="Calibrated focal length in x (pixels)")
    parser.add_argument("--fy", type=float, help="Calibrated focal length in y (pixels)")
    parser.add_argument("--cx", type=float, help="Calibrated principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Calibrated principal point y (pixels)")
    parser.add_argument("--sample-every", type=int, metavar="N")
    parser.add_argument("--max-mapping-frames", type=int)
    parser.add_argument("--translation-step", type=float)
    parser.add_argument("--point-cloud-stride", type=int)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--model", help="Hugging Face Depth Anything model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--minimum-matches", type=int)
    parser.add_argument("--ransac-probability", type=float)
    parser.add_argument("--ransac-threshold", type=float, metavar="PIXELS")
    parser.add_argument("--minimum-inliers", type=int)
    parser.add_argument("--minimum-inlier-ratio", type=float)
    parser.add_argument("--output-dir", type=Path, help="Root for generated outputs")
    return parser.parse_args()


def setting(cli_value: Any, config: dict[str, Any], name: str, fallback: Any) -> Any:
    return cli_value if cli_value is not None else config.get(name, fallback)


def select_mapping_frames(
    video: Path, sample_every: int, maximum: int
) -> tuple[list[MappingFrame], float]:
    frames: list[MappingFrame] = []
    source_fps = 0.0
    for frame in VideoLoader(video, sample_every):
        source_fps = frame.source_fps
        frames.append(MappingFrame(
            image=frame.image,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
        ))
        if len(frames) >= maximum:
            break
    if not frames:
        raise RuntimeError(f"No mapping frames could be read from video: {video}")
    return frames, source_fps


def create_output_directory(root: Path, video: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / f"relative_map_{video.stem}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_trajectory(directory: Path, result: RelativeMapResult) -> None:
    np.save(directory / "trajectory_relative.npy", result.trajectory_positions)
    with (directory / "trajectory_relative.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("frame_index", "x", "y", "z", "accepted"))
        for frame_index, position in zip(
            result.trajectory_frame_indices, result.trajectory_positions
        ):
            writer.writerow((
                int(frame_index), float(position[0]), float(position[1]),
                float(position[2]), "true"
            ))


def save_frame_statistics(directory: Path, result: RelativeMapResult) -> None:
    with (directory / "frame_stats.jsonl").open("w", encoding="utf-8") as file:
        for statistics in result.frame_statistics:
            file.write(json.dumps(statistics.to_dict()) + "\n")


def save_outputs(
    directory: Path, result: RelativeMapResult, metadata: dict[str, Any]
) -> Path:
    np.save(directory / "global_points_relative.npy", result.fused_cloud.points)
    np.save(directory / "global_colors_rgb.npy", result.fused_cloud.colors)
    ply_path = write_ascii_ply(
        directory / "global_relative_map.ply",
        result.fused_cloud.points,
        result.fused_cloud.colors,
    )
    save_trajectory(directory, result)
    save_frame_statistics(directory, result)
    with (directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return ply_path


def main() -> int:
    args = parse_args()
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {args.video}")
        config = load_config(args.config)
        model_config = config.get("model", {})
        motion_config = config.get("motion", {})
        map_config = config.get("map", {})
        output_config = config.get("output", {})
        if not all(isinstance(section, dict) for section in (
            model_config, motion_config, map_config, output_config
        )):
            raise ValueError("model, motion, map, and output config sections must be mappings")
        camera_matrix = camera_matrix_from_args(args, config)

        sample_every = int(setting(args.sample_every, map_config, "sample_every", 10))
        maximum = int(setting(
            args.max_mapping_frames, map_config, "max_mapping_frames", 5
        ))
        translation_step = float(setting(
            args.translation_step, map_config, "translation_step", 1.0
        ))
        cloud_stride = int(setting(
            args.point_cloud_stride, map_config, "point_cloud_stride", 6
        ))
        voxel_size = float(setting(args.voxel_size, map_config, "voxel_size", 0.05))
        if sample_every < 1 or maximum < 1 or cloud_stride < 1:
            raise ValueError("sampling, maximum frame count, and cloud stride must be positive")

        selected_frames, source_fps = select_mapping_frames(
            args.video, sample_every, maximum
        )
        print(
            f"Selected {len(selected_frames)} mapping frames: "
            + ", ".join(str(frame.frame_index) for frame in selected_frames)
        )

        model_name = args.model or model_config.get("name", DEFAULT_MODEL)
        device = args.device or model_config.get("device", "auto")
        print(f"Loading {model_name} on {device}...")
        depth_estimator = DepthEstimator(model_name=model_name, device=device)
        builder = RelativeMapBuilder(
            depth_estimator=depth_estimator,
            feature_tracker=build_tracker(args, motion_config),
            motion_estimator=build_motion_estimator(args, motion_config),
            camera_matrix=camera_matrix,
            translation_step=translation_step,
            point_cloud_stride=cloud_stride,
            voxel_size=voxel_size,
        )
        result = builder.build(selected_frames)

        output_root = args.output_dir or Path(output_config.get("directory", "outputs"))
        run_directory = create_output_directory(output_root / "relative_map", args.video)
        metadata = {
            "map_type": "relative_multi_frame",
            "is_metric": False,
            "depth_type": "relative",
            "translation_scale": "arbitrary_relative_step",
            "translation_step": translation_step,
            "coordinate_units": "relative_map_units",
            "voxel_units": "relative_map_units",
            "source": str(args.video.resolve()),
            "source_fps": source_fps,
            "image_width": result.image_width,
            "image_height": result.image_height,
            "camera_intrinsics": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
            "model": depth_estimator.model_name,
            "device": depth_estimator.device,
            "sample_every": sample_every,
            "max_mapping_frames": maximum,
            "point_cloud_stride": cloud_stride,
            "voxel_size": voxel_size,
            "sampled_frames": result.sampled_frame_count,
            "accepted_frames": result.accepted_frame_count,
            "rejected_frames": result.rejected_frame_count,
            "trajectory_pose_count": result.trajectory_positions.shape[0],
            "initial_map_point_count": result.raw_fused_point_count,
            "final_map_point_count": result.fused_cloud.output_point_count,
            "trajectory_format": "accepted frames only; rejected frames are in frame_stats.jsonl",
            "note": "Map coordinates are arbitrary relative units, not metres.",
        }
        ply_path = save_outputs(run_directory, result, metadata)

        print("Relative multi-frame map completed")
        print(f"Sampled frames: {result.sampled_frame_count}")
        print(f"Accepted mapping frames: {result.accepted_frame_count}")
        print(f"Rejected frames: {result.rejected_frame_count}")
        print(f"Raw fused points: {result.raw_fused_point_count}")
        print(f"Downsampled map points: {result.fused_cloud.output_point_count}")
        print(f"Relative trajectory poses: {result.trajectory_positions.shape[0]}")
        print("WARNING: Map coordinates use arbitrary relative units. The map is NOT metric.")
        print(f"Final map: {ply_path.resolve()}")
        print(f"Outputs: {run_directory.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Relative map failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
