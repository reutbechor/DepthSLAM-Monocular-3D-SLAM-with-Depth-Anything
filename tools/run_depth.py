"""CLI for image and sampled-video Depth Anything V2 inference."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run: python -m pip install -r requirements.txt"
    ) from exc

from src.depth_estimator import DEFAULT_MODEL, DepthEstimator
from src.video_loader import VideoLoader
from src.visualization import colorize_depth, make_side_by_side

IMAGES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEOS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything V2 relative-depth inference")
    parser.add_argument("source", type=Path)
    parser.add_argument("--input-type", choices=("auto", "image", "video"), default="auto")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--sample-every", type=int, metavar="N")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def detect_type(source: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if source.suffix.lower() in IMAGES:
        return "image"
    if source.suffix.lower() in VIDEOS:
        return "video"
    raise ValueError("Unknown extension; pass --input-type image or --input-type video")


def create_run_dir(root: Path, source: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"{source.stem}_{stamp}"
    for name in ("rgb", "depth_raw", "depth_vis", "side_by_side"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    return run_dir


def metadata(source: Path, index: int, seconds: float, image: np.ndarray,
             estimator: DepthEstimator) -> dict[str, Any]:
    height, width = image.shape[:2]
    return {
        "source_path": str(source.resolve()), "frame_index": index,
        "timestamp_seconds": round(seconds, 6), "frame_width": width,
        "frame_height": height, "model": estimator.model_name,
        "device": estimator.device,
        "depth_type": estimator.depth_type,
        "is_metric": estimator.is_metric,
        "representation": estimator.representation,
        "geometry_ready": estimator.is_metric,
        "note": (
            "Relative raw values are disparity-like and are not camera Z."
            if not estimator.is_metric
            else "Metric model output represents model-predicted camera depth."
        ),
    }


def save_sample(run_dir: Path, stem: str, image: np.ndarray, depth: np.ndarray,
                record: dict[str, Any]) -> None:
    depth_vis = colorize_depth(depth)
    comparison = make_side_by_side(image, depth_vis)
    paths = {
        "rgb": run_dir / "rgb" / f"{stem}.png",
        "depth_vis": run_dir / "depth_vis" / f"{stem}_depth.png",
        "side_by_side": run_dir / "side_by_side" / f"{stem}_comparison.png",
    }
    images = {"rgb": image, "depth_vis": depth_vis, "side_by_side": comparison}
    for label, path in paths.items():
        if not cv2.imwrite(str(path), images[label]):
            raise RuntimeError(f"OpenCV failed to write: {path}")
    raw_path = run_dir / "depth_raw" / f"{stem}_depth.npy"
    np.save(raw_path, depth.astype(np.float32, copy=False))
    record["outputs"] = {key: str(path.relative_to(run_dir)) for key, path in paths.items()}
    record["outputs"]["depth_raw"] = str(raw_path.relative_to(run_dir))
    with (run_dir / "metadata.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def run_image(source: Path, run_dir: Path, estimator: DepthEstimator) -> int:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {source}")
    record = metadata(source, 0, 0.0, image, estimator)
    record["input_type"] = "image"
    save_sample(run_dir, source.stem, image, estimator.predict(image), record)
    return 1


def run_video(source: Path, run_dir: Path, estimator: DepthEstimator, every: int) -> int:
    count = 0
    for frame in VideoLoader(source, every):
        record = metadata(source, frame.frame_index, frame.timestamp_seconds,
                          frame.image, estimator)
        record.update({"input_type": "video", "source_fps": round(frame.source_fps, 6),
                       "sample_every_n_frames": every})
        save_sample(run_dir, f"frame_{frame.frame_index:06d}", frame.image,
                    estimator.predict(frame.image), record)
        count += 1
        print(f"Processed source frame {frame.frame_index}", flush=True)
    if count == 0:
        raise RuntimeError(f"No frames could be read from video: {source}")
    return count


def main() -> int:
    args = parse_args()
    try:
        if not args.source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {args.source}")
        config = load_config(args.config)
        model_cfg, video_cfg, output_cfg = (config.get(key, {}) for key in
                                             ("model", "video", "output"))
        model = args.model or model_cfg.get("name", DEFAULT_MODEL)
        device = args.device or model_cfg.get("device", "auto")
        every = args.sample_every if args.sample_every is not None else video_cfg.get(
            "sample_every_n_frames", 1)
        if not isinstance(every, int) or every < 1:
            raise ValueError("--sample-every must be at least 1")
        input_type = detect_type(args.source, args.input_type)
        output_root = args.output_dir or Path(output_cfg.get("directory", "outputs"))

        print(f"Loading {model} on {device}...")
        estimator = DepthEstimator(model, device)
        print(f"Using device: {estimator.device}")
        run_dir = create_run_dir(output_root, args.source)
        count = (run_image(args.source, run_dir, estimator) if input_type == "image"
                 else run_video(args.source, run_dir, estimator, every))
        print(f"Saved {count} prediction(s) to: {run_dir.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
