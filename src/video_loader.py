"""OpenCV video reading and deterministic frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    image: np.ndarray
    frame_index: int
    timestamp_seconds: float
    source_fps: float


def should_sample_frame(frame_index: int, sample_every_n_frames: int) -> bool:
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames must be at least 1")
    return frame_index % sample_every_n_frames == 0


class VideoLoader:
    """Yield every Nth video frame as an OpenCV BGR image."""

    def __init__(self, path: str | Path, sample_every_n_frames: int = 1) -> None:
        self.path = Path(path)
        if sample_every_n_frames < 1:
            raise ValueError("sample_every_n_frames must be at least 1")
        self.sample_every_n_frames = sample_every_n_frames

    def __iter__(self) -> Iterator[VideoFrame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video: {self.path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if should_sample_frame(frame_index, self.sample_every_n_frames):
                    timestamp = (
                        frame_index / source_fps
                        if source_fps > 0
                        else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    )
                    yield VideoFrame(image, frame_index, timestamp, source_fps)
                frame_index += 1
        finally:
            capture.release()
