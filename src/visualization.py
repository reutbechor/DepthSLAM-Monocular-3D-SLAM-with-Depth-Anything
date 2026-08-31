"""Depth and feature-match visualization helpers using NumPy and OpenCV."""

import cv2
import numpy as np


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize one depth map to uint8, ignoring non-finite values."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional NumPy array")
    finite = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if not finite.any():
        return normalized
    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    if maximum > minimum:
        scaled = (depth[finite] - minimum) / (maximum - minimum)
        normalized[finite] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return normalized


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize relative depth as a BGR image."""
    return cv2.applyColorMap(normalize_depth(depth), cv2.COLORMAP_INFERNO)


def make_side_by_side(rgb_bgr: np.ndarray, depth_bgr: np.ndarray) -> np.ndarray:
    """Place a BGR source frame and depth visualization side-by-side."""
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must be an HxWx3 image")
    if depth_bgr.ndim != 3 or depth_bgr.shape[2] != 3:
        raise ValueError("depth_bgr must be an HxWx3 image")
    if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
        depth_bgr = cv2.resize(
            depth_bgr,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.hstack((rgb_bgr, depth_bgr))


def draw_feature_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Draw all Lowe matches in orange and pose inliers in green."""
    def as_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("match visualization images must be non-empty")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError("match visualization expects grayscale, BGR, or BGRA images")

    first = as_bgr(image1)
    second = as_bgr(image2)
    if inlier_mask is None:
        mask = np.zeros(len(matches), dtype=bool)
    else:
        mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if mask.size != len(matches):
            raise ValueError("inlier_mask length must equal the number of matches")

    banner_height = 32
    height = max(first.shape[0], second.shape[0])
    width = first.shape[1] + second.shape[1]
    canvas = np.zeros((height + banner_height, width, 3), dtype=np.uint8)
    canvas[banner_height : banner_height + first.shape[0], : first.shape[1]] = first
    canvas[
        banner_height : banner_height + second.shape[0],
        first.shape[1] :,
    ] = second

    # Draw outliers first so accepted pose inliers remain prominent.
    for desired_inlier, color in ((False, (0, 165, 255)), (True, (0, 255, 0))):
        for index, match in enumerate(matches):
            if bool(mask[index]) != desired_inlier:
                continue
            point1 = keypoints1[match.queryIdx].pt
            point2 = keypoints2[match.trainIdx].pt
            start = (int(round(point1[0])), int(round(point1[1])) + banner_height)
            end = (
                int(round(point2[0])) + first.shape[1],
                int(round(point2[1])) + banner_height,
            )
            cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, start, 3, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, end, 3, color, 1, cv2.LINE_AA)

    legend = "green: pose inlier | orange: Lowe-ratio match / outlier"
    cv2.putText(canvas, legend, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
