"""SIFT feature detection and Lowe-ratio descriptor matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


class FeatureTrackingError(RuntimeError):
    """Raised when reliable feature correspondences cannot be produced."""


@dataclass(frozen=True)
class FeatureMatchStatistics:
    keypoints_image1: int
    keypoints_image2: int
    knn_match_groups: int
    good_matches: int
    ratio_threshold: float


@dataclass(frozen=True)
class FeatureMatchResult:
    keypoints1: list[cv2.KeyPoint]
    keypoints2: list[cv2.KeyPoint]
    good_matches: list[cv2.DMatch]
    points1: np.ndarray
    points2: np.ndarray
    statistics: FeatureMatchStatistics


class FeatureTracker:
    """Detect SIFT features and match them with BFMatcher and Lowe's test."""

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        max_features: int | None = None,
        minimum_matches: int = 8,
    ) -> None:
        if not 0.0 < ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must be between 0 and 1")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None")
        if minimum_matches < 1:
            raise ValueError("minimum_matches must be at least 1")
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not provide SIFT_create")

        self.ratio_threshold = ratio_threshold
        self.max_features = max_features
        self.minimum_matches = minimum_matches
        self._detector = cv2.SIFT_create(nfeatures=max_features or 0)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    @staticmethod
    def _to_grayscale(image: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise FeatureTrackingError(f"{label} is empty or is not a NumPy image")
        if image.ndim == 2:
            return image
        if image.ndim != 3:
            raise FeatureTrackingError(f"{label} must be a grayscale, BGR, or BGRA image")
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise FeatureTrackingError(f"{label} has an unsupported channel count")

    def detect_and_describe(
        self, image: np.ndarray, label: str = "image"
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Return SIFT keypoints and float descriptors for one image."""
        grayscale = self._to_grayscale(image, label)
        try:
            keypoints, descriptors = self._detector.detectAndCompute(grayscale, None)
        except cv2.error as exc:
            raise FeatureTrackingError(f"SIFT failed for {label}: {exc}") from exc
        if descriptors is None or not keypoints:
            raise FeatureTrackingError(f"No SIFT descriptors were found in {label}")
        return list(keypoints), descriptors

    @staticmethod
    def filter_ratio_matches(
        knn_matches: Sequence[Sequence[cv2.DMatch]], ratio_threshold: float
    ) -> list[cv2.DMatch]:
        """Keep a match when its best distance is clearly below the second best."""
        if not 0.0 < ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must be between 0 and 1")
        good_matches: list[cv2.DMatch] = []
        for candidates in knn_matches:
            if len(candidates) < 2:
                continue
            best, second_best = candidates[0], candidates[1]
            if best.distance < ratio_threshold * second_best.distance:
                good_matches.append(best)
        return good_matches

    def match(self, image1: np.ndarray, image2: np.ndarray) -> FeatureMatchResult:
        """Return reliable SIFT correspondences between two images."""
        keypoints1, descriptors1 = self.detect_and_describe(image1, "image 1")
        keypoints2, descriptors2 = self.detect_and_describe(image2, "image 2")
        try:
            knn_matches = self._matcher.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error as exc:
            raise FeatureTrackingError(f"Descriptor matching failed: {exc}") from exc
        good_matches = self.filter_ratio_matches(knn_matches, self.ratio_threshold)
        if len(good_matches) < self.minimum_matches:
            raise FeatureTrackingError(
                f"Only {len(good_matches)} Lowe-ratio matches were found; "
                f"at least {self.minimum_matches} are required"
            )

        points1 = np.asarray(
            [keypoints1[match.queryIdx].pt for match in good_matches], dtype=np.float32
        )
        points2 = np.asarray(
            [keypoints2[match.trainIdx].pt for match in good_matches], dtype=np.float32
        )
        statistics = FeatureMatchStatistics(
            keypoints_image1=len(keypoints1),
            keypoints_image2=len(keypoints2),
            knn_match_groups=len(knn_matches),
            good_matches=len(good_matches),
            ratio_threshold=self.ratio_threshold,
        )
        return FeatureMatchResult(
            keypoints1, keypoints2, good_matches, points1, points2, statistics
        )
