import unittest

import cv2
import numpy as np

from src.feature_tracker import FeatureTracker, FeatureTrackingError


class FeatureTrackerTests(unittest.TestCase):
    def test_lowe_ratio_filtering(self) -> None:
        passing = (
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10.0),
            cv2.DMatch(_queryIdx=0, _trainIdx=1, _distance=20.0),
        )
        failing = (
            cv2.DMatch(_queryIdx=1, _trainIdx=2, _distance=18.0),
            cv2.DMatch(_queryIdx=1, _trainIdx=3, _distance=20.0),
        )
        incomplete = (cv2.DMatch(_queryIdx=2, _trainIdx=4, _distance=5.0),)

        result = FeatureTracker.filter_ratio_matches(
            [passing, failing, incomplete], ratio_threshold=0.75
        )

        self.assertEqual(result, [passing[0]])

    def test_empty_image_is_rejected(self) -> None:
        tracker = FeatureTracker()
        with self.assertRaisesRegex(FeatureTrackingError, "empty"):
            tracker.detect_and_describe(np.empty((0, 0), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
