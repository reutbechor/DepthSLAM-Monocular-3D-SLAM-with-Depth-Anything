import unittest

import numpy as np

from src.keyframe_selector import (
    KeyframeSelector,
    KeyframeThresholds,
    feature_displacement_statistics,
    rotation_angle_degrees,
)


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


class KeyframeSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points1 = np.zeros((100, 2), dtype=np.float64)
        self.mask = np.ones(100, dtype=bool)
        self.thresholds = KeyframeThresholds(
            min_good_matches=100,
            min_geometric_inliers=80,
            min_geometric_inlier_ratio=0.4,
            min_median_feature_displacement_px=8.0,
            min_rotation_deg=1.0,
            max_frames_without_keyframe=3,
        )
        self.selector = KeyframeSelector(self.thresholds)

    def evaluate(
        self,
        displacement: float,
        rotation: np.ndarray | None = None,
        frames_since: int = 1,
        good_matches: int = 100,
        geometric_inliers: int = 100,
        geometric_inlier_ratio: float = 1.0,
    ):
        points2 = self.points1.copy()
        points2[:, 0] += displacement
        return self.selector.evaluate(
            points1=self.points1,
            points2=points2,
            inlier_mask=self.mask,
            rotation=np.eye(3) if rotation is None else rotation,
            good_matches=good_matches,
            geometric_inliers=geometric_inliers,
            geometric_inlier_ratio=geometric_inlier_ratio,
            frames_since_last_keyframe=frames_since,
        )

    def test_first_frame_is_always_a_keyframe(self) -> None:
        result = self.selector.initial_frame()

        self.assertTrue(result.selected)
        self.assertEqual(result.reason, "initial_frame")

    def test_tiny_displacement_and_rotation_is_skipped(self) -> None:
        result = self.evaluate(1.0)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "insufficient_keyframe_motion")

    def test_sufficient_displacement_selects_keyframe(self) -> None:
        result = self.evaluate(8.0)

        self.assertTrue(result.selected)
        self.assertEqual(result.reason, "sufficient_feature_displacement")

    def test_sufficient_rotation_selects_keyframe(self) -> None:
        result = self.evaluate(0.0, rotation_z(1.5))

        self.assertTrue(result.selected)
        self.assertEqual(result.reason, "sufficient_rotation")

    def test_unreliable_geometry_is_rejected_not_skipped(self) -> None:
        result = self.evaluate(20.0, good_matches=99)

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.reason, "keyframe_good_matches")

    def test_max_frame_gap_forces_keyframe_attempt(self) -> None:
        result = self.evaluate(0.0, frames_since=3)

        self.assertTrue(result.selected)
        self.assertEqual(result.reason, "max_frame_gap")

    def test_disabled_mode_selects_tiny_motion(self) -> None:
        selector = KeyframeSelector(KeyframeThresholds(enabled=False))
        result = selector.evaluate(
            points1=self.points1,
            points2=self.points1,
            inlier_mask=self.mask,
            rotation=np.eye(3),
            good_matches=1,
            geometric_inliers=1,
            geometric_inlier_ratio=0.01,
            frames_since_last_keyframe=1,
        )

        self.assertTrue(result.selected)
        self.assertEqual(result.reason, "keyframe_selection_disabled")

    def test_rotation_angle_is_numerically_stable(self) -> None:
        slightly_high_trace = np.eye(3)
        slightly_high_trace[0, 0] += 1e-12

        self.assertEqual(rotation_angle_degrees(slightly_high_trace), 0.0)
        self.assertAlmostEqual(rotation_angle_degrees(rotation_z(45.0)), 45.0)

    def test_feature_displacement_uses_only_inliers(self) -> None:
        points2 = self.points1.copy()
        points2[:90, 0] = 2.0
        points2[90:, 0] = 1000.0
        mask = self.mask.copy()
        mask[90:] = False

        median, p75, p90 = feature_displacement_statistics(
            self.points1, points2, mask
        )

        self.assertEqual((median, p75, p90), (2.0, 2.0, 2.0))

    def test_decision_is_deterministic(self) -> None:
        first = self.evaluate(12.0, rotation_z(2.0), frames_since=2)
        second = self.evaluate(12.0, rotation_z(2.0), frames_since=2)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
