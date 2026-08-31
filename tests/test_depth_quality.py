import unittest

import numpy as np

from src.depth_quality import DepthQualityThresholds, assess_depth_alignment_quality
from src.depth_types import CameraDepth


def aligned_depth(
    values: np.ndarray,
    *,
    denominator_rejects: int = 0,
) -> CameraDepth:
    return CameraDepth(
        values=np.asarray(values, dtype=np.float32).reshape(10, 10),
        depth_type="relative",
        is_metric=False,
        representation="relative_camera_z_proxy",
        model_name="synthetic",
        coordinate_units="relative_depth_units",
        conversion="reciprocal_affine_disparity_alignment",
        alignment_method="scale_and_shift",
        disparity_scale=2.0,
        disparity_shift=0.5,
        denominator_epsilon=0.001,
        rejected_small_denominator_count=denominator_rejects,
    )


class DepthQualityTests(unittest.TestCase):
    def assess(
        self,
        depth: CameraDepth,
        *,
        inputs: int = 1000,
        inliers: int = 600,
        thresholds: DepthQualityThresholds | None = None,
    ):
        return assess_depth_alignment_quality(
            depth,
            alignment_input_correspondences=inputs,
            alignment_inliers=inliers,
            thresholds=thresholds or DepthQualityThresholds(),
        )

    def test_good_depth_alignment_is_accepted(self) -> None:
        result = self.assess(aligned_depth(np.linspace(1.0, 2.0, 100)))

        self.assertTrue(result.accepted)
        self.assertIsNone(result.rejection_reason)
        self.assertEqual(result.metrics.total_depth_candidates, 100)

    def test_high_denominator_reject_ratio_is_rejected(self) -> None:
        values = np.concatenate((np.ones(40), np.full(60, np.nan)))
        result = self.assess(
            aligned_depth(values, denominator_rejects=60),
            thresholds=DepthQualityThresholds(min_valid_depth_ratio=None),
        )

        self.assertEqual(result.rejection_reason, "depth_denominator_reject_ratio")
        self.assertAlmostEqual(result.metrics.denominator_rejection_ratio, 0.6)

    def test_low_valid_depth_ratio_is_rejected(self) -> None:
        values = np.concatenate((np.ones(40), np.full(60, np.nan)))
        result = self.assess(
            aligned_depth(values),
            thresholds=DepthQualityThresholds(
                max_denominator_reject_ratio=None,
            ),
        )

        self.assertEqual(result.rejection_reason, "depth_valid_ratio")
        self.assertAlmostEqual(result.metrics.valid_aligned_depth_ratio, 0.4)

    def test_too_few_alignment_inliers_is_rejected(self) -> None:
        result = self.assess(
            aligned_depth(np.ones(100)),
            inputs=1000,
            inliers=499,
        )

        self.assertEqual(result.rejection_reason, "depth_alignment_inliers")

    def test_low_alignment_inlier_ratio_is_rejected(self) -> None:
        result = self.assess(
            aligned_depth(np.ones(100)),
            inputs=2000,
            inliers=500,
        )

        self.assertEqual(result.rejection_reason, "depth_alignment_inlier_ratio")
        self.assertAlmostEqual(result.metrics.alignment_inlier_ratio, 0.25)

    def test_pathological_relative_z_distribution_is_rejected(self) -> None:
        values = np.concatenate((np.ones(99), [10000.0]))
        result = self.assess(aligned_depth(values))

        self.assertEqual(result.rejection_reason, "depth_z_distribution")
        self.assertGreater(result.metrics.relative_z_p99_over_median, 50.0)

    def test_assessment_is_deterministic(self) -> None:
        depth = aligned_depth(np.linspace(0.5, 4.0, 100))

        first = self.assess(depth)
        second = self.assess(depth)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
