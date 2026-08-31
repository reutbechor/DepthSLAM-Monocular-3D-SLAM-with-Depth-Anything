import unittest

import numpy as np

from src.depth_alignment import align_prediction_to_pose
from src.depth_pose_estimator import DepthPoseEstimateResult
from src.depth_types import DepthPrediction


class DepthAlignmentTests(unittest.TestCase):
    def test_affine_disparity_alignment_recovers_camera_z(self) -> None:
        points = np.array(
            [[10, 10], [20, 10], [30, 10], [40, 10],
             [10, 20], [20, 20], [30, 20], [40, 20]],
            dtype=float,
        )
        z = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0])
        objects = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
        raw = np.full((32, 52), np.nan, dtype=np.float32)
        disparity_scale = 4.0
        disparity_shift = 0.7
        raw[points[:, 1].astype(int), points[:, 0].astype(int)] = (
            disparity_scale / z + disparity_shift
        )
        prediction = DepthPrediction(
            raw, "relative", False, "relative_inverse_depth", "synthetic"
        )
        pose = DepthPoseEstimateResult(
            success=True,
            message="synthetic",
            rotation=np.eye(3),
            translation=np.zeros(3),
            inlier_mask=np.ones(8, dtype=bool),
            geometric_inlier_count=8,
            valid_depth_correspondences=8,
            pnp_inliers=8,
            pnp_inlier_ratio=1.0,
            reprojection_rmse_pixels=0.0,
            reprojection_median_pixels=0.0,
            translation_magnitude=0.0,
            translation_units="relative_depth_units",
            correspondence_match_indices=np.arange(8),
            object_points_previous=objects,
            image_points_current=points,
            pnp_inlier_correspondence_mask=np.ones(8, dtype=bool),
        )

        result = align_prediction_to_pose(prediction, points, pose)

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.method, "scale_and_shift")
        self.assertAlmostEqual(result.disparity_scale, disparity_scale, places=5)
        self.assertAlmostEqual(result.disparity_shift, disparity_shift, places=5)
        recovered = result.camera_depth.values[
            points[:, 1].astype(int), points[:, 0].astype(int)
        ]
        np.testing.assert_allclose(recovered, z, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
