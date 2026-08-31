import unittest

import numpy as np

from src.depth_geometry import DepthGeometryProcessor


class DepthGeometryTests(unittest.TestCase):
    def test_bilinear_sampling_and_backprojection(self) -> None:
        depth = np.array([[1.0, 2.0], [3.0, 5.0]])
        pixels = np.array([[0.5, 0.5]])
        result = DepthGeometryProcessor("bilinear").process(
            pixels, np.array([True]), depth, np.eye(3)
        )

        self.assertEqual(result.original_match_count, 1)
        self.assertEqual(result.pose_inlier_count, 1)
        self.assertEqual(result.valid_depth_sample_count, 1)
        np.testing.assert_allclose(result.sampled_relative_depths, [2.75])
        np.testing.assert_allclose(result.points_3d_relative, [[1.375, 1.375, 2.75]])
        np.testing.assert_array_equal(result.valid_match_indices, [0])

    def test_invalid_and_out_of_bounds_samples_are_excluded(self) -> None:
        depth = np.array(
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, -1.0]]
        )
        pixels = np.array(
            [
                [0.0, 0.0],   # valid pose inlier and depth
                [1.0, 1.0],   # invalid NaN depth
                [2.0, 2.0],   # invalid negative depth
                [-0.1, 1.0],  # outside image
                [1.0, 0.0],   # valid depth but not a pose inlier
            ]
        )
        pose_mask = np.array([True, True, True, True, False])

        result = DepthGeometryProcessor("nearest").process(
            pixels, pose_mask, depth, np.eye(3)
        )

        self.assertEqual(result.original_match_count, 5)
        self.assertEqual(result.pose_inlier_count, 4)
        self.assertEqual(result.valid_depth_sample_count, 1)
        np.testing.assert_allclose(result.valid_pixel_coordinates, [[0.0, 0.0]])
        np.testing.assert_allclose(result.sampled_relative_depths, [1.0])
        np.testing.assert_array_equal(result.valid_match_indices, [0])
        np.testing.assert_array_equal(
            result.valid_match_mask, [True, False, False, False, False]
        )


if __name__ == "__main__":
    unittest.main()
