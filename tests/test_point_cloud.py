import unittest

import numpy as np

from src.point_cloud import generate_colored_point_cloud
from src.depth_types import CameraDepth


def camera_depth(values: np.ndarray) -> CameraDepth:
    clean = np.asarray(values, dtype=np.float32).copy()
    clean[~np.isfinite(clean) | (clean <= 0.0)] = np.nan
    return CameraDepth(
        clean, "relative", False, "relative_camera_z_proxy", "synthetic",
        "relative_depth_units", "synthetic", "none",
    )


class PointCloudTests(unittest.TestCase):
    def test_dense_points_and_rgb_colors(self) -> None:
        image_rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [10, 20, 30]],
            ],
            dtype=np.uint8,
        )
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = generate_colored_point_cloud(
            image_rgb, camera_depth(depth), np.eye(3)
        )

        expected_points = np.array(
            [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0],
             [0.0, 3.0, 3.0], [4.0, 4.0, 4.0]]
        )
        np.testing.assert_allclose(result.points, expected_points)
        np.testing.assert_array_equal(result.colors, image_rgb.reshape(-1, 3))
        self.assertEqual(result.sampled_pixel_count, 4)
        self.assertEqual(result.valid_point_count, 4)
        self.assertEqual(result.points.shape, (4, 3))
        self.assertEqual(result.colors.dtype, np.uint8)
        self.assertEqual(result.coordinate_units, "relative_depth_units")

    def test_stride_controls_sample_count(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.ones((8, 8), dtype=np.float32)

        result = generate_colored_point_cloud(
            image, camera_depth(depth), np.eye(3), stride=2
        )

        self.assertEqual(result.sampled_pixel_count, 16)
        self.assertEqual(result.valid_point_count, 16)
        self.assertEqual(result.stride, 2)

    def test_invalid_depths_are_removed_with_matching_colors(self) -> None:
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        depth = np.array([[1.0, np.nan, 0.0], [np.inf, -1.0, 2.0]])

        result = generate_colored_point_cloud(image, camera_depth(depth), np.eye(3))

        self.assertEqual(result.sampled_pixel_count, 6)
        self.assertEqual(result.valid_point_count, 2)
        np.testing.assert_allclose(result.points, [[0.0, 0.0, 1.0], [4.0, 2.0, 2.0]])
        np.testing.assert_array_equal(result.colors, [image[0, 0], image[1, 2]])
        np.testing.assert_array_equal(result.valid_pixel_coordinates, [[0, 0], [2, 1]])


if __name__ == "__main__":
    unittest.main()
