import unittest

import numpy as np

from src.backprojection import backproject_pixels


class BackprojectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [[2.0, 0.0, 10.0], [0.0, 4.0, 20.0], [0.0, 0.0, 1.0]]
        )

    def test_known_pixels_and_depths(self) -> None:
        pixels = np.array([[12.0, 24.0], [8.0, 16.0]])
        depths = np.array([3.0, 2.0])

        points = backproject_pixels(pixels, depths, self.camera_matrix)

        expected = np.array([[3.0, 3.0, 3.0], [-2.0, -2.0, 2.0]])
        np.testing.assert_allclose(points, expected)
        self.assertEqual(points.shape, (2, 3))

    def test_principal_point_maps_to_optical_axis(self) -> None:
        point = backproject_pixels(
            np.array([[10.0, 20.0]]), np.array([[7.5]]), self.camera_matrix
        )

        np.testing.assert_allclose(point, [[0.0, 0.0, 7.5]])

    def test_invalid_depths_are_rejected(self) -> None:
        pixel = np.array([[10.0, 20.0]])
        for invalid_depth in (np.nan, np.inf, 0.0, -1.0):
            with self.subTest(depth=invalid_depth):
                with self.assertRaisesRegex(ValueError, "finite, positive"):
                    backproject_pixels(pixel, np.array([invalid_depth]), self.camera_matrix)

    def test_invalid_intrinsics_are_rejected(self) -> None:
        invalid = self.camera_matrix.copy()
        invalid[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive fx"):
            backproject_pixels(np.array([[1.0, 1.0]]), np.array([1.0]), invalid)


if __name__ == "__main__":
    unittest.main()
