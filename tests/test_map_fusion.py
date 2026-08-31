import unittest

import numpy as np

from src.map_fusion import voxel_downsample


class MapFusionTests(unittest.TestCase):
    def test_same_voxel_averages_position_and_rgb(self) -> None:
        points = np.array([[0.01, 0.01, 0.0], [0.02, 0.03, 0.0]])
        colors = np.array([[10, 20, 30], [30, 40, 50]], dtype=np.uint8)

        result = voxel_downsample(points, colors, voxel_size=0.1)

        self.assertEqual(result.input_point_count, 2)
        self.assertEqual(result.output_point_count, 1)
        np.testing.assert_allclose(result.points, [[0.015, 0.02, 0.0]])
        np.testing.assert_array_equal(result.colors, [[20, 30, 40]])

    def test_distinct_voxels_remain_distinct(self) -> None:
        points = np.array([[0.01, 0.0, 0.0], [0.11, 0.0, 0.0]])
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        result = voxel_downsample(points, colors, voxel_size=0.1)

        self.assertEqual(result.output_point_count, 2)
        np.testing.assert_allclose(result.points, points)
        np.testing.assert_array_equal(result.colors, colors)

    def test_invalid_clouds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            voxel_downsample(
                np.array([[np.nan, 0.0, 0.0]]),
                np.array([[0, 0, 0]], dtype=np.uint8),
                0.1,
            )
        with self.assertRaisesRegex(ValueError, "equal length"):
            voxel_downsample(
                np.zeros((2, 3)), np.zeros((1, 3), dtype=np.uint8), 0.1
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            voxel_downsample(
                np.zeros((1, 3)), np.zeros((1, 3), dtype=np.float32), 0.1
            )


if __name__ == "__main__":
    unittest.main()
