import unittest

import numpy as np

from src.robust_filtering import filter_global_radius


class RobustFilteringTests(unittest.TestCase):
    @staticmethod
    def synthetic_cloud() -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.linspace(-0.1, 0.1, 10)
        cluster = np.column_stack((coordinates, coordinates**2, -coordinates))
        points = np.vstack((cluster, [[1000.0, -1000.0, 500.0]]))
        colors = np.arange(points.size, dtype=np.uint8).reshape(-1, 3)
        return points, colors

    def test_global_isolated_point_is_removed_and_cluster_remains(self) -> None:
        points, colors = self.synthetic_cloud()

        result = filter_global_radius(points, colors, percentile=90.0)

        self.assertEqual(result.input_count, 11)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(result.output_count, 10)
        np.testing.assert_allclose(result.points, points[:-1])
        np.testing.assert_array_equal(result.colors, colors[:-1])
        self.assertGreaterEqual(result.points_outside_diagnostic_radius, 1)

    def test_global_filter_is_deterministic(self) -> None:
        points, colors = self.synthetic_cloud()

        first = filter_global_radius(points, colors, percentile=90.0)
        second = filter_global_radius(points, colors, percentile=90.0)

        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.colors, second.colors)
        self.assertEqual(first.distance_threshold, second.distance_threshold)
        self.assertEqual(
            first.coordinate_statistics_after,
            second.coordinate_statistics_after,
        )


if __name__ == "__main__":
    unittest.main()
