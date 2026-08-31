import unittest

import numpy as np

from src.depth_types import DepthPrediction


class DepthTypeTests(unittest.TestCase):
    def test_larger_relative_prediction_becomes_nearer_camera_z(self) -> None:
        prediction = DepthPrediction(
            np.array([[4.0, 2.0]], dtype=np.float32),
            "relative",
            False,
            "relative_inverse_depth",
            "synthetic",
        )

        depth = prediction.to_camera_depth()

        self.assertLess(depth.values[0, 0], depth.values[0, 1])
        np.testing.assert_allclose(depth.values, [[0.75, 1.5]])
        self.assertFalse(depth.is_metric)
        self.assertEqual(depth.representation, "relative_camera_z_proxy")

    def test_invalid_and_near_zero_disparity_become_nan(self) -> None:
        prediction = DepthPrediction(
            np.array([[2.0, 0.0, -1.0, np.inf]], dtype=np.float32),
            "relative",
            False,
            "relative_inverse_depth",
            "synthetic",
        )

        depth = prediction.to_camera_depth()

        self.assertTrue(np.isfinite(depth.values[0, 0]))
        self.assertTrue(np.isnan(depth.values[0, 1:]).all())


if __name__ == "__main__":
    unittest.main()
