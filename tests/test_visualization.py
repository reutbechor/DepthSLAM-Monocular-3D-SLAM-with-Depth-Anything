import unittest

import numpy as np

from src.visualization import colorize_depth, make_side_by_side, normalize_depth


class VisualizationTests(unittest.TestCase):
    def test_normalize_full_range(self) -> None:
        result = normalize_depth(np.array([[2, 3], [4, 6]], dtype=np.float32))
        self.assertEqual((int(result.min()), int(result.max())), (0, 255))

    def test_constant_depth(self) -> None:
        self.assertTrue(np.all(normalize_depth(np.ones((2, 2))) == 0))

    def test_side_by_side_shape(self) -> None:
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        vis = colorize_depth(np.arange(20, dtype=np.float32).reshape(4, 5))
        self.assertEqual(make_side_by_side(rgb, vis).shape, (4, 10, 3))


if __name__ == "__main__":
    unittest.main()
