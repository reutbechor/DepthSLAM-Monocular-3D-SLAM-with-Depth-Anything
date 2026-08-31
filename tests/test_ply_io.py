import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ply_io import write_ascii_ply


class PlyIoTests(unittest.TestCase):
    def test_ascii_ply_header_count_and_rgb_order(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-1.5, 0.0, 4.25]])
        colors = np.array([[255, 0, 7], [3, 4, 5]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_ascii_ply(path, points, colors)
            lines = path.read_text(encoding="ascii").splitlines()

        self.assertEqual(lines[0], "ply")
        self.assertEqual(lines[1], "format ascii 1.0")
        self.assertIn("element vertex 2", lines)
        self.assertIn("property uchar red", lines)
        self.assertIn("property uchar green", lines)
        self.assertIn("property uchar blue", lines)
        data_start = lines.index("end_header") + 1
        self.assertEqual(lines[data_start], "1 2 3 255 0 7")
        self.assertEqual(lines[data_start + 1], "-1.5 0 4.25 3 4 5")

    def test_point_color_length_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "equal length"):
                write_ascii_ply(
                    Path(directory) / "bad.ply",
                    np.zeros((2, 3)),
                    np.zeros((1, 3), dtype=np.uint8),
                )


if __name__ == "__main__":
    unittest.main()
