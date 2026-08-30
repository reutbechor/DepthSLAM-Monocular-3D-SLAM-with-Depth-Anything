import unittest

from src.video_loader import should_sample_frame


class VideoSamplingTests(unittest.TestCase):
    def test_every_third_frame(self) -> None:
        self.assertEqual([i for i in range(8) if should_sample_frame(i, 3)], [0, 3, 6])

    def test_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            should_sample_frame(0, 0)


if __name__ == "__main__":
    unittest.main()
