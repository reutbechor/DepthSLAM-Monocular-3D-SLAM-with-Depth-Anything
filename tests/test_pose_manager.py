import unittest

import numpy as np

from src.pose_manager import PoseManager
from src.transforms import invert_transform, make_transform, transform_points


class PoseManagerTests(unittest.TestCase):
    def test_identity_origin_and_sequential_relative_steps(self) -> None:
        manager = PoseManager(relative_translation_step=2.0)

        np.testing.assert_allclose(manager.current_pose(), np.eye(4))
        first = manager.add_relative_pose(np.eye(3), np.array([4.0, 0.0, 0.0]))
        second = manager.add_relative_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))

        np.testing.assert_allclose(first[:3, 3], [-2.0, 0.0, 0.0])
        np.testing.assert_allclose(second[:3, 3], [-4.0, 0.0, 0.0])
        np.testing.assert_allclose(
            manager.trajectory_positions(),
            [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
        )

    def test_opencv_relative_transform_is_inverted(self) -> None:
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        direction = np.array([1.0, 0.0, 0.0])
        manager = PoseManager(relative_translation_step=1.0)

        actual = manager.add_relative_pose(rotation, direction)
        expected = invert_transform(make_transform(rotation, direction))

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)
        # The previous camera origin appears at t in the current camera frame.
        # Mapping that known current-frame point back to world must recover origin.
        transformed = transform_points(
            np.array([[1.0, 0.0, 0.0]]), actual[:3, :3], actual[:3, 3]
        )
        np.testing.assert_allclose(transformed, [[0.0, 0.0, 0.0]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
