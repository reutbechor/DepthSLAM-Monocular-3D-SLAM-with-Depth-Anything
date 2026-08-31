import unittest

import numpy as np

from src.transforms import invert_transform, make_transform, transform_points


class TransformTests(unittest.TestCase):
    def test_identity_rotation_and_translation(self) -> None:
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        translation = np.array([10.0, -2.0, 0.5])

        transformed = transform_points(points, np.eye(3), translation)

        np.testing.assert_allclose(transformed, points + translation)

    def test_make_and_invert_transform(self) -> None:
        rotation = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        transform = make_transform(rotation, np.array([[1.0], [2.0], [3.0]]))
        inverse = invert_transform(transform)

        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
