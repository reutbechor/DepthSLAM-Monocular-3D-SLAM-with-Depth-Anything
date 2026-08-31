import unittest
from types import SimpleNamespace

from tools.run_relative_map import scientific_metadata


class RelativeMapMetadataTests(unittest.TestCase):
    def test_relative_pnp_labels_are_non_metric_and_not_fixed_step(self) -> None:
        result = SimpleNamespace(
            is_metric=False,
            depth_type="relative",
            depth_representation="relative_inverse_depth",
            translation_units="relative_depth_units",
        )

        metadata = scientific_metadata(result, "depth-pnp", 1.0)

        self.assertFalse(metadata["is_metric"])
        self.assertEqual(metadata["depth_representation"], "relative_inverse_depth")
        self.assertEqual(metadata["scale_estimation_method"], "depth_pnp")
        self.assertEqual(metadata["translation_units"], "relative_depth_units")
        self.assertEqual(
            metadata["depth_alignment_method"],
            "scale_and_shift_per_accepted_pair",
        )
        self.assertIsNone(metadata["translation_step"])

    def test_fixed_step_metadata_is_explicitly_debug(self) -> None:
        result = SimpleNamespace(
            is_metric=False,
            depth_type="relative",
            depth_representation="relative_inverse_depth",
            translation_units="relative_depth_units",
        )

        metadata = scientific_metadata(result, "fixed-step", 0.75)

        self.assertEqual(metadata["scale_estimation_method"], "fixed_step_debug")
        self.assertEqual(metadata["translation_scale"], "arbitrary_fixed_step_debug")
        self.assertEqual(metadata["translation_step"], 0.75)


if __name__ == "__main__":
    unittest.main()
