import unittest
from types import SimpleNamespace

from tools.run_relative_map import robustness_metadata, scientific_metadata


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

    def test_robustness_metadata_reports_pre_and_post_counts(self) -> None:
        axis_statistics = {
            axis: {name: 0.0 for name in (
                "min", "p1", "p5", "median", "p95", "p99", "max"
            )}
            for axis in ("x", "y", "z")
        }
        result = SimpleNamespace(
            raw_fused_point_count=200,
            voxel_downsampled_point_count=100,
            global_filter=SimpleNamespace(
                method="median_center_distance_percentile",
                percentile=99.5,
                distance_threshold=5.0,
                input_count=100,
                rejected_count=1,
                output_count=99,
                coordinate_statistics_before=axis_statistics,
                coordinate_statistics_after=axis_statistics,
                robust_center=[0.0, 0.0, 0.0],
                distance_statistics=axis_statistics["x"],
                diagnostic_robust_radius=4.0,
                points_outside_diagnostic_radius=2,
            ),
        )

        metadata = robustness_metadata(result)

        self.assertEqual(metadata["raw_fused_point_count"], 200)
        self.assertEqual(metadata["voxel_downsampled_point_count"], 100)
        filtering = metadata["global_outlier_filter"]
        self.assertEqual(filtering["points_before"], 100)
        self.assertEqual(filtering["points_rejected"], 1)
        self.assertEqual(filtering["points_after"], 99)


if __name__ == "__main__":
    unittest.main()
