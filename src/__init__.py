"""Reusable depth inference, relative geometry, and transform components."""

from .depth_estimator import DepthEstimator
from .depth_pose_estimator import DepthPoseEstimateResult, DepthPoseEstimator
from .depth_quality import (
    DepthAlignmentQualityMetrics,
    DepthQualityAssessment,
    DepthQualityThresholds,
    assess_depth_alignment_quality,
    measure_depth_alignment_quality,
)
from .depth_types import CameraDepth, DepthPrediction
from .depth_geometry import DepthGeometryProcessor
from .feature_tracker import FeatureTracker
from .evaluation import (
    EvaluationResult,
    FRAME_METRIC_COLUMNS,
    evaluate_run_directory,
    generate_evaluation_plots,
    write_evaluation_outputs,
)
from .map_builder import MappingFrame, RelativeMapBuilder, RelativeMapResult
from .map_fusion import FusedPointCloud, RelativeMapFusion, voxel_downsample
from .keyframe_selector import (
    KeyframeMotionMetrics,
    KeyframeSelectionResult,
    KeyframeSelector,
    KeyframeThresholds,
    feature_displacement_statistics,
    rotation_angle_degrees,
)
from .motion_estimator import MotionEstimator
from .point_cloud import PointCloudResult, generate_colored_point_cloud
from .pose_manager import PoseManager
from .robust_filtering import (
    GlobalOutlierFilterResult,
    coordinate_statistics,
    distribution_statistics,
    filter_depth_range,
    filter_global_radius,
)
from .transforms import invert_transform, make_transform, transform_points
from .trajectory_refinement import (
    JumpDetectionResult,
    TrajectoryData,
    TrajectoryDiagnostics,
    TrajectoryMetrics,
    TrajectoryRefinementResult,
    detect_suspicious_jumps,
    load_map_trajectory,
    refine_trajectory,
    trajectory_diagnostics,
    trajectory_metrics,
    write_refinement_outputs,
)

__all__ = [
    "DepthEstimator",
    "DepthPoseEstimateResult",
    "DepthPoseEstimator",
    "DepthAlignmentQualityMetrics",
    "DepthQualityAssessment",
    "DepthQualityThresholds",
    "assess_depth_alignment_quality",
    "measure_depth_alignment_quality",
    "CameraDepth",
    "DepthPrediction",
    "DepthGeometryProcessor",
    "FeatureTracker",
    "EvaluationResult",
    "FRAME_METRIC_COLUMNS",
    "evaluate_run_directory",
    "generate_evaluation_plots",
    "write_evaluation_outputs",
    "MappingFrame",
    "RelativeMapBuilder",
    "RelativeMapResult",
    "FusedPointCloud",
    "RelativeMapFusion",
    "voxel_downsample",
    "KeyframeMotionMetrics",
    "KeyframeSelectionResult",
    "KeyframeSelector",
    "KeyframeThresholds",
    "feature_displacement_statistics",
    "rotation_angle_degrees",
    "MotionEstimator",
    "PointCloudResult",
    "generate_colored_point_cloud",
    "PoseManager",
    "GlobalOutlierFilterResult",
    "coordinate_statistics",
    "distribution_statistics",
    "filter_depth_range",
    "filter_global_radius",
    "invert_transform",
    "make_transform",
    "transform_points",
    "JumpDetectionResult",
    "TrajectoryData",
    "TrajectoryDiagnostics",
    "TrajectoryMetrics",
    "TrajectoryRefinementResult",
    "detect_suspicious_jumps",
    "load_map_trajectory",
    "refine_trajectory",
    "trajectory_diagnostics",
    "trajectory_metrics",
    "write_refinement_outputs",
]
