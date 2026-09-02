"""Optional pose-only optimization over a small accepted-keyframe window.

The canonical convention is ``T_A_from_B`` and stored poses are
``T_world_from_camera``. Depth, intrinsics, landmarks, and scale are fixed.
SciPy is intentionally not required; a small deterministic NumPy
Levenberg-Marquardt solver with robust residual weights is used instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from .backprojection import validate_camera_matrix
from .depth_geometry import DepthGeometryProcessor
from .depth_types import CameraDepth
from .transforms import make_transform


@dataclass(frozen=True)
class SlidingWindowPoseOptimizationConfig:
    enabled: bool = False
    window_size: int = 3
    minimum_observations: int = 300
    minimum_relative_median_improvement: float = 0.05
    maximum_rotation_change_deg: float = 2.0
    maximum_translation_change_relative: float = 0.50
    robust_loss: str = "huber"
    f_scale_px: float = 2.0
    max_nfev: int = 100
    finite_difference_step: float = 1e-6
    projection_depth_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if not 0.0 <= self.minimum_relative_median_improvement <= 1.0:
            raise ValueError("minimum_relative_median_improvement must be in [0, 1]")
        if self.maximum_rotation_change_deg <= 0.0:
            raise ValueError("maximum_rotation_change_deg must be positive")
        if self.maximum_translation_change_relative <= 0.0:
            raise ValueError("maximum_translation_change_relative must be positive")
        if self.robust_loss not in {"huber", "soft_l1"}:
            raise ValueError("robust_loss must be 'huber' or 'soft_l1'")
        if self.f_scale_px <= 0.0 or self.max_nfev < 1:
            raise ValueError("f_scale_px and max_nfev must be positive")
        if self.finite_difference_step <= 0.0 or self.projection_depth_epsilon <= 0.0:
            raise ValueError("numerical steps must be positive")


@dataclass(frozen=True)
class PoseWindowFrame:
    frame_index: int
    world_from_camera: np.ndarray


@dataclass(frozen=True)
class ReprojectionEdge:
    source_frame_index: int
    target_frame_index: int
    source_points_camera: np.ndarray
    target_pixels: np.ndarray
    geometric_support: str = "pnp_inliers"

    @property
    def observation_count(self) -> int:
        return int(np.asarray(self.source_points_camera).shape[0])


@dataclass(frozen=True)
class ReprojectionMetrics:
    observation_count: int
    median: float | None
    mean: float | None
    rmse: float | None
    p90: float | None


@dataclass(frozen=True)
class PoseChangeDiagnostics:
    frame_index: int
    rotation_change_deg: float
    translation_change: float
    original_step_magnitude: float
    translation_change_relative: float
    camera_center_before: tuple[float, float, float]
    camera_center_after: tuple[float, float, float]


@dataclass(frozen=True)
class SolverResult:
    success: bool
    message: str
    parameters: np.ndarray
    cost: float
    function_evaluations: int


@dataclass(frozen=True)
class SlidingWindowPoseOptimizationResult:
    attempted: bool
    accepted: bool
    reason: str
    window_frame_indices: tuple[int, ...]
    fixed_frame_index: int | None
    optimized_frame_indices: tuple[int, ...]
    edges_used: tuple[tuple[int, int], ...]
    observations_per_edge: dict[str, int]
    total_observation_count: int
    solver_success: bool
    solver_message: str
    solver_function_evaluations: int
    cost_before: float | None
    cost_after: float | None
    metrics_before: ReprojectionMetrics
    metrics_after: ReprojectionMetrics
    relative_median_improvement: float | None
    per_edge_before: dict[str, ReprojectionMetrics]
    per_edge_after: dict[str, ReprojectionMetrics]
    pose_changes: tuple[PoseChangeDiagnostics, ...]
    baseline_world_from_camera: tuple[np.ndarray, ...]
    proposed_world_from_camera: tuple[np.ndarray, ...]
    selected_world_from_camera: tuple[np.ndarray, ...]
    coordinate_scale: str = "relative_non_metric"
    pose_convention: str = "T_world_from_camera"
    fixed_depth: bool = True
    fixed_intrinsics: bool = True
    fixed_scale: bool = True

    def report_dict(self) -> dict[str, object]:
        return {
            "window_frames": list(self.window_frame_indices),
            "fixed_frame": self.fixed_frame_index,
            "optimized_frames": list(self.optimized_frame_indices),
            "edges_used": [f"{source}->{target}" for source, target in self.edges_used],
            "observations_per_edge": dict(self.observations_per_edge),
            "total_observation_count": self.total_observation_count,
            "solver_success": self.solver_success,
            "solver_message": self.solver_message,
            "solver_function_evaluations": self.solver_function_evaluations,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "reprojection_before": asdict(self.metrics_before),
            "reprojection_after": asdict(self.metrics_after),
            "relative_median_improvement": self.relative_median_improvement,
            "per_edge_before": {
                name: asdict(metrics) for name, metrics in self.per_edge_before.items()
            },
            "per_edge_after": {
                name: asdict(metrics) for name, metrics in self.per_edge_after.items()
            },
            "pose_changes": [asdict(change) for change in self.pose_changes],
            "optimization_accepted": self.accepted,
            "acceptance_reason": self.reason,
            "pose_convention": self.pose_convention,
            "coordinate_scale": self.coordinate_scale,
            "is_metric": False,
            "fixed_quantities": {
                "depth": self.fixed_depth,
                "intrinsics": self.fixed_intrinsics,
                "scale": self.fixed_scale,
                "first_pose": True,
            },
            "limitations": [
                "pose-only local optimization",
                "relative/non-metric depth and translation",
                "no landmark or depth optimization",
                "no loop closure or long-term drift correction",
                "lower reprojection error does not prove absolute accuracy",
            ],
        }


@dataclass(frozen=True)
class _PreparedEdge:
    source_frame_index: int
    target_frame_index: int
    source_points_camera: np.ndarray
    target_pixels: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.source_frame_index}->{self.target_frame_index}"

    @property
    def count(self) -> int:
        return int(self.source_points_camera.shape[0])


def _validate_pose(transform: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(transform, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return pose


def build_reprojection_edge(
    source_frame_index: int,
    target_frame_index: int,
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    supported_match_mask: np.ndarray,
    source_camera_depth: CameraDepth,
    camera_matrix: np.ndarray,
    *,
    sampling_method: str = "bilinear",
    geometric_support: str = "pnp_inliers",
) -> ReprojectionEdge:
    """Backproject supported source matches without modifying source depth."""

    source = np.asarray(source_pixels, dtype=np.float64)
    target = np.asarray(target_pixels, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("source_pixels and target_pixels must be equal Nx2 arrays")
    mask = np.asarray(supported_match_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != source.shape[0]:
        raise ValueError("supported_match_mask length must equal match count")
    geometry = DepthGeometryProcessor(sampling_method).process(
        source, mask, source_camera_depth, camera_matrix
    )
    target_selected = target[geometry.valid_match_indices]
    finite = np.isfinite(target_selected).all(axis=1)
    return ReprojectionEdge(
        source_frame_index=int(source_frame_index),
        target_frame_index=int(target_frame_index),
        source_points_camera=geometry.points_3d_relative[finite].copy(),
        target_pixels=target_selected[finite].copy(),
        geometric_support=str(geometric_support),
    )


def _pose_to_parameters(pose: np.ndarray) -> np.ndarray:
    validated = _validate_pose(pose, "world_from_camera pose")
    rotation_vector, _ = cv2.Rodrigues(validated[:3, :3])
    return np.concatenate((rotation_vector.reshape(3), validated[:3, 3]))


def _parameters_to_pose(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("pose parameters must be six finite values")
    rotation, _ = cv2.Rodrigues(values[:3])
    return make_transform(rotation, values[3:])


def _project_edge(
    edge: _PreparedEdge,
    poses: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    depth_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_pose = poses[edge.source_frame_index]
    target_pose = poses[edge.target_frame_index]
    source = edge.source_points_camera
    world = (source_pose[:3, :3] @ source.T).T + source_pose[:3, 3]
    target = (target_pose[:3, :3].T @ (world - target_pose[:3, 3]).T).T
    valid = np.isfinite(target).all(axis=1) & (target[:, 2] > depth_epsilon)
    projection = np.full((source.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = target[valid, 0] / target[valid, 2]
        y = target[valid, 1] / target[valid, 2]
        projection[valid, 0] = camera_matrix[0, 0] * x + camera_matrix[0, 2]
        projection[valid, 1] = camera_matrix[1, 1] * y + camera_matrix[1, 2]
    return projection, valid


def _reprojection_metrics(residual_xy: np.ndarray) -> ReprojectionMetrics:
    residuals = np.asarray(residual_xy, dtype=np.float64)
    if residuals.size == 0:
        return ReprojectionMetrics(0, None, None, None, None)
    norms = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
    return ReprojectionMetrics(
        observation_count=norms.size,
        median=float(np.median(norms)),
        mean=float(np.mean(norms)),
        rmse=float(np.sqrt(np.mean(norms ** 2))),
        p90=float(np.percentile(norms, 90.0)),
    )


def _robust_cost(residuals: np.ndarray, loss: str, scale: float) -> float:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    if loss == "huber":
        terms = np.where(
            absolute <= scale,
            0.5 * values ** 2,
            scale * (absolute - 0.5 * scale),
        )
    else:
        terms = scale ** 2 * (np.sqrt(1.0 + (values / scale) ** 2) - 1.0)
    return float(np.sum(terms))


def _robust_sqrt_weights(residuals: np.ndarray, loss: str, scale: float) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    if loss == "huber":
        weights = np.ones_like(values)
        outside = absolute > scale
        weights[outside] = scale / absolute[outside]
    else:
        weights = 1.0 / np.sqrt(1.0 + (values / scale) ** 2)
    return np.sqrt(weights)


def solve_numpy_least_squares(
    residual_function: Callable[[np.ndarray], np.ndarray],
    initial_parameters: np.ndarray,
    config: SlidingWindowPoseOptimizationConfig,
) -> SolverResult:
    """Deterministic finite-difference LM with Huber/soft-L1 IRLS weights."""

    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    residuals = np.asarray(residual_function(parameters), dtype=np.float64).reshape(-1)
    evaluations = 1
    if residuals.size == 0 or not np.isfinite(residuals).all():
        return SolverResult(False, "initial residuals are invalid", parameters, float("inf"), evaluations)
    cost = _robust_cost(residuals, config.robust_loss, config.f_scale_px)
    if np.max(np.abs(residuals)) < 1e-10:
        return SolverResult(True, "initial solution already has zero residual", parameters, cost, evaluations)

    damping = 1e-3
    accepted_steps = 0
    converged = False
    message = "maximum evaluation budget reached"
    variable_count = parameters.size
    while evaluations + variable_count + 1 <= config.max_nfev:
        jacobian = np.empty((residuals.size, variable_count), dtype=np.float64)
        for column in range(variable_count):
            step = config.finite_difference_step * max(1.0, abs(parameters[column]))
            shifted = parameters.copy()
            shifted[column] += step
            shifted_residuals = np.asarray(
                residual_function(shifted), dtype=np.float64
            ).reshape(-1)
            evaluations += 1
            if shifted_residuals.shape != residuals.shape or not np.isfinite(
                shifted_residuals
            ).all():
                return SolverResult(
                    False, "finite-difference residuals are invalid",
                    parameters, cost, evaluations,
                )
            jacobian[:, column] = (shifted_residuals - residuals) / step

        sqrt_weights = _robust_sqrt_weights(
            residuals, config.robust_loss, config.f_scale_px
        )
        weighted_jacobian = jacobian * sqrt_weights[:, None]
        weighted_residuals = residuals * sqrt_weights
        normal = weighted_jacobian.T @ weighted_jacobian
        diagonal = np.maximum(np.diag(normal), 1.0)
        system = normal + damping * np.diag(diagonal)
        gradient = weighted_jacobian.T @ weighted_residuals
        try:
            delta = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(system, -gradient, rcond=None)
        if not np.isfinite(delta).all():
            return SolverResult(False, "solver produced a non-finite step", parameters, cost, evaluations)

        candidate = parameters + delta
        candidate_residuals = np.asarray(
            residual_function(candidate), dtype=np.float64
        ).reshape(-1)
        evaluations += 1
        if not np.isfinite(candidate_residuals).all():
            damping *= 10.0
            continue
        candidate_cost = _robust_cost(
            candidate_residuals, config.robust_loss, config.f_scale_px
        )
        if candidate_cost < cost:
            relative_cost_change = (cost - candidate_cost) / max(cost, 1.0)
            parameters = candidate
            residuals = candidate_residuals
            cost = candidate_cost
            accepted_steps += 1
            damping = max(damping / 3.0, 1e-12)
            if np.linalg.norm(delta) < 1e-9 or relative_cost_change < 1e-10:
                converged = True
                message = "converged"
                break
        else:
            damping = min(damping * 10.0, 1e12)

    success = converged or accepted_steps > 0
    if success and not converged:
        message = "evaluation budget reached after valid descent"
    elif not success:
        message = "solver found no finite descent step"
    return SolverResult(success, message, parameters, cost, evaluations)


def _rotation_change_degrees(before: np.ndarray, after: np.ndarray) -> float:
    difference = after @ before.T
    cosine = np.clip((np.trace(difference) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class SlidingWindowPoseOptimizer:
    """Optimize later ``T_world_from_camera`` poses while fixing the first."""

    def __init__(
        self,
        config: SlidingWindowPoseOptimizationConfig | None = None,
        solver: Callable[
            [Callable[[np.ndarray], np.ndarray], np.ndarray, SlidingWindowPoseOptimizationConfig],
            SolverResult,
        ] | None = None,
    ) -> None:
        self.config = config or SlidingWindowPoseOptimizationConfig()
        self.solver = solver or solve_numpy_least_squares

    def _fallback(
        self,
        reason: str,
        frames: list[PoseWindowFrame],
        *,
        attempted: bool,
        edges: list[_PreparedEdge] | None = None,
        solver_result: SolverResult | None = None,
        before_residuals: np.ndarray | None = None,
        after_residuals: np.ndarray | None = None,
        before_per_edge: dict[str, ReprojectionMetrics] | None = None,
        after_per_edge: dict[str, ReprojectionMetrics] | None = None,
        pose_changes: tuple[PoseChangeDiagnostics, ...] = (),
        proposed: tuple[np.ndarray, ...] | None = None,
    ) -> SlidingWindowPoseOptimizationResult:
        baseline = tuple(frame.world_from_camera.copy() for frame in frames)
        prepared = edges or []
        before = np.empty((0, 2)) if before_residuals is None else before_residuals
        after = before if after_residuals is None else after_residuals
        before_metrics = _reprojection_metrics(before)
        after_metrics = _reprojection_metrics(after)
        improvement = None
        if (
            before_metrics.median is not None
            and after_metrics.median is not None
            and before_metrics.median > np.finfo(float).eps
        ):
            improvement = (
                before_metrics.median - after_metrics.median
            ) / before_metrics.median
        return SlidingWindowPoseOptimizationResult(
            attempted=attempted,
            accepted=False,
            reason=reason,
            window_frame_indices=tuple(frame.frame_index for frame in frames),
            fixed_frame_index=(None if not frames else frames[0].frame_index),
            optimized_frame_indices=tuple(frame.frame_index for frame in frames[1:]),
            edges_used=tuple((edge.source_frame_index, edge.target_frame_index) for edge in prepared),
            observations_per_edge={edge.name: edge.count for edge in prepared},
            total_observation_count=sum(edge.count for edge in prepared),
            solver_success=False if solver_result is None else solver_result.success,
            solver_message=(reason if solver_result is None else solver_result.message),
            solver_function_evaluations=(0 if solver_result is None else solver_result.function_evaluations),
            cost_before=(None if before_residuals is None else _robust_cost(before.reshape(-1), self.config.robust_loss, self.config.f_scale_px)),
            cost_after=(None if after_residuals is None else _robust_cost(after.reshape(-1), self.config.robust_loss, self.config.f_scale_px)),
            metrics_before=before_metrics,
            metrics_after=after_metrics,
            relative_median_improvement=improvement,
            per_edge_before=before_per_edge or {},
            per_edge_after=after_per_edge or {},
            pose_changes=pose_changes,
            baseline_world_from_camera=baseline,
            proposed_world_from_camera=baseline if proposed is None else proposed,
            selected_world_from_camera=baseline,
        )

    def optimize(
        self,
        frames: list[PoseWindowFrame] | tuple[PoseWindowFrame, ...],
        edges: list[ReprojectionEdge] | tuple[ReprojectionEdge, ...],
        camera_matrix: np.ndarray,
        image_size: tuple[int, int],
    ) -> SlidingWindowPoseOptimizationResult:
        frame_list = list(frames)
        if not self.config.enabled:
            return self._fallback(
                "sliding_window_pose_optimization_disabled", frame_list,
                attempted=False,
            )
        if len(frame_list) < 2:
            return self._fallback(
                "sliding_window_insufficient_frames", frame_list, attempted=True
            )
        if len(frame_list) > self.config.window_size:
            frame_list = frame_list[-self.config.window_size:]
        intrinsics = validate_camera_matrix(camera_matrix)
        width, height = (int(image_size[0]), int(image_size[1]))
        if width < 1 or height < 1:
            raise ValueError("image_size must contain positive width and height")
        frame_indices = {frame.frame_index for frame in frame_list}
        baseline_by_index: dict[int, np.ndarray] = {}
        for frame in frame_list:
            baseline_by_index[frame.frame_index] = _validate_pose(
                frame.world_from_camera, f"frame {frame.frame_index} pose"
            ).copy()

        prepared: list[_PreparedEdge] = []
        for edge in edges:
            if edge.source_frame_index not in frame_indices or edge.target_frame_index not in frame_indices:
                continue
            source = np.asarray(edge.source_points_camera, dtype=np.float64)
            target = np.asarray(edge.target_pixels, dtype=np.float64)
            if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != (source.shape[0], 2):
                raise ValueError("edge points must be Nx3 source and Nx2 target arrays")
            base_valid = (
                np.isfinite(source).all(axis=1)
                & (source[:, 2] > 0.0)
                & np.isfinite(target).all(axis=1)
                & (target[:, 0] >= 0.0)
                & (target[:, 0] < width)
                & (target[:, 1] >= 0.0)
                & (target[:, 1] < height)
            )
            candidate = _PreparedEdge(
                edge.source_frame_index,
                edge.target_frame_index,
                source[base_valid].copy(),
                target[base_valid].copy(),
            )
            if candidate.count == 0:
                continue
            _, projected_valid = _project_edge(
                candidate,
                baseline_by_index,
                intrinsics,
                self.config.projection_depth_epsilon,
            )
            candidate = _PreparedEdge(
                candidate.source_frame_index,
                candidate.target_frame_index,
                candidate.source_points_camera[projected_valid],
                candidate.target_pixels[projected_valid],
            )
            if candidate.count:
                prepared.append(candidate)

        total = sum(edge.count for edge in prepared)
        if total < self.config.minimum_observations:
            return self._fallback(
                "sliding_window_insufficient_observations", frame_list,
                attempted=True, edges=prepared,
            )

        optimized_frames = frame_list[1:]
        initial_parameters = np.concatenate([
            _pose_to_parameters(frame.world_from_camera) for frame in optimized_frames
        ])

        def poses_from_parameters(parameters: np.ndarray) -> dict[int, np.ndarray]:
            values = np.asarray(parameters, dtype=np.float64).reshape(-1)
            if values.size != 6 * len(optimized_frames):
                raise ValueError("optimizer parameter count does not match window")
            poses = {index: pose.copy() for index, pose in baseline_by_index.items()}
            for offset, frame in enumerate(optimized_frames):
                poses[frame.frame_index] = _parameters_to_pose(
                    values[offset * 6:(offset + 1) * 6]
                )
            return poses

        invalid_penalty = self.config.f_scale_px * 100.0

        def residual_blocks(parameters: np.ndarray) -> list[np.ndarray]:
            poses = poses_from_parameters(parameters)
            blocks: list[np.ndarray] = []
            for edge in prepared:
                projected, valid = _project_edge(
                    edge, poses, intrinsics, self.config.projection_depth_epsilon
                )
                residual = edge.target_pixels - projected
                residual[~valid] = invalid_penalty
                residual[~np.isfinite(residual)] = invalid_penalty
                blocks.append(residual)
            return blocks

        def residual_function(parameters: np.ndarray) -> np.ndarray:
            return np.concatenate(residual_blocks(parameters), axis=0).reshape(-1)

        before_blocks = residual_blocks(initial_parameters)
        before_residuals = np.concatenate(before_blocks, axis=0)
        before_per_edge = {
            edge.name: _reprojection_metrics(block)
            for edge, block in zip(prepared, before_blocks)
        }
        try:
            solver_result = self.solver(
                residual_function, initial_parameters, self.config
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            return self._fallback(
                "sliding_window_solver_failed", frame_list, attempted=True,
                edges=prepared, before_residuals=before_residuals,
                before_per_edge=before_per_edge,
                solver_result=SolverResult(
                    False, str(exc), initial_parameters,
                    _robust_cost(before_residuals.reshape(-1), self.config.robust_loss, self.config.f_scale_px),
                    0,
                ),
            )
        if not solver_result.success or not np.isfinite(solver_result.parameters).all():
            return self._fallback(
                "sliding_window_solver_failed", frame_list, attempted=True,
                edges=prepared, solver_result=solver_result,
                before_residuals=before_residuals,
                before_per_edge=before_per_edge,
            )

        try:
            proposed_by_index = poses_from_parameters(solver_result.parameters)
        except ValueError:
            return self._fallback(
                "sliding_window_invalid_pose", frame_list, attempted=True,
                edges=prepared, solver_result=solver_result,
                before_residuals=before_residuals,
                before_per_edge=before_per_edge,
            )
        proposed = tuple(proposed_by_index[frame.frame_index] for frame in frame_list)
        after_blocks = residual_blocks(solver_result.parameters)
        after_residuals = np.concatenate(after_blocks, axis=0)
        after_per_edge = {
            edge.name: _reprojection_metrics(block)
            for edge, block in zip(prepared, after_blocks)
        }
        before_metrics = _reprojection_metrics(before_residuals)
        after_metrics = _reprojection_metrics(after_residuals)
        assert before_metrics.median is not None and after_metrics.median is not None
        improvement = (
            before_metrics.median - after_metrics.median
        ) / max(before_metrics.median, np.finfo(float).eps)

        changes: list[PoseChangeDiagnostics] = []
        for index, frame in enumerate(frame_list[1:], start=1):
            before_pose = baseline_by_index[frame.frame_index]
            after_pose = proposed_by_index[frame.frame_index]
            previous_before = baseline_by_index[frame_list[index - 1].frame_index]
            original_step = float(np.linalg.norm(
                before_pose[:3, 3] - previous_before[:3, 3]
            ))
            translation_change = float(np.linalg.norm(
                after_pose[:3, 3] - before_pose[:3, 3]
            ))
            changes.append(PoseChangeDiagnostics(
                frame_index=frame.frame_index,
                rotation_change_deg=_rotation_change_degrees(
                    before_pose[:3, :3], after_pose[:3, :3]
                ),
                translation_change=translation_change,
                original_step_magnitude=original_step,
                translation_change_relative=(
                    translation_change / max(original_step, np.finfo(float).eps)
                ),
                camera_center_before=tuple(float(v) for v in before_pose[:3, 3]),
                camera_center_after=tuple(float(v) for v in after_pose[:3, 3]),
            ))

        common = dict(
            attempted=True,
            edges=prepared,
            solver_result=solver_result,
            before_residuals=before_residuals,
            after_residuals=after_residuals,
            before_per_edge=before_per_edge,
            after_per_edge=after_per_edge,
            pose_changes=tuple(changes),
            proposed=proposed,
        )
        if improvement <= 0.0:
            return self._fallback(
                "sliding_window_median_not_improved", frame_list, **common
            )
        if improvement < self.config.minimum_relative_median_improvement:
            return self._fallback(
                "sliding_window_improvement_below_threshold", frame_list, **common
            )
        if any(
            change.rotation_change_deg > self.config.maximum_rotation_change_deg
            for change in changes
        ):
            return self._fallback(
                "sliding_window_excessive_rotation_change", frame_list, **common
            )
        if any(
            change.translation_change_relative
            > self.config.maximum_translation_change_relative
            for change in changes
        ):
            return self._fallback(
                "sliding_window_excessive_translation_change", frame_list, **common
            )

        baseline = tuple(frame.world_from_camera.copy() for frame in frame_list)
        return SlidingWindowPoseOptimizationResult(
            attempted=True,
            accepted=True,
            reason="sliding_window_pose_optimization_accepted",
            window_frame_indices=tuple(frame.frame_index for frame in frame_list),
            fixed_frame_index=frame_list[0].frame_index,
            optimized_frame_indices=tuple(frame.frame_index for frame in frame_list[1:]),
            edges_used=tuple((edge.source_frame_index, edge.target_frame_index) for edge in prepared),
            observations_per_edge={edge.name: edge.count for edge in prepared},
            total_observation_count=total,
            solver_success=True,
            solver_message=solver_result.message,
            solver_function_evaluations=solver_result.function_evaluations,
            cost_before=_robust_cost(before_residuals.reshape(-1), self.config.robust_loss, self.config.f_scale_px),
            cost_after=_robust_cost(after_residuals.reshape(-1), self.config.robust_loss, self.config.f_scale_px),
            metrics_before=before_metrics,
            metrics_after=after_metrics,
            relative_median_improvement=improvement,
            per_edge_before=before_per_edge,
            per_edge_after=after_per_edge,
            pose_changes=tuple(changes),
            baseline_world_from_camera=baseline,
            proposed_world_from_camera=proposed,
            selected_world_from_camera=proposed,
        )
