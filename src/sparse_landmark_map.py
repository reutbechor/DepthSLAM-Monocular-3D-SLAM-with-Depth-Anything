"""Optional sparse multi-view landmark mapping from accepted keyframes.

The dense mapper remains the source of camera poses.  This module reuses
geometrically verified SIFT associations, builds persistent feature tracks,
and triangulates landmarks in the same relative, non-metric world frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import json
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ply_io import write_ascii_ply


@dataclass(frozen=True)
class SparseLandmarkConfig:
    enabled: bool = False
    minimum_track_length: int = 2
    minimum_triangulation_angle_deg: float = 1.0
    maximum_reprojection_error_px: float = 3.0
    minimum_observations_after_validation: int = 2
    maximum_landmark_distance_percentile: float | None = 99.5

    def __post_init__(self) -> None:
        if self.minimum_track_length < 2:
            raise ValueError("minimum_track_length must be at least 2")
        if self.minimum_observations_after_validation < 2:
            raise ValueError("minimum_observations_after_validation must be at least 2")
        if self.minimum_triangulation_angle_deg < 0.0:
            raise ValueError("minimum_triangulation_angle_deg must be non-negative")
        if self.maximum_reprojection_error_px <= 0.0:
            raise ValueError("maximum_reprojection_error_px must be positive")
        if (
            self.maximum_landmark_distance_percentile is not None
            and not 0.0 < self.maximum_landmark_distance_percentile <= 100.0
        ):
            raise ValueError(
                "maximum_landmark_distance_percentile must be in (0, 100] or null"
            )


@dataclass(frozen=True)
class SparseKeyframe:
    frame_index: int
    image_bgr: np.ndarray
    world_from_camera: np.ndarray


@dataclass(frozen=True)
class PairwiseLandmarkMatches:
    previous_frame_index: int
    current_frame_index: int
    previous_keypoint_indices: np.ndarray
    current_keypoint_indices: np.ndarray
    previous_pixels: np.ndarray
    current_pixels: np.ndarray
    previous_keypoint_count: int
    current_keypoint_count: int

    def __post_init__(self) -> None:
        count = np.asarray(self.previous_keypoint_indices).reshape(-1).shape[0]
        if any(
            value.shape[0] != count
            for value in (
                np.asarray(self.current_keypoint_indices).reshape(-1),
                np.asarray(self.previous_pixels),
                np.asarray(self.current_pixels),
            )
        ):
            raise ValueError("pairwise sparse-match arrays must have equal length")


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    keypoint_index: int
    pixel: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "keypoint_index": self.keypoint_index,
            "pixel": list(self.pixel),
        }


@dataclass(frozen=True)
class FeatureTrack:
    track_id: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class SparseLandmark:
    landmark_id: int
    track_id: int
    world_xyz: tuple[float, float, float] | None
    color_rgb: tuple[int, int, int] | None
    observations: tuple[TrackObservation, ...]
    reprojection_errors_px: tuple[float, ...]
    reprojection_error_median_px: float | None
    reprojection_error_rmse_px: float | None
    triangulation_angle_deg: float | None
    triangulation_frame_indices: tuple[int, int] | None
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "track_id": self.track_id,
            "world_xyz": None if self.world_xyz is None else list(self.world_xyz),
            "color_rgb": None if self.color_rgb is None else list(self.color_rgb),
            "observation_count": len(self.observations),
            "source_frame_indices": [item.frame_index for item in self.observations],
            "observations": [item.to_dict() for item in self.observations],
            "reprojection_errors_px": list(self.reprojection_errors_px),
            "reprojection_error_median_px": self.reprojection_error_median_px,
            "reprojection_error_rmse_px": self.reprojection_error_rmse_px,
            "triangulation_angle_deg": self.triangulation_angle_deg,
            "triangulation_frame_indices": (
                None
                if self.triangulation_frame_indices is None
                else list(self.triangulation_frame_indices)
            ),
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class SparseLandmarkResult:
    enabled: bool
    points: np.ndarray
    colors: np.ndarray
    tracks: tuple[FeatureTrack, ...]
    landmarks: tuple[SparseLandmark, ...]
    diagnostics: dict[str, Any]

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "map_representation": "sparse_multi_view_feature_geometry",
            "coordinate_scale": "relative_non_metric",
            "is_metric": False,
            "color_method": "median_rgb_across_track_observations",
            **self.diagnostics,
        }


def projection_matrix(
    camera_matrix: np.ndarray, world_from_camera: np.ndarray
) -> np.ndarray:
    """Return K [R|t] after converting stored world-from-camera to camera-from-world."""
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("camera matrix must be 3x3 and pose must be 4x4")
    camera_from_world = np.linalg.inv(pose)
    return intrinsic @ camera_from_world[:3, :]


def build_feature_tracks(
    pairwise_matches: Iterable[PairwiseLandmarkMatches],
) -> tuple[tuple[FeatureTrack, ...], int]:
    """Build deterministic tracks while rejecting associations with frame conflicts."""
    tracks: dict[int, dict[tuple[int, int], TrackObservation]] = {}
    assignment: dict[tuple[int, int], int] = {}
    rejected_conflicts = 0
    next_id = 0

    pairs = sorted(
        pairwise_matches,
        key=lambda item: (item.previous_frame_index, item.current_frame_index),
    )
    for pair in pairs:
        rows = zip(
            np.asarray(pair.previous_keypoint_indices, dtype=np.int64).reshape(-1),
            np.asarray(pair.current_keypoint_indices, dtype=np.int64).reshape(-1),
            np.asarray(pair.previous_pixels, dtype=np.float64),
            np.asarray(pair.current_pixels, dtype=np.float64),
        )
        for previous_kp, current_kp, previous_pixel, current_pixel in rows:
            first_key = (pair.previous_frame_index, int(previous_kp))
            second_key = (pair.current_frame_index, int(current_kp))
            first = TrackObservation(
                first_key[0], first_key[1], tuple(map(float, previous_pixel))
            )
            second = TrackObservation(
                second_key[0], second_key[1], tuple(map(float, current_pixel))
            )
            first_track = assignment.get(first_key)
            second_track = assignment.get(second_key)

            if first_track is None and second_track is None:
                tracks[next_id] = {first_key: first, second_key: second}
                assignment[first_key] = assignment[second_key] = next_id
                next_id += 1
                continue
            if first_track is not None and second_track is None:
                target = tracks[first_track]
                if any(obs.frame_index == second.frame_index for obs in target.values()):
                    rejected_conflicts += 1
                    continue
                target[second_key] = second
                assignment[second_key] = first_track
                continue
            if first_track is None and second_track is not None:
                target = tracks[second_track]
                if any(obs.frame_index == first.frame_index for obs in target.values()):
                    rejected_conflicts += 1
                    continue
                target[first_key] = first
                assignment[first_key] = second_track
                continue
            if first_track == second_track:
                continue

            assert first_track is not None and second_track is not None
            winner, loser = sorted((first_track, second_track))
            winner_frames = {obs.frame_index for obs in tracks[winner].values()}
            loser_frames = {obs.frame_index for obs in tracks[loser].values()}
            if winner_frames & loser_frames:
                rejected_conflicts += 1
                continue
            for key, observation in tracks[loser].items():
                tracks[winner][key] = observation
                assignment[key] = winner
            del tracks[loser]

    output = []
    for track_id in sorted(tracks):
        observations = tuple(
            sorted(
                tracks[track_id].values(),
                key=lambda item: (item.frame_index, item.keypoint_index),
            )
        )
        output.append(FeatureTrack(track_id, observations))
    return tuple(output), rejected_conflicts


def _camera_center(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64)[:3, 3]


def _largest_baseline_pair(
    observations: tuple[TrackObservation, ...],
    frames: dict[int, SparseKeyframe],
) -> tuple[TrackObservation, TrackObservation]:
    candidates: list[tuple[float, int, int, TrackObservation, TrackObservation]] = []
    for i, first in enumerate(observations[:-1]):
        for second in observations[i + 1 :]:
            baseline = np.linalg.norm(
                _camera_center(frames[first.frame_index].world_from_camera)
                - _camera_center(frames[second.frame_index].world_from_camera)
            )
            candidates.append(
                (float(baseline), -first.frame_index, -second.frame_index, first, second)
            )
    if not candidates:
        raise ValueError("at least two observations are required")
    return max(candidates, key=lambda item: item[:3])[3:]


def triangulate_observations(
    first: TrackObservation,
    second: TrackObservation,
    frames: dict[int, SparseKeyframe],
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    p1 = projection_matrix(
        camera_matrix, frames[first.frame_index].world_from_camera
    )
    p2 = projection_matrix(
        camera_matrix, frames[second.frame_index].world_from_camera
    )
    homogeneous = cv2.triangulatePoints(
        p1,
        p2,
        np.asarray(first.pixel, dtype=np.float64).reshape(2, 1),
        np.asarray(second.pixel, dtype=np.float64).reshape(2, 1),
    ).reshape(4)
    if not np.all(np.isfinite(homogeneous)) or abs(homogeneous[3]) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.all(np.isfinite(point)) else None


def _camera_point(point_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    camera_from_world = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    return camera_from_world[:3, :3] @ point_world + camera_from_world[:3, 3]


def _reproject(
    point_world: np.ndarray, pose: np.ndarray, camera_matrix: np.ndarray
) -> tuple[np.ndarray, float]:
    camera_point = _camera_point(point_world, pose)
    projected = np.asarray(camera_matrix, dtype=np.float64) @ camera_point
    return projected[:2] / projected[2], float(camera_point[2])


def triangulation_angle_degrees(
    point_world: np.ndarray, first_pose: np.ndarray, second_pose: np.ndarray
) -> float:
    ray1 = point_world - _camera_center(first_pose)
    ray2 = point_world - _camera_center(second_pose)
    norm = np.linalg.norm(ray1) * np.linalg.norm(ray2)
    if norm <= 1e-15:
        return 0.0
    cosine = float(np.clip(np.dot(ray1, ray2) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def landmark_color_rgb(
    observations: tuple[TrackObservation, ...],
    frames: dict[int, SparseKeyframe],
) -> tuple[int, int, int]:
    """Use the channel-wise median RGB color across in-bounds observations."""
    samples: list[np.ndarray] = []
    for observation in observations:
        image = frames[observation.frame_index].image_bgr
        x = int(round(observation.pixel[0]))
        y = int(round(observation.pixel[1]))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            samples.append(image[y, x, ::-1].astype(np.float64))
    if not samples:
        return (255, 255, 255)
    color = np.rint(np.median(np.vstack(samples), axis=0)).astype(np.uint8)
    return tuple(int(value) for value in color)


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "rmse": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "p90": float(np.percentile(array, 90.0)),
    }


class SparseLandmarkMapper:
    def __init__(
        self, camera_matrix: np.ndarray, config: SparseLandmarkConfig | None = None
    ) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.config = config or SparseLandmarkConfig()

    def build(
        self,
        keyframes: Iterable[SparseKeyframe],
        pairwise_matches: Iterable[PairwiseLandmarkMatches],
    ) -> SparseLandmarkResult:
        if not self.config.enabled:
            return SparseLandmarkResult(
                False,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.uint8),
                (),
                (),
                {},
            )
        frames = {item.frame_index: item for item in keyframes}
        pairs = tuple(pairwise_matches)
        tracks, conflicts = build_feature_tracks(pairs)
        keypoint_counts: dict[int, int] = {}
        for pair in pairs:
            keypoint_counts[pair.previous_frame_index] = pair.previous_keypoint_count
            keypoint_counts[pair.current_frame_index] = pair.current_keypoint_count

        landmarks: list[SparseLandmark] = []
        rejection_counts = {
            "rejected_insufficient_track_length": 0,
            "rejected_invalid_triangulation": 0,
            "rejected_behind_camera": 0,
            "rejected_low_angle": 0,
            "rejected_reprojection": 0,
            "rejected_far_distance": 0,
        }
        candidate_indices: list[int] = []
        for track in tracks:
            observations = tuple(
                item for item in track.observations if item.frame_index in frames
            )
            status = "accepted"
            point: np.ndarray | None = None
            errors: list[float] = []
            angle: float | None = None
            pair_indices: tuple[int, int] | None = None
            color: tuple[int, int, int] | None = None
            if len(observations) < max(
                self.config.minimum_track_length,
                self.config.minimum_observations_after_validation,
            ):
                status = "rejected_insufficient_track_length"
            else:
                first, second = _largest_baseline_pair(observations, frames)
                pair_indices = (first.frame_index, second.frame_index)
                point = triangulate_observations(
                    first, second, frames, self.camera_matrix
                )
                if point is None:
                    status = "rejected_invalid_triangulation"
                else:
                    for observation in observations:
                        pixel, depth = _reproject(
                            point,
                            frames[observation.frame_index].world_from_camera,
                            self.camera_matrix,
                        )
                        if not np.isfinite(depth) or depth <= 1e-8:
                            status = "rejected_behind_camera"
                            break
                        errors.append(float(np.linalg.norm(
                            pixel - np.asarray(observation.pixel, dtype=np.float64)
                        )))
                    if status == "accepted":
                        angle = triangulation_angle_degrees(
                            point,
                            frames[first.frame_index].world_from_camera,
                            frames[second.frame_index].world_from_camera,
                        )
                        if angle < self.config.minimum_triangulation_angle_deg:
                            status = "rejected_low_angle"
                        elif (
                            not errors
                            or max(errors)
                            > self.config.maximum_reprojection_error_px
                        ):
                            status = "rejected_reprojection"
                        else:
                            color = landmark_color_rgb(observations, frames)
            rejection_counts[status] = rejection_counts.get(status, 0) + (
                0 if status == "accepted" else 1
            )
            median = None if not errors else float(np.median(errors))
            rmse = None if not errors else float(np.sqrt(np.mean(np.square(errors))))
            landmark = SparseLandmark(
                landmark_id=len(landmarks),
                track_id=track.track_id,
                world_xyz=None if point is None else tuple(map(float, point)),
                color_rgb=color,
                observations=observations,
                reprojection_errors_px=tuple(errors),
                reprojection_error_median_px=median,
                reprojection_error_rmse_px=rmse,
                triangulation_angle_deg=angle,
                triangulation_frame_indices=pair_indices,
                validation_status=status,
            )
            landmarks.append(landmark)
            if status == "accepted":
                candidate_indices.append(len(landmarks) - 1)

        if candidate_indices and self.config.maximum_landmark_distance_percentile is not None:
            candidate_points = np.asarray(
                [landmarks[index].world_xyz for index in candidate_indices],
                dtype=np.float64,
            )
            camera_center = np.median(
                np.vstack([_camera_center(item.world_from_camera) for item in frames.values()]),
                axis=0,
            )
            distances = np.linalg.norm(candidate_points - camera_center, axis=1)
            limit = float(np.percentile(
                distances, self.config.maximum_landmark_distance_percentile
            ))
            for index, distance in zip(candidate_indices, distances):
                if distance > limit:
                    landmarks[index] = replace(
                        landmarks[index], validation_status="rejected_far_distance"
                    )
                    rejection_counts["rejected_far_distance"] += 1

        accepted = [item for item in landmarks if item.validation_status == "accepted"]
        points = np.asarray([item.world_xyz for item in accepted], dtype=np.float64).reshape(-1, 3)
        colors = np.asarray([item.color_rgb for item in accepted], dtype=np.uint8).reshape(-1, 3)
        all_errors = [value for item in accepted for value in item.reprojection_errors_px]
        angles = [
            float(item.triangulation_angle_deg)
            for item in accepted
            if item.triangulation_angle_deg is not None
        ]
        lengths = [len(item.observations) for item in tracks]
        observation_counts = [len(item.observations) for item in accepted]
        angle_summary = {
            "min": None,
            "median": None,
            "p90": None,
        }
        if angles:
            angle_summary = {
                "min": float(np.min(angles)),
                "median": float(np.median(angles)),
                "p90": float(np.percentile(angles, 90.0)),
            }
        diagnostics: dict[str, Any] = {
            "accepted_keyframes": len(frames),
            "total_sift_keypoints": int(sum(keypoint_counts.values())),
            "pairwise_reliable_matches": int(sum(
                np.asarray(item.previous_keypoint_indices).size for item in pairs
            )),
            "pairwise_reliable_matches_by_pair": {
                f"{item.previous_frame_index}->{item.current_frame_index}": int(
                    np.asarray(item.previous_keypoint_indices).size
                )
                for item in pairs
            },
            "track_count": len(tracks),
            "track_conflict_rejection_count": conflicts,
            "track_length_distribution": {
                "2": sum(length == 2 for length in lengths),
                "3": sum(length == 3 for length in lengths),
                "4+": sum(length >= 4 for length in lengths),
            },
            "candidate_landmarks": sum(
                length >= max(
                    self.config.minimum_track_length,
                    self.config.minimum_observations_after_validation,
                )
                for length in lengths
            ),
            **rejection_counts,
            "final_landmark_count": len(accepted),
            "reprojection_error_px": _metric_summary(all_errors),
            "triangulation_angle_deg": angle_summary,
            "landmark_observation_count": {
                "min": None if not observation_counts else int(min(observation_counts)),
                "median": None if not observation_counts else float(np.median(observation_counts)),
                "max": None if not observation_counts else int(max(observation_counts)),
            },
            "configuration": asdict(self.config),
        }
        return SparseLandmarkResult(
            True, points, colors, tracks, tuple(landmarks), diagnostics
        )


def _equal_axes(axis: plt.Axes, values: np.ndarray) -> None:
    if values.size == 0:
        return
    lower, upper = np.min(values, axis=0), np.max(values, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _draw_sparse(
    axis: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray | None,
    view: str,
) -> None:
    if points.shape[0]:
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors.astype(np.float64) / 255.0, s=5, linewidths=0,
        )
    combined = points
    if trajectory is not None and trajectory.shape[0]:
        axis.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            "-o", color="#e63946", markersize=3, linewidth=1.5,
        )
        combined = trajectory if not points.shape[0] else np.vstack((points, trajectory))
    elevation, azimuth = {
        "front": (0.0, -90.0), "oblique": (24.0, -58.0), "top": (90.0, -90.0)
    }[view]
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("X (relative)")
    axis.set_ylabel("Y (relative)")
    axis.set_zlabel("Z (relative)")
    axis.grid(True, alpha=0.2)
    _equal_axes(axis, combined)


def save_sparse_landmark_outputs(
    directory: str | Path,
    result: SparseLandmarkResult,
    trajectory_positions: np.ndarray,
    representative_image_bgr: np.ndarray,
    dense_points: np.ndarray,
    dense_colors: np.ndarray,
) -> dict[str, Path]:
    """Save scientific arrays, track records, and sparse/dense comparison views."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "ply": output / "sparse_landmarks.ply",
        "points_npy": output / "sparse_landmarks.npy",
        "colors_npy": output / "sparse_landmark_colors.npy",
        "tracks_json": output / "sparse_landmark_tracks.json",
        "front": output / "sparse_map_front.png",
        "oblique": output / "sparse_map_oblique.png",
        "top": output / "sparse_map_top.png",
        "with_trajectory": output / "sparse_map_with_trajectory_3d.png",
        "dense_vs_sparse": output / "dense_vs_sparse_overview.png",
    }
    write_ascii_ply(paths["ply"], result.points, result.colors)
    np.save(paths["points_npy"], result.points)
    np.save(paths["colors_npy"], result.colors)
    with paths["tracks_json"].open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metadata": result.metadata_dict(),
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "observation_count": len(track.observations),
                        "observations": [item.to_dict() for item in track.observations],
                    }
                    for track in result.tracks
                ],
                "landmarks": [item.to_dict() for item in result.landmarks],
            },
            file,
            indent=2,
        )

    trajectory = np.asarray(trajectory_positions, dtype=np.float64)
    for view in ("front", "oblique", "top"):
        figure = plt.figure(figsize=(8, 7), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        _draw_sparse(axis, result.points, result.colors, trajectory, view)
        axis.set_title("Sparse multi-view landmark map + trajectory (relative, non-metric)")
        figure.savefig(paths[view], dpi=150)
        plt.close(figure)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    _draw_sparse(axis, result.points, result.colors, trajectory, "oblique")
    axis.set_title("Sparse landmarks with camera trajectory (relative, non-metric)")
    figure.savefig(paths["with_trajectory"], dpi=150)
    plt.close(figure)

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    rgb_axis = figure.add_subplot(2, 2, 1)
    rgb_axis.imshow(np.asarray(representative_image_bgr)[:, :, ::-1])
    rgb_axis.set_title("Representative RGB frame")
    rgb_axis.axis("off")
    dense_axis = figure.add_subplot(2, 2, 2, projection="3d")
    dense = np.asarray(dense_points)
    dense_rgb = np.asarray(dense_colors)
    if dense.shape[0]:
        indices = np.linspace(0, dense.shape[0] - 1, min(20_000, dense.shape[0]), dtype=int)
        dense_axis.scatter(
            dense[indices, 0], dense[indices, 1], dense[indices, 2],
            c=dense_rgb[indices].astype(np.float64) / 255.0, s=1, linewidths=0,
        )
    dense_axis.view_init(elev=24.0, azim=-58.0)
    dense_axis.set_title("Dense relative-depth map")
    sparse_axis = figure.add_subplot(2, 2, 3, projection="3d")
    _draw_sparse(sparse_axis, result.points, result.colors, None, "oblique")
    sparse_axis.set_title("Sparse multi-view landmark map")
    combined_axis = figure.add_subplot(2, 2, 4, projection="3d")
    _draw_sparse(combined_axis, result.points, result.colors, trajectory, "oblique")
    combined_axis.set_title("Sparse landmarks + camera trajectory")
    figure.suptitle("Dense vs sparse reconstruction (relative geometry, non-metric)")
    figure.savefig(paths["dense_vs_sparse"], dpi=150)
    plt.close(figure)
    return paths
