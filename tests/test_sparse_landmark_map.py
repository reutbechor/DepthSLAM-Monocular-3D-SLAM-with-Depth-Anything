import json

import cv2
import numpy as np
import pytest

from src.sparse_landmark_map import (
    PairwiseLandmarkMatches,
    SparseKeyframe,
    SparseLandmarkConfig,
    SparseLandmarkMapper,
    TrackObservation,
    build_feature_tracks,
    landmark_color_rgb,
    projection_matrix,
    save_sparse_landmark_outputs,
    triangulate_observations,
)


K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def pose(center_x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = center_x
    return value


def project(point: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    homogeneous = np.append(point, 1.0)
    image = projection_matrix(K, world_from_camera) @ homogeneous
    return image[:2] / image[2]


def keyframe(index: int, center_x: float, color_bgr=(10, 20, 30)) -> SparseKeyframe:
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = color_bgr
    return SparseKeyframe(index, image, pose(center_x))


def pair(
    first_frame: int,
    second_frame: int,
    first_indices: list[int],
    second_indices: list[int],
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
) -> PairwiseLandmarkMatches:
    return PairwiseLandmarkMatches(
        first_frame,
        second_frame,
        np.asarray(first_indices),
        np.asarray(second_indices),
        np.asarray(first_pixels, dtype=np.float64),
        np.asarray(second_pixels, dtype=np.float64),
        20,
        20,
    )


def mapper(**overrides) -> SparseLandmarkMapper:
    return SparseLandmarkMapper(
        K,
        SparseLandmarkConfig(
            enabled=True,
            maximum_landmark_distance_percentile=None,
            **overrides,
        ),
    )


def one_track_inputs(point=(0.2, -0.1, 5.0), centers=(0.0, 1.0)):
    xyz = np.asarray(point, dtype=np.float64)
    frames = [keyframe(index, center) for index, center in enumerate(centers)]
    pixels = [project(xyz, frame.world_from_camera) for frame in frames]
    pairs = [
        pair(i, i + 1, [7 + i], [8 + i], [pixels[i]], [pixels[i + 1]])
        for i in range(len(frames) - 1)
    ]
    return xyz, frames, pairs


def test_two_camera_triangulation_recovers_known_world_point() -> None:
    expected, frames, _ = one_track_inputs()
    by_index = {item.frame_index: item for item in frames}
    observations = [
        TrackObservation(item.frame_index, 0, tuple(project(expected, item.world_from_camera)))
        for item in frames
    ]
    actual = triangulate_observations(observations[0], observations[1], by_index, K)
    assert actual is not None
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_projection_matrix_inverts_world_from_camera_pose() -> None:
    projection = projection_matrix(K, pose(2.0))
    expected_extrinsic = np.hstack((np.eye(3), np.array([[-2.0], [0.0], [0.0]])))
    np.testing.assert_allclose(projection, K @ expected_extrinsic)


def test_landmark_behind_camera_is_rejected() -> None:
    _, frames, pairs = one_track_inputs(point=(0.0, 0.0, -5.0))
    result = mapper(minimum_triangulation_angle_deg=0.0).build(frames, pairs)
    assert result.diagnostics["rejected_behind_camera"] == 1
    assert result.points.shape == (0, 3)


def test_small_triangulation_angle_is_rejected() -> None:
    _, frames, pairs = one_track_inputs(centers=(0.0, 0.001))
    result = mapper(minimum_triangulation_angle_deg=1.0).build(frames, pairs)
    assert result.diagnostics["rejected_low_angle"] == 1


def test_high_reprojection_error_in_long_track_is_rejected() -> None:
    _, frames, pairs = one_track_inputs(centers=(0.0, 1.0, 2.0))
    corrupted = pairs[1]
    pairs[1] = pair(
        1,
        2,
        [corrupted.previous_keypoint_indices[0]],
        [corrupted.current_keypoint_indices[0]],
        corrupted.previous_pixels,
        corrupted.current_pixels + np.array([[20.0, 0.0]]),
    )
    result = mapper(maximum_reprojection_error_px=3.0).build(frames, pairs)
    assert result.diagnostics["rejected_reprojection"] == 1
    assert max(result.landmarks[0].reprojection_errors_px) > 3.0


def test_valid_three_frame_track_becomes_one_landmark() -> None:
    expected, frames, pairs = one_track_inputs(centers=(0.0, 1.0, 2.0))
    result = mapper(minimum_track_length=3).build(frames, pairs)
    assert result.diagnostics["track_count"] == 1
    assert result.diagnostics["track_length_distribution"] == {"2": 0, "3": 1, "4+": 0}
    assert result.points.shape == (1, 3)
    np.testing.assert_allclose(result.points[0], expected, atol=1e-8)


def test_conflicting_association_is_rejected_deterministically() -> None:
    pixels = np.zeros((2, 2))
    tracks, conflicts = build_feature_tracks([
        pair(0, 1, [1, 2], [3, 4], pixels, pixels),
        pair(1, 2, [3, 4], [5, 5], pixels, pixels),
    ])
    assert conflicts == 1
    assert [track.track_id for track in tracks] == [0, 1]
    assert [len(track.observations) for track in tracks] == [3, 2]


def test_one_keypoint_is_never_assigned_to_multiple_tracks() -> None:
    pixels = np.zeros((2, 2))
    tracks, _ = build_feature_tracks([
        pair(0, 1, [1, 2], [3, 4], pixels, pixels),
        pair(1, 2, [3, 4], [5, 5], pixels, pixels),
    ])
    keys = [
        (observation.frame_index, observation.keypoint_index)
        for track in tracks
        for observation in track.observations
    ]
    assert len(keys) == len(set(keys))


def test_landmark_color_is_median_rgb_across_observations() -> None:
    frames = {
        0: keyframe(0, 0.0, (10, 20, 30)),
        1: keyframe(1, 1.0, (30, 40, 50)),
        2: keyframe(2, 2.0, (50, 60, 70)),
    }
    observations = tuple(TrackObservation(i, i, (10.0, 10.0)) for i in frames)
    assert landmark_color_rgb(observations, frames) == (50, 40, 30)


def test_sparse_build_does_not_mutate_input_poses_or_dense_arrays() -> None:
    _, frames, pairs = one_track_inputs()
    poses_before = [item.world_from_camera.copy() for item in frames]
    dense = np.arange(12, dtype=np.float64).reshape(4, 3)
    dense_before = dense.copy()
    mapper().build(frames, pairs)
    for item, before in zip(frames, poses_before):
        np.testing.assert_array_equal(item.world_from_camera, before)
    np.testing.assert_array_equal(dense, dense_before)


def test_disabled_mode_returns_no_sparse_geometry() -> None:
    _, frames, pairs = one_track_inputs()
    result = SparseLandmarkMapper(K, SparseLandmarkConfig(enabled=False)).build(
        frames, pairs
    )
    assert not result.enabled
    assert result.points.shape == (0, 3)
    assert result.tracks == ()


def test_relative_nonmetric_metadata_is_explicit() -> None:
    _, frames, pairs = one_track_inputs()
    metadata = mapper().build(frames, pairs).metadata_dict()
    assert metadata["coordinate_scale"] == "relative_non_metric"
    assert metadata["is_metric"] is False


def test_sparse_outputs_have_expected_point_color_count(tmp_path) -> None:
    _, frames, pairs = one_track_inputs()
    result = mapper().build(frames, pairs)
    dense = np.array([[0.0, 0.0, 1.0]])
    paths = save_sparse_landmark_outputs(
        tmp_path,
        result,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        frames[0].image_bgr,
        dense,
        np.array([[255, 0, 0]], dtype=np.uint8),
    )
    assert np.load(paths["points_npy"]).shape == (1, 3)
    assert np.load(paths["colors_npy"]).shape == (1, 3)
    header = paths["ply"].read_text(encoding="ascii").split("end_header")[0]
    assert "element vertex 1" in header
    payload = json.loads(paths["tracks_json"].read_text(encoding="utf-8"))
    assert payload["landmarks"][0]["observation_count"] == 2
    assert all(path.is_file() for path in paths.values())
