# Transform Convention and Map-Fusion Audit

## Canonical notation

This audit uses `T_A_from_B` to mean that a point expressed in coordinate
system B is transformed into coordinate system A:

```text
p_A = T_A_from_B p_B
p_A = R_A_from_B @ p_B + t_A_from_B
```

Points in the code are stored as `N x 3` row arrays. The mathematically
equivalent NumPy expression is:

```python
points_A = points_B @ R_A_from_B.T + t_A_from_B
```

Therefore, `points @ R.T + t` and `(R @ points.T).T + t` are correct. Using
`points @ R + t` would apply the inverse rotation for an orthonormal `R` and
would be incorrect for this convention.

The translation `t_A_from_B` is the origin of B expressed in coordinates of A.
It is not generally the camera center in world coordinates. For
`T_camera_from_world = [R | t]`, the camera center is `-R.T @ t`. For an
already-inverted `T_world_from_camera`, the camera center is its translation
column.

## Pose-composition derivation

OpenCV returns the local pose used by this project as:

```text
p_current = T_current_from_previous p_previous
```

Consequently:

```text
p_previous = inverse(T_current_from_previous) p_current
p_world = T_world_from_previous p_previous
```

Substitution gives the required accumulated pose:

```text
T_world_from_current =
    T_world_from_previous @ inverse(T_current_from_previous)
```

This is the exact equation implemented by `PoseManager.compose_world_pose`.
The inverse and right-multiplication are both required.

## Component audit

| Component | Variable/function | Returned or consumed transform | Meaning and equation | Used by | Expected | Actual | Status |
|---|---|---|---|---|---|---|---|
| `motion_estimator.py` | `MotionEstimator.estimate(points1, points2, K)` | `rotation`, `translation_direction` | With `points1=previous` and `points2=current`, `recoverPose` gives `p_current = R @ p_previous + t`; `t` is a unit direction in current coordinates | Keyframe motion gate; fixed-step debug pose | `T_current_from_previous` | `T_current_from_previous` | Verified |
| `depth_pose_estimator.py` | `DepthPoseEstimator.estimate` | `rotation`, `translation` | PnP object points are backprojected in the previous camera and image points belong to current: `p_current = R @ p_previous + t`; `t` is expressed in current coordinates | Depth alignment and pose accumulation | `T_current_from_previous` | `T_current_from_previous` | Verified |
| `depth_alignment.py` | `align_prediction_to_pose` | consumes PnP `R,t` | Computes `(R @ previous_points.T).T + t` to obtain geometric current-camera Z | Current-frame disparity scale/shift fit | Previous to current | Previous to current | Verified |
| `pose_refinement_3d.py` | `build_3d_correspondences` | previous/current camera points | Builds matched `p_previous` and `p_current` without changing frames | 3D refinement | Explicit previous/current arrays | Explicit previous/current arrays | Verified |
| `pose_refinement_3d.py` | `kabsch_rigid_transform` / `robust_rigid_alignment` | `rotation`, `translation` | Estimates `target = R @ source + t`, where source is previous and target is current | Optional refinement | `T_current_from_previous` | `T_current_from_previous` | Verified |
| `pose_manager.py` | `compose_world_pose` | `T_world_from_current` | `T_world_from_previous @ inverse(T_current_from_previous)` | All accepted camera poses | Camera to world | Camera to world | Verified |
| `pose_manager.py` | `trajectory_positions` | translation of each stored pose | Stored poses are `T_world_from_camera`, so the translation is the camera center in world coordinates | Trajectory output | Camera center | Camera center | Verified |
| `transforms.py` | `transform_points` | transformed `N x 3` points | `(R @ points.T).T + t`, equal to `points @ R.T + t` | Fusion and diagnostics | Column-vector transform applied to row storage | Correct equivalent | Verified |
| `point_cloud.py` / `backprojection.py` | `generate_colored_point_cloud` / `backproject_pixels` | camera-frame point cloud | `(X,Y,Z)` is produced in the camera whose image/depth generated the cloud | Map builder | `p_camera` | `p_camera` | Verified |
| `map_builder.py` | accepted `world_pose` | `T_world_from_camera` | Returned by `PoseManager` after inverting the local OpenCV pose | Point-cloud fusion | Camera to world | Camera to world | Verified |
| `map_builder.py` | `world_points` | world-frame points | `transform_points(camera_cloud.points, world_pose.R, world_pose.t)` | `RelativeMapFusion.add` | `p_world = R_world_from_camera @ p_camera + t_world_from_camera` | Same equation | Verified |
| `map_fusion.py` | `RelativeMapFusion.add` | no transform | Copies already-world-frame points; finalization only voxel-averages them | Final relative map | No implicit frame change | No implicit frame change | Verified |
| `pose_refinement_3d.py` | `build_pair_alignment_clouds` | consumes `T_world_from_previous/current` | Independently maps both camera clouds to world with `transform_points` | Diagnostic comparison | Camera to world | Camera to world | Verified |
| `temporal_depth_normalization.py` | `matched_world_residual_statistics` | consumes `T_world_from_previous/current` | Independently maps matched camera points to world | Diagnostic-only residuals | Camera to world | Camera to world | Verified |
| `pose_chain_diagnostics.py` | `direct_world_pose_from_reference_to_current` | `T_world_from_current` | Reuses `PoseManager.compose_world_pose` for direct PnP poses | Diagnostic-only chain comparison | Same composition as mapping | Same composition | Verified |

## OpenCV convention checks

### `solvePnP`

The actual object points passed by `DepthPoseEstimator` are created by
backprojecting `matches.points1` with the previous accepted camera's Z map.
They are therefore expressed in the previous camera frame. The corresponding
image points are `matches.points2`, measured in the current image. OpenCV's
projection equation is:

```text
p_current = R @ p_previous + t
pixel_current = project(K, p_current)
```

Thus the returned pose is `T_current_from_previous`, not a camera-to-world
pose. A deterministic synthetic PnP test recovers a known `R,t` in this exact
direction.

### `recoverPose`

`MapBuilder` passes `matches.points1` as previous and `matches.points2` as
current. For an essential matrix `E = [t]_x R`, `recoverPose(E, points1,
points2, K)` satisfies `p_current = R @ p_previous + t`, with translation only
known as a direction. A deterministic calibrated two-view test confirms the
returned rotation and translation sign/direction.

## Fusion and row-vector audit

Every production transformation of an `N x 3` point array routes through
`transform_points`. Its implementation is `(R @ points.T).T + t`. There is no
production use of the erroneous `points @ R + t` form. `RelativeMapFusion`
does not apply another transform, so points are neither inverted nor
double-transformed during fusion.

The synthetic three-camera landmark test constructs one fixed world landmark,
projects it into three known camera frames, accumulates the same local poses
through `PoseManager`, transforms each observation back to world, and verifies
that all observations fuse at one coordinate.

## Audit outcome

The algebra, implementation, and deterministic tests agree. No inversion,
multiplication-order, translation-frame, camera-center, row-vector, or fusion
convention inconsistency has been demonstrated. Mapping behavior is therefore
left unchanged. The optional `--transform-audit` switch only records explicit
matrices and matched-point discrepancies in `transform_trace.json`.

This conclusion concerns transform conventions only. It does not establish
that estimated poses or relative depths are accurate. If real-data local and
world residuals agree numerically while the map still fans out, estimation
error or temporally inconsistent relative geometry remains more likely than a
coordinate-frame implementation bug.

## Real three-frame trace

One diagnostic run used `data/drone_new.mp4`, frames 0/15/30, the supplied
intrinsics `(fx, fy, cx, cy) = (800, 800, 636, 321)`, CPU inference, keyframes,
sampling interval 15, three candidates, and point-cloud stride 4. Optional 3D
refinement, depth stabilization, temporal normalization, pose-chain analysis,
and drift analysis were disabled.

| Frame | Camera center in world `(x, y, z)` |
|---:|---|
| 0 | `(0.000000, 0.000000, 0.000000)` |
| 15 | `(0.007538, -0.012173, 0.010458)` |
| 30 | `(0.011125, -0.026684, 0.024544)` |

The saved trace contains each explicit `T_current_from_previous`, its inverse,
the accumulated `T_world_from_camera`, and the camera center.

| Pair | Correspondences | Local median / RMSE | World median / RMSE | Maximum per-match residual difference |
|---|---:|---|---|---:|
| 0 -> 15 | 2,882 | `0.019291 / 0.075664` | `0.019291 / 0.075664` | `1.67e-15` |
| 15 -> 30 | 3,000 | `0.019053 / 0.075983` | `0.019053 / 0.075983` | `8.88e-16` |

The local discrepancy `||T_current_from_previous p_previous - p_current||`
and the discrepancy after independently transforming both observations into
world coordinates are equal to floating-point precision. This directly checks
the real PnP-to-accumulation-to-fusion path and provides no evidence of a
direction, inverse, multiplication-order, or row-vector bug.
