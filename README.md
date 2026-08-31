# DepthSLAM – Monocular 3D SLAM with Depth Anything

This university project will eventually explore monocular 3D SLAM using depth
estimated from ordinary RGB images. The repository currently contains the
validated depth-estimation stages plus an incremental relative mapping
prototype for short videos. **It is not a complete SLAM system and it does not
produce a metric map.**

## Current status

**Stage 1 is complete and validated.** Image inference and video inference both
work. Video validation with `--sample-every 10` successfully processed 17
sampled frames. The expected RGB, raw depth, visualization, comparison, and
metadata outputs were created.

The Stage 2 two-frame motion module is implemented, covered by offline unit and
synthetic-geometry tests, and exercised on the calibrated local frame pair used
for Stage 3 validation.

Stage 3 connects Frame 1 relative depth to pose-inlier features and produces a
small set of relative 3D points in the Frame 1 camera coordinate system. It does
not create or accumulate a map. The integrated CPU run was validated on
`data/frame1.jpg` and `data/frame2.jpg` with the documented intrinsics.

Stage 4 adds a dense colored relative point cloud for one frame plus standalone
rigid-transform utilities. The single-frame CPU pipeline is validated on
`data/frame1.jpg`. It still does not fuse frames or create a global map.

Stage 5 composes accepted pairwise motions, transforms selected-frame clouds to
the first camera's world frame, and voxel-downsamples them into one relative
map. Its new components and small end-to-end CPU run are validated; details are
in the Stage 5 section below.

## Depth Anything and this project

[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) is a
monocular depth model: it predicts which regions are nearer or farther from one
ordinary RGB image. This project defaults to the lightweight, 24.8M-parameter
**Depth Anything V2 Small** model.

The default checkpoint produces a **relative disparity/inverse-depth-like
prediction, not metric camera depth**. Larger raw values represent nearer
structure. Values are not distances in metres; scale and shift are ambiguous,
especially across separate frames. Raw predictions must not be used directly as
the pinhole `Z` coordinate. See [the depth-semantics audit](docs/depth_semantics.md).

Stage 1:

- loads `depth-anything/Depth-Anything-V2-Small-hf` through Transformers, a path
  documented by the official Depth Anything V2 repository;
- accepts a single image;
- accepts an OpenCV-readable video;
- supports configurable video-frame sampling with `--sample-every`;
- runs Depth Anything V2 Small relative disparity-like inference;
- saves the original RGB frame;
- saves the raw float32 relative inverse-depth prediction as `.npy`;
- saves a colorized depth visualization;
- saves an RGB/depth side-by-side comparison;
- writes one frame record to `metadata.jsonl`.

`src/depth_estimator.py` isolates the reusable `DepthEstimator.predict(image)`
interface for later stages. The official repository and weights are not copied.

## Setup

Python 3.10 or newer is recommended; the validated Windows setup used Python
3.13.5. From the repository root on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
```

`requirements.txt` contains runtime dependencies. `requirements-dev.txt` adds
the test dependency while including all runtime requirements.

On macOS/Linux use `source .venv/bin/activate` and replace `py` with `python3`
in the commands. The first inference automatically downloads the official
`depth-anything/Depth-Anything-V2-Small-hf` checkpoint from Hugging Face; later
runs use its local cache. Internet is needed once unless the weights are already
cached.

Hugging Face authentication is optional for this public checkpoint. Downloads
work without an `HF_TOKEN`, although unauthenticated requests can have lower
rate limits. On Windows, Hugging Face may warn that its cache cannot use
symbolic links unless Developer Mode or administrator privileges are enabled.
This warning is harmless and does not block inference; the cache may simply use
more disk space.

## Run

Image:

```powershell
py tools\run_depth.py data\test.jpg --input-type image
```

Video, processing frames 0, 10, 20, and so on:

```powershell
py tools\run_depth.py data\test.mp4 --input-type video --sample-every 10
```

Common extensions are auto-detected if `--input-type` is omitted. Options include
`--config config/default.yaml`, `--output-dir outputs`, `--device auto|cpu|cuda`,
and `--model <hugging-face-model-id>`.

Run lightweight tests without model weights:

```powershell
py -m pytest
```

### CPU and CUDA behavior

With the default `device: auto`, the estimator uses CUDA when PyTorch reports an
available CUDA device and otherwise uses CPU. `--device cpu` forces CPU, while
`--device cuda` requests CUDA and returns a clear error if CUDA is unavailable.
Stage 1 was validated on CPU only; **CUDA has not been tested**.

## Stage 1 Validated Environment

- Operating system: Windows
- Python: 3.13.5
- Inference device: CPU
- Stage 1 tests at validation: 5 pytest tests passed
- Image inference: passed
- Video inference: passed, including 17 frames sampled with `--sample-every 10`
- CLI help: passed

## Outputs

Each run creates a timestamped directory without overwriting earlier results:

```text
outputs/<source-name>_<timestamp>/
|-- rgb/
|-- depth_raw/      # float32 raw relative inverse-depth .npy files
|-- depth_vis/      # per-frame normalized color images
|-- side_by_side/   # original and depth visualization
`-- metadata.jsonl  # one record per processed frame
```

Metadata contains source path, source frame index, video timestamp, width,
height, model, device, depth type, and video sampling details. Images use frame
index `0` and timestamp `0.0`. Visualization is normalized independently per
frame; its colors are not quantitatively comparable between frames. Use `.npy`
files for unquantized predictions.

Defaults are in `config/default.yaml`. A sampling interval of `1` processes all
frames; `10` processes every tenth frame.

## Stage 2: two-frame visual motion estimation

Stage 2 estimates relative camera motion between exactly two RGB frames using
classical calibrated two-view geometry:

```text
Frame 1 + Frame 2
        |
        v
      SIFT
        |
        v
descriptor matching (BFMatcher, L2)
        |
        v
Lowe ratio test
        |
        v
Essential Matrix + RANSAC
        |
        v
recoverPose
        |
        v
relative rotation R + translation direction t
```

SIFT detects distinctive image locations and describes the local appearance
around them. BFMatcher compares those descriptors. The Lowe ratio test rejects
an ambiguous match when its best candidate is not clearly better than its
second-best candidate.

The Essential Matrix represents the epipolar geometry between two views from a
calibrated camera. RANSAC estimates it while rejecting correspondence outliers.
OpenCV's `recoverPose` then recovers relative rotation `R` and translation
direction `t` from the inlier correspondences.

The length of `t` is **unknown**. A monocular two-view pair provides translation
only up to scale, so Stage 2 does not report metres or metric camera motion.
Depth Anything predictions are not used in this motion estimate.

### Camera intrinsics

Stage 2 requires calibrated pixel intrinsics `fx`, `fy`, `cx`, and `cy`. The
values in `config/default.yaml` are intentionally `null`; the project does not
invent calibration values. Set measured values in the config or pass all four
on the command line.

Example:

```powershell
py tools\run_motion.py data\frame1.jpg data\frame2.jpg --fx 1000 --fy 1000 --cx 960 --cy 540
```

The tool prints feature and inlier statistics, `R`, and the unknown-scale `t`
direction. It saves a debug image to:

```text
outputs/motion/motion_<frame1>_<frame2>_<timestamp>/
`-- matches.png  # inliers green; other Lowe-ratio matches orange
```

Thresholds for the Lowe test, RANSAC, minimum matches, and minimum inliers are
configurable in `config/default.yaml` or through `py tools\run_motion.py --help`.
The command returns a non-zero exit code when feature matching or pose recovery
fails.

Stage 2 is visual motion estimation between two frames only. It does not
accumulate a trajectory and is not a full SLAM system.

## Stage 3: depth-assisted relative 3D geometry

The three implemented stages now connect as follows:

```text
Stage 1: RGB frame -> Depth Anything relative inverse-depth prediction
Stage 2: two RGB frames -> relative R and unknown-scale t direction
Stage 3: pose-inlier Frame 1 pixels + explicit relative camera-Z proxy + intrinsics
         -> relative Frame 1 camera-frame 3D points
```

Stage 3 now converts the raw disparity-like prediction to a typed camera-Z proxy
before sampling it at feature matches accepted by Stage 2. It then backprojects
every valid sample with the calibrated pinhole model:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = reciprocal relative-disparity proxy
```

Pixels outside the depth image and samples with `NaN`, infinity, zero, or
negative depth are removed. No replacement depth is invented. Bilinear sampling
is the default; nearest-neighbor sampling is also available.

The conversion preserves near/far ordering but its initial zero-shift assumption
cannot remove the checkpoint's affine ambiguity. `X`, `Y`, and `Z` remain in
`relative_depth_units`, not metres. Stage 3 does not recover monocular metric
scale, and the Stage 2 translation remains an unknown-scale direction.

### Run the integrated pipeline

Provide calibration values measured for the source camera:

```powershell
py tools\run_depth_geometry.py data\frame1.jpg data\frame2.jpg --fx 730 --fy 730 --cx 636 --cy 321
```

The earlier CPU run produced 5,604 good matches and 5,357 pose inliers. The
corrected Stage 3 uses the same correspondences but no longer interprets raw
model scores directly as `Z`.

This command reuses `FeatureTracker`, `MotionEstimator`, and `DepthEstimator`;
the depth sampling and backprojection code never loads a model itself. Use
`--sampling nearest` to override bilinear sampling, or inspect all options with:

```powershell
py tools\run_depth_geometry.py --help
```

Each invocation creates a timestamped directory:

```text
outputs/depth_geometry/depth_geometry_<frame1>_<frame2>_<timestamp>/
|-- depth_raw.npy
|-- camera_z.npy
|-- depth_vis.png
|-- matches.png
|-- feature_points_2d.npy
|-- feature_depths.npy
|-- points_3d_relative.npy
`-- metadata.json
```

`metadata.json` explicitly separates the raw `relative_inverse_depth` prediction
from the geometry-ready `relative_camera_z_proxy`, records
`coordinate_units: relative_depth_units`, and states that translation scale is
unknown. No global point cloud or PLY file is produced.

Stage 3 creates relative two-frame geometry only. It is not a global SLAM map.

## Stage 4: dense colored relative point clouds

Stage 4 has two independent parts:

```text
Stage 4A:
RGB + typed reciprocal-disparity camera-Z proxy + camera intrinsics
    -> sampled dense colored relative point cloud

Stage 4B:
validated rigid coordinate transformations
    -> infrastructure for a later map-fusion stage
```

A point cloud stores an `(X, Y, Z)` coordinate and an RGB color for every valid
sampled pixel. The generator reuses Stage 3's pinhole backprojection, discards
non-finite or non-positive depth, and keeps each source pixel's red, green, and
blue values.

The `stride` controls subsampling in both image axes. `stride=1` uses every
pixel, while `stride=4` uses pixels 0, 4, 8, and so on in each row and column.
Larger strides produce smaller files and require less memory.

Points are expressed in the input camera coordinate frame. Their units are
`relative_depth_units`. The raw model prediction is stored separately and is
never passed directly to backprojection. **The point cloud is not metric and
its coordinates are not metres.**

### Run single-frame point-cloud generation

Use real calibration values for the source camera:

```powershell
py tools\run_point_cloud.py data\frame1.jpg --fx 730 --fy 730 --cx 636 --cy 321 --stride 4
```

The corrected CPU run used a 1272x642 image and produced 51,198 sampled pixels
and 51,198 valid points. Camera-Z proxy statistics were: minimum 0.514849, 5th
percentile 0.609269, median 1.000000, 95th percentile 3.343912, and maximum
9.244245 relative units. The earlier invalid raw-as-Z interpretation had
minimum/median/maximum 0.309172/2.858060/5.551260.

The output is timestamped:

```text
outputs/point_cloud/point_cloud_<frame>_<timestamp>/
|-- depth_raw.npy
|-- camera_z.npy
|-- depth_vis.png
|-- points_3d_relative.npy
|-- colors_rgb.npy
|-- cloud_relative.ply
`-- metadata.json
```

PLY is a simple point-cloud file format. This project writes an ASCII PLY with
`x y z red green blue` per vertex and no viewer dependency. The colors are RGB,
not OpenCV's BGR ordering.

### Coordinate transforms

`src/transforms.py` provides `transform_points`, `make_transform`, and
`invert_transform`. Translation values supplied to these helpers must already
use units compatible with the points. The unknown-scale unit translation
direction from Stage 2 is **not** automatically compatible with Stage 4 relative
depth and is never applied automatically.

These helpers are infrastructure only. Stage 4 performs no multi-frame fusion,
trajectory accumulation, or global mapping.

## Stage 5: incremental relative multi-frame mapping

Stage 5 connects the existing modules for a short, sampled video sequence:

```text
sampled RGB frames
    -> pairwise visual motion between consecutive accepted frames
    -> Essential Matrix geometric inlier filtering
    -> previous-frame camera-Z proxy + solvePnPRansac
    -> scaled relative R,t and sequential pose accumulation
    -> current-frame affine disparity scale/shift alignment
    -> dense colored relative point cloud
    -> camera cloud transformed into the common world frame
    -> cloud fusion and NumPy voxel downsampling
    -> relative multi-frame map and relative camera trajectory
```

The first sampled frame is accepted at identity and defines the world origin.
OpenCV `recoverPose` describes the previous camera coordinates transformed into
the current camera coordinates: `X_current = R @ X_previous + t`. The
`PoseManager` stores the opposite convention, `T_world_camera`, so it inverts
each accepted relative transform before composing it with the previous stored
pose. Tests cover this convention with synthetic transforms and points.

The original Stage 5 multiplied the Essential Matrix translation direction by a
fixed `translation_step` and treated raw relative predictions directly as `Z`.
That combined two incompatible arbitrary scales and reversed the disparity
near/far ordering, producing fanned or separated copies of the scene.

The preferred `depth-pnp` path now backprojects pose-inlier points from the
previous accepted frame and uses their current-frame pixels with
`solvePnPRansac`. PnP returns `R` and the full translation vector in the same
units as the previous camera-Z proxy. The current frame's independently
predicted disparity is then aligned by robustly fitting
`d_current = a / Z_geometric + b`; its cloud uses `Z_current = a / (d_current -
b)`. This is scale-and-shift alignment in disparity space.

The first frame still establishes an arbitrary relative scale, so relative-mode
translation is in `relative_depth_units`, not metres. `--scale-mode fixed-step`
retains the old translation-direction behavior solely for explicit debugging.
It is never a silent fallback and its metadata says `fixed_step_debug`.

Mapping frames are selected by a fixed source-frame interval. This is simple
frame sampling, not loop-closure-aware keyframe selection. A frame is rejected
when feature matching, geometric filtering, depth-assisted PnP, disparity
alignment, or cloud generation fails. A rejected frame receives no invented
pose or cloud; the next sampled frame is matched against the most recent
accepted frame, and its reason is saved in `frame_stats.jsonl`.

Voxel downsampling assigns finite points with
`floor(point / voxel_size)` and averages the positions and RGB colors in every
occupied voxel. It reduces duplicate/dense points deterministically without an
Open3D dependency. `voxel_size` is in arbitrary relative map units, not metres.

### Run relative mapping

Use intrinsics calibrated for the source video. This deliberately small CPU
example samples every twentieth source frame and considers at most five mapping
frames:

```powershell
py tools\run_relative_map.py data\drone.mp4 --fx 730 --fy 730 --cx 636 --cy 321 --sample-every 20 --max-mapping-frames 5 --point-cloud-stride 8 --voxel-size 0.05 --device cpu
```

Defaults are under `map:` in `config/default.yaml`. Inspect all CLI options with:

```powershell
py tools\run_relative_map.py --help
```

Each invocation writes a new timestamped directory:

```text
outputs/relative_map/relative_map_<source>_<timestamp>/
|-- global_relative_map.ply       # ASCII x y z red green blue
|-- global_points_relative.npy    # finite Nx3 relative map coordinates
|-- global_colors_rgb.npy         # matching Nx3 uint8 RGB colors
|-- trajectory_relative.csv       # accepted frames only
|-- trajectory_relative.npy       # accepted camera positions, Nx3
|-- frame_stats.jsonl             # accepted/rejected status and reason per sample
`-- metadata.json                 # parameters, counts, and non-metric labels
```

The trajectory CSV includes accepted frames only; rejected frames are retained
in `frame_stats.jsonl`. Metadata explicitly records
`map_type: relative_multi_frame`, `is_metric: false`, `depth_type: relative`,
`scale_estimation_method: depth_pnp`, `translation_step: null`, and
`depth_alignment_method: scale_and_shift_per_accepted_pair`.

Stage 5 is an incremental relative mapping prototype, not a complete production
SLAM system. Because motion is composed only from consecutive accepted pairs,
small errors accumulate as drift; there is no global correction.

### Stage 5 validated run

The corrected command above was executed on Windows with Python 3.13.5 and CPU
inference using source frames 0, 20, 40, 60, and 80. All five were accepted.
The four scaled translation magnitudes were 0.025817, 0.026099, 0.029411, and
0.029453 relative-depth units; PnP retained 2,162, 2,024, 2,419, and 2,173
inliers. The map contained 63,974 raw fused points and 16,051 points after voxel
downsampling. Saved points and trajectory were finite, RGB remained `uint8`,
and metadata was verified with `is_metric: false`, `scale_mode: depth-pnp`, and
`translation_step: null`. CUDA was not tested.

For a focused two-frame report, run:

```powershell
py tools\run_depth_pose.py data\frame1.jpg data\frame2.jpg --fx 730 --fy 730 --cx 636 --cy 321 --device cpu
```

The validated pair had 5,604 matches, 5,357 geometric inliers, 5,357 usable
depth correspondences, 4,095 PnP inliers, 1.491902 px reprojection RMSE, and a
translation magnitude of 0.01293350 relative-depth units. Translation is not in
metres.

### Metric outdoor model status

Depth Anything V2 publishes an official Small outdoor metric checkpoint trained
on synthetic Virtual KITTI 2 with an 80 m maximum depth. That repository ships
a standalone `.pth` checkpoint for the official `metric_depth` code, not the
Transformers configuration used here. It has therefore not been silently
substituted or loaded through an unsupported conversion. Even a future official
integration would be model-predicted metric depth, not guaranteed accuracy for
DJI aerial footage.

## Current limitations

- The default relative model predicts affine-ambiguous disparity-like values,
  not camera Z or metric depth.
- First-frame reciprocal conversion assumes zero disparity shift; it is only a
  relative-Z proxy. Per-frame affine alignment reduces but cannot eliminate
  model inconsistency.
- Depth estimates are inferred independently and can change across frames.
- PnP translation is in propagated relative-depth units, not metres.
- Fixed-step translation remains available only as explicit debug behavior.
- Sequential pose accumulation drifts over time.
- There is no loop closure, bundle adjustment, pose graph optimization, global
  relocalization, IMU/GNSS fusion, GUI, or ground-truth evaluation.
- Mapping-frame selection is fixed sampling, not sophisticated keyframe logic.
- Stage 2 accepts two still frames; it is not a full-video motion or SLAM tool.
- Stage 3 points remain local to the Frame 1 camera and use relative depth units.
- Stage 4 exports a single-camera relative PLY; Stage 5 fuses multiple clouds
  only in arbitrary relative map coordinates.
- Approximate or incorrect camera intrinsics can distort motion and 3D geometry.
- The validation intrinsics are approximate for the DJI footage and are not a
  substitute for camera calibration or lens-distortion correction.
- Dynamic objects can violate the static-scene assumptions of Essential Matrix,
  PnP, depth alignment, and cloud fusion.
- The relative model and Virtual-KITTI metric model are not specifically trained
  or validated for this aerial domain.
- CPU inference is supported but can be slow.
- Transformers output can differ slightly from the original repository's
  OpenCV preprocessing/upsampling path, as the official authors note.

## References

- [Official Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Official metric-depth documentation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
- [Official outdoor Small metric checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small)
- [Official Small Transformers checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414)
