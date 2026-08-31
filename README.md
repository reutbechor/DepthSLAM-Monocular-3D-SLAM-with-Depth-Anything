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

Stage 6 adds visual keyframe selection before candidate depth inference and map
fusion. It preserves every Stage 5 reliability gate and is validated on a
30-candidate CPU run. It is a selection layer, not loop closure or global map
optimization.

**Stage 7 is complete and validated.** It evaluates saved Stage 5/6 artifacts,
exports stable frame/run metrics, a relative trajectory, conservative reports,
and optional plots. It does not modify poses or maps and does not add a SLAM
optimization algorithm.

**Stage 8 is complete and validated.** It adds optional, position-only local trajectory
diagnostics and refinement while preserving the raw accepted-pose trajectory.
It is not loop closure, bundle adjustment, or ground-truth correction.

**Stage 9 is the final engineering stage.** It packages the existing mapper,
evaluation, and optional trajectory refinement into one reproducible CLI and
one verified final-run directory. It does not introduce a new SLAM algorithm.

## Quick Start / Final Pipeline

After completing the setup below, run the full existing pipeline once:

```powershell
py tools\run_pipeline.py data\video.mp4 --fx 800 --fy 800 --cx 636 --cy 321 --sample-every 5 --max-candidate-frames 30 --point-cloud-stride 8 --device cpu --keyframes --refine-trajectory
```

The CLI runs relative mapping once, evaluates its saved artifacts, optionally
refines the saved position trajectory, and creates a timestamped directory
under `outputs/final_pipeline/`. Its main files are:

- `final_summary.json`: compact scientific and numerical run result;
- `run_manifest.json`: exact arguments, environment versions, input metadata,
  and resolved thresholds;
- `FINAL_REPORT.md`: conservative machine-generated experiment report;
- `artifacts.json`: verified index of mapping, evaluation, refinement, and final
  files;
- `mapping/`, `evaluation/`, and optionally `trajectory_refinement/`.

The four intrinsics are required and are recorded as manually supplied and
approximate. The `800/800/636/321` example is uncalibrated and specific to the
current DJI experiment. A different camera requires appropriate intrinsics;
incorrect values can warp the recovered geometry. The CLI always prints this
warning and never labels manual values as calibrated.

### Final architecture

```text
monocular RGB video
  -> sampled frame candidates
  -> SIFT feature matching
  -> Essential Matrix + RANSAC geometric filtering
  -> visual keyframe selection
  -> Depth Anything V2 relative inverse-depth inference
  -> explicit relative camera-Z conversion
  -> depth-assisted PnP translation scale
  -> per-pair depth scale/shift alignment
  -> geometric and depth quality gates
  -> sequential camera-pose accumulation
  -> colored camera-frame point clouds
  -> world transformation and voxel fusion
  -> relative, non-metric 3D map
  -> artifact-based evaluation
  -> optional position-only trajectory refinement
  -> verified final report and reproducibility manifest
```

This remains a sequential monocular relative-mapping prototype. It has no loop
closure, bundle adjustment, pose-graph optimization, GNSS/GPS integration, or
GUI, and it makes no absolute-accuracy claim.

### Final pipeline validated run

The Quick Start command was executed on `data/drone_new.mp4` with the shown
intrinsics, CPU, sampling interval 5, 30 candidate frames, point-cloud stride
8, keyframes enabled, and trajectory refinement enabled. The video is 1272 x
642 at 30 FPS with 173 frames.

The final run accepted 13 keyframes, skipped 0 candidates, rejected 17, ran 14
depth inferences, saved 13 trajectory poses, and produced 34,013 final map
points. Mean/median PnP inlier ratio was 0.743173/0.729670; mean/median
reprojection RMSE was 1.402504/1.428752 pixels; and mean/median depth-alignment
inlier ratio was 0.969722/0.975904. Rejections were
`geometric_filtering: 16` and `depth_z_distribution: 1`.

Jump-aware refinement completed with zero suspicious jumps and zero modified
poses. Observed wall times were 100.878 seconds for mapping, 0.950 seconds for
evaluation, 0.226 seconds for refinement, and 102.152 seconds end-to-end. The
mapping time includes blocked Hugging Face network checks before the cached
checkpoint loaded, so these values are environment-specific observations, not
benchmarks. The artifact index verified 28 referenced files. The full suite
passed with **93 tests and 4 subtests**, and `run_pipeline.py --help` passed.

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

The robustness-validation CPU run used a 1272x642 image and produced 51,198
sampled valid depths. Its relative-Z 1st/99th percentile filter removed 1,024
tail samples and exported 50,174 points. Pre-filter camera-Z proxy statistics
were min/p1/p5/median/p95/p99/max =
0.515010/0.560803/0.608739/1.000667/3.372398/5.750152/9.131558 relative units.
The denominator guard rejected no samples in this unaligned first-frame case.

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
    -> reject non-finite or numerically unsafe disparity denominators
    -> pre-filter depth-alignment quality gate (accept or reject frame)
    -> dense colored relative point cloud
    -> optional relative-Z percentile tail suppression
    -> camera cloud transformed into the common world frame
    -> cloud fusion and NumPy voxel downsampling
    -> optional robust-center distance percentile filtering
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

### Stage 5 numerical robustness controls

The affine conversion `Z = a / (raw_disparity - b)` rejects a sample when its
denominator is non-finite, is not greater than the configured positive epsilon,
or produces non-finite/non-positive Z. Invalid samples become missing values;
they are never clamped to an invented distance. The default
`disparity_denominator_epsilon` is `0.001`.

Relative-mode point clouds then keep the configured central Z percentile range
(`1.0` through `99.0` by default). This is optional tail/outlier suppression
for usable visualization, not ground-truth depth correction. It is not applied
automatically to metric predictions. The final voxelized map can similarly keep
points within the configured center-distance percentile (`99.5` by default),
measured from the coordinate-wise median map center.

Both filters are explicit and auditable. `frame_stats.jsonl` records alignment
`a`/`b`, the denominator epsilon and minimum absolute denominator, rejection
counts, Z min/p1/p5/median/p95/p99/max, percentile bounds, and cloud counts.
`metadata.json` preserves raw fused, voxel-downsampled, and final counts; global
X/Y/Z percentiles before and after filtering; center-distance percentiles; and
a median-plus-scaled-MAD radius diagnostic. Use
`--disparity-denominator-epsilon`, `--depth-percentile-low`,
`--depth-percentile-high`, and `--global-outlier-percentile` to override the
defaults.
Set both depth percentile values to `null` to disable the per-frame filter, or
set `global_outlier_percentile: null` to disable the final global filter while
retaining all diagnostics.

### Depth-alignment acceptance gates

Good PnP inlier counts and reprojection error establish pose quality, but they
do not guarantee that the independently predicted relative depth aligns
reliably. Before generating or filtering a candidate's dense cloud, the mapper
therefore measures full-resolution, pre-filter alignment health and applies
these configurable defaults from `map:` in `config/default.yaml`:

```yaml
min_valid_depth_ratio: 0.60
max_denominator_reject_ratio: 0.30
min_depth_alignment_inliers: 500
min_depth_alignment_inlier_ratio: 0.30
max_relative_z_p99_over_median: 50.0
```

The measurements include total depth candidates, valid aligned depths,
denominator rejects and their ratios, robust alignment correspondences and
inliers, aligned Z p1/median/p99, p99/median, and affine scale/shift. These are
heuristic quality gates for relative-depth reconstruction, not learned
confidence scores or metric guarantees. Each threshold can be disabled with
`null` in the YAML configuration.

If any enabled gate fails, `frame_stats.jsonl` records an exact
`rejection_reason`, such as `depth_denominator_reject_ratio`,
`depth_valid_ratio`, `depth_alignment_inliers`,
`depth_alignment_inlier_ratio`, or `depth_z_distribution`. The candidate adds
neither a cloud nor a pose and does not replace the previous accepted-frame
reference. The next sample is still matched against the last accepted frame.
There is no fixed-step fallback. Rejecting later frames when temporal relative
depth degrades is a valid safer outcome than contaminating the map.

The p1/p99 percentile cloud filter remains a separate post-acceptance
visualization cleanup. A frame is never considered reliable merely because its
extreme points can be filtered away.

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
|-- trajectory_relative.csv       # accepted keyframes only
|-- trajectory_relative.npy       # accepted camera positions, Nx3
|-- frame_stats.jsonl             # frame status, Z/alignment/filter diagnostics
`-- metadata.json                 # parameters, raw/filter counts, global statistics
```

The trajectory CSV includes accepted keyframes only; skipped and rejected
candidates are retained in `frame_stats.jsonl`. Metadata explicitly records
the keyframe mode, `is_metric: false`, `depth_type: relative`,
`scale_estimation_method: depth_pnp`, `translation_step: null`, and
`depth_alignment_method: scale_and_shift_per_accepted_pair`.

Stage 5 is an incremental relative mapping prototype, not a complete production
SLAM system. Because motion is composed only from consecutive accepted pairs,
small errors accumulate as drift; there is no global correction.

### Stage 5 validated run

The robustness command above was executed on Windows with Python 3.13.5 and CPU
inference using source frames 0, 20, 40, 60, and 80. All five were accepted.
The four scaled translation magnitudes were 0.025817, 0.026099, 0.029411, and
0.028908 relative-depth units; PnP retained 2,162, 2,024, 2,419, and 2,204
inliers with RMSE 1.645, 1.731, 1.806, and 1.798 pixels. The map contained
62,633 raw fused points and 15,339 after voxel downsampling. The configured
99.5th center-distance percentile filter rejected 77 points and exported
15,262. The exported points and trajectory were finite, RGB remained `uint8`,
and the PLY vertex count matched metadata. The full suite passed 45 tests. CUDA
was not tested.

### Depth-quality gate validation

The quality-gated mapper was also executed on CPU with:

```powershell
py tools\run_relative_map.py data\drone_new.mp4 --fx 800 --fy 800 --cx 636 --cy 321 --sample-every 15 --max-mapping-frames 8 --point-cloud-stride 8 --device cpu
```

Frames 0, 15, 30, 45, 60, 75, 90, and 105 were all accepted under the initial
documented thresholds. This is recorded rather than hidden: the worst observed
denominator-reject ratio was 0.039150 (below 0.30), the lowest valid-depth ratio
was 0.960850 (above 0.60), the lowest non-origin alignment-inlier ratio was
0.897641 (above 0.30), every alignment had more than 500 inliers, and the worst
relative-Z p99/median ratio was 33.215475 (below 50.0). Therefore no quality
gate fired in this particular run. The run produced 99,841 raw fused points,
21,315 voxelized points, and 21,208 final points after 107 global outliers were
removed. The full suite passed 53 tests after this correction. CUDA was not
tested.

These initial thresholds are deliberately visible and configurable. Tightening
them changes the scientific acceptance policy and should be justified for the
dataset; the implementation does not silently tune them merely to produce more
accepted or rejected frames.

## Stage 6: visual keyframe selection

A keyframe is a candidate image selected to contribute a pose and dense cloud
to the mapped trajectory. Mapping every sampled frame can add nearly duplicate
geometry and unnecessary depth inference, so Stage 6 first compares each
candidate against the last accepted keyframe using information already
available from feature matching and Essential-Matrix geometry:

```text
sampled candidate
    -> match against last accepted keyframe
    -> Essential Matrix and geometric inliers
    -> keyframe selector
       -> skip redundant candidate before depth inference, or
       -> attempt the unchanged Stage 5 PnP/depth/quality/fusion pipeline
```

The first valid frame is always an `accepted_keyframe` with reason
`initial_frame`. Later candidates must first satisfy configurable minimum good
matches, geometric inliers, and geometric-inlier ratio. A reliable candidate is
selected when its median geometric-inlier pixel displacement reaches the
configured threshold, its recovered rotation reaches the rotation threshold,
or the maximum candidate gap forces an attempt. The forced-gap rule does not
bypass PnP or depth-quality validation.

Pixel displacement is computed as the median of
`sqrt((u2-u1)^2 + (v2-v1)^2)` over geometric inliers; p75 and p90 are diagnostic
only. It is an image-space heuristic whose meaning depends on image resolution,
scene depth, motion, and camera. Essential-Matrix translation magnitude is not
used as distance because monocular translation scale is unknown.

Frame states are intentionally distinct:

- `accepted_keyframe`: passed selection and the complete Stage 5 pipeline.
- `skipped_non_keyframe`: reliable visual geometry but insufficient new motion;
  no depth inference, pose, cloud, or reference update occurs.
- `rejected`: feature/geometry, PnP, alignment, depth-quality, or cloud
  reliability failed.

Both skipped and rejected candidates remain compared against the last accepted
keyframe on the next attempt. Metadata records the state, decision reason,
matches, geometric statistics, median/p75/p90 displacement, rotation, candidate
gap, depth-execution flag, and downstream diagnostics when executed.

Defaults are under `keyframes:` in `config/default.yaml`. Selection is enabled
by default. Use `--no-keyframes` to reproduce Stage 5 fixed-sample attempts as
closely as possible, or override the compact CLI controls:

```powershell
py tools\run_relative_map.py data\drone_new.mp4 --fx 800 --fy 800 --cx 636 --cy 321 --sample-every 5 --max-candidate-frames 30 --point-cloud-stride 8 --device cpu --keyframes

py tools\run_relative_map.py data\drone_new.mp4 --fx 800 --fy 800 --cx 636 --cy 321 --sample-every 5 --max-candidate-frames 30 --point-cloud-stride 8 --device cpu --no-keyframes
```

### Stage 6 validated run and Stage 5 comparison

The keyframe-enabled command above was executed on Windows, Python 3.13.5, and
CPU. Of 30 candidates, 13 became accepted keyframes, 17 were rejected, and none
were skipped for insufficient keyframe motion. Fourteen candidates executed
Depth Anything: the 13 accepted keyframes plus frame 145, which was rejected by
the existing `depth_z_distribution` gate. The remaining 16 candidates failed
Essential-Matrix pose reliability before selection or depth inference. The map
contained 154,138 raw fused points, 34,184 voxelized points, and 34,013 final
points.

The matching `--no-keyframes` comparison was also executed. It produced the
same 30 candidates, 14 depth calls, 13 accepted frames, and 34,013 final points.
On this particular video, every geometrically reliable candidate already had
median displacement above 8 px, while lower-motion candidates failed geometric
pose recovery first. Therefore this validation demonstrates correct selection
and accounting but does not demonstrate a runtime or depth-call reduction. No
runtime improvement is claimed.

Keyframe selection does not resolve scale ambiguity, make depth metric, remove
sequential drift, provide loop closure, guarantee optimal map coverage, or
replace bundle adjustment. Its thresholds are dataset- and resolution-dependent
heuristics for computational and mapping selection.

## Stage 7: evaluation and reporting

Stage 7 reads `metadata.json`, `frame_stats.jsonl`, and
`trajectory_relative.csv` from a Stage 5/6 map run. It aggregates the artifacts
without re-running SLAM logic when an existing run directory is supplied. Given
a video instead, the CLI first invokes the existing relative-map pipeline and
then evaluates the newly saved artifacts.

Frame metrics include candidate status and rejection reason, feature matches,
geometric inliers, keyframe motion, PnP inliers and reprojection residuals,
relative translation magnitude, depth-alignment quality, depth-distribution
quality, and contributed cloud points. Run metrics report accepted, skipped,
and rejected counts; mean/median/range summaries; rejection-reason counts; map
point counts; and lightweight stage timings.

These metrics measure different forms of **internal consistency**:

- Geometric and PnP inlier ratios describe how many tested correspondences
  agree with the fitted models.
- Reprojection RMSE is the pixel residual between matched observations and the
  fitted PnP pose. It is not absolute trajectory accuracy.
- Depth-alignment ratios describe agreement in the per-pair affine alignment
  of relative inverse-depth predictions.
- Denominator rejection and valid-depth ratios expose unsafe or invalid
  relative-Z conversions.

`drone_new.mp4` has no known external ground-truth trajectory. Stage 7 therefore
does **not** calculate ATE or RPE, does not call reprojection error trajectory
accuracy, and does not claim absolute map accuracy. Both trajectory coordinates
and translation magnitudes are explicitly saved as `relative_depth_units`, not
metres.

Run mapping and evaluation together:

```powershell
py tools\run_evaluation.py data\drone_new.mp4 --fx 800 --fy 800 --cx 636 --cy 321 --sample-every 5 --max-candidate-frames 30 --point-cloud-stride 8 --device cpu --keyframes
```

The numeric intrinsics in this command are manually supplied and are labeled
approximate by default. Use `--no-intrinsics-approximate` only for values from a
valid calibration. To evaluate an existing map run without rerunning mapping:

```powershell
py tools\run_evaluation.py outputs\relative_map\relative_map_drone_new_<timestamp>
```

Use `--no-plots` for a report without matplotlib figures. Each invocation
creates a timestamped directory:

```text
outputs/evaluation/evaluation_<source>_<timestamp>/
|-- frame_metrics.csv
|-- summary.json
|-- trajectory.csv
|-- evaluation_report.txt
|-- geometric_inlier_ratio.png
|-- pnp_inlier_ratio.png
|-- reprojection_rmse.png
|-- depth_alignment_inlier_ratio.png
|-- denominator_rejection_ratio.png
|-- frame_status.png
|-- trajectory_xz.png
`-- mapping/                         # present only for video-driven evaluation
```

Missing CSV values mean that a frame never reached the corresponding stage.
`summary.json` records the model, device, scale mode, intrinsics and their
provenance, keyframe settings, quality thresholds, map counts, units, runtime,
and scientific limitations. `trajectory.csv` contains accepted keyframes only,
aligned with their frame indices and timestamps.

### Stage 7 validated run

The exact command above was executed on Windows with Python 3.13.5 and CPU.
The full suite passed with **71 tests and 4 subtests**; `run_evaluation.py
--help` also passed. The 30 candidates produced 13 accepted keyframes, 0 skipped
non-keyframes, 17 rejected frames, 14 depth inferences, 13 trajectory poses,
and 34,013 final map points.

For that run, geometric inlier ratio mean/median was
0.380201/0.153467; PnP inlier ratio mean/median was 0.743173/0.729670;
reprojection RMSE mean/median was 1.402504/1.428752 pixels; and depth-alignment
inlier ratio mean/median was 0.969722/0.975904. The maximum denominator
rejection ratio was 0.191795. Rejections were `geometric_filtering: 16` and
`depth_z_distribution: 1`.

The observed end-to-end wall time was 105.588 seconds, including process
startup, model/cache checks, and blocked Hugging Face HEAD retries before the
cached checkpoint loaded. Instrumented depth inference totaled 23.332 seconds
(1.667 seconds per inference). These are observations from one run, not precise
benchmarks: CPU load, hardware, model loading, cache state, network checks, and
file I/O affect the measurements.

## Stage 8: lightweight trajectory refinement

Sequential pose estimates can accumulate small local inconsistencies that
appear as jitter, sharp direction changes, or isolated translation steps.
Stage 8 analyzes an already saved accepted-keyframe trajectory and can apply a
small, optional local average to positions. It does not rerun Depth Anything,
feature matching, motion estimation, or mapping.

For each consecutive accepted pose, Stage 8 computes translation-step
magnitude. Interior poses also receive direction-change and second-difference
diagnostics. These are geometric diagnostics in relative trajectory units;
they are not physical velocity or acceleration.

Suspicious steps use a robust threshold:

```text
threshold = median_step + mad_multiplier * 1.4826 * MAD(step)
```

MAD is the median absolute deviation. The default multiplier of `4.0` is a
heuristic, not a metric-distance limit. A zero MAD is handled by using the
median itself as the threshold, with only strictly larger steps flagged.

Two position-only modes are available:

- `jump_aware` is the default. It smooths only an interior pose targeted by a
  suspicious incoming step. Normal neighbors and both endpoints remain
  unchanged.
- `moving_average` applies the configured `[0.25, 0.50, 0.25]` weights to every
  interior position. It may suppress real motion as well as noise.

Rotations are not modified because averaging rotation matrices element-wise
does not preserve valid SO(3) rotations. All output coordinates remain
`relative_depth_units`, not metres. The original map-run files are never
overwritten.

Refine an existing map run and generate both plots:

```powershell
py tools\run_trajectory_refinement.py outputs\relative_map\relative_map_drone_new_<timestamp> --mode jump_aware --mad-multiplier 4.0 --plots
```

The Stage 8 CLI is an explicit opt-in operation; `trajectory_refinement.enabled`
remains `false` in `config/default.yaml` so refinement is never silently added
to mapping. Each invocation writes a new timestamped directory:

```text
outputs/trajectory_refinement/trajectory_refinement_<run>_<timestamp>/
|-- trajectory_raw.csv
|-- trajectory_raw.npy
|-- trajectory_refined.csv
|-- trajectory_refined.npy
|-- trajectory_diagnostics.csv
|-- trajectory_raw_vs_refined_xz.png
|-- trajectory_step_magnitude.png
`-- refinement_summary.json
```

Raw and refined CSV rows preserve accepted frame order and timestamps and carry
both `trajectory_type` and `trajectory_units`. The summary records the robust
threshold, suspicious pose/frame indices, modified poses, before/after
smoothness metrics, and scientific limitations.

Stage 7 can include a matching refinement summary without changing its original
metric definitions:

```powershell
py tools\run_evaluation.py outputs\relative_map\relative_map_drone_new_<timestamp> --refinement-dir outputs\trajectory_refinement\trajectory_refinement_<run>_<timestamp>
```

The refined trajectory is an analysis artifact only. The dense fused map still
comes from the raw accepted poses; Stage 8 does not move map points or claim a
refined map. A shorter or smoother path is not automatically more accurate.
Smoothing does not recover ground truth, resolve scale ambiguity, correct
accumulated drift, or replace loop closure, bundle adjustment, or pose-graph
optimization.

### Stage 8 validated run

The saved 13-pose trajectory from the Stage 7 `drone_new` run was refined
without rerunning the model or mapper, using `jump_aware`, MAD multiplier `4.0`,
and weights `[0.25, 0.50, 0.25]`. The median step was 0.012718 relative units,
MAD was 0.000901, and the robust threshold was 0.018060. The maximum observed
step was 0.017889, so no jump exceeded the threshold. Zero poses were modified;
this unchanged result is the intended conservative behavior, not a failed run.

Both raw and refined trajectories contained 13 poses. Their mean/median step
magnitudes were 0.013722/0.012718, step standard deviation was 0.002411, and
maximum step was 0.017889. Mean/median/maximum second differences were
0.006312/0.005699/0.013029. Both relative path lengths were 0.164666. No claim
of trajectory accuracy follows from these smoothness values.

The full suite passed with **84 tests and 4 subtests**, and both the Stage 8 CLI
help and Stage 7 `--refinement-dir` integration were executed successfully.

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
- Robust filtering removes numerical and relative-depth outliers; it does not
  make the map metric or correct the underlying depth estimates.
- Excessive percentile filtering can hide genuine scene geometry.
- Depth estimates are inferred independently and can change across frames.
- PnP translation is in propagated relative-depth units, not metres.
- Fixed-step translation remains available only as explicit debug behavior.
- Sequential pose accumulation drifts over time.
- Outlier filtering does not replace bundle adjustment or loop closure.
- There is no loop closure, bundle adjustment, pose graph optimization, global
  relocalization, IMU/GNSS fusion, GUI, or ground-truth evaluation.
- Stage 7 reports artifact-based internal consistency only; without an external
  ground-truth trajectory it cannot report ATE, RPE, absolute drift, or map
  accuracy.
- Stage 8 is local position smoothing only. It does not refine rotations or the
  fused map, and smoothness must not be interpreted as trajectory accuracy.
- Stage 6 keyframe decisions use heuristic image motion and may skip useful
  geometry or retain redundant views when thresholds are poorly chosen.
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
