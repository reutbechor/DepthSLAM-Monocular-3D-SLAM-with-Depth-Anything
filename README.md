# DepthSLAM – Monocular 3D SLAM with Depth Anything

This university project will eventually explore monocular 3D SLAM using depth
estimated from ordinary RGB images. The repository currently contains the
validated depth-estimation module and a small two-frame visual motion module.
**No full SLAM functionality exists yet.**

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

This status applies only to Stage 1 depth estimation; it does not imply that a
3D SLAM pipeline has been implemented or validated.

## Depth Anything and this project

[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) is a
monocular depth model: it predicts which regions are nearer or farther from one
ordinary RGB image. This project defaults to the lightweight, 24.8M-parameter
**Depth Anything V2 Small** model.

The default checkpoint produces **relative depth, not metric depth**. Values are
not distances in metres; scale and shift are ambiguous, especially across
separate frames.

Stage 1:

- loads `depth-anything/Depth-Anything-V2-Small-hf` through Transformers, a path
  documented by the official Depth Anything V2 repository;
- accepts a single image;
- accepts an OpenCV-readable video;
- supports configurable video-frame sampling with `--sample-every`;
- runs Depth Anything V2 Small relative-depth inference;
- saves the original RGB frame;
- saves the raw float32 relative-depth map as `.npy`;
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
|-- depth_raw/      # float32 relative-depth .npy files
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
Stage 1: RGB frame -> Depth Anything relative depth
Stage 2: two RGB frames -> relative R and unknown-scale t direction
Stage 3: pose-inlier Frame 1 pixels + relative depth + camera intrinsics
         -> relative Frame 1 camera-frame 3D points
```

Stage 3 samples Frame 1's relative depth at feature matches accepted by Stage
2's pose geometry. It then backprojects every valid sample with the calibrated
pinhole model:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = relative depth
```

Pixels outside the depth image and samples with `NaN`, infinity, zero, or
negative depth are removed. No replacement depth is invented. Bilinear sampling
is the default; nearest-neighbor sampling is also available.

Because Depth Anything V2 Small produces **relative depth**, `X`, `Y`, and `Z`
are in `relative_depth_units`, not metres. Stage 3 does not recover monocular
metric scale, and the Stage 2 translation remains an unknown-scale direction.

### Run the integrated pipeline

Provide calibration values measured for the source camera:

```powershell
py tools\run_depth_geometry.py data\frame1.jpg data\frame2.jpg --fx 730 --fy 730 --cx 636 --cy 321
```

The validated CPU run produced 5,604 good matches, 5,357 pose inliers, 5,357
valid relative-depth samples, and 5,357 relative 3D points. These counts describe
relative geometry only and do not indicate metric reconstruction.

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
|-- depth_vis.png
|-- matches.png
|-- feature_points_2d.npy
|-- feature_depths.npy
|-- points_3d_relative.npy
`-- metadata.json
```

`metadata.json` explicitly records `depth_type: relative`,
`coordinate_units: relative_depth_units`, the intrinsics and counts, and that
translation scale is unknown. No global point cloud or PLY file is produced.

Stage 3 creates relative two-frame geometry only. It is not a global SLAM map.

## Current limitations

- Depth is relative and uncalibrated, not metric.
- Video frames are inferred independently with no temporal consistency step.
- There is no trajectory accumulation, metric scale recovery, visual odometry
  pipeline, map fusion, keyframe handling, loop closure, bundle adjustment,
  point-cloud generation, GUI, or evaluation.
- Stage 2 accepts two still frames; it is not a full-video motion or SLAM tool.
- Stage 3 points remain local to the Frame 1 camera and use relative depth units.
- CPU inference is supported but can be slow.
- Transformers output can differ slightly from the original repository's
  OpenCV preprocessing/upsampling path, as the official authors note.

## References

- [Official Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Official Small Transformers checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414)
