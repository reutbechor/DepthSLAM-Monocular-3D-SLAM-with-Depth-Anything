# DepthSLAM – Monocular 3D SLAM with Depth Anything

This university project will eventually explore monocular 3D SLAM using depth
estimated from ordinary RGB images. The repository currently contains only the
reusable depth-estimation module. **No SLAM functionality exists yet.**

## Current status

**Stage 1 is complete and validated.** Image inference and video inference both
work. Video validation with `--sample-every 10` successfully processed 17
sampled frames. The expected RGB, raw depth, visualization, comparison, and
metadata outputs were created.

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

## Validated Environment

- Operating system: Windows
- Python: 3.13.5
- Inference device: CPU
- Tests: 5 pytest tests passed
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

## Current limitations

- Depth is relative and uncalibrated, not metric.
- Video frames are inferred independently with no temporal consistency step.
- There is no SLAM, visual odometry, feature matching, camera motion estimation,
  map fusion, keyframe handling, loop closure, GUI, or evaluation.
- CPU inference is supported but can be slow.
- Transformers output can differ slightly from the original repository's
  OpenCV preprocessing/upsampling path, as the official authors note.

## References

- [Official Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Official Small Transformers checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414)
