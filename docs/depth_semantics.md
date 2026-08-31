# Depth Anything V2 output semantics and geometry decision

## Evidence audited

- The [official Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
  calls the standard Small checkpoint a relative-depth model.
- The [official demo code](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/app.py)
  labels its saved raw 16-bit result as output that can be considered disparity.
- The [official model implementation](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py)
  applies a non-negative ReLU head to the relative model output. The repository
  does not convert this output to camera-axis distance.
- The [Transformers checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
  is tagged as relative depth. Its loaded configuration declares
  `depth_estimation_type: relative`; Transformers uses a ReLU relative head.
- The [official metric-depth documentation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
  uses a different fine-tuned DPT head and states that its output is depth in
  metres. Outdoor models are trained on synthetic Virtual KITTI 2 with a maximum
  depth of 80 m.

## Conclusion

For `depth-anything/Depth-Anything-V2-Small-hf`, `predicted_depth` is a
non-negative, relative, disparity/inverse-depth-like score. Larger values mean
nearer structure; it is not metric camera Z. The old operation
`Z = predicted_depth` reversed the near/far ordering required by pinhole
backprojection and was scientifically invalid.

The corrected relative path constructs an explicit camera-Z proxy:

```text
Z_relative = disparity_scale / (raw_prediction - disparity_shift)
```

For the first frame, `disparity_shift = 0` and `disparity_scale` is the median
positive raw score. This preserves ordering and sets a nominal relative scale,
but it assumes zero disparity shift. Because the relative prediction is affine
ambiguous, this first-frame proxy is not guaranteed to be Euclidean depth and
must never be called metric.

For each later accepted frame, depth-assisted PnP transforms previous-frame 3D
correspondences into the current camera. A robust fit of
`raw_disparity = a * (1 / geometric_Z) + b` estimates the current frame's scale
and shift. The aligned camera-Z proxy is `a / (raw_disparity - b)`. This
propagates the first frame's arbitrary relative units while accounting for
independent per-frame affine disparity variation. It does not recover metres.

## Metric outdoor model decision

The official `Depth-Anything-V2-Metric-VKITTI-Small` repository contains an
official PyTorch `.pth` checkpoint intended for the official cloned
`metric_depth` code. It does not contain the Transformers `config.json` and
preprocessor files needed by this project's `AutoModelForDepthEstimation` path.
It is therefore not silently substituted or loaded through an unsupported
conversion. A future explicit integration may depend on the official repository
as an external component. Even then, Virtual KITTI 2 training and the 80 m
range do not guarantee accurate metric depth for DJI aerial imagery.
