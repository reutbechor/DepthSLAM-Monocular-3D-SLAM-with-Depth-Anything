"""Presentation-only filtering and plotting helpers for relative geometry.

These helpers never feed data back into the SLAM pipeline.  They create
display-friendly copies while preserving the scientific point-cloud outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .ply_io import write_ascii_ply


@dataclass(frozen=True)
class DisplayCleaningResult:
    """A deterministic, presentation-only subset of a point cloud."""

    points: np.ndarray
    colors: np.ndarray | None
    raw_count: int
    display_count: int
    removed_count: int
    nonfinite_removed: int
    z_percentile_removed: int
    center_distance_removed: int
    z_bounds: tuple[float, float] | None
    center_distance_limit: float | None


@dataclass(frozen=True)
class CloudVisualArtifacts:
    """Paths and counts produced for one visual point-cloud export."""

    raw_paths: tuple[Path, ...]
    display_path: Path | None
    preview_paths: tuple[Path, ...]
    cleaning: DisplayCleaningResult


def display_cleaning_metadata(
    cleaning: DisplayCleaningResult,
    *,
    raw_artifact: str,
    compatibility_artifact: str,
    display_artifact: str | None,
) -> dict[str, object]:
    """Build the common, explicit raw-versus-display metadata block."""

    return {
        "presentation_only": True,
        "scientific_raw_artifact": raw_artifact,
        "backward_compatible_raw_artifact": compatibility_artifact,
        "display_artifact": display_artifact,
        "raw_point_count": cleaning.raw_count,
        "display_filtered_point_count": cleaning.display_count,
        "removed_for_display_count": cleaning.removed_count,
        # Short aliases retained for convenient programmatic consumption.
        "display_point_count": cleaning.display_count,
        "removed_point_count": cleaning.removed_count,
        "nonfinite_removed": cleaning.nonfinite_removed,
        "z_percentile_removed": cleaning.z_percentile_removed,
        "center_distance_removed": cleaning.center_distance_removed,
        "z_bounds": cleaning.z_bounds,
        "center_distance_limit": cleaning.center_distance_limit,
        "coordinate_scale": "relative_non_metric",
    }


def clean_point_cloud_for_display(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    z_percentiles: tuple[float, float] | None = (1.0, 99.0),
    center_distance_percentile: float | None = 99.5,
    center_mad_multiplier: float = 6.0,
) -> DisplayCleaningResult:
    """Return a conservative visual-only subset using robust statistics.

    Filtering is applied in this order: non-finite points, optional Z bounds,
    then distance from the median 3D center.  The final radius is the larger
    of a requested percentile and a median-plus-MAD bound, which keeps the
    operation conservative and deterministic.
    """

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {xyz.shape}")

    rgb: np.ndarray | None = None
    if colors is not None:
        rgb = np.asarray(colors)
        if rgb.ndim != 2 or rgb.shape != xyz.shape:
            raise ValueError(f"colors must match points shape {xyz.shape}, got {rgb.shape}")

    raw_count = int(xyz.shape[0])
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    if rgb is not None and np.issubdtype(rgb.dtype, np.floating):
        finite_mask &= np.all(np.isfinite(rgb), axis=1)
    xyz = xyz[finite_mask]
    if rgb is not None:
        rgb = rgb[finite_mask]
    nonfinite_removed = raw_count - int(xyz.shape[0])

    z_bounds: tuple[float, float] | None = None
    z_removed = 0
    if z_percentiles is not None and xyz.shape[0] > 0:
        low, high = map(float, z_percentiles)
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("z_percentiles must satisfy 0 <= low < high <= 100")
        lower, upper = np.percentile(xyz[:, 2], [low, high])
        z_bounds = (float(lower), float(upper))
        keep = (xyz[:, 2] >= lower) & (xyz[:, 2] <= upper)
        z_removed = int(xyz.shape[0] - np.count_nonzero(keep))
        xyz = xyz[keep]
        if rgb is not None:
            rgb = rgb[keep]

    center_limit: float | None = None
    center_removed = 0
    if xyz.shape[0] > 0 and center_distance_percentile is not None:
        percentile = float(center_distance_percentile)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("center_distance_percentile must be in (0, 100]")
        if center_mad_multiplier < 0.0:
            raise ValueError("center_mad_multiplier must be non-negative")

        center = np.median(xyz, axis=0)
        distances = np.linalg.norm(xyz - center, axis=1)
        distance_median = float(np.median(distances))
        distance_mad = float(np.median(np.abs(distances - distance_median)))
        mad_limit = distance_median + float(center_mad_multiplier) * 1.4826 * distance_mad
        percentile_limit = float(np.percentile(distances, percentile))
        center_limit = max(mad_limit, percentile_limit)
        keep = distances <= center_limit
        center_removed = int(xyz.shape[0] - np.count_nonzero(keep))
        xyz = xyz[keep]
        if rgb is not None:
            rgb = rgb[keep]

    display_count = int(xyz.shape[0])
    return DisplayCleaningResult(
        points=xyz,
        colors=rgb,
        raw_count=raw_count,
        display_count=display_count,
        removed_count=raw_count - display_count,
        nonfinite_removed=nonfinite_removed,
        z_percentile_removed=z_removed,
        center_distance_removed=center_removed,
        z_bounds=z_bounds,
        center_distance_limit=center_limit,
    )


def _subsample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _scatter_colors(colors: np.ndarray | None, indices: np.ndarray) -> np.ndarray | str:
    if colors is None:
        return "#58a6ff"
    selected = np.asarray(colors)[indices].astype(np.float64)
    if selected.size and np.nanmax(selected) > 1.0:
        selected /= 255.0
    return np.clip(selected, 0.0, 1.0)


def _set_equal_3d_axes(axis: plt.Axes, points: np.ndarray) -> None:
    if points.shape[0] == 0:
        return
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:  # pragma: no cover - old Matplotlib compatibility
        pass


def _plot_point_cloud(
    axis: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    view: str,
    max_points: int,
) -> None:
    indices = _subsample_indices(points.shape[0], max_points)
    shown = points[indices]
    if shown.shape[0] > 0:
        axis.scatter(
            shown[:, 0], shown[:, 1], shown[:, 2],
            c=_scatter_colors(colors, indices), s=1.0, alpha=0.8,
            linewidths=0, depthshade=False,
        )

    views = {
        "front": (0.0, -90.0),
        "oblique": (24.0, -58.0),
        "top": (90.0, -90.0),
    }
    if view not in views:
        raise ValueError(f"unsupported point-cloud view: {view}")
    elevation, azimuth = views[view]
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("X (relative)")
    axis.set_ylabel("Y (relative)")
    axis.set_zlabel("Z (relative)")
    axis.grid(True, alpha=0.2)
    _set_equal_3d_axes(axis, shown)


def save_point_cloud_preview(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    view: str = "oblique",
    title: str = "Relative point cloud (display only, non-metric)",
    max_points: int = 40_000,
) -> Path:
    """Render a point cloud to a deterministic PNG view."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    figure.patch.set_facecolor("#f2f2f2")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#f7f7f7")
    _plot_point_cloud(axis, np.asarray(points), colors, view=view, max_points=max_points)
    axis.set_title(title)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_depth_stabilization_comparison(
    directory: str | Path,
    unstabilized_points: np.ndarray,
    unstabilized_colors: np.ndarray,
    stabilized_points: np.ndarray,
    stabilized_colors: np.ndarray,
    *,
    preview_max_points: int = 40_000,
) -> dict[str, Path]:
    """Save matched raw-map comparisons without display-only point filtering."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "unstabilized_ply": output / "global_relative_map_unstabilized.ply",
        "stabilized_ply": output / "global_relative_map_stabilized.ply",
        "unstabilized_oblique": output / "global_map_unstabilized_oblique.png",
        "stabilized_oblique": output / "global_map_stabilized_oblique.png",
        "unstabilized_top": output / "global_map_unstabilized_top.png",
        "stabilized_top": output / "global_map_stabilized_top.png",
    }
    write_ascii_ply(
        paths["unstabilized_ply"], unstabilized_points, unstabilized_colors
    )
    write_ascii_ply(paths["stabilized_ply"], stabilized_points, stabilized_colors)
    for variant, points, colors in (
        ("unstabilized", unstabilized_points, unstabilized_colors),
        ("stabilized", stabilized_points, stabilized_colors),
    ):
        title = f"{variant.capitalize()} relative map (non-metric)"
        for view in ("oblique", "top"):
            save_point_cloud_preview(
                paths[f"{variant}_{view}"],
                points,
                colors,
                view=view,
                title=title,
                max_points=preview_max_points,
            )
    return paths


def save_temporal_depth_normalization_comparison(
    directory: str | Path,
    baseline_points: np.ndarray,
    baseline_colors: np.ndarray,
    normalized_points: np.ndarray,
    normalized_colors: np.ndarray,
    *,
    preview_max_points: int = 40_000,
) -> dict[str, Path]:
    """Save matched baseline/normalized maps without display-only filtering."""

    output = Path(directory)
    paths = {
        "baseline_ply": output / "global_relative_map_baseline.ply",
        "temporal_normalized_ply": (
            output / "global_relative_map_temporal_normalized.ply"
        ),
    }
    write_ascii_ply(paths["baseline_ply"], baseline_points, baseline_colors)
    write_ascii_ply(
        paths["temporal_normalized_ply"], normalized_points, normalized_colors
    )
    for variant, points, colors in (
        ("baseline", baseline_points, baseline_colors),
        ("temporal_normalized", normalized_points, normalized_colors),
    ):
        title = (
            "Baseline relative map (non-metric)"
            if variant == "baseline"
            else "Temporal-normalized relative map (non-metric)"
        )
        for view in ("front", "oblique", "top"):
            key = f"{variant}_{view}"
            paths[key] = output / f"global_map_{variant}_{view}.png"
            save_point_cloud_preview(
                paths[key], points, colors, view=view, title=title,
                max_points=preview_max_points,
            )
    return paths


def save_cloud_visual_artifacts(
    directory: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    raw_filenames: Sequence[str],
    display_filename: str,
    preview_prefix: str,
    title: str,
    save_display_clean: bool = True,
    save_previews: bool = True,
    z_percentiles: tuple[float, float] | None = (1.0, 99.0),
    center_distance_percentile: float | None = 99.5,
    center_mad_multiplier: float = 6.0,
    preview_max_points: int = 40_000,
) -> CloudVisualArtifacts:
    """Write preserved raw PLY files plus optional display-only artifacts."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    raw_paths = tuple(output / name for name in raw_filenames)
    for raw_path in raw_paths:
        write_ascii_ply(raw_path, points, colors)

    cleaning = clean_point_cloud_for_display(
        points,
        colors,
        z_percentiles=z_percentiles,
        center_distance_percentile=center_distance_percentile,
        center_mad_multiplier=center_mad_multiplier,
    )
    display_path: Path | None = None
    if save_display_clean:
        display_path = output / display_filename
        write_ascii_ply(display_path, cleaning.points, cleaning.colors)

    preview_paths: list[Path] = []
    if save_previews:
        for view in ("front", "oblique", "top"):
            preview_paths.append(
                save_point_cloud_preview(
                    output / f"{preview_prefix}_{view}.png",
                    cleaning.points,
                    cleaning.colors,
                    view=view,
                    title=title,
                    max_points=preview_max_points,
                )
            )

    return CloudVisualArtifacts(
        raw_paths=raw_paths,
        display_path=display_path,
        preview_paths=tuple(preview_paths),
        cleaning=cleaning,
    )


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        return array[:, :, ::-1]
    return array


def save_rgb_depth_side_by_side(
    path: str | Path,
    image_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    *,
    title: str = "RGB and Depth Anything V2 relative depth (non-metric)",
) -> Path:
    """Save a titled RGB/depth comparison suitable for reports."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(_bgr_to_rgb(image_bgr))
    axes[0].set_title("RGB input")
    axes[1].imshow(_bgr_to_rgb(depth_bgr))
    axes[1].set_title("Relative depth (non-metric)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def save_trajectory_previews(
    directory: str | Path,
    trajectory_xyz: np.ndarray,
    *,
    prefix: str = "trajectory",
) -> tuple[Path, Path, Path]:
    """Save XZ, XY, and 3D views of a relative camera trajectory."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = np.asarray(trajectory_xyz, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"trajectory must have shape (N, 3), got {trajectory.shape}")

    paths: list[Path] = []
    for horizontal, vertical, suffix in ((0, 2, "xz"), (0, 1, "xy")):
        path = output / f"{prefix}_{suffix}.png"
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        axis.plot(trajectory[:, horizontal], trajectory[:, vertical], "-o", markersize=3)
        axis.set_xlabel(f"{'XYZ'[horizontal]} (relative)")
        axis.set_ylabel(f"{'XYZ'[vertical]} (relative)")
        axis.set_title(f"Camera trajectory {suffix.upper()} (relative, non-metric)")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)

    path_3d = output / f"{prefix}_3d.png"
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "-o", markersize=3)
    axis.set_xlabel("X (relative)")
    axis.set_ylabel("Y (relative)")
    axis.set_zlabel("Z (relative)")
    axis.set_title("Camera trajectory 3D (relative, non-metric)")
    _set_equal_3d_axes(axis, trajectory)
    axis.grid(True, alpha=0.3)
    figure.savefig(path_3d, dpi=150)
    plt.close(figure)
    paths.append(path_3d)
    return paths[0], paths[1], paths[2]


def save_map_overview_panel(
    path: str | Path,
    image_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    trajectory_xyz: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    max_points: int = 20_000,
) -> Path:
    """Save a compact RGB, depth, trajectory, and map presentation panel."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    rgb_axis = figure.add_subplot(2, 2, 1)
    rgb_axis.imshow(_bgr_to_rgb(image_bgr))
    rgb_axis.set_title("Representative RGB frame")
    rgb_axis.axis("off")

    depth_axis = figure.add_subplot(2, 2, 2)
    depth_axis.imshow(_bgr_to_rgb(depth_bgr))
    depth_axis.set_title("Relative depth (non-metric)")
    depth_axis.axis("off")

    trajectory = np.asarray(trajectory_xyz)
    trajectory_axis = figure.add_subplot(2, 2, 3)
    trajectory_axis.plot(trajectory[:, 0], trajectory[:, 2], "-o", markersize=3)
    trajectory_axis.set_xlabel("X (relative)")
    trajectory_axis.set_ylabel("Z (relative)")
    trajectory_axis.set_title("Camera trajectory XZ")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True, alpha=0.3)

    cloud_axis = figure.add_subplot(2, 2, 4, projection="3d")
    _plot_point_cloud(
        cloud_axis, np.asarray(points), colors, view="oblique", max_points=max_points
    )
    cloud_axis.set_title("Display-cleaned relative map")

    figure.suptitle("DepthSLAM visual overview — relative geometry, not metric scale")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
