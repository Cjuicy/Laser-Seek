"""Analytically exact synthetic scenes for the IDEA-001 L2 checkpoints.

Why analytic scenes
-------------------
The IDEA-001 question is *why* `geometry` and `atomic` move ATE where `depth` does not. Answering
it needs the intermediate stages to have known-correct answers, and a real dataset cannot supply
those: nobody knows the true depth-layer partition of a Sintel frame. These fixtures therefore
render a world made of explicit planes, where the ground-truth per-pixel layer id, the true depth
and the true inter-window per-layer scale are all exact by construction.

The camera model is an ideal pinhole with known intrinsics, and rendering is analytic ray/plane
intersection, so there is no rasterisation error and no randomness beyond the seeds used to place
the planes. Every scenario targets one concrete failure mode described in the Idea document:

===========  ==========================================================================
F1 crease    Two planes whose depth is continuous across the shared edge but whose normals
             differ by 40 degrees. A criterion that only looks at depth differences merges
             them; a normal-aware criterion keeps them apart. This is the only fixture that
             isolates `geometry`'s normal condition.
F2 far field A single large plane with a slow depth gradient, plus a second plane at a
             slightly different depth. Emulates the far-field fragmentation that pure
             depth-difference merging produces when the depth range is large.
F3 occluder  A thin pole standing in front of a background plane. The initial Felzenszwalb
             atoms span the pole and the background on both sides, so the coarse layer is
             wrong and only a geometry-aware merge can refuse to join them.
F4 scale     Two windows of one world. Window 2 multiplies every layer's depth by a KNOWN
             per-layer factor, which is exactly the monocular scale ambiguity LSA exists to
             undo. The fixture also carries `merged_scale`, a single global factor, so that
             "how much of the per-layer variation can one global scale absorb" is measurable.
===========  ==========================================================================

Ground truth produced per frame: `depth`, `points` (local camera-frame point map),
`layer_label` (the true plane index), `normal`, `valid_mask`, and a metric `scale` per layer
relative to the window-1 world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Plane:
    """A world plane given by a point and a unit normal, plus a readable layer name.

    The plane is oriented so that the region it covers is the set of points whose offset along
    `normal` is negative; `tangent_u` / `tangent_v` span the covered rectangle around `origin`.
    """

    name: str
    origin: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    half_extent_u: float
    half_extent_v: float


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def make_plane(
    name: str,
    origin: Sequence[float],
    normal: Sequence[float],
    half_u: float = 12.0,
    half_v: float = 12.0,
) -> Plane:
    normal = _unit(normal)
    # Any vector not parallel to the normal gives a stable tangent basis.
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent_u = _unit(np.cross(normal, seed))
    tangent_v = _unit(np.cross(normal, tangent_u))
    return Plane(
        name=name,
        origin=np.asarray(origin, dtype=np.float64),
        normal=normal,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        half_extent_u=float(half_u),
        half_extent_v=float(half_v),
    )


@dataclass
class Render:
    depth: np.ndarray            # (H, W) z along the camera axis
    points: np.ndarray           # (H, W, 3) local camera-frame points
    layer_label: np.ndarray      # (H, W) ground-truth plane index, -1 where nothing was hit
    normal: np.ndarray           # (H, W, 3) unit surface normal in the camera frame
    valid_mask: np.ndarray       # (H, W) bool
    intrinsic: np.ndarray        # (3, 3)
    layer_names: list[str] = field(default_factory=list)


@dataclass
class WindowRender:
    frames: list[Render]
    scales: dict[int, float]     # true metric scale of each layer relative to window-1 world
    global_scale: float          # the single scale that maps this window onto window 1


def _intrinsic(width: int, height: int, focal: float) -> np.ndarray:
    return np.array(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation(axis: Sequence[float], degrees: float) -> np.ndarray:
    axis = _unit(axis)
    angle = np.deg2rad(degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def render_planes(
    planes: Sequence[Plane],
    *,
    width: int = 64,
    height: int = 48,
    focal: float = 60.0,
    camera_pose: np.ndarray | None = None,
    layer_scales: dict[int, float] | None = None,
    max_depth: float = 60.0,
) -> Render:
    """Render the world planes from one camera by analytic ray/plane intersection.

    `layer_scales` scales each plane's world geometry before rendering, which is how the fixture
    injects the monocular per-layer scale ambiguity: a plane rendered with scale `s` sits at `s`
    times its window-1 distance, and no global transform of the resulting point map can undo that.
    """
    if camera_pose is None:
        camera_pose = np.eye(4, dtype=np.float64)
    camera_pose = np.asarray(camera_pose, dtype=np.float64)

    intrinsic = _intrinsic(width, height, focal)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    direction_cam = np.stack(
        [
            (grid_x - intrinsic[0, 2]) / focal,
            (grid_y - intrinsic[1, 2]) / focal,
            np.ones_like(grid_x),
        ],
        axis=-1,
    )

    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    direction_world = direction_cam @ rotation.T

    best_depth = np.full((height, width), np.inf)
    best_label = np.full((height, width), -1, dtype=np.int64)
    best_normal = np.zeros((height, width, 3), dtype=np.float64)

    for index, plane in enumerate(planes):
        scale = 1.0 if not layer_scales else float(layer_scales.get(index, 1.0))
        origin = plane.origin * scale
        # Offset of the camera from the plane, along the plane normal.
        camera_offset = float(np.dot(translation - origin, plane.normal))
        ray_offset = direction_world @ plane.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.where(np.abs(ray_offset) > 1e-12, -camera_offset / ray_offset, np.inf)
        hit_point_world = translation + distance[..., None] * direction_world
        offset = hit_point_world - origin
        along_u = offset @ plane.tangent_u
        along_v = offset @ plane.tangent_v
        inside = (
            (np.abs(along_u) <= plane.half_extent_u * scale)
            & (np.abs(along_v) <= plane.half_extent_v * scale)
        )
        # Depth is measured along the camera axis; the ray has unit z in the camera frame.
        depth = np.where(np.isfinite(distance), distance, np.inf)
        candidate = inside & (depth > 1e-6) & (depth < max_depth) & (depth < best_depth)
        best_depth = np.where(candidate, depth, best_depth)
        best_label = np.where(candidate, index, best_label)
        normal_cam = rotation.T @ plane.normal
        if np.dot(normal_cam, [0.0, 0.0, 1.0]) > 0:
            normal_cam = -normal_cam
        best_normal = np.where(candidate[..., None], normal_cam, best_normal)

    valid = np.isfinite(best_depth) & (best_label >= 0)
    depth = np.where(valid, best_depth, np.nan)
    points = direction_cam * np.where(valid, best_depth, np.nan)[..., None]
    return Render(
        depth=depth,
        points=points,
        layer_label=best_label,
        normal=best_normal,
        valid_mask=valid,
        intrinsic=intrinsic,
        layer_names=[plane.name for plane in planes],
    )


def render_window(
    planes: Sequence[Plane],
    *,
    frame_count: int,
    base_layer_scales: dict[int, float] | None = None,
    global_scale: float = 1.0,
    motion_translation: Sequence[float] = (0.012, 0.0, 0.0),
    motion_rotation_degrees: float = 0.15,
    width: int = 64,
    height: int = 48,
    focal: float = 60.0,
) -> WindowRender:
    """Render `frame_count` frames of one window with a small per-frame camera motion.

    The motion is what makes the fixture non-trivial: it shifts each layer's mask by a pixel or
    two between frames, so the cross-window intersection masks are proper subsets and the per-edge
    IRLS has to cope with partially overlapping regions, exactly as in a real window.

    It must stay small. The overlap test uses the LAST `overlap` frames of the anchor window
    against the FIRST `overlap` frames of the next one, which are `window_size - overlap` frames
    apart; a motion large enough to make those two blocks disagree beyond the IoU threshold leaves
    the correspondence graph empty, and then the per-edge stages legitimately have nothing to
    measure. Real 30 fps footage does not move that far between adjacent frames.
    """
    frames: list[Render] = []
    for frame in range(frame_count):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rotation([0.0, 1.0, 0.0], motion_rotation_degrees * frame)
        pose[:3, 3] = np.asarray(motion_translation, dtype=np.float64) * frame
        frames.append(
            render_planes(
                planes,
                width=width,
                height=height,
                focal=focal,
                camera_pose=pose,
                layer_scales=base_layer_scales,
            )
        )
    applied = 1.0 if not base_layer_scales else float(np.mean(list(base_layer_scales.values())))
    return WindowRender(
        frames=frames,
        scales=dict(base_layer_scales or {}),
        global_scale=float(global_scale) * applied,
    )


# --------------------------------------------------------------------------------------------
# The four scenarios.
# --------------------------------------------------------------------------------------------


def scenario_crease(**kwargs) -> tuple[list[Plane], dict]:
    """F1: depth-continuous but normal-discontinuous at a shared crease.

    Both planes meet at the line `x = 0, z = 4` and both recede away from the camera across that
    line, so crossing the crease changes the depth *gradient* and the normal but not the depth
    itself. A merge rule that compares mean depths of the regions either side sees a small
    difference and fuses them; a rule that also compares normals refuses.
    """
    # The ramp covers the left of the view and recedes across it; the background is a broad
    # fronto-parallel surface right behind. Their shared boundary moves a couple of pixels between
    # frames because of the per-frame camera motion, which is what makes the cross-window
    # intersection masks proper subsets rather than identical partitions.
    # The crease sits at (x=0.6, z=7). The ramp tilts 45 degrees about the y axis so that it
    # recedes away from the camera on the left of the crease, while the background stays
    # fronto-parallel on the right. Their normals therefore differ by exactly 45 degrees while the
    # depth is continuous along the crease, which is what makes this fixture specific to the
    # normal condition.
    ramp = make_plane(
        "ramp", [0.6, 0.0, 7.0], [np.sin(np.deg2rad(45.0)), 0.0, -np.cos(np.deg2rad(45.0))],
        half_u=9.0, half_v=16.0,
    )
    background = make_plane(
        "background", [0.0, 0.0, 8.0], [0.0, 0.0, -1.0], half_u=60.0, half_v=60.0
    )
    planes = [ramp, background]
    meta = {
        "name": "F1-crease",
        "intent": "depth is continuous across the crease; only the normal changes",
        "expected": {
            "depth": "fuses across the crease (no normal condition exists)",
            "geometry": "refuses to fuse across the crease (normal angle condition)",
            "atomic": "refuses to fuse across the crease (3D gap condition)",
        },
    }
    return planes, meta


def scenario_far_field(**kwargs) -> tuple[list[Plane], dict]:
    """F2: a deep depth range with a large near surface and a distant background.

    The far surface is the same world plane as the near one, just much further away, so the
    relative depth difference between the two layers stays well below the absolute merge threshold
    that `depth_merge_thresh * full_depth_range` produces for a deep scene. That is precisely the
    fragmentation mechanism the Idea document describes: the threshold is scaled by the whole
    scene depth range, so a far object at a modest relative offset gets fused into whatever is
    behind it.
    """
    # A fronto-parallel surface in the middle of a tilted backdrop. The backdrop's tilt gives the
    # scene a depth range, and `depth_merge_thresh * range` is an ABSOLUTE threshold, so a modest
    # offset between the two surfaces falls under it and the baseline fuses them. The normals
    # differ, which is the only thing a geometry-aware rule can use here.
    near = make_plane("near", [0.0, 0.0, 5.6], [0.0, 0.0, -1.0], half_u=0.85, half_v=0.75)
    far = make_plane(
        "far", [0.0, 0.0, 6.6], [-np.sin(np.deg2rad(25.0)), 0.0, -np.cos(np.deg2rad(25.0))],
        half_u=40.0, half_v=40.0,
    )
    planes = [near, far]
    meta = {
        "name": "F2-far-field",
        "intent": "a deep depth range inflates the absolute merge threshold",
        "expected": {
            "depth": "fuses the near surface into the far background (relative offset is small)",
            "geometry": "same absolute depth rule plus normal/confidence conditions",
            "atomic": "scale-normalised gap separates the near surface from the far background",
        },
    }
    return planes, meta


def scenario_occluder(**kwargs) -> tuple[list[Plane], dict]:
    """F3: a thin pole in front of a distant background, the classic wrong-coarse-layer case."""
    # At z=2.4 one world unit spans about 25 pixels, so half_u=0.25 makes the pole roughly
    # 12 pixels wide: wide enough for Felzenszwalb to produce a dedicated atom, narrow enough to
    # be a genuine thin occluder.
    pole = make_plane("pole", [0.0, 0.0, 2.4], [0.0, 0.0, -1.0], half_u=0.25, half_v=3.0)
    background = make_plane("background", [0.0, 0.0, 7.0], [0.0, 0.0, -1.0], half_u=30.0, half_v=30.0)
    planes = [pole, background]
    meta = {
        "name": "F3-occluder",
        "intent": "the initial atoms span pole and background, so the coarse layer is wrong",
        "expected": {
            "depth": "the coarse layer fuses the pole into the background",
            "geometry": "keeps the pole separate (depth difference and normal both differ)",
            "atomic": "keeps the pole separate through the cross-layer gap limit",
        },
    }
    return planes, meta


def scenario_scale(**kwargs) -> tuple[list[Plane], dict]:
    """F4: two windows of one world with a KNOWN per-layer scale factor in the second window."""
    background = make_plane("background", [0.0, 0.0, 8.0], [0.0, 0.0, -1.0], half_u=60.0, half_v=60.0)
    foreground = make_plane("foreground", [0.0, 0.0, 3.5], [0.0, 0.0, -1.0], half_u=0.9, half_v=0.9)
    planes = [foreground, background]
    meta = {
        "name": "F4-known-scale",
        "intent": "measure how well each method lets LSA recover the true per-layer scales",
        "expected": {
            "all": "the ground-truth-partition landing error is the reference floor",
        },
    }
    return planes, meta


SCENARIOS = {
    "F1-crease": scenario_crease,
    "F2-far-field": scenario_far_field,
    "F3-occluder": scenario_occluder,
    "F4-known-scale": scenario_scale,
}


def build_pair(
    scenario: str,
    *,
    frame_count: int = 4,
    overlap: int = 2,
    per_layer_warp: dict[int, float] | None = None,
    global_warp: float = 1.35,
    width: int = 64,
    height: int = 48,
    focal: float = 60.0,
) -> tuple[WindowRender, WindowRender, dict]:
    """Build the anchor window and a second window whose depths carry a known scale change.

    `per_layer_warp` gives the true per-layer depth factor of window 2 relative to window 1; it
    defaults to `global_warp` for every layer, which is the case a single scalar scale could fix.
    `global_warp` is the scalar that `align_cam_pts_irls` is expected to recover.
    """
    build = SCENARIOS[scenario]
    planes, meta = build()
    layer_count = len(planes)

    anchor = render_window(
        planes,
        frame_count=frame_count,
        width=width,
        height=height,
        focal=focal,
    )

    if per_layer_warp is None:
        per_layer_warp = {index: global_warp for index in range(layer_count)}
    missing = [index for index in range(layer_count) if index not in per_layer_warp]
    if missing:
        raise ValueError(f"per_layer_warp is missing layers {missing}")

    target = render_window(
        planes,
        frame_count=frame_count,
        base_layer_scales=per_layer_warp,
        global_scale=global_warp,
        width=width,
        height=height,
        focal=focal,
    )

    meta = dict(meta)
    meta.update(
        {
            "frame_count": frame_count,
            "overlap": overlap,
            "layer_names": [plane.name for plane in planes],
            "per_layer_warp": {int(k): float(v) for k, v in per_layer_warp.items()},
            "global_warp": float(global_warp),
            "width": width,
            "height": height,
        }
    )
    return anchor, target, meta


def stack_points(render: Render, frames: Iterable[int]) -> np.ndarray:
    """Stack selected frames' local point maps into `(N, H, W, 3)`."""
    return np.stack([render.frames[index].points for index in frames])


def stack_depth(render: Render, frames: Iterable[int]) -> np.ndarray:
    return np.stack([render.frames[index].depth for index in frames])


def stack_labels(render: Render, frames: Iterable[int]) -> np.ndarray:
    return np.stack([render.frames[index].layer_label for index in frames])


def layer_masks(render: Render, frame_index: int, layer_count: int) -> list[np.ndarray]:
    """Boolean mask per ground-truth layer for one frame."""
    labels = render.frames[frame_index].layer_label
    return [labels == index for index in range(layer_count)]


__all__ = [
    "Plane",
    "Render",
    "WindowRender",
    "make_plane",
    "render_planes",
    "render_window",
    "build_pair",
    "stack_points",
    "stack_depth",
    "stack_labels",
    "layer_masks",
    "SCENARIOS",
]
