"""Guarded one-pass splitting for merged atomic geometry regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .geometry import compute_normals_cross_np, compute_normals_sobel_np


# --- Inlined instead of `from pipeline.config import AtomicSplitMode` --------
# The reference module imports `AtomicSplitMode` from the package
# `pipeline.config`. That package does not exist in this repository (it was not
# ported) and no target module offers an equivalent, so the enum is inlined here
# verbatim from the reference `pipeline/config.py`. Inlining keeps both the
# `AtomicSplitMode(split_mode)` validation (including the
# "unknown atomic split_mode" ValueError) and the `is AtomicSplitMode.NONE`
# identity check behaving exactly as in the reference.
class AtomicSplitMode(str, Enum):
    NONE = "none"
    CONSERVATIVE = "conservative"
    NORMAL_ONLY = "normal_only"
# ---------------------------------------------------------------------------


NORMAL_BARRIER_RAD = np.deg2rad(30.0)
MAX_LEAVES = 4
MIN_CHILD_FRACTION = 0.02
EPS = 1e-8


@dataclass(frozen=True)
class SplitDiagnostics:
    split_mode: str
    split_parent_count: int = 0
    split_proposed_count: int = 0
    split_accepted_count: int = 0
    split_added_regions: int = 0
    split_score_mean: float = 0.0
    split_score_max: float = 0.0
    split_reject_no_markers: int = 0
    split_reject_small_child: int = 0
    split_reject_low_score: int = 0
    split_runtime_ms: float = 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.intp, copy=False)


def _normal_map(point_map, normal_method):
    if normal_method == "cross":
        normals = compute_normals_cross_np(point_map)
    elif normal_method == "sobel":
        normals = compute_normals_sobel_np(point_map)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")

    valid = np.isfinite(point_map).all(axis=-1)
    valid &= np.isfinite(normals).all(axis=-1)
    valid &= np.linalg.norm(normals, axis=-1) > EPS
    try:
        import cv2

        filtered = cv2.medianBlur(
            normals.astype(np.float32, copy=False),
            3,
        )
    except ImportError:
        filtered = np.stack(
            [
                ndimage.median_filter(
                    normals[..., channel],
                    size=3,
                    mode="nearest",
                )
                for channel in range(3)
            ],
            axis=-1,
        )
    norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
    filtered = np.divide(
        filtered,
        norm,
        out=np.zeros_like(filtered),
        where=norm > EPS,
    )
    valid &= np.linalg.norm(filtered, axis=-1) > EPS
    filtered[~valid] = 0.0
    return filtered.astype(np.float32, copy=False), valid


def _normal_edge_map(normals, valid):
    edge = np.zeros(normals.shape[:2], dtype=np.float32)
    pairs = (
        (
            normals[:, :-1],
            normals[:, 1:],
            valid[:, :-1],
            valid[:, 1:],
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            normals[:-1],
            normals[1:],
            valid[:-1],
            valid[1:],
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    )
    for (
        left,
        right,
        valid_left,
        valid_right,
        left_slice,
        right_slice,
    ) in pairs:
        pair_valid = valid_left & valid_right
        dot = np.sum(left * right, axis=-1)
        angle = np.zeros(dot.shape, dtype=np.float32)
        angle[pair_valid] = np.arccos(
            np.clip(np.abs(dot[pair_valid]), 0.0, 1.0)
        )
        edge[left_slice] = np.maximum(edge[left_slice], angle)
        edge[right_slice] = np.maximum(edge[right_slice], angle)
    return edge


def _normal_dispersion(normals, mask):
    selected = normals[mask]
    if selected.size == 0:
        return np.nan
    moment = np.einsum("ni,nj->ij", selected, selected)
    moment = moment / selected.shape[0]
    return float(max(0.0, 1.0 - np.linalg.eigvalsh(moment)[-1]))


def _minimum_child_area(parent_area, seg_min_size):
    return max(
        int(seg_min_size),
        int(np.ceil(MIN_CHILD_FRACTION * parent_area)),
    )


def _markers(parent_mask, valid, normal_edge, min_marker_area):
    seed_mask = parent_mask & valid & (normal_edge < NORMAL_BARRIER_RAD)
    components, count = ndimage.label(seed_mask)
    if count < 2:
        return np.zeros(parent_mask.shape, dtype=np.int32), count

    sizes = np.bincount(
        components.reshape(-1),
        minlength=count + 1,
    )[1:]
    component_ids = np.arange(1, count + 1)
    eligible = sizes >= min_marker_area
    sizes = sizes[eligible]
    component_ids = component_ids[eligible]
    if component_ids.size < 2:
        return (
            np.zeros(parent_mask.shape, dtype=np.int32),
            component_ids.size,
        )

    order = np.lexsort((component_ids, -sizes))[:MAX_LEAVES]
    selected = component_ids[order]
    markers = np.zeros(parent_mask.shape, dtype=np.int32)
    for marker_id, component_id in enumerate(selected, start=1):
        markers[components == component_id] = marker_id
    return markers, selected.size


def _rgb_float(rgb_image, shape):
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image)
    if rgb.shape != (*shape, 3):
        raise ValueError("rgb_image must have shape (H, W, 3)")
    rgb = rgb.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
    if max_value > 1.0:
        rgb = rgb / 255.0
    return rgb


def _edge_fields(point_map, rgb_image, atom_labels, atom_scales):
    """Return right/down RGB and locally normalized 3D edge strengths."""

    rgb = _rgb_float(rgb_image, point_map.shape[:2])
    fields = {}
    for axis, first_slice, second_slice in (
        (
            "right",
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            "down",
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    ):
        first_points = point_map[first_slice]
        second_points = point_map[second_slice]
        with np.errstate(invalid="ignore", over="ignore"):
            distance = np.linalg.norm(
                first_points - second_points,
                axis=-1,
            )
        first_atom = atom_labels[first_slice]
        second_atom = atom_labels[second_slice]
        denominator = np.sqrt(
            atom_scales[first_atom] * atom_scales[second_atom]
        )
        gap = np.zeros(distance.shape, dtype=np.float32)
        valid_gap = (
            np.isfinite(distance)
            & np.isfinite(denominator)
            & (denominator > EPS)
        )
        np.divide(distance, denominator, out=gap, where=valid_gap)
        fields[f"gap_{axis}"] = gap

        if rgb is None:
            fields[f"rgb_{axis}"] = None
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                rgb_edge = np.linalg.norm(
                    rgb[first_slice] - rgb[second_slice],
                    axis=-1,
                )
            fields[f"rgb_{axis}"] = np.nan_to_num(
                rgb_edge,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
    return fields


def _field_contrast(candidate, parent_mask, horizontal, vertical):
    if horizontal is None or vertical is None:
        return 0.0

    pair_parent_h = parent_mask[:, :-1] & parent_mask[:, 1:]
    pair_parent_v = parent_mask[:-1] & parent_mask[1:]
    boundary_h = pair_parent_h & (
        candidate[:, :-1] != candidate[:, 1:]
    )
    boundary_v = pair_parent_v & (
        candidate[:-1] != candidate[1:]
    )
    interior_h = pair_parent_h & (
        candidate[:, :-1] == candidate[:, 1:]
    )
    interior_v = pair_parent_v & (
        candidate[:-1] == candidate[1:]
    )

    boundary = np.concatenate(
        (horizontal[boundary_h], vertical[boundary_v])
    )
    interior = np.concatenate(
        (horizontal[interior_h], vertical[interior_v])
    )
    boundary = boundary[np.isfinite(boundary)]
    interior = interior[np.isfinite(interior)]
    if boundary.size == 0:
        return 0.0
    boundary_strength = float(np.median(boundary))
    interior_strength = (
        float(np.quantile(interior, 0.75)) if interior.size else 0.0
    )
    ratio = boundary_strength / max(interior_strength, EPS)
    return float(
        np.clip((ratio - 1.0) / (ratio + 1.0), 0.0, 1.0)
    )


def _normal_gain(normals, valid, parent_mask, candidate):
    parent_valid = parent_mask & valid
    parent_dispersion = _normal_dispersion(normals, parent_valid)
    if not np.isfinite(parent_dispersion) or parent_dispersion <= EPS:
        return 0.0

    child_total = 0
    child_weighted_dispersion = 0.0
    for child_id in np.unique(candidate[parent_mask]):
        child_valid = parent_valid & (candidate == child_id)
        child_count = int(np.count_nonzero(child_valid))
        child_dispersion = _normal_dispersion(normals, child_valid)
        if child_count == 0 or not np.isfinite(child_dispersion):
            continue
        child_total += child_count
        child_weighted_dispersion += child_count * child_dispersion
    if child_total == 0:
        return 0.0
    after = child_weighted_dispersion / child_total
    return float(
        np.clip(
            (parent_dispersion - after) / parent_dispersion,
            0.0,
            1.0,
        )
    )


def _validate_inputs(
    point_map,
    rgb_image,
    auto_labels,
    atom_labels,
    atom_scales,
):
    point_map = np.asarray(point_map)
    auto_labels = np.asarray(auto_labels)
    atom_labels = np.asarray(atom_labels)
    atom_scales = np.asarray(atom_scales, dtype=np.float64)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape (H, W, 3)")
    if auto_labels.shape != point_map.shape[:2]:
        raise ValueError("auto_labels must have shape (H, W)")
    if auto_labels.size and np.min(auto_labels) < 0:
        raise ValueError("auto_labels must be non-negative")
    if atom_labels.shape != point_map.shape[:2]:
        raise ValueError("atom_labels must have shape (H, W)")
    if atom_labels.size and (
        np.min(atom_labels) < 0
        or np.max(atom_labels) >= atom_scales.size
    ):
        raise ValueError(
            "atom_scales must contain one entry per compact atom label"
        )
    if rgb_image is not None and np.asarray(rgb_image).shape != (
        *point_map.shape[:2],
        3,
    ):
        raise ValueError("rgb_image must have shape (H, W, 3)")
    return (
        point_map,
        _compact_labels(auto_labels),
        atom_labels.astype(np.intp, copy=False),
        atom_scales,
    )


def refine_auto_regions(
    point_map,
    rgb_image,
    auto_labels,
    atom_labels,
    atom_scales,
    seg_min_size,
    normal_method,
    split_score_threshold,
    split_mode,
):
    """Split each original merged region at most once."""

    start = time.perf_counter()
    try:
        split_mode = AtomicSplitMode(split_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown atomic split_mode: {split_mode!r}") from exc
    if (
        not np.isfinite(split_score_threshold)
        or split_score_threshold < 0
    ):
        raise ValueError(
            "split_score_threshold must be finite and non-negative"
        )
    if int(seg_min_size) <= 0:
        raise ValueError("seg_min_size must be positive")

    point_map, labels, atom_labels, atom_scales = _validate_inputs(
        point_map,
        rgb_image,
        auto_labels,
        atom_labels,
        atom_scales,
    )
    if split_mode is AtomicSplitMode.NONE:
        return labels, SplitDiagnostics(
            split_mode=split_mode.value,
            split_runtime_ms=(time.perf_counter() - start) * 1000.0,
        )

    normals, valid = _normal_map(point_map, normal_method)
    normal_edge = _normal_edge_map(normals, valid)
    result = labels.copy()
    next_label = int(result.max()) + 1 if result.size else 0
    parent_count = proposed_count = accepted_count = added_regions = 0
    reject_no_markers = reject_small_child = reject_low_score = 0
    scores = []

    parent_areas = np.bincount(labels.reshape(-1))
    parent_crops = ndimage.find_objects(labels + 1)
    for parent_id, crop in enumerate(parent_crops):
        if crop is None:
            continue
        parent_area = int(parent_areas[parent_id])
        min_child_area = _minimum_child_area(
            parent_area,
            seg_min_size,
        )
        if parent_area < 2 * min_child_area:
            continue
        parent_count += 1

        crop_parent = labels[crop] == parent_id
        crop_valid = valid[crop]
        crop_edge = normal_edge[crop]
        markers, marker_count = _markers(
            crop_parent,
            crop_valid,
            crop_edge,
            min_child_area,
        )
        if marker_count < 2:
            reject_no_markers += 1
            continue
        proposed_count += 1

        elevation = crop_edge.copy()
        invalid_crop = crop_parent & ~crop_valid
        valid_elevation = elevation[crop_parent & crop_valid]
        invalid_level = (
            float(np.max(valid_elevation))
            if valid_elevation.size
            else NORMAL_BARRIER_RAD
        )
        elevation[invalid_crop] = invalid_level
        candidate = watershed(
            elevation,
            markers=markers,
            mask=crop_parent,
        )
        child_ids, child_sizes = np.unique(
            candidate[crop_parent],
            return_counts=True,
        )
        nonzero = child_ids > 0
        child_ids = child_ids[nonzero]
        child_sizes = child_sizes[nonzero]
        if (
            child_ids.size < 2
            or child_ids.size > MAX_LEAVES
            or np.any(child_sizes < min_child_area)
            or int(np.sum(child_sizes)) != parent_area
        ):
            reject_small_child += 1
            continue

        crop_normals = normals[crop]
        normal_gain = _normal_gain(
            crop_normals,
            crop_valid,
            crop_parent,
            candidate,
        )
        if split_mode is AtomicSplitMode.NORMAL_ONLY:
            score = normal_gain
        else:
            crop_rgb = (
                None
                if rgb_image is None
                else np.asarray(rgb_image)[crop]
            )
            edge_fields = _edge_fields(
                point_map[crop],
                crop_rgb,
                atom_labels[crop],
                atom_scales,
            )
            rgb_confirmation = _field_contrast(
                candidate,
                crop_parent,
                edge_fields["rgb_right"],
                edge_fields["rgb_down"],
            )
            gap_confirmation = _field_contrast(
                candidate,
                crop_parent,
                edge_fields["gap_right"],
                edge_fields["gap_down"],
            )
            score = normal_gain * max(
                rgb_confirmation,
                gap_confirmation,
            )

        scores.append(score)
        if not np.isfinite(score) or score < split_score_threshold:
            reject_low_score += 1
            continue

        accepted_count += 1
        ordered_children = sorted(
            zip(child_ids.tolist(), child_sizes.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        keep_child = ordered_children[0][0]
        result_crop = result[crop]
        result_crop[
            crop_parent & (candidate == keep_child)
        ] = parent_id
        for child_id, _ in ordered_children[1:]:
            result_crop[
                crop_parent & (candidate == child_id)
            ] = next_label
            next_label += 1
            added_regions += 1

    compact = _compact_labels(result)
    diagnostics = SplitDiagnostics(
        split_mode=split_mode.value,
        split_parent_count=parent_count,
        split_proposed_count=proposed_count,
        split_accepted_count=accepted_count,
        split_added_regions=added_regions,
        split_score_mean=float(np.mean(scores)) if scores else 0.0,
        split_score_max=float(np.max(scores)) if scores else 0.0,
        split_reject_no_markers=reject_no_markers,
        split_reject_small_child=reject_small_child,
        split_reject_low_score=reject_low_score,
        split_runtime_ms=(time.perf_counter() - start) * 1000.0,
    )
    return compact, diagnostics
