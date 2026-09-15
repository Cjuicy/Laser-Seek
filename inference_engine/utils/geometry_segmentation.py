"""Geometry-aware segmentation label generation from LASER-Geometry."""

import warnings

import numpy as np
from skimage.segmentation import felzenszwalb

from .geometry import build_geometry_info_np
from .segmentation_trace import SegmentationStages
from .confidence import select_numpy_top_confidence_mask


def _select_batch_item(value, batch_idx, single_frame_ndim):
    if value is None or batch_idx is None:
        return value
    value = np.asarray(value)
    if value.ndim == single_frame_ndim + 1:
        return value[batch_idx]
    return value


def compute_region_geometry_descriptors(labels, depth, geometry_info, conf=None):
    descriptors = {}
    normals = geometry_info["normal"]

    for label_id in np.unique(labels):
        mask = labels == label_id
        if mask.sum() == 0:
            continue

        area = int(mask.sum())
        mean_depth = float(np.nanmean(depth[mask]))
        region_normals = normals[mask]
        mean_normal = np.nanmean(region_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
        normal_variance = float(
            np.nanmean(np.linalg.norm(region_normals - mean_normal, axis=-1))
        )
        mean_conf = float(np.nanmean(conf[mask])) if conf is not None else 1.0
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        descriptors[label_id] = {
            "area": area,
            "mean_depth": mean_depth,
            "mean_normal": mean_normal,
            "normal_variance": normal_variance,
            "mean_conf": mean_conf,
            "bbox": bbox,
        }

    return descriptors


def should_merge_geometry(
    desc_a,
    desc_b,
    depth_thresh,
    normal_cos_thresh,
    conf_thresh=None,
):
    depth_diff = abs(desc_a["mean_depth"] - desc_b["mean_depth"])
    normal_sim = float(np.dot(desc_a["mean_normal"], desc_b["mean_normal"]))

    if depth_diff > depth_thresh:
        return False
    if normal_sim < normal_cos_thresh:
        return False
    if conf_thresh is not None:
        if min(desc_a["mean_conf"], desc_b["mean_conf"]) < conf_thresh:
            return False
    return True


def merge_regions_geometry(
    labels,
    depth,
    geometry_info,
    conf=None,
    depth_thresh=None,
    normal_thresh_deg=20.0,
    conf_thresh=None,
):
    descriptors = compute_region_geometry_descriptors(
        labels,
        depth,
        geometry_info,
        conf=conf,
    )
    normal_cos_thresh = np.cos(np.deg2rad(normal_thresh_deg))

    if depth_thresh is None:
        depth_range = np.nanmax(depth) - np.nanmin(depth)
        depth_thresh = 0.05 * depth_range

    parent = {int(label_id): int(label_id) for label_id in descriptors}

    def find(label_id):
        label_id = int(label_id)
        while parent[label_id] != label_id:
            parent[label_id] = parent[parent[label_id]]
            label_id = parent[label_id]
        return label_id

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    right_pairs = np.stack(
        [labels[:, :-1].reshape(-1), labels[:, 1:].reshape(-1)],
        axis=-1,
    )
    down_pairs = np.stack(
        [labels[:-1, :].reshape(-1), labels[1:, :].reshape(-1)],
        axis=-1,
    )
    adjacent_pairs = np.concatenate([right_pairs, down_pairs], axis=0)
    adjacent_pairs = adjacent_pairs[adjacent_pairs[:, 0] != adjacent_pairs[:, 1]]
    if adjacent_pairs.size > 0:
        adjacent_pairs = np.unique(
            np.sort(adjacent_pairs.astype(np.int64), axis=1),
            axis=0,
        )

    for label_left, label_right in adjacent_pairs:
        if label_left not in descriptors or label_right not in descriptors:
            continue
        if should_merge_geometry(
            descriptors[label_left],
            descriptors[label_right],
            depth_thresh=depth_thresh,
            normal_cos_thresh=normal_cos_thresh,
            conf_thresh=conf_thresh,
        ):
            union(label_left, label_right)

    roots = np.array(
        [find(label_id) for label_id in labels.reshape(-1)],
        dtype=np.int64,
    )
    _, relabeled = np.unique(roots, return_inverse=True)
    return relabeled.reshape(labels.shape).astype(np.intp, copy=False)


def segment_geometry_felzenszwalb_rag_stages(
    depth_map,
    conf_map=None,
    intrinsic=None,
    point_map=None,
    confidence_keep_ratio=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
    seg_scale=200,
    seg_sigma=1.0,
    seg_min_size=300,
    normal_method="cross",
    batch_idx=None,
):
    conf_map = _select_batch_item(conf_map, batch_idx, depth_map.ndim)
    point_map = _select_batch_item(point_map, batch_idx, depth_map.ndim + 1)
    intrinsic = _select_batch_item(intrinsic, batch_idx, 2)

    geometry_info = build_geometry_info_np(
        depth=depth_map,
        conf=conf_map,
        intrinsic=intrinsic,
        points=point_map,
        normal_method=normal_method,
    )

    depth_norm = depth_map - np.nanmin(depth_map)
    depth_norm = depth_norm / (np.nanmax(depth_norm) + 1e-8)
    geom_img = np.concatenate(
        [depth_norm[..., None], geometry_info["normal"]],
        axis=-1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Got image with third dimension of 4. "
                "This image will be interpreted as a multichannel 2d image"
            ),
            category=RuntimeWarning,
        )
        labels = felzenszwalb(
            geom_img,
            scale=seg_scale,
            sigma=seg_sigma,
            min_size=seg_min_size,
            channel_axis=-1,
        )

    if conf_map is not None and confidence_keep_ratio is not None:
        high_conf_mask = select_numpy_top_confidence_mask(
            conf_map,
            confidence_keep_ratio,
        )
        conf_thresh = float(np.min(conf_map[high_conf_mask]))
        if high_conf_mask.sum() > 0:
            depth_range = (
                np.nanmax(depth_map[high_conf_mask])
                - np.nanmin(depth_map[high_conf_mask])
            )
            depth_thresh = depth_merge_thresh * depth_range
        else:
            depth_thresh = None
    else:
        conf_thresh = float("nan")
        high_conf_mask = np.ones(depth_map.shape, dtype=bool)
        depth_thresh = None

    merged_labels = merge_regions_geometry(
        labels,
        depth_map,
        geometry_info,
        conf=conf_map,
        depth_thresh=depth_thresh,
        normal_thresh_deg=normal_thresh_deg,
    )
    return SegmentationStages(
        initial_labels=np.asarray(labels, dtype=np.intp),
        merged_labels=np.asarray(merged_labels, dtype=np.intp),
        confidence_threshold=conf_thresh,
        high_confidence_mask=high_conf_mask,
    )


def geometry_merge_thresholds(
    depth_map,
    conf_map=None,
    confidence_keep_ratio=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
):
    """Return the two merge thresholds this module would use, without segmenting.

    Exists so the diagnostics can report the criterion that was actually in force instead of
    re-deriving it. A re-derivation could silently drift from the segmentation path; reading the
    same expression cannot.
    """
    high_conf_mask = None
    depth_thresh = None
    if conf_map is not None and confidence_keep_ratio is not None:
        high_conf_mask = select_numpy_top_confidence_mask(
            conf_map,
            confidence_keep_ratio,
        )
        if high_conf_mask.sum() > 0:
            depth_range = (
                np.nanmax(depth_map[high_conf_mask]) - np.nanmin(depth_map[high_conf_mask])
            )
            depth_thresh = depth_merge_thresh * depth_range
    return {
        "depth_thresh": depth_thresh,
        "normal_cos_thresh": float(np.cos(np.deg2rad(normal_thresh_deg))),
        "conf_thresh": (
            float(np.min(conf_map[high_conf_mask])) if high_conf_mask is not None else None
        ),
    }


def segment_geometry_felzenszwalb_rag(
    depth_map,
    conf_map=None,
    intrinsic=None,
    point_map=None,
    confidence_keep_ratio=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
    seg_scale=200,
    seg_sigma=1.0,
    seg_min_size=300,
    normal_method="cross",
    batch_idx=None,
):
    return segment_geometry_felzenszwalb_rag_stages(
        depth_map,
        conf_map=conf_map,
        intrinsic=intrinsic,
        point_map=point_map,
        confidence_keep_ratio=confidence_keep_ratio,
        depth_merge_thresh=depth_merge_thresh,
        normal_thresh_deg=normal_thresh_deg,
        seg_scale=seg_scale,
        seg_sigma=seg_sigma,
        seg_min_size=seg_min_size,
        normal_method=normal_method,
        batch_idx=batch_idx,
    ).merged_labels


def segment_geometry_felzenszwalb_rag_baseline_params_stages(
    depth_map,
    conf_map=None,
    intrinsic=None,
    point_map=None,
    confidence_keep_ratio=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
    seg_scale=300,
    seg_sigma=1.1,
    seg_min_size=500,
    normal_method="cross",
    batch_idx=None,
):
    return segment_geometry_felzenszwalb_rag_stages(
        depth_map,
        conf_map=conf_map,
        intrinsic=intrinsic,
        point_map=point_map,
        confidence_keep_ratio=confidence_keep_ratio,
        depth_merge_thresh=depth_merge_thresh,
        normal_thresh_deg=normal_thresh_deg,
        seg_scale=seg_scale,
        seg_sigma=seg_sigma,
        seg_min_size=seg_min_size,
        normal_method=normal_method,
        batch_idx=batch_idx,
    )


def segment_geometry_felzenszwalb_rag_baseline_params(
    depth_map,
    conf_map=None,
    intrinsic=None,
    point_map=None,
    confidence_keep_ratio=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
    seg_scale=300,
    seg_sigma=1.1,
    seg_min_size=500,
    normal_method="cross",
    batch_idx=None,
):
    return segment_geometry_felzenszwalb_rag_baseline_params_stages(
        depth_map,
        conf_map=conf_map,
        intrinsic=intrinsic,
        point_map=point_map,
        confidence_keep_ratio=confidence_keep_ratio,
        depth_merge_thresh=depth_merge_thresh,
        normal_thresh_deg=normal_thresh_deg,
        seg_scale=seg_scale,
        seg_sigma=seg_sigma,
        seg_min_size=seg_min_size,
        normal_method=normal_method,
        batch_idx=batch_idx,
    ).merged_labels
