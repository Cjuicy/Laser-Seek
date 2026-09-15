"""Layer-wise scale alignment: per-frame regions, temporal matching, per-pixel scale mask.

This module is the whole LSA stage of the streaming engine. Three initial segmentation methods
(`depth`, `geometry`, `atomic`) enter at `make_sp_graph` and are indistinguishable from the
temporal-matching step onward, which is exactly why a segmentation change can move ATE without
being observable: the four stages between "regions" and "scale mask" used to return nothing.

IDEA-001 (`docs/dsh/ideas/IDEA-001-segmentation-method-attribution.md`) adds:

* method dispatch in `make_sp_graph` — default `method='depth'` reproduces the Baseline exactly;
* observation points OP-3 (temporal matching), OP-4 (per-edge IRLS) and OP-5 (propagation and the
  final mask), all of which read state that already exists and change no return value.

Every observation is inert: with `diagnostics=None` (the default) the code paths are the same
operations in the same order as before.
"""

import numpy as np
import torch

from .depth import (
    segment_depth_felzenszwalb_rag,
    match_segmentation_seq,
    assign_overlap_window_depth_scale,
)
from ..segmentation_config import (
    METHOD_ATOMIC,
    METHOD_DEPTH,
    METHOD_GEOMETRY,
    SegmentationConfig,
)
from ..segmentation_diagnostics import (
    OP_MERGE_EVIDENCE,
    OP_SCALE_MASK,
    OP_SEGMENTATION_EXIT,
    OP_TEMPORAL_MATCH,
    summarize_labels,
    summarize_scale_mask,
    summarize_scale_propagation,
    summarize_segmentation_exit,
)


def refine_depth_segments(
        src_pcd,
        tgt_pcd,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4,
        irls=None,
        diagnostics=None,
):
    """
    src_pcd: previous window pcd
    tgt_pcd: current window pcd
    src_sp_graphs: previous window superpixel graph
    overlap: window overlap size
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding

    `irls` and `diagnostics` are forwarded to the temporal-matching and per-edge stages. Without
    forwarding them the observation points downstream of this call record nothing, and an empty
    OP-3/OP-4 section is indistinguishable from "the stage found no correspondences" — which is
    exactly the ambiguity the observation points exist to remove.
    """
    src_depth = src_pcd[..., -1]
    tgt_depth = tgt_pcd[..., -1]

    tgt_scale_mask = align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh,
        irls=irls,
        diagnostics=diagnostics,
    )

    return torch.from_numpy(tgt_scale_mask[..., None])


def segmentation_config_or_default(segmentation=None) -> SegmentationConfig:
    """Return the supplied config, or the locked default one.

    Reading the shipped YAML here keeps `make_sp_graph` usable on its own (tests, the sliding
    window path) while still funnelling production runs through one parameter source.
    """
    if segmentation is not None:
        return segmentation
    from ..segmentation_config import load_segmentation_config

    return load_segmentation_config()


def segment_labels_for_method(
        method,
        depth,
        *,
        segmentation,
        point_map=None,
        intrinsic=None,
        conf_map=None,
        batch_idx=None,
):
    """Produce one frame's labels with the configured initial segmentation method.

    Returns `(labels, diagnostics_dict)`.

    * `depth` — the Baseline: Felzenszwalb on the depth map, then the absolute mean-depth merge.
      The observation values come from the split-out `_stages` function so that OP-1 can report the
      atom count, the coarse-layer count and the merge threshold.
    * `geometry` — a 4-channel `[normalized depth, normal, normal, normal]` image, then a merge
      gated by depth difference, region-mean normal angle and confidence.
    * `atomic` — the depth atoms and coarse layers are kept as a prior, and a scale-normalised 3D
      boundary gap decides connectivity; an optional split stage may then separate regions.

    `point_map` is required by `geometry` and `atomic`; a missing one is a programming error.
    """
    if method == METHOD_DEPTH:
        from .depth import segment_depth_felzenszwalb_rag_stages

        # `batch_idx` indexes the BATCHED arrays the segmentation functions receive. Memory order
        # makes that index also the right one for a per-frame depth slice, so the macro-step can
        # hand `make_sp_graph` the window's overlap slice of confidence — which is all the Baseline
        # ever saw — while still segmenting every frame of the window.
        stages = segment_depth_felzenszwalb_rag_stages(
            depth,
            segmentation.depth_merge_thresh,
            conf_map=conf_map,
            confidence_keep_ratio=segmentation.confidence_keep_ratio,
            seg_scale=segmentation.felzenszwalb.scale,
            seg_sigma=segmentation.felzenszwalb.sigma,
            seg_min_size=segmentation.felzenszwalb.min_size,
            batch_idx=batch_idx,
            confidence_quantile_method=segmentation.confidence_quantile_method,
        )
        info = {
            # `initial_labels` are the Felzenszwalb atoms, `merged_labels` the depth-DSU coarse
            # layers. For the Baseline the coarse layer IS the final output, so OP-1 sees all
            # three levels collapse to two stages here.
            "initial_labels": stages.initial_labels,
            "coarse_labels": stages.merged_labels,
            "merged_labels": stages.merged_labels,
            "merge_threshold": stages.merge_threshold,
            "high_confidence_mask": stages.high_confidence_mask,
        }
        return stages.merged_labels, info

    if point_map is None:
        raise ValueError(
            f"method={method!r} requires point_map; only 'depth' can run from depth alone"
        )

    # `make_sp_graph` slices the frame before calling here, so `conf_map` and `point_map` arrive as
    # single frames and must NOT be indexed by `batch_idx` again. The segmentation functions accept
    # a 2-D confidence map directly and skip their own batch selection in that case.
    frame_conf = None if conf_map is None else np.asarray(conf_map)
    if frame_conf is not None and frame_conf.ndim == 3:
        frame_conf = frame_conf[batch_idx] if batch_idx is not None else frame_conf[0]
    frame_points = np.asarray(point_map)
    if frame_points.ndim == 4:
        frame_points = frame_points[batch_idx] if batch_idx is not None else frame_points[0]

    if method == METHOD_GEOMETRY:
        from .geometry_segmentation import segment_geometry_felzenszwalb_rag

        labels = segment_geometry_felzenszwalb_rag(
            depth,
            conf_map=frame_conf,
            intrinsic=intrinsic,
            point_map=frame_points,
            confidence_keep_ratio=segmentation.confidence_keep_ratio,
            depth_merge_thresh=segmentation.depth_merge_thresh,
            normal_thresh_deg=segmentation.geometry.normal_threshold_degrees,
            seg_scale=segmentation.felzenszwalb.scale,
            seg_sigma=segmentation.felzenszwalb.sigma,
            seg_min_size=segmentation.felzenszwalb.min_size,
            normal_method=segmentation.geometry.normal_method,
        )
        # The geometry method merges only; its region count can only fall. The comparison against
        # the depth atoms it started from is reported at OP-1 by the caller.
        return np.asarray(labels), {}

    if method == METHOD_ATOMIC:
        from .layer_atomic_geometry import segment_point_map_atomic

        labels, split_diagnostics = segment_point_map_atomic(
            frame_points,
            depth_merge_thresh=segmentation.depth_merge_thresh,
            normal_method=segmentation.geometry.normal_method,
            split_score_threshold=segmentation.atomic.split_score_threshold,
            split_mode=segmentation.atomic.split_mode,
            conf_map=frame_conf,
            confidence_keep_ratio=segmentation.confidence_keep_ratio,
            seg_scale=segmentation.felzenszwalb.scale,
            seg_sigma=segmentation.felzenszwalb.sigma,
            seg_min_size=segmentation.felzenszwalb.min_size,
            confidence_quantile_method=segmentation.confidence_quantile_method,
        )
        return np.asarray(labels), {"split_diagnostics": split_diagnostics}

    raise ValueError(f"unsupported segmentation method: {method!r}")


def _record_merge_evidence(method, labels, depth, point_map, conf_map, info, config, diagnostics):
    """OP-2. Re-apply the method's own criterion to the boundaries it kept.

    A merge criterion is invisible in its own output: two labels sitting side by side do not say
    whether the method decided they were similar, or never considered them. This function measures
    the *decision* directly by evaluating the criterion on every 4-connected boundary between two
    different labels of the final partition, and reporting the distribution of that criterion's
    value plus how many boundaries the criterion would have refused.

    Those refused boundaries are not automatically errors — a criterion may be conservative by
    design, and `atomic`'s gap depends on per-atom scales that are not recoverable from labels
    alone, which is why its depth-gap numbers here are a weaker proxy than the other two. What the
    count does give is a comparable, per-method quantity for "how tightly did this method merge",
    which is not visible in a region count.

    `info` carries the thresholds the method actually used, so the same locked configuration is
    reflected rather than a re-derivation that could drift from it.
    """
    if diagnostics is None:
        return
    from ..segmentation_diagnostics import OP_MERGE_EVIDENCE, summarize_merge_evidence

    labels = np.asarray(labels)
    depth = np.asarray(depth)
    total = 0
    violating = 0
    depth_gaps: list[np.ndarray] = []
    normal_angles: list[np.ndarray] = []
    mean_depth: dict[int, float] = {}
    mean_conf: dict[int, float] = {}
    cos_threshold = None
    conf_threshold = None
    merge_threshold = info.get("merge_threshold") if info else None

    if method == METHOD_GEOMETRY and info:
        cos_threshold = info.get("normal_cos_thresh")
        conf_threshold = info.get("conf_thresh")
        if cos_threshold is None:
            from .geometry_segmentation import geometry_merge_thresholds

            thresholds = geometry_merge_thresholds(
                depth,
                conf_map=conf_map,
                confidence_keep_ratio=config.confidence_keep_ratio,
                depth_merge_thresh=config.depth_merge_thresh,
                normal_thresh_deg=config.geometry.normal_threshold_degrees,
            )
            cos_threshold = thresholds["normal_cos_thresh"]
            conf_threshold = thresholds["conf_thresh"]

    labels_flat = labels.reshape(-1)
    for label in np.unique(labels_flat):
        mask = labels_flat == label
        mean_depth[int(label)] = float(np.mean(depth.reshape(-1)[mask]))
        if conf_map is not None:
            mean_conf[int(label)] = float(np.mean(np.asarray(conf_map).reshape(-1)[mask]))

    normals = None
    if method == METHOD_GEOMETRY and point_map is not None and cos_threshold is not None:
        from .geometry import build_geometry_info_np

        normals = build_geometry_info_np(
            depth, points=np.asarray(point_map), normal_method=config.geometry.normal_method
        )["normal"]

    for axis in (0, 1):
        left = [slice(None)] * 2
        right = [slice(None)] * 2
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        a = labels[left]
        b = labels[right]
        separated = a != b
        if not separated.any():
            continue
        gaps = np.abs(depth[left][separated] - depth[right][separated])
        depth_gaps.append(gaps)
        total += int(separated.sum())

        if method == METHOD_GEOMETRY and normals is not None:
            left_normal = normals[left][separated]
            right_normal = normals[right][separated]
            cosine = (left_normal * right_normal).sum(axis=-1)
            normal_angles.append(cosine)
            # The method accepted this boundary, so by construction the normal condition passed
            # where it was evaluated. Recomputing it here catches the case where it was NOT
            # evaluated — an unbounded region, a missing confidence map — which is the silent way
            # a criterion stops doing anything.
            violating += int((cosine < float(cos_threshold)).sum())
        elif merge_threshold is not None:
            # For `depth` and as a proxy for `atomic`, the criterion is an absolute mean-depth
            # difference and the recorded boundary gap is the strongest label-level analogue.
            violating += int((gaps > float(merge_threshold)).sum())

    diagnostics.record_stage(
        OP_MERGE_EVIDENCE,
        summarize_merge_evidence(
            method=method,
            adjacent_pairs=total,
            merged_pairs=total - violating,
            rejected_pairs=violating,
            criteria={
                "boundary_depth_gap": np.concatenate(depth_gaps) if depth_gaps else None,
                "boundary_normal_cosine": (
                    np.concatenate(normal_angles) if normal_angles else None
                ),
            },
            extra={
                "final_region_count": int(np.unique(labels).size),
                "merge_threshold": merge_threshold,
                "normal_cos_threshold": cos_threshold,
                "conf_threshold": conf_threshold,
                "atoms_considered": (
                    int(np.unique(info["initial_labels"]).size)
                    if info and info.get("initial_labels") is not None
                    else None
                ),
            },
        ),
    )


def make_sp_graph(
        depth,
        depth_merge_thresh=0.1,
        conf_map=None,
        top_conf_percentile=None,
        corr_iou_thresh=0.3,
        method=None,
        point_map=None,
        intrinsic=None,
        segmentation=None,
        diagnostics=None,
):
    """Build the per-frame Vertex graph used by LSA.

    Backwards compatibility: called with the original arguments (no `method`), this behaves exactly
    as before. `top_conf_percentile` is retained for that reason and is interpreted with the
    historical semantics — it IS the quantile level, so `confidence_keep_ratio = top_conf_percentile`.
    When `method` is given it comes from a `SegmentationConfig`, which carries a keep ratio
    directly and takes precedence.

    `depth` is `(N, H, W)`; `point_map` is `(N, H, W, 3)` and required by `geometry` / `atomic`.
    """
    config = None
    if method is not None or segmentation is not None:
        config = segmentation_config_or_default(segmentation)
        if method is not None:
            if method != config.method:
                config = config.with_method(method)
        resolved_method = config.method
        keep_ratio = config.confidence_keep_ratio
        quantile_method = config.confidence_quantile_method
        merge_thresh = config.depth_merge_thresh
        iou_thresh = config.corr_iou_thresh_intra
    else:
        # Historical call form. `top_conf_percentile` was already the quantile level.
        resolved_method = METHOD_DEPTH
        if top_conf_percentile is not None:
            keep_ratio = float(top_conf_percentile)
        else:
            keep_ratio = None
        quantile_method = "nearest"
        merge_thresh = depth_merge_thresh
        iou_thresh = corr_iou_thresh
        config = None

    depth = np.asarray(depth)
    if depth.ndim == 2:
        depth = depth[None]
    frames = int(depth.shape[0])

    point_maps = None if point_map is None else np.asarray(point_map)
    if point_maps is not None and point_maps.ndim == 3:
        point_maps = point_maps[None]

    label_frames = []
    per_frame_diagnostics = []

    def _frame_conf(frame_index: int):
        """Confidence for one frame, tolerating a shorter confidence array than the depth stack.

        The Baseline only ever had confidence for the OVERLAP frames while it segmented every frame
        of the window, so the two lengths legitimately differ. Frames without their own confidence
        reuse the last available one, which is stated here rather than raising an index error deep
        inside a segmentation loop.
        """
        if conf_map is None:
            return None
        conf_array = np.asarray(conf_map)
        if conf_array.ndim == 2:
            return conf_array
        if frame_index < conf_array.shape[0]:
            return conf_array[frame_index]
        return conf_array[-1]

    for frame_index in range(frames):
        frame_depth = depth[frame_index]
        frame_conf = _frame_conf(frame_index)
        frame_points = None
        if point_maps is not None:
            frame_points = point_maps[frame_index]

        if resolved_method == METHOD_DEPTH and config is None:
            labels = segment_depth_felzenszwalb_rag(
                frame_depth,
                depth_merge_thresh=merge_thresh,
                conf_map=frame_conf,
                confidence_keep_ratio=keep_ratio,
                confidence_quantile_method=quantile_method,
            )
            info = None
        else:
            labels, info = segment_labels_for_method(
                resolved_method,
                frame_depth,
                segmentation=config,
                point_map=frame_points,
                intrinsic=intrinsic,
                conf_map=frame_conf,
                # `frame_conf` here is already the frame for `frame_index`; passing the index lets
                # `geometry` and `atomic` slice it again only when it really is still batched.
                batch_idx=frame_index,
            )
        labels = np.asarray(labels)
        label_frames.append(labels)

        if diagnostics is not None and resolved_method in (METHOD_GEOMETRY, METHOD_ATOMIC):
            _record_merge_evidence(
                resolved_method,
                labels,
                frame_depth,
                frame_points,
                frame_conf,
                info,
                config,
                diagnostics,
            )

        if diagnostics is not None:
            frame_report = summarize_labels(labels, prefix="final")
            if info:
                if "merge_threshold" in info:
                    high_mask = info.get("high_confidence_mask")
                    frame_report.update(
                        summarize_segmentation_exit(
                            initial_labels=info.get("initial_labels"),
                            coarse_labels=info.get("coarse_labels"),
                            merged_labels=labels,
                            merge_threshold=info.get("merge_threshold"),
                            high_confidence_count=(
                                int(high_mask.sum()) if high_mask is not None else None
                            ),
                            high_confidence_fraction=(
                                float(high_mask.mean()) if high_mask is not None else None
                            ),
                            depth_range=float(
                                np.max(frame_depth) - np.min(frame_depth)
                            ),
                        )
                    )
                    diagnostics.record_stage(OP_SEGMENTATION_EXIT, frame_report)
                else:
                    frame_report["method"] = resolved_method
                    for key, value in (info.get("extra") or {}).items():
                        frame_report[key] = value
                    diagnostics.record_stage(OP_MERGE_EVIDENCE, frame_report)
            if info and "split_diagnostics" in info:
                split = info["split_diagnostics"]
                as_dict = split.as_dict() if hasattr(split, "as_dict") else dict(split)
                frame_report.update(as_dict)
                diagnostics.record_stage(OP_MERGE_EVIDENCE, frame_report)
            per_frame_diagnostics.append(frame_report)

    sp_graph = match_segmentation_seq(
        np.stack(label_frames),
        iou_thresh=iou_thresh,
        diagnostics=diagnostics,
    )

    return sp_graph


def align_adjacent_windows_depth_segments(
        src_depth,  # N, H, W
        tgt_depth,  # N, H, W
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4,
        irls=None,
        diagnostics=None,
):
    """
    src_depth: previous window depth map
    tgt_depth: current window depth map
    src_sp_graphs: previous window superpixel graph (nested list of Vertex)
    tgt_sp_graphs: current window superpixel graph
    overlap: window overlap size
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding

    Return:
        depth_scale_mask: N, H, W for current window pcd

    The mask arithmetic below is intentionally unchanged: `_get_scale_mask` averages the scales
    propagated from several parents by their IoU weights, and falls back to 1.0 when a vertex has
    no cached scale. Both behaviours are part of the Baseline, and OP-5 only *counts* them.
    """

    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache['scale']) > 0:
            iou_wts = np.asarray(parent.cache['iou'])
            prop_scale = np.dot(np.asarray(parent.cache['scale']), iou_wts / np.sum(iou_wts))
            child.cache['iou'].append(edge_wt)
            child.cache['scale'].append(prop_scale)

    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache['scale']) > 0:
            iou_wts = np.asarray(cache['iou'])
            mu_scale = np.dot(np.asarray(cache['scale']), iou_wts / np.sum(iou_wts))
        else:
            mu_scale = 1.0
        return mask * mu_scale

    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]

    for sp_graph in src_sp_graphs_overlap:
        for v in sp_graph:
            v.remove_all_edges()

    if diagnostics is not None:
        diagnostics.record_stage(
            OP_TEMPORAL_MATCH,
            {
                "overlap": int(overlap),
                "corr_iou_thresh_inter": float(corr_iou_thresh),
                "source_frames": int(len(src_sp_graphs_overlap)),
                "target_frames": int(len(tgt_sp_graphs_overlap)),
                "source_vertices": int(sum(len(g) for g in src_sp_graphs_overlap)),
                "target_vertices": int(sum(len(g) for g in tgt_sp_graphs_overlap)),
            },
        )

    # sptial scale initilaization
    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=corr_iou_thresh,
        irls=irls,
        diagnostics=diagnostics,
    )
    # temporal scale propagation
    for tgt_graph_layer in tgt_sp_graphs:
        for v in tgt_graph_layer:
            v.propagate_data_once(_propagate_scale_cache)

    mask_seq = []
    for sp_graph in tgt_sp_graphs:
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for v in sp_graph[1:]:
            mask_frame += v.data_cache_op(_get_scale_mask)
        mask_seq.append(mask_frame)

    if diagnostics is not None:
        _record_scale_propagation(tgt_sp_graphs, diagnostics)

    return np.stack(mask_seq)


def _record_scale_propagation(tgt_sp_graphs, diagnostics) -> None:
    """OP-5. Report the Vertex scale-cache state that `_get_scale_mask` just consumed."""
    counts = []
    scales = []
    weights = []
    for sp_graph in tgt_sp_graphs:
        for vertex in sp_graph:
            cache = vertex.cache or {}
            vertex_scales = list(cache.get("scale") or [])
            counts.append(len(vertex_scales))
            scales.append(vertex_scales)
            weights.append(list(cache.get("iou") or []))
    diagnostics.record_stage(
        OP_SCALE_MASK,
        summarize_scale_propagation(
            vertex_scale_counts=counts,
            vertex_scale_values=scales,
            vertex_iou_weights=weights,
        ),
    )


def record_scale_mask(mask: np.ndarray, diagnostics) -> None:
    """OP-5 companion. Called by the macro-step with the mask that is about to be applied."""
    if diagnostics is None:
        return
    array = np.asarray(mask)
    diagnostics.record_stage(OP_SCALE_MASK, summarize_scale_mask(array))


__all__ = [
    "refine_depth_segments",
    "make_sp_graph",
    "align_adjacent_windows_depth_segments",
    "segment_labels_for_method",
    "segmentation_config_or_default",
    "record_scale_mask",
]
