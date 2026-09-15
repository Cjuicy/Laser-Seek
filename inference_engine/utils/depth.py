"""Depth-map segmentation, temporal region matching and per-layer scale estimation.

The Baseline chain lives here:

    segment_depth_felzenszwalb_rag  ->  match_segmentation_seq
    ->  connect_bipartite_sp_graphs  ->  align_depth_irls

IDEA-001 changes this module in three behaviour-neutral ways:

1. `segment_depth_felzenszwalb_rag` is split into `segment_depth_felzenszwalb_rag_stages`, which
   returns the intermediate labels and the merge threshold. The old function remains as a thin
   wrapper whose output is element-wise identical, which is the Regression gate for the whole Idea.
2. The confidence selection parameter is renamed from the quantile level (`top_conf_percentile`)
   to a keep ratio (`confidence_keep_ratio`) so it matches the Reference implementation and the
   `SegmentationConfig`. The numerics are unchanged: the keep ratio IS the quantile level, i.e.
   `confidence_keep_ratio = 1 - top_conf_percentile` in the historical CLI naming.
3. Observation points OP-1 (segmentation exit, emitted through the returned stages), OP-3
   (temporal matching) and OP-4 (per-edge IRLS) are inserted. Every one of them is inert: with
   `diagnostics=None` no extra work happens and no return value changes.

`align_depth_irls` and `pairwise_iou` are untouched apart from an optional result-detail return,
because they belong to LSA and are shared by all three methods.
"""

import os
import numpy as np
from skimage.segmentation import felzenszwalb
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fast_seg import fast_graph_segmentation
from ._segmentation_cy import merge_regions
from .confidence import select_numpy_top_confidence_mask
from .segmentation_trace import SegmentationStages
from pi3.utils.graph import Vertex


def align_depth_irls(
        src_depth,
        tgt_depth,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6,
        return_info=False
):
    """Estimate the scalar depth scale that maps `src_depth` onto `tgt_depth`.

    `return_info=True` additionally returns a small dict describing how the solve went. That
    detail is what OP-4 reports, and it is the only way to tell an edge whose estimate converged
    from one pinned at `clamp_min` because its intersection mask was degenerate.
    """
    sample_count = None
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]
    sample_count = int(np.asarray(src_depth).size)

    num = np.nanmean(tgt_depth)
    den = np.nanmean(src_depth)
    s_d = np.maximum(num / den, clamp_min)
    initial = float(s_d)

    iterations = 0
    converged = False
    for _ in range(iters):
        iterations += 1
        d_res = s_d * src_depth - tgt_depth
        res = abs(d_res) + eps
        w = 1.0 / res

        num = (w * src_depth * tgt_depth).sum()
        den = (w * src_depth ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = bool(abs(s_d_new - s_d) < stop_tol)
        s_d = s_d_new

        if converged:
            break  # early stopping

    if not return_info:
        return s_d

    final = float(s_d)
    info = {
        "scale": final,
        "initial_scale": initial,
        "iterations": int(iterations),
        "converged": bool(converged),
        "clamped": bool(np.isclose(final, clamp_min, rtol=0.0, atol=1e-15)),
        "sample_count": sample_count,
        "nonfinite_samples": int(sample_count - np.isfinite(np.asarray(src_depth)).sum()),
    }
    return s_d, info


def segment_depth_felzenszwalb_rag_stages(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        confidence_keep_ratio=None,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=500,
        batch_idx=None,
        confidence_quantile_method="nearest",
):
    """Baseline depth segmentation, exposing its intermediate stages.

    Returns `SegmentationStages(initial_labels, merged_labels, confidence_threshold,
    high_confidence_mask, merge_threshold, depth_range)`. `merged_labels` is element-wise
    identical to what `segment_depth_felzenszwalb_rag` returns for the same inputs.

    `confidence_keep_ratio` is the complement of the retained fraction: the pixels kept are those
    with `conf >= quantile(conf, 1 - keep_ratio)`. This is exactly the historical behaviour, where
    the engine passed `1 - top_conf_percentile` as the quantile level, so a historical
    `--top_conf_percentile 0.3` is `confidence_keep_ratio=0.7` (the top 30% kept).
    `confidence_quantile_method` defaults to `"nearest"`, the Baseline's rule; the Reference
    implementation uses `"higher"`, and a comparison must not change this silently.

    When confidence is unavailable the threshold is NaN and every pixel is used, which is the
    Baseline behaviour (and what makes the merge threshold collapse to NaN for an all-NaN depth
    map — recorded as a Baseline observation, deliberately not repaired here).
    """
    initial_labels = felzenszwalb(
        depth_map, scale=seg_scale, sigma=seg_sigma, min_size=seg_min_size
    )

    if conf_map is not None and confidence_keep_ratio is not None:
        conf_map = np.asarray(conf_map)
        frame_conf = conf_map[batch_idx] if (batch_idx is not None and conf_map.ndim == depth_map.ndim + 1) else conf_map
        high_confidence_mask = select_numpy_top_confidence_mask(
            frame_conf,
            confidence_keep_ratio,
            method=confidence_quantile_method,
        )
        # Derived from the mask rather than recomputed, so the reported threshold and the selected
        # pixels can never disagree. `keep_ratio` selects the top `1 - keep_ratio` fraction (see
        # utils/confidence.py).
        conf_thresh = float(np.min(frame_conf[high_confidence_mask]))
        conf_depth = depth_map[high_confidence_mask]
    else:
        conf_thresh = float("nan")
        high_confidence_mask = np.ones(depth_map.shape, dtype=bool)
        conf_depth = depth_map

    depth_range = float(np.max(conf_depth) - np.min(conf_depth))
    merge_thresh = depth_merge_thresh * depth_range

    merged_labels = merge_regions(initial_labels, depth_map, merge_thresh)
    return SegmentationStages(
        initial_labels=np.asarray(initial_labels),
        merged_labels=np.asarray(merged_labels),
        confidence_threshold=conf_thresh,
        high_confidence_mask=high_confidence_mask,
        merge_threshold=float(merge_thresh),
        depth_range=depth_range,
    )


def segment_depth_felzenszwalb_rag(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        confidence_keep_ratio=None,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=500,
        batch_idx=None,
        confidence_quantile_method="nearest",
):
    """Baseline depth segmentation. Thin wrapper whose output equals the pre-IDEA-001 function."""
    return segment_depth_felzenszwalb_rag_stages(
        depth_map,
        depth_merge_thresh,
        conf_map=conf_map,
        confidence_keep_ratio=confidence_keep_ratio,
        seg_scale=seg_scale,
        seg_sigma=seg_sigma,
        seg_min_size=seg_min_size,
        batch_idx=batch_idx,
        confidence_quantile_method=confidence_quantile_method,
    ).merged_labels


def segment_depth_graph_fast(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        top_conf_percentile=None,
        batch_idx=None
):
    if conf_map is not None and top_conf_percentile is not None:
        conf_map = conf_map[batch_idx]
        conf_thresh = np.quantile(conf_map.reshape(-1), top_conf_percentile, method='nearest')
        conf_depth = depth_map[conf_map >= conf_thresh]
    else:
        conf_depth = depth_map
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))
    return fast_graph_segmentation(depth_map, merge_thresh)


def pairwise_intersection_ratio(mask1, mask2):
    """
    Highest pairwise intersection ratio for assigning correspondence
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    area1 = np.maximum(area1, 1)
    area2 = np.maximum(area2, 1)

    ratios1 = inter / area1
    ratios2 = inter / area2
    min_rel_inter = np.minimum(ratios1, ratios2)
    max_rel_inter = np.maximum(ratios1, ratios2)

    return min_rel_inter, max_rel_inter  # N, M


def pairwise_iou(mask1, mask2):
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    union = area1 + area2 - inter

    iou = inter / np.maximum(union, 1)
    return iou


def match_segmentation_seq(labels, iou_thresh=0.4, diagnostics=None):
    def get_seg_vertices(seg):
        seg_ids = np.unique(seg)
        masks = seg[None, :, :] == seg_ids[:, None, None]
        seg_vertices_ = [Vertex(data=m, default_cache={'iou': [], 'scale': []}) for m in masks]
        return seg_vertices_  # , masks

    # root = get_seg_vertices(labels[0])
    sp_graph = [get_seg_vertices(labels[0])]

    for seg_map in labels[1:]:
        seg_vertices = get_seg_vertices(seg_map)
        connect_bipartite_sp_graphs(
            sp_graph[-1],
            seg_vertices,
            iou_thresh=iou_thresh,
            diagnostics=diagnostics,
            frame_index=len(sp_graph),
            relates_to_anchor=False,
        )
        sp_graph.append(seg_vertices)
        # prev_mask = cur_mask

    # for v in root:
    #     v.cut_edge_threshold(inter_thresh)
    return sp_graph


def connect_bipartite_sp_graphs(
        graph1,
        graph2,
        iou_thresh=0.3,
        diagnostics=None,
        frame_index=None,
        relates_to_anchor=True,
):
    masks1 = np.stack([v.data for v in graph1])
    masks2 = np.stack([v.data for v in graph2])

    iou = pairwise_iou(masks1, masks2)
    matchable = iou >= iou_thresh
    graph1_indices, graph2_indices = np.nonzero(matchable)

    if diagnostics is not None:
        from ..segmentation_diagnostics import OP_TEMPORAL_MATCH, summarize_temporal_match_per_frame

        diagnostics.record_frame(
            frame_index if frame_index is not None else -1,
            summarize_temporal_match_per_frame(
                frame_index=frame_index if frame_index is not None else -1,
                source_vertices=len(graph1),
                target_vertices=len(graph2),
                iou=iou,
                threshold=iou_thresh,
                relates_to_anchor=bool(relates_to_anchor),
            ),
            stage=OP_TEMPORAL_MATCH,
        )

    for v1, v2 in zip(graph1_indices, graph2_indices):
        graph1[v1].add_edge(graph2[v2], iou[v1, v2])


def _edge_scale_worker(
        src_depth,
        tgt_depth,
        src_vertex,
        irls=None,
        diagnostics=None,
        frame_index=None,
):
    irls = irls or {}
    src_mask = src_vertex.data
    for tgt_v, tgt_iou in zip(src_vertex.connectivity, src_vertex.edge_weights):
        tgt_mask = tgt_v.data
        inter_mask = src_mask & tgt_mask
        tgt2src_s, info = align_depth_irls(
            tgt_depth,
            src_depth,
            inter_mask,
            return_info=True,
            **irls,
        )
        tgt_v.cache['iou'].append(tgt_iou)
        tgt_v.cache['scale'].append(tgt2src_s)
        if diagnostics is not None:
            from ..segmentation_diagnostics import OP_EDGE_IRLS

            diagnostics.record_frame(
                frame_index if frame_index is not None else -1,
                {
                    "inter_mask_area": info["sample_count"],
                    "scale": info["scale"],
                    "initial_scale": info["initial_scale"],
                    "iterations": info["iterations"],
                    "converged": info["converged"],
                    "clamped": info["clamped"],
                    "iou": float(tgt_iou),
                },
                # Per-edge rows are grouped under its own observation point, so a reader can tell
                # them apart from OP-3's per-frame-pair rows.
                stage=OP_EDGE_IRLS,
            )


def assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=0.4,
        n_jobs=1,
        irls=None,
        diagnostics=None,
):
    for frame_index, (src_sp_graph, tgt_sp_graph) in enumerate(
        zip(src_sp_graphs_overlap, tgt_sp_graphs_overlap)
    ):
        connect_bipartite_sp_graphs(
            src_sp_graph,
            tgt_sp_graph,
            iou_thresh=iou_thresh,
            diagnostics=diagnostics,
            frame_index=frame_index,
            relates_to_anchor=True,
        )

    for idx, src_graph in enumerate(src_sp_graphs_overlap):
        n_jobs = min(os.cpu_count(), len(src_graph)) if n_jobs is None else n_jobs
        if n_jobs == 1:
            for src_v in src_graph:
                _edge_scale_worker(
                    src_depth_overlap[idx],
                    tgt_depth_overlap[idx],
                    src_v,
                    irls=irls,
                    diagnostics=diagnostics,
                    frame_index=idx,
                )
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                promises = [
                    ex.submit(
                        _edge_scale_worker,
                        src_depth_overlap[idx],
                        tgt_depth_overlap[idx],
                        src_v,
                        irls,
                        diagnostics,
                        idx,
                    )
                    for src_v in src_graph
                ]
                for promise in as_completed(promises):
                    promise.result()
