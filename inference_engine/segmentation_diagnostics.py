"""Observation points OP-1 … OP-6 for the segmentation / LSA chain (IDEA-001).

Why this module exists
----------------------
Between "per-frame regions" and "per-pixel scale mask" the Baseline pipeline exposes nothing:
`make_sp_graph` -> `match_segmentation_seq` -> `assign_overlap_window_depth_scale` ->
`_get_scale_mask` collapse four stages into one tensor. ATE is therefore the only visible number,
and an ATE improvement cannot be attributed to any single stage. Every function here reads state
that already exists on the data path and turns it into finite scalars, so that the intermediate
steps of the IDEA-001 hypotheses become observable.

Hard rules
----------
* Behaviour neutrality: nothing in this module may change a label, a scale, a mask or a return
  value. Every entry point is a pure function of its inputs.
* Off means off: when `DiagnosticsConfig.active` is false, callers must not invoke these
  functions at all. `NullDiagnosticsSink` exists for callers that prefer to stay branch-free.
* Scalars only: the written JSON contains integers, floats, booleans and short strings. Arrays are
  written separately and only when explicitly requested, because a per-window label array is tens
  of megabytes.
* Never raise into the hot path: sinks must not fail a reconstruction. `WindowDiagnostics` stores
  what it is given; unreadable optional values are skipped rather than raised.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------
# Constants naming the observation points, so a reader can map a diagnostic key to the design.
# --------------------------------------------------------------------------------------------

OP_SEGMENTATION_EXIT = "OP-1"   # utils/depth.py::segment_depth_felzenszwalb_rag_stages return
OP_MERGE_EVIDENCE = "OP-2"      # geometry / atomic merge and split decisions
OP_TEMPORAL_MATCH = "OP-3"      # utils/depth.py::connect_bipartite_sp_graphs
OP_EDGE_IRLS = "OP-4"           # utils/depth.py::_edge_scale_worker -> align_depth_irls
OP_SCALE_MASK = "OP-5"          # utils/lsa.py::_get_scale_mask / _propagate_scale_cache
OP_OVERLAP_CONSISTENCY = "OP-6"  # inference_utils.py::run_lsa_refinement, before/after the mask


def _finite_or_none(value: Any) -> Any:
    """Coerce numpy scalars to JSON-safe Python values; non-finite floats become None."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def region_areas(labels: np.ndarray) -> np.ndarray:
    """Pixel area of every label that actually occurs, in ascending label order.

    Must not be `np.bincount(...)` indexed by label value: the Baseline's Cython `merge_regions`
    renumbers regions as `root + 1`, and a DSU root set is not contiguous, so `bincount` produces
    holes that inflate the region count and add phantom zero-area regions. `np.unique` counts only
    labels that are present, which is what every consumer here means by "region count".

    Raises on a non-2-D input rather than guessing, because a silently wrong histogram is worse
    than a failed reconstruction.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    _, counts = np.unique(labels, return_counts=True)
    return counts.astype(np.int64, copy=False)


def _quantiles(values: np.ndarray, probs: Sequence[float]) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{int(p * 100):02d}": None for p in probs}
    computed = np.quantile(values, list(probs))
    return {
        f"p{int(p * 100):02d}": float(v) for p, v in zip(probs, np.atleast_1d(computed))
    }


# --------------------------------------------------------------------------------------------
# OP-1: the segmentation exit. Answers "how many regions, of what size, did this method produce".
# --------------------------------------------------------------------------------------------


def summarize_labels(labels: np.ndarray, *, prefix: str = "labels") -> dict[str, Any]:
    """Region-count and area-distribution summary of one label map."""
    areas = region_areas(labels)
    pixels = int(np.asarray(labels).size)
    summary: dict[str, Any] = {
        f"{prefix}_region_count": int(areas.size),
        f"{prefix}_pixels": pixels,
        f"{prefix}_regions_per_megapixel": (
            float(areas.size) / (pixels / 1.0e6) if pixels else None
        ),
        f"{prefix}_area_min": int(areas.min()) if areas.size else None,
        f"{prefix}_area_max": int(areas.max()) if areas.size else None,
        f"{prefix}_area_mean": float(areas.mean()) if areas.size else None,
        f"{prefix}_area_median": float(np.median(areas)) if areas.size else None,
    }
    if areas.size:
        summary[f"{prefix}_area_p10"] = float(np.quantile(areas, 0.10))
        summary[f"{prefix}_area_p90"] = float(np.quantile(areas, 0.90))
        # A method whose regions are mostly tiny cannot give the per-edge IRLS a usable mask.
        summary[f"{prefix}_tiny_region_fraction"] = float(
            np.mean(areas < 64)
        )
    else:
        summary[f"{prefix}_area_p10"] = None
        summary[f"{prefix}_area_p90"] = None
        summary[f"{prefix}_tiny_region_fraction"] = None
    return summary


def summarize_segmentation_exit(
    *,
    initial_labels: np.ndarray | None = None,
    coarse_labels: np.ndarray | None = None,
    merged_labels: np.ndarray | None = None,
    merge_threshold: float | None = None,
    high_confidence_count: int | None = None,
    high_confidence_fraction: float | None = None,
    depth_range: float | None = None,
) -> dict[str, Any]:
    """OP-1. Records what each segmentation stage produced for one frame.

    `initial_labels` are the Felzenszwalb atoms, `coarse_labels` the mean-depth merged layers, and
    `merged_labels` the final output of the method (for `depth` the two are the same object).
    Region counts are recorded for every stage so the *direction* of each stage's effect is
    visible: `atomic` may reconnect across coarse layers, and its optional split may raise the
    final count.
    """
    summary: dict[str, Any] = {
        "merge_threshold": _finite_or_none(merge_threshold),
        "high_confidence_count": (
            int(high_confidence_count) if high_confidence_count is not None else None
        ),
        "high_confidence_fraction": _finite_or_none(high_confidence_fraction),
        "depth_range": _finite_or_none(depth_range),
    }
    for name, labels in (
        ("initial", initial_labels),
        ("coarse", coarse_labels),
        ("merged", merged_labels),
    ):
        if labels is None:
            summary[f"{name}_region_count"] = None
            continue
        summary.update(summarize_labels(np.asarray(labels), prefix=name))
    initial_count = summary.get("initial_region_count")
    merged_count = summary.get("merged_region_count")
    if initial_count is not None and merged_count is not None and initial_count > 0:
        summary["merge_reduction_ratio"] = 1.0 - float(merged_count) / float(initial_count)
    else:
        summary["merge_reduction_ratio"] = None
    return summary


# --------------------------------------------------------------------------------------------
# OP-2: the merge / split decision evidence produced inside a method.
# --------------------------------------------------------------------------------------------


def summarize_merge_evidence(
    *,
    method: str,
    adjacent_pairs: int | None = None,
    merged_pairs: int | None = None,
    rejected_pairs: int | None = None,
    criterion_values: np.ndarray | None = None,
    criterion_name: str | None = None,
    criteria: Mapping[str, np.ndarray] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """OP-2. Records *why* a method merged or refused to merge adjacent regions.

    This is the point that distinguishes the three mechanisms rather than only their outcomes:
    `depth` compares an absolute mean-depth difference, `geometry` adds a normal-angle condition
    and a confidence condition, and `atomic` uses a scale-normalised 3D boundary gap. Recording the
    criterion distribution lets a later checkpoint test whether the mechanism acted as designed
    (for example: on a crease fixture the geometry criterion must reject the pair that
    `depth` accepts).
    """
    summary: dict[str, Any] = {
        "method": str(method),
        "adjacent_pairs": int(adjacent_pairs) if adjacent_pairs is not None else None,
        "merged_pairs": int(merged_pairs) if merged_pairs is not None else None,
        "rejected_pairs": int(rejected_pairs) if rejected_pairs is not None else None,
    }
    if adjacent_pairs:
        summary["merge_acceptance_ratio"] = (
            float(merged_pairs) / float(adjacent_pairs) if merged_pairs is not None else None
        )
    else:
        summary["merge_acceptance_ratio"] = None

    blocks: list[tuple[str, np.ndarray | None]] = []
    if criterion_values is not None:
        blocks.append((criterion_name or "criterion", criterion_values))
    if criteria:
        blocks.extend((str(name), values) for name, values in criteria.items())
    for name, values in blocks:
        stats = _quantiles(np.asarray(values, dtype=np.float64), (0.05, 0.25, 0.5, 0.75, 0.95))
        for key, value in stats.items():
            summary[f"{name}_{key}"] = value
    if extra:
        for key, value in extra.items():
            summary[str(key)] = _finite_or_none(value)
    return summary


def summarize_split_decision(
    *,
    split_mode: str,
    parent_count: int | None = None,
    proposed: int | None = None,
    accepted: int | None = None,
    reject_no_markers: int | None = None,
    reject_small_child: int | None = None,
    reject_low_score: int | None = None,
    added_regions: int | None = None,
) -> dict[str, Any]:
    """OP-2 companion for `atomic`'s optional split — the only stage that can RAISE region count."""
    return {
        "split_mode": str(split_mode),
        "split_parent_count": int(parent_count) if parent_count is not None else None,
        "split_proposed_count": int(proposed) if proposed is not None else None,
        "split_accepted_count": int(accepted) if accepted is not None else None,
        "split_reject_no_markers": (
            int(reject_no_markers) if reject_no_markers is not None else None
        ),
        "split_reject_small_child": (
            int(reject_small_child) if reject_small_child is not None else None
        ),
        "split_reject_low_score": (
            int(reject_low_score) if reject_low_score is not None else None
        ),
        "split_added_regions": int(added_regions) if added_regions is not None else None,
    }


# --------------------------------------------------------------------------------------------
# OP-3: the cross-window correspondence stage.
# --------------------------------------------------------------------------------------------


def summarize_temporal_match_per_frame(
    *,
    frame_index: int,
    source_vertices: int,
    target_vertices: int,
    iou: np.ndarray,
    threshold: float,
    relates_to_anchor: bool,
) -> dict[str, Any]:
    """OP-3 for one frame pair. `iou` is the (N_source, M_target) matrix already computed.

    The matched-vertex ratio and the degree distribution are the quantities that decide whether a
    method's larger regions still find correspondences at all: a method that merges regions into a
    few huge blobs can lose edges entirely, which would silently push every unmatched pixel onto
    the `mu_scale = 1.0` fallback in `_get_scale_mask` instead of correcting it.
    """
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2:
        raise ValueError("iou must be a 2-D matrix")
    matchable = iou >= threshold
    n_rows, n_cols = iou.shape
    degrees = matchable.sum(axis=1)
    summary: dict[str, Any] = {
        "frame_index": int(frame_index),
        "relates_to_anchor": bool(relates_to_anchor),
        "match_iou_threshold": float(threshold),
        "source_vertices": int(source_vertices),
        "target_vertices": int(target_vertices),
        "match_edge_count": int(matchable.sum()),
        "match_rejected_pairs": int(matchable.size - matchable.sum()),
        "match_rejected_ratio": (
            float(matchable.size - matchable.sum()) / float(matchable.size)
            if matchable.size
            else None
        ),
        "matched_source_vertex_ratio": (
            float((degrees > 0).sum()) / float(n_rows) if n_rows else None
        ),
        "matched_target_vertex_ratio": (
            float((matchable.sum(axis=0) > 0).sum()) / float(n_cols) if n_cols else None
        ),
        "degree_mean": float(degrees.mean()) if n_rows else None,
        "degree_max": int(degrees.max()) if n_rows else None,
        "degree_zero_count": int((degrees == 0).sum()) if n_rows else None,
    }
    iou_values = iou[np.isfinite(iou)]
    summary["match_iou_max"] = float(iou_values.max()) if iou_values.size else None
    summary.update(
        {
            f"match_iou_{key}": value
            for key, value in _quantiles(iou_values, (0.5, 0.9, 0.99)).items()
        }
    )
    return summary


# --------------------------------------------------------------------------------------------
# OP-4: the per-edge layer-scale estimation.
# --------------------------------------------------------------------------------------------


def summarize_edge_scale(
    *,
    edges: Sequence[Mapping[str, Any]],
    clamp_min: float,
) -> dict[str, Any]:
    """OP-4. Aggregate per-edge IRLS outcomes for one window.

    Each element of `edges` is expected to carry the keys produced by the LSA stage:
    `inter_mask_area`, `scale`, `iterations`, `converged`, `clamped`, `iou`. Missing keys are
    skipped instead of raising, so an instrumentation gap degrades to a smaller report rather than
    a failed run.
    """
    if not edges:
        return {
            "edge_scale_count": 0,
            "edge_scale_empty_mask_count": 0,
            "edge_scale_clamped_count": 0,
            "edge_scale_nonconverged_count": 0,
        }

    def values(key: str) -> np.ndarray:
        collected = [
            float(edge[key])
            for edge in edges
            if edge.get(key) is not None and np.isfinite(float(edge[key]))
        ]
        return np.asarray(collected, dtype=np.float64)

    scales = values("scale")
    areas = values("inter_mask_area")
    iterations = values("iterations")
    clamped = sum(1 for edge in edges if edge.get("clamped"))
    nonconverged = sum(1 for edge in edges if edge.get("converged") is False)
    empty_masks = sum(1 for edge in edges if float(edge.get("inter_mask_area", 1.0)) <= 0.0)

    summary: dict[str, Any] = {
        "edge_scale_count": int(len(edges)),
        "edge_scale_empty_mask_count": int(empty_masks),
        "edge_scale_empty_mask_ratio": float(empty_masks) / float(len(edges)),
        "edge_scale_clamped_count": int(clamped),
        "edge_scale_nonconverged_count": int(nonconverged),
        "edge_scale_iterations_mean": float(iterations.mean()) if iterations.size else None,
        "edge_scale_clamp_min": float(clamp_min),
    }
    for key, value in _quantiles(scales, (0.05, 0.25, 0.5, 0.75, 0.95)).items():
        summary[f"edge_scale_{key}"] = value
    for key, value in _quantiles(areas, (0.05, 0.5, 0.95)).items():
        summary[f"edge_mask_area_{key}"] = value
    if scales.size:
        summary["edge_scale_std"] = float(scales.std())
        summary["edge_scale_deviation_from_one_mean"] = float(np.mean(np.abs(scales - 1.0)))
        # A scale pinned at clamp_min is the signature of a degenerate intersection mask.
        summary["edge_scale_at_clamp_ratio"] = float(
            np.mean(np.isclose(scales, clamp_min, rtol=0.0, atol=1e-12))
        )
    else:
        summary["edge_scale_std"] = None
        summary["edge_scale_deviation_from_one_mean"] = None
        summary["edge_scale_at_clamp_ratio"] = None
    return summary


# --------------------------------------------------------------------------------------------
# OP-5: propagation and the final per-pixel mask.
# --------------------------------------------------------------------------------------------


def summarize_scale_propagation(
    *,
    vertex_scale_counts: Sequence[int],
    vertex_scale_values: Sequence[Sequence[float]],
    vertex_iou_weights: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """OP-5. Records the `Vertex` scale cache state that `_get_scale_mask` consumes.

    `vertex_scale_counts == 0` is the critical case: `_get_scale_mask` falls back to
    `mu_scale = 1.0`, i.e. the region is left uncorrected. Counting it converts "the improvement
    came from LSA" into a measurable claim about how many regions were actually reached.
    """
    counts = np.asarray(list(vertex_scale_counts), dtype=np.float64)
    total = int(counts.size)
    empty = int((counts == 0).sum()) if total else 0

    flat_scales: list[float] = []
    for values in vertex_scale_values:
        for value in values:
            value = float(value)
            if math.isfinite(value):
                flat_scales.append(value)
    scales = np.asarray(flat_scales, dtype=np.float64)

    summary: dict[str, Any] = {
        "propagation_vertex_count": total,
        "propagation_empty_cache_count": empty,
        "propagation_empty_cache_ratio": float(empty) / float(total) if total else None,
        "propagation_sources_mean": float(counts.mean()) if total else None,
        "propagation_sources_max": int(counts.max()) if total else None,
        "propagation_multi_source_count": (
            int((counts > 1).sum()) if total else None
        ),
        "propagation_scale_count": int(scales.size),
    }
    if scales.size:
        summary["propagation_scale_mean"] = float(scales.mean())
        summary["propagation_scale_std"] = float(scales.std())
        summary["propagation_scale_deviation_from_one_mean"] = float(
            np.mean(np.abs(scales - 1.0))
        )
    else:
        summary["propagation_scale_mean"] = None
        summary["propagation_scale_std"] = None
        summary["propagation_scale_deviation_from_one_mean"] = None

    if vertex_iou_weights:
        concentrations: list[float] = []
        for weights in vertex_iou_weights:
            arr = np.asarray(list(weights), dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            total_weight = arr.sum()
            if arr.size and total_weight > 0:
                normalized = arr / total_weight
                concentrations.append(float(np.max(normalized)))
        summary["propagation_iou_weight_concentration_mean"] = (
            float(np.mean(concentrations)) if concentrations else None
        )
    else:
        summary["propagation_iou_weight_concentration_mean"] = None
    return summary


def summarize_scale_mask(mask: np.ndarray, *, nonidentity_tolerance: float = 1e-6) -> dict[str, Any]:
    """OP-5 companion. Statistics of the per-pixel multiplicative mask actually applied.

    `nonidentity_ratio` is the share of pixels that received a real correction. If a method
    improves ATE while this number stays flat, the improvement did not travel through LSA.
    """
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    finite = mask[np.isfinite(mask)]
    summary: dict[str, Any] = {
        "mask_pixels": int(mask.size),
        "mask_nonfinite_count": int(mask.size - finite.size),
        "mask_nonidentity_tolerance": float(nonidentity_tolerance),
    }
    if finite.size == 0:
        return summary
    summary.update(
        {
            "mask_mean": float(finite.mean()),
            "mask_std": float(finite.std()),
            "mask_nonidentity_ratio": float(
                np.mean(np.abs(finite - 1.0) > nonidentity_tolerance)
            ),
            "mask_deviation_from_one_mean": float(np.mean(np.abs(finite - 1.0))),
        }
    )
    for key, value in _quantiles(finite, (0.01, 0.5, 0.99)).items():
        summary[f"mask_{key}"] = value
    return summary


# --------------------------------------------------------------------------------------------
# OP-6: did the correction actually make the overlap geometry more consistent?
# --------------------------------------------------------------------------------------------


def overlap_depth_consistency(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    relative_tolerance: float = 0.05,
    stride: int = 4,
    region_mask: np.ndarray | None = None,
    also_absolute: bool = True,
) -> dict[str, Any]:
    """OP-6. Measure how consistently the two windows scale the SAME overlap frames.

    `source_points` is the anchor window's already-registered overlap block and `target_points`
    the current window's overlap block, both in the same pixel layout. The same physical frame
    appears in both, so `target_z / source_z` is that frame's effective depth scale between the
    two windows. Two properties decide whether the registration is sound:

    * **consistency** — the scale must be near-constant over the frame. `consistency_rate` is the
      share of sampled pixels whose local scale stays within `relative_tolerance` of the frame's
      median scale. The default tolerance of 5% matches the PI3 geometry-warp default.
    * **level** — the median scale itself. It is *not* required to be 1: LSA exists precisely to
      change per-layer scales. Comparing `median_scales` before and after the mask is what shows
      whether the correction moved the two windows into agreement.

    No camera poses are needed, which keeps this free of projection conventions. `also_absolute`
    adds the share of pixels whose raw depth difference is within tolerance: a sanity signal that
    is expected to be low whenever a non-trivial scale is in play.

    A caller that wants a per-region rate passes that region's boolean mask through `region_mask`.
    """
    src_pts = np.asarray(source_points, dtype=np.float64)
    tgt_pts = np.asarray(target_points, dtype=np.float64)
    if src_pts.shape != tgt_pts.shape:
        raise ValueError(
            "source_points and target_points must have the same shape, got "
            f"{src_pts.shape} and {tgt_pts.shape}"
        )
    if src_pts.ndim != 4 or src_pts.shape[-1] != 3:
        raise ValueError("points must have shape (N, H, W, 3)")

    frames = int(src_pts.shape[0])
    stride = max(1, int(stride))
    region = None if region_mask is None else np.asarray(region_mask, dtype=bool)
    if region is not None and region.shape != src_pts.shape[1:3]:
        raise ValueError("region_mask must match the frame resolution")

    summary: dict[str, Any] = {
        "consistency_frames": frames,
        "consistency_relative_tolerance": float(relative_tolerance),
        "consistency_stride": int(stride),
    }

    pairs = 0
    consistent = 0
    absolute_consistent = 0
    median_scales: list[float] = []
    spread: list[float] = []
    residuals: list[np.ndarray] = []

    for frame in range(frames):
        source = src_pts[frame][::stride, ::stride].reshape(-1, 3)
        target = tgt_pts[frame][::stride, ::stride].reshape(-1, 3)
        if region is not None:
            selected = region[::stride, ::stride].reshape(-1)
            source = source[selected]
            target = target[selected]
        src_z = source[:, 2]
        tgt_z = target[:, 2]
        valid = (
            np.isfinite(src_z)
            & np.isfinite(tgt_z)
            & (np.abs(src_z) > 1e-6)
            & (np.abs(tgt_z) > 1e-6)
        )
        if not valid.any():
            continue
        src_z = src_z[valid]
        tgt_z = tgt_z[valid]
        local_scale = tgt_z / src_z
        median_scale = float(np.median(local_scale))
        if not math.isfinite(median_scale) or median_scale <= 0.0:
            continue
        relative = np.abs(local_scale - median_scale) / abs(median_scale)
        residuals.append(relative)
        pairs += int(relative.size)
        consistent += int((relative < relative_tolerance).sum())
        median_scales.append(median_scale)
        spread.append(float(np.quantile(relative, 0.90)))
        if also_absolute:
            absolute = np.abs(tgt_z - src_z) / np.maximum(np.abs(tgt_z), 1e-6)
            absolute_consistent += int((absolute < relative_tolerance).sum())

    summary["consistency_pairs"] = int(pairs)
    summary["consistency_consistent"] = int(consistent)
    summary["consistency_rate"] = float(consistent) / float(pairs) if pairs else None
    if median_scales:
        summary["consistency_median_scale"] = float(np.median(median_scales))
        summary["consistency_median_scale_std"] = float(np.std(median_scales))
        summary["consistency_window_scale_spread_p90"] = float(np.mean(spread))
    else:
        summary["consistency_median_scale"] = None
        summary["consistency_median_scale_std"] = None
        summary["consistency_window_scale_spread_p90"] = None
    if also_absolute:
        summary["consistency_absolute_rate"] = (
            float(absolute_consistent) / float(pairs) if pairs else None
        )
    if residuals:
        merged = np.concatenate(residuals)
        for key, value in _quantiles(merged, (0.5, 0.9, 0.95)).items():
            summary[f"consistency_residual_{key}"] = value
        summary["consistency_residual_mean"] = float(merged.mean())
    else:
        summary["consistency_residual_p50"] = None
        summary["consistency_residual_p90"] = None
        summary["consistency_residual_p95"] = None
        summary["consistency_residual_mean"] = None
    return summary


def mask_consistency_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """OP-6 delta between the before-mask and after-mask measurements.

    A positive `consistency_rate_delta` is evidence that the LSA correction actually made the
    overlap geometry more self-consistent. A flat or negative delta falsifies the claim that the
    improvement travels through LSA, whatever ATE happens to do.
    """

    def _delta(key: str) -> float | None:
        left = before.get(key)
        right = after.get(key)
        if left is None or right is None:
            return None
        return float(right) - float(left)

    def _ratio(key: str) -> float | None:
        left = before.get(key)
        right = after.get(key)
        if left in (None, 0) or right is None:
            return None
        return float(right) / float(left)

    return {
        "consistency_rate_before": before.get("consistency_rate"),
        "consistency_rate_after": after.get("consistency_rate"),
        "consistency_rate_delta": _delta("consistency_rate"),
        "consistency_median_scale_before": before.get("consistency_median_scale"),
        "consistency_median_scale_after": after.get("consistency_median_scale"),
        "consistency_median_scale_delta": _delta("consistency_median_scale"),
        "consistency_median_scale_ratio": _ratio("consistency_median_scale"),
        "consistency_scale_spread_before": before.get("consistency_window_scale_spread_p90"),
        "consistency_scale_spread_after": after.get("consistency_window_scale_spread_p90"),
        "consistency_scale_spread_delta": _delta("consistency_window_scale_spread_p90"),
        "consistency_residual_mean_before": before.get("consistency_residual_mean"),
        "consistency_residual_mean_after": after.get("consistency_residual_mean"),
        "consistency_residual_mean_delta": _delta("consistency_residual_mean"),
    }


# --------------------------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------------------------


@dataclass
class WindowDiagnostics:
    """All diagnostics collected for one window, keyed by observation point.

    Per-frame records are grouped by stage so that OP-3 (temporal matching, one record per frame
    pair) and OP-4 (per-edge IRLS, many records per frame pair) do not mix in the output.
    """

    window_id: int
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_stage: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def per_frame(self) -> list[dict[str, Any]]:
        """Flat view of every per-frame record, in stage-insertion order."""
        flat: list[dict[str, Any]] = []
        for records in self.per_stage.values():
            flat.extend(records)
        return flat

    def record_stage(self, stage: str, values: Mapping[str, Any]) -> None:
        """Merge `values` into the block for `stage`, e.g. `OP-1`, `OP-3`."""
        block = self.stages.setdefault(str(stage), {})
        for key, value in values.items():
            block[str(key)] = _finite_or_none(value)

    def record_frame(self, stage: str, frame_index: int, values: Mapping[str, Any]) -> None:
        bucket = self.per_stage.setdefault(str(stage), [])
        payload = {"stage": str(stage), "frame_index": int(frame_index)}
        payload.update({str(key): _finite_or_none(value) for key, value in values.items()})
        bucket.append(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": int(self.window_id),
            "stages": {
                stage: {key: _finite_or_none(value) for key, value in block.items()}
                for stage, block in self.stages.items()
            },
            "per_stage": {
                stage: [
                    {key: _finite_or_none(value) for key, value in record.items()}
                    for record in records
                ]
                for stage, records in self.per_stage.items()
            },
            "per_frame": [
                {key: _finite_or_none(value) for key, value in record.items()}
                for record in self.per_frame
            ],
            "notes": list(self.notes),
        }


def _sanitize(value: Any) -> Any:
    """Recursively make a payload JSON-safe.

    Diagnostics must never fail a reconstruction, so a stray non-finite float becomes `null`
    rather than raising inside `json.dumps`.
    """
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class DiagnosticsSink:
    """Thread-safe collector that writes one JSON file per window.

    The streaming engine runs inference and registration on separate threads and the segmentation
    helper is thread-parallel, so appends are guarded. Writing happens once per window on the
    registration thread, next to where the window's cache file is written.

    When diagnostics are disabled no sink is created at all: callers check
    `DiagnosticsConfig.active` and pass `None`, which is why every consumer here tolerates `None`.
    """

    def __init__(self, config: Any, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: WindowDiagnostics | None = None

    # -- window lifecycle -----------------------------------------------------------------

    def begin_window(self, window_id: int) -> WindowDiagnostics:
        window = WindowDiagnostics(window_id=int(window_id))
        with self._lock:
            self._current = window
        return window

    def current_window(self, window_id: int | None = None) -> WindowDiagnostics:
        with self._lock:
            window = self._current
        if window is None:
            window = self.begin_window(0 if window_id is None else window_id)
        return window

    def record_stage(self, stage: str, values: Mapping[str, Any]) -> None:
        self.current_window().record_stage(stage, values)

    def record_frame(
        self, frame_index: int, values: Mapping[str, Any], stage: str = "frame"
    ) -> None:
        """Append one per-frame record. Safe to call from worker threads."""
        window = self.current_window()
        with self._lock:
            window.record_frame(stage, frame_index, values)

    def publish_frame_stages(self) -> None:
        """Mirror every per-frame group into `stages` as a discoverable marker.

        OP-3 produces one row per frame pair and OP-4 one row per edge, so both live in
        `per_stage`. Without a matching `stages` entry, a reader scanning that map would conclude
        the observation point never fired — and an instrumentation gap would then look exactly
        like a real negative result.
        """
        window = self.current_window()
        for stage, rows in window.per_stage.items():
            if rows:
                self.record_stage(stage, {"frame_level": True, "row_count": int(len(rows))})

    def note(self, message: str) -> None:
        window = self.current_window()
        with self._lock:
            window.notes.append(str(message))

    # -- persistence ----------------------------------------------------------------------

    def write_window(self, window_id: int | None = None) -> Path | None:
        """Write the current window, or a named previously-collected one if it is still current."""
        with self._lock:
            window = self._current
        if window is None:
            return None
        if window_id is not None and window.window_id != int(window_id):
            # A different window is current; its own diagnostics were already written when it
            # finished, so there is nothing to write for the requested id here.
            return None
        return self.write(window)

    def write(self, window: WindowDiagnostics) -> Path:
        path = self.output_dir / f"window_{int(window.window_id):04d}.json"
        payload = json.dumps(
            _sanitize(window.to_dict()), indent=2, sort_keys=True, allow_nan=False
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
        return path

    def write_arrays(self, window_id: int, arrays: Mapping[str, np.ndarray]) -> list[Path]:
        """Write full-resolution arrays, only when the observation level asks for them."""
        if not getattr(self.config, "wants_arrays", False):
            return []
        written: list[Path] = []
        array_dir = self.output_dir / f"window_{int(window_id):04d}_arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            path = array_dir / f"{name}.npy"
            np.save(path, np.asarray(array))
            written.append(path)
        return written


class NullDiagnosticsSink:
    """No-op sink for callers that prefer branch-free instrumentation."""

    config = None
    output_dir = None

    def begin_window(self, window_id: int) -> WindowDiagnostics:
        return WindowDiagnostics(window_id=int(window_id))

    def current_window(self, window_id: int | None = None) -> WindowDiagnostics:
        return WindowDiagnostics(window_id=0 if window_id is None else int(window_id))

    def record_stage(self, stage: str, values: Mapping[str, Any]) -> None:
        return None

    def record_frame(
        self, frame_index: int, values: Mapping[str, Any], stage: str = "frame"
    ) -> None:
        return None

    def note(self, message: str) -> None:
        return None

    def write_window(self, window_id: int | None = None) -> Path | None:
        return None

    def write(self, window: WindowDiagnostics) -> Path | None:
        return None

    def write_arrays(self, window_id: int, arrays: Mapping[str, np.ndarray]) -> list[Path]:
        return []



def load_window_diagnostics(path: str | Path) -> dict[str, Any]:
    """Read back one window JSON. Used by the L3/L4 attribution tooling and the checks."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_window_diagnostics(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a window payload into `stage.key` scalars for tabular attribution.

    Per-frame records are collapsed with a stage-qualified mean/max/min so that one window becomes
    one row of the L4 attribution table. Counts are summed naturally by the mean of a constant
    series only when that series is constant, so count-like keys keep all three aggregates.
    """
    flat: dict[str, Any] = {}
    for stage, block in (payload.get("stages") or {}).items():
        for key, value in block.items():
            flat[f"{prefix}{stage}.{key}"] = value

    for stage, records in (payload.get("per_stage") or {}).items():
        if not records:
            continue
        keys = {key for record in records for key in record}
        for key in sorted(keys):
            if key in ("stage", "frame_index"):
                continue
            values = [
                float(record[key])
                for record in records
                if isinstance(record.get(key), (int, float))
                and not isinstance(record.get(key), bool)
            ]
            if not values:
                continue
            flat[f"{prefix}{stage}.{key}_mean"] = sum(values) / float(len(values))
            flat[f"{prefix}{stage}.{key}_max"] = max(values)
            flat[f"{prefix}{stage}.{key}_min"] = min(values)
    return flat


__all__ = [
    "OP_SEGMENTATION_EXIT",
    "OP_MERGE_EVIDENCE",
    "OP_TEMPORAL_MATCH",
    "OP_EDGE_IRLS",
    "OP_SCALE_MASK",
    "OP_OVERLAP_CONSISTENCY",
    "region_areas",
    "summarize_labels",
    "summarize_segmentation_exit",
    "summarize_merge_evidence",
    "summarize_split_decision",
    "summarize_temporal_match_per_frame",
    "summarize_edge_scale",
    "summarize_scale_propagation",
    "summarize_scale_mask",
    "overlap_depth_consistency",
    "mask_consistency_delta",
    "WindowDiagnostics",
    "DiagnosticsSink",
    "NullDiagnosticsSink",
    "load_window_diagnostics",
    "flatten_window_diagnostics",
]
