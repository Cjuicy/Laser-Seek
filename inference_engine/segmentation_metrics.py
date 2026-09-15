"""L2 metrics: turning the synthetic ground truth into the L2-A / L2-B checkpoints.

The point of these functions is attribution, not scoring. IDEA-001 asks *why* a segmentation method
moves ATE, and the only way to answer that without ATE is to split the pipeline's error into the
part the segmentation contributes and the part that is inherent to LSA:

* **L2-A segmentation quality** measures a method against the ground-truth layer partition, with no
  LSA involved at all. Different metrics here say different things: ARI and VI measure the
  partition as a whole, while `boundary_violation_rate` measures only the *decision* the merge
  criterion made at each shared boundary, which is what the three methods actually disagree about.
* **L2-B landing quality** feeds a partition into the LSA scale estimator and measures how far the
  recovered per-layer scale is from the known truth. Running it twice — once with the ground-truth
  partition and once with the method's own partition — separates the two error sources:
  `method_error - floor_error` is the segmentation's own contribution.

Nothing here is used by the reconstruction pipeline. These are measurement functions.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:  # scikit-learn is already a project dependency through the evaluation stack
    from sklearn.metrics import adjusted_rand_score as _ari
    from sklearn.metrics import adjusted_mutual_info_score as _ami
except Exception:  # pragma: no cover - only hit in a stripped environment
    _ari = None
    _ami = None


# --------------------------------------------------------------------------------------------
# L2-A: segmentation quality against a known layer partition
# --------------------------------------------------------------------------------------------


def partition_similarity(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    """ARI, AMI and the region-count ratio of one label map against the ground-truth partition.

    ARI is chance-corrected, so a method that merges everything scores near 0 rather than being
    rewarded for the dominant layer's pixel share.
    """
    predicted = np.asarray(predicted).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    valid = truth >= 0
    predicted = predicted[valid]
    truth = truth[valid]
    if predicted.size == 0:
        return {"ari": None, "ami": None, "region_count_ratio": None}
    result: dict[str, float | None] = {
        "ari": float(_ari(truth, predicted)) if _ari else None,
        "ami": float(_ami(truth, predicted)) if _ami else None,
    }
    predicted_regions = int(np.unique(predicted).size)
    truth_regions = int(np.unique(truth).size)
    result["region_count_ratio"] = (
        float(predicted_regions) / float(truth_regions) if truth_regions else None
    )
    result["predicted_region_count"] = predicted_regions
    result["truth_region_count"] = truth_regions
    return result


def dominant_layer_purity(
    predicted: np.ndarray, truth: np.ndarray
) -> dict[str, float | None]:
    """For each predicted region, the share of its pixels belonging to its dominant true layer.

    Reported as the pixel-weighted mean, plus the fraction of regions that are perfectly pure. A
    method whose regions are fewer but purer is doing exactly what the mechanism claims.
    """
    predicted = np.asarray(predicted).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    valid = truth >= 0
    predicted = predicted[valid]
    truth = truth[valid]
    if predicted.size == 0:
        return {"purity_pixel_weighted": None, "pure_region_fraction": None}
    purities = []
    weights = []
    for label in np.unique(predicted):
        mask = predicted == label
        _, counts = np.unique(truth[mask], return_counts=True)
        purities.append(float(counts.max()) / float(mask.sum()))
        weights.append(int(mask.sum()))
    purities = np.asarray(purities)
    weights = np.asarray(weights, dtype=np.float64)
    return {
        "purity_pixel_weighted": float(np.average(purities, weights=weights)),
        "pure_region_fraction": float(np.mean(purities >= 0.999)),
    }


def boundary_violation_rate(
    labels: np.ndarray,
    truth: np.ndarray,
    geometry: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """The share of true same-layer adjacencies that the method left split.

    A ground-truth layer boundary and a method's region boundary sit at *different pixels*, so
    comparing the two boundary maps elementwise is meaningless: even an exactly-correct partition
    disagrees with the truth map about where the boundary runs. The well-defined question is the
    decision. Take every 4-connected pair of pixels that truly belong to the same layer; the method
    either keeps them together or separates them. This measures the fraction it separated.

    That distinguishes two errors a region count cannot tell apart:

    * **over-fragmentation** — one true layer split into several regions. Raises this metric, and is
      the mechanism the Idea document names for unstable per-layer scale estimates.
    * **over-merging** — one region spanning several true layers. Reported separately as
      `cross_layer_region_ratio`, because a single merged region covers arbitrarily many
      truly-same-layer pairs and would otherwise swamp the first quantity.

    `geometry` may carry `depth` and `normal` maps so the depth gap and normal angle at each split
    can be described, which is what says whether the split was avoidable given what the criterion
    could see.
    """
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    if labels.shape != truth.shape:
        raise ValueError("labels and truth must have the same shape")

    depth = None if geometry is None else geometry.get("depth")
    normal = None if geometry is None else geometry.get("normal")

    same_layer_pairs = 0
    split_pairs = 0
    gap_samples: list[np.ndarray] = []
    normal_samples: list[np.ndarray] = []

    for axis in (0, 1):
        left = [slice(None)] * 2
        right = [slice(None)] * 2
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        a_label, b_label = labels[left], labels[right]
        a_truth, b_truth = truth[left], truth[right]
        valid = (a_truth >= 0) & (b_truth >= 0)
        same_layer = valid & (a_truth == b_truth)
        split = same_layer & (a_label != b_label)
        same_layer_pairs += int(same_layer.sum())
        split_pairs += int(split.sum())
        if split.any() and depth is not None:
            gap_samples.append(np.abs(depth[left][split] - depth[right][split]))
        if split.any() and normal is not None:
            cosine = np.abs((normal[left][split] * normal[right][split]).sum(axis=-1))
            normal_samples.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    per_region_layers: list[int] = []
    for label in np.unique(labels):
        mask = labels == label
        present = truth[mask & (truth >= 0)]
        if present.size:
            per_region_layers.append(int(np.unique(present).size))
    cross_layer_regions = sum(1 for count in per_region_layers if count > 1)

    summary: dict[str, float] = {
        "same_layer_pairs": int(same_layer_pairs),
        "split_pairs": int(split_pairs),
        "boundary_violation_rate": (
            float(split_pairs) / float(same_layer_pairs) if same_layer_pairs else 0.0
        ),
        "cross_layer_region_count": int(cross_layer_regions),
        "cross_layer_region_ratio": (
            float(cross_layer_regions) / float(len(per_region_layers))
            if per_region_layers
            else 0.0
        ),
    }
    if gap_samples:
        merged = np.concatenate(gap_samples)
        summary["split_depth_gap_mean"] = float(merged.mean())
        summary["split_depth_gap_max"] = float(merged.max())
    if normal_samples:
        merged = np.concatenate(normal_samples)
        summary["split_normal_angle_mean"] = float(merged.mean())
        summary["split_normal_angle_max"] = float(merged.max())
    return summary


def region_area_stats(labels: np.ndarray) -> dict[str, float | None]:
    """Area distribution, the direct measure of the fragmentation the Idea document describes.

    Counts only labels that actually occur. `np.bincount` would also count the holes left by the
    Baseline's non-contiguous `root + 1` labels and report invisible zero-area regions as real
    ones, which inflates every region-count comparison between the three methods.
    """
    labels = np.asarray(labels)
    _, areas = np.unique(labels.reshape(-1), return_counts=True)
    areas = areas.astype(np.float64)
    if areas.size == 0:
        return {"region_count": 0, "area_mean": None, "area_median": None, "tiny_fraction": None}
    pixels = float(labels.size)
    return {
        "region_count": int(areas.size),
        "area_mean": float(areas.mean()),
        "area_median": float(np.median(areas)),
        "area_min": float(areas.min()),
        "tiny_fraction": float(np.mean(areas < 64)),
        "regions_per_megapixel": float(areas.size) / (pixels / 1.0e6),
    }


def evaluate_segmentation(
    labels: np.ndarray,
    truth: np.ndarray,
    geometry: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float | None]:
    """The complete L2-A checkpoint for one frame."""
    report: dict[str, float | None] = {}
    report.update(partition_similarity(labels, truth))
    report.update(dominant_layer_purity(labels, truth))
    report.update(boundary_violation_rate(labels, truth, geometry))
    report.update(region_area_stats(labels))
    return report


# --------------------------------------------------------------------------------------------
# L2-B: landing quality -- how well LSA recovers a KNOWN per-layer scale
# --------------------------------------------------------------------------------------------


def layer_scale_recovery(
    scale_predictions: Sequence[float],
    scale_truth: Sequence[float],
) -> dict[str, float | None]:
    """Compare recovered per-layer scales against the fixture's known factors."""
    predicted = np.asarray(list(scale_predictions), dtype=np.float64)
    truth = np.asarray(list(scale_truth), dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("scale_predictions and scale_truth must have the same length")
    relative = np.abs(predicted - truth) / np.maximum(np.abs(truth), 1e-12)
    return {
        "layer_count": int(truth.size),
        "scale_truth_first": float(truth[0]) if truth.size else None,
        "scale_relative_error_mean": float(relative.mean()) if relative.size else None,
        "scale_relative_error_max": float(relative.max()) if relative.size else None,
        "scale_recovered_mean": float(predicted.mean()) if predicted.size else None,
        "scale_truth_mean": float(truth.mean()) if truth.size else None,
    }


def landing_error(
    points_before: np.ndarray,
    points_after: np.ndarray,
    points_truth: np.ndarray,
    region_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """How far the corrected point map lands from the analytically true one.

    `relative_error` is the depth-relative RMS distance, so it is comparable across scenes. The
    `improvement` field is the fraction of the original error removed: 0 means the correction did
    nothing, negative means it made the geometry worse than doing nothing at all.
    """
    before = np.asarray(points_before, dtype=np.float64)
    after = np.asarray(points_after, dtype=np.float64)
    truth = np.asarray(points_truth, dtype=np.float64)
    if before.shape != truth.shape or after.shape != truth.shape:
        raise ValueError("all point maps must have the same shape")

    mask = np.isfinite(truth).all(axis=-1) & np.isfinite(before).all(axis=-1) & np.isfinite(after).all(axis=-1)
    if region_mask is not None:
        mask &= np.asarray(region_mask, dtype=bool)
    if not mask.any():
        return {
            "relative_error_before": None,
            "relative_error_after": None,
            "improvement": None,
            "pixels": 0,
        }

    scale = np.maximum(np.abs(truth[..., 2][mask]), 1e-9)
    error_before = np.sqrt(((before[mask] - truth[mask]) ** 2).sum(-1)) / scale
    error_after = np.sqrt(((after[mask] - truth[mask]) ** 2).sum(-1)) / scale
    rms_before = float(np.sqrt((error_before ** 2).mean()))
    rms_after = float(np.sqrt((error_after ** 2).mean()))
    return {
        "relative_error_before": rms_before,
        "relative_error_after": rms_after,
        "improvement": (rms_before - rms_after) / rms_before if rms_before > 0 else None,
        "pixels": int(mask.sum()),
    }


def attribution_delta(
    method_error: float | None, floor_error: float | None
) -> dict[str, float | None]:
    """Separate the segmentation's contribution from the error inherent to LSA.

    `floor_error` comes from running the same LSA stage on the ground-truth partition. Any error
    there is LSA's own; anything the method adds on top is what the segmentation is responsible
    for. This is the number that answers "how much better is this method, and where does the
    difference come from" independently of ATE.
    """
    if method_error is None or floor_error is None:
        return {"method_error": method_error, "floor_error": floor_error, "segmentation_cost": None}
    return {
        "method_error": float(method_error),
        "floor_error": float(floor_error),
        "segmentation_cost": float(method_error) - float(floor_error),
    }


# --------------------------------------------------------------------------------------------
# Cross-method comparison
# --------------------------------------------------------------------------------------------


def compare_methods(
    per_method: Mapping[str, Mapping[str, float | None]],
    key: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, object]:
    """Rank methods on one metric and name the winner, without hiding ties."""
    values = {name: report.get(key) for name, report in per_method.items()}
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return {"metric": key, "values": values, "best": None, "spread": None}
    best = (max if higher_is_better else min)(present, key=present.get)
    ordered = sorted(present.values())
    return {
        "metric": key,
        "values": values,
        "best": best,
        "spread": float(ordered[-1] - ordered[0]),
        "higher_is_better": bool(higher_is_better),
    }


__all__ = [
    "partition_similarity",
    "dominant_layer_purity",
    "boundary_violation_rate",
    "region_area_stats",
    "evaluate_segmentation",
    "layer_scale_recovery",
    "landing_error",
    "attribution_delta",
    "compare_methods",
]
