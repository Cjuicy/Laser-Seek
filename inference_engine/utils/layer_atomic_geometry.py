from dataclasses import dataclass

import numpy as np

from .depth import segment_depth_felzenszwalb_rag_stages
from .post_merge_split import SplitDiagnostics, refine_auto_regions


@dataclass(frozen=True)
class AtomMergeResult:
    labels: np.ndarray
    atom_labels: np.ndarray
    atom_scales: np.ndarray


def _compact_labels(labels):
    labels = np.asarray(labels)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape)


class _DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while node != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller = min(left_root, right_root)
            larger = max(left_root, right_root)
            self.parent[larger] = smaller


def _merge_layer_atoms_with_metadata(
    point_map,
    initial_labels,
    coarse_labels,
    depth_merge_thresh,
) -> AtomMergeResult:
    point_map = np.asarray(point_map)
    initial_labels = np.asarray(initial_labels)
    coarse_labels = np.asarray(coarse_labels)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape (H, W, 3)")
    if initial_labels.shape != point_map.shape[:2]:
        raise ValueError("initial_labels must have shape (H, W)")
    if coarse_labels.shape != point_map.shape[:2]:
        raise ValueError("coarse_labels must have shape (H, W)")

    atom_labels = _compact_labels(initial_labels)
    n_atoms = int(atom_labels.max()) + 1
    scale_sums = np.zeros(n_atoms, dtype=np.float64)
    scale_counts = np.zeros(n_atoms, dtype=np.int64)
    boundary_codes = []
    boundary_distances = []

    directions = (
        (point_map[:, :-1], point_map[:, 1:], atom_labels[:, :-1], atom_labels[:, 1:]),
        (point_map[:-1, :], point_map[1:, :], atom_labels[:-1, :], atom_labels[1:, :]),
    )
    for points_a, points_b, atoms_a, atoms_b in directions:
        with np.errstate(invalid="ignore", over="ignore"):
            differences = points_a - points_b
            distances = np.linalg.norm(differences, axis=-1)
        del differences

        finite = np.isfinite(distances)
        internal = atoms_a == atoms_b
        valid_internal = finite & internal & (distances > 0)
        scale_sums += np.bincount(
            atoms_a[valid_internal],
            weights=distances[valid_internal],
            minlength=n_atoms,
        )
        scale_counts += np.bincount(
            atoms_a[valid_internal],
            minlength=n_atoms,
        )

        valid_boundary = finite & ~internal
        boundary_left = np.minimum(atoms_a[valid_boundary], atoms_b[valid_boundary])
        boundary_right = np.maximum(atoms_a[valid_boundary], atoms_b[valid_boundary])
        boundary_codes.append(boundary_left * n_atoms + boundary_right)
        boundary_distances.append(distances[valid_boundary])
        del distances

    scales = np.zeros(n_atoms, dtype=np.float64)
    np.divide(
        scale_sums,
        scale_counts,
        out=scales,
        where=scale_counts > 0,
    )
    valid_scales = (scale_counts > 0) & np.isfinite(scales) & (scales > 0)

    if boundary_codes:
        all_codes = np.concatenate(boundary_codes)
        all_distances = np.concatenate(boundary_distances)
    else:
        all_codes = np.empty(0, dtype=np.int64)
        all_distances = np.empty(0, dtype=np.float64)

    dsu = _DisjointSet(n_atoms)
    if all_codes.size:
        unique_codes, inverse = np.unique(all_codes, return_inverse=True)
        pair_counts = np.bincount(inverse)
        pair_gaps = np.bincount(inverse, weights=all_distances) / pair_counts
        pair_left = unique_codes // n_atoms
        pair_right = unique_codes % n_atoms

        atom_ids, first_indices = np.unique(atom_labels.reshape(-1), return_index=True)
        atom_coarse = np.empty(n_atoms, dtype=coarse_labels.dtype)
        atom_coarse[atom_ids] = coarse_labels.reshape(-1)[first_indices]

        denominator = np.sqrt(scales[pair_left]) * np.sqrt(scales[pair_right])
        valid_pair = (
            valid_scales[pair_left]
            & valid_scales[pair_right]
            & np.isfinite(pair_gaps)
            & np.isfinite(denominator)
            & (denominator > 0)
        )
        normalized_gap = np.full(pair_gaps.shape, np.inf, dtype=np.float64)
        with np.errstate(invalid="ignore", over="ignore"):
            np.divide(
                pair_gaps,
                denominator,
                out=normalized_gap,
                where=valid_pair,
            )
        same_coarse = atom_coarse[pair_left] == atom_coarse[pair_right]
        limits = np.where(same_coarse, 1.0 + depth_merge_thresh, 1.0)
        should_merge = valid_pair & (normalized_gap <= limits)

        for left, right in zip(pair_left[should_merge], pair_right[should_merge], strict=True):
            dsu.union(int(left), int(right))

    roots = np.asarray([dsu.find(atom) for atom in range(n_atoms)])
    labels = _compact_labels(roots[atom_labels])
    return AtomMergeResult(labels, atom_labels, scales)


def merge_layer_atoms(
    point_map,
    initial_labels,
    coarse_labels,
    depth_merge_thresh,
) -> np.ndarray:
    return _merge_layer_atoms_with_metadata(
        point_map,
        initial_labels,
        coarse_labels,
        depth_merge_thresh,
    ).labels


def _depth_stage_labels(stages):
    """Return `(initial_labels, coarse_labels)` from a depth-stages result.

    The reference `.depth::segment_depth_felzenszwalb_rag_stages` returns the tuple
    `(initial_labels, coarse_labels, merge_threshold)`, which the reference unpacks inline. In the
    target repository that function returns a `SegmentationStages` record instead, whose
    `initial_labels` / `merged_labels` fields carry exactly those first two values. This adapter
    accepts both shapes so the call site below stays the reference's single positional call with
    unchanged meaning; it exists only because the target's return shape differs from the
    reference's.
    """
    if hasattr(stages, "initial_labels") and hasattr(stages, "merged_labels"):
        return stages.initial_labels, stages.merged_labels
    initial_labels, coarse_labels, _ = stages
    return initial_labels, coarse_labels


def _select_rgb_frame(rgb_images, batch_idx, height, width):
    if rgb_images is None:
        return None
    rgb = np.asarray(rgb_images)
    if rgb.ndim == 4:
        if batch_idx is None:
            raise ValueError("batch_idx is required for batched rgb_images")
        rgb = rgb[batch_idx]
    if rgb.shape == (3, height, width):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape != (height, width, 3):
        raise ValueError(
            "rgb_images must contain RGB frames aligned with point_map"
        )
    return np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)


def segment_point_map_atomic(
    point_map,
    depth_merge_thresh,
    rgb_images=None,
    normal_method="cross",
    split_score_threshold=0.10,
    split_mode="conservative",
    conf_map=None,
    confidence_keep_ratio=None,
    seg_scale=300,
    seg_sigma=1.1,
    seg_min_size=500,
    batch_idx=None,
    confidence_quantile_method="nearest",
) -> np.ndarray:
    point_map = np.asarray(point_map)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape (H, W, 3)")

    depth_map = point_map[..., -1]
    initial_labels, coarse_labels = _depth_stage_labels(
        segment_depth_felzenszwalb_rag_stages(
            depth_map,
            depth_merge_thresh,
            conf_map,
            confidence_keep_ratio,
            seg_scale,
            seg_sigma,
            seg_min_size,
            batch_idx,
            # IDEA-001 addition: the locked configuration carries one quantile rule for all three
            # methods, so the atomic branch must honour it too. Appended last so that the eight
            # positional arguments of the reference call site keep their exact positions.
            confidence_quantile_method,
        )
    )
    merged = _merge_layer_atoms_with_metadata(
        point_map,
        initial_labels,
        coarse_labels,
        depth_merge_thresh,
    )
    rgb_image = _select_rgb_frame(
        rgb_images,
        batch_idx,
        point_map.shape[0],
        point_map.shape[1],
    )
    return refine_auto_regions(
        point_map,
        rgb_image,
        merged.labels,
        merged.atom_labels,
        merged.atom_scales,
        seg_min_size=seg_min_size,
        normal_method=normal_method,
        split_score_threshold=split_score_threshold,
        split_mode=split_mode,
    )
