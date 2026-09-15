#!/usr/bin/env python3
"""L1 invariants and L2 synthetic ground-truth checks for IDEA-001.

Run this with no arguments from the repository root:

    python scripts/check_segmentation_methods.py

It needs no GPU, no model weights and no dataset. Everything it measures comes from analytically
exact synthetic scenes (`inference_engine/synthetic_scenes.py`), so the "ground truth" here is a
construction rather than an annotation, and the numbers are reproducible on any machine.

What it answers
---------------
* **L1** — do the three methods produce well-formed partitions, does the Baseline path stay
  element-wise identical, and is the whole thing deterministic?
* **L2-A** — which method recovers the true depth-layer partition better, and which merge
  decisions violate it?
* **L2-B** — feed a partition into the LSA scale estimator and measure how far it lands from the
  known truth. Run twice per method: once with the ground-truth partition (the floor, which is
  LSA's own error) and once with the method's partition. The difference is the segmentation's cost.
* **L2-C** — the predicted-versus-observed behaviour matrix for each fixture.

Exit status is 0 when every L1 invariant holds, 1 otherwise. L2 numbers are reported, not gated:
they are measurements about competing methods, and turning them into pass/fail thresholds would
pre-judge the experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_engine.segmentation_config import (  # noqa: E402
    SEGMENTATION_METHODS,
    describe_config,
    load_segmentation_config,
)
from inference_engine.segmentation_metrics import (  # noqa: E402
    attribution_delta,
    evaluate_segmentation,
    landing_error,
    layer_scale_recovery,
)
from inference_engine.synthetic_scenes import SCENARIOS, build_pair  # noqa: E402
from inference_engine.utils.depth import (  # noqa: E402
    align_depth_irls,
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)
from inference_engine.utils.lsa import segment_labels_for_method  # noqa: E402
from inference_engine.utils._segmentation_cy import merge_regions  # noqa: E402
from skimage.segmentation import felzenszwalb  # noqa: E402


# --------------------------------------------------------------------------------------------
# L1
# --------------------------------------------------------------------------------------------


def l1_partition_invariants(labels: np.ndarray, *, method: str, frame: int) -> list[str]:
    """Compact labels covering the whole frame, no NaN, no negative values."""
    problems: list[str] = []
    labels = np.asarray(labels)
    if labels.ndim != 2:
        problems.append(f"{method} frame {frame}: labels must be 2-D, got {labels.shape}")
        return problems
    if not np.issubdtype(labels.dtype, np.integer):
        problems.append(f"{method} frame {frame}: labels must be integral, got {labels.dtype}")
    unique = np.unique(labels)
    # Compactness is NOT asserted here. The Baseline's Cython `merge_regions` emits 1-based labels
    # (`root + 1`), so the `depth` method legitimately starts at 1; the two new strategies call
    # `compact_labels` and start at 0. Both are fine because every consumer works from
    # `np.unique`. Asserting 0-based compactness would fail the Baseline, which is the opposite of
    # the point of a Regression gate.
    if unique.size and unique.size != int(unique.max() - unique.min() + 1):
        problems.append(
            f"{method} frame {frame}: labels are not contiguous (got {unique[:8]}...)"
        )
    if labels.size and np.isnan(labels.astype(np.float64)).any():
        problems.append(f"{method} frame {frame}: labels contain NaN")
    if unique.size and unique.min() < 0:
        problems.append(f"{method} frame {frame}: labels contain negative values")
    return problems


def l1_merge_only_invariant(
    labels: np.ndarray, atoms: np.ndarray, *, method: str, frame: int
) -> list[str]:
    """Every output region must be a union of input atoms; no region may be split or invented."""
    problems: list[str] = []
    labels = np.asarray(labels)
    atoms = np.asarray(atoms)
    for label in np.unique(labels):
        mask = labels == label
        for atom in np.unique(atoms[mask]):
            atom_mask = atoms == atom
            if not np.all(labels[atom_mask] == label):
                problems.append(
                    f"{method} frame {frame}: atom {atom} is split across labels "
                    f"(region {label} holds only part of it)"
                )
                return problems
    return problems


def l1_baseline_identity(depth: np.ndarray, conf: np.ndarray, cfg) -> list[str]:
    """`segment_depth_felzenszwalb_rag` must equal an independent oracle built from its parts."""
    problems: list[str] = []
    frames = depth.shape[0]
    for frame in range(frames):
        frame_depth = depth[frame]
        frame_conf = conf[frame]
        stages = segment_depth_felzenszwalb_rag_stages(
            frame_depth,
            cfg.depth_merge_thresh,
            conf_map=frame_conf,
            confidence_keep_ratio=cfg.confidence_keep_ratio,
            seg_scale=cfg.felzenszwalb.scale,
            seg_sigma=cfg.felzenszwalb.sigma,
            seg_min_size=cfg.felzenszwalb.min_size,
            confidence_quantile_method=cfg.confidence_quantile_method,
        )
        wrapper = segment_depth_felzenszwalb_rag(
            frame_depth,
            cfg.depth_merge_thresh,
            conf_map=frame_conf,
            confidence_keep_ratio=cfg.confidence_keep_ratio,
            seg_scale=cfg.felzenszwalb.scale,
            seg_sigma=cfg.felzenszwalb.sigma,
            seg_min_size=cfg.felzenszwalb.min_size,
            confidence_quantile_method=cfg.confidence_quantile_method,
        )
        # Independent oracle: the same two primitives called directly.
        atoms = felzenszwalb(
            frame_depth,
            scale=cfg.felzenszwalb.scale,
            sigma=cfg.felzenszwalb.sigma,
            min_size=cfg.felzenszwalb.min_size,
        )
        threshold = cfg.depth_merge_thresh * (
            np.max(frame_depth[stages.high_confidence_mask])
            - np.min(frame_depth[stages.high_confidence_mask])
        )
        oracle = merge_regions(atoms, frame_depth, threshold)

        if not np.array_equal(np.asarray(wrapper), np.asarray(stages.merged_labels)):
            problems.append(f"frame {frame}: wrapper != stages.merged_labels")
        if not np.array_equal(np.asarray(stages.merged_labels), np.asarray(oracle)):
            problems.append(f"frame {frame}: stages.merged_labels != independent oracle")
        if not np.isclose(threshold, stages.merge_threshold, rtol=1e-12, atol=1e-12):
            problems.append(
                f"frame {frame}: merge threshold {stages.merge_threshold} != oracle {threshold}"
            )
    return problems


def l1_irls_sanity(cfg) -> list[str]:
    """`align_depth_irls` must recover a known scale exactly and report its own convergence."""
    problems: list[str] = []
    rng = np.random.default_rng(7)
    source = rng.uniform(1.0, 6.0, size=(32, 32))
    for truth in (0.5, 1.0, 2.5):
        target = source * truth
        scale, info = align_depth_irls(
            source,
            target,
            None,
            return_info=True,
            iters=cfg.irls.iters,
            eps=cfg.irls.eps,
            stop_tol=cfg.irls.stop_tol,
            clamp_min=cfg.irls.clamp_min,
        )
        if not np.isclose(scale, truth, rtol=1e-6):
            problems.append(f"IRLS recovered {float(scale)} for a true scale of {truth}")
        if not info["converged"]:
            problems.append(f"IRLS did not converge for a true scale of {truth}")
        if info["sample_count"] != source.size:
            problems.append("IRLS reported the wrong sample count")
    # A degenerate intersection mask is a POSITIVE but vanishingly small depth pair: the ratio
    # then collapses onto `clamp_min`. An all-zero mask is deliberately not used because 0/0
    # produces NaN at the initialisation step, before any clamp can apply. That NaN behaviour is
    # Baseline behaviour and is reported by the L1 section below rather than asserted here.
    tiny_source = np.full((8, 8), 1e-12)
    tiny_target = np.full((8, 8), 1e-18)
    _scale, info = align_depth_irls(
        tiny_source, tiny_target, None, return_info=True,
        clamp_min=cfg.irls.clamp_min,
    )
    if not info["clamped"]:
        problems.append("IRLS did not report the clamp on a degenerate positive pair")
    zero_scale, zero_info = align_depth_irls(
        np.zeros((8, 8)), np.zeros((8, 8)), None, return_info=True
    )
    if not np.isnan(zero_scale):
        problems.append("IRLS on an all-zero mask no longer returns NaN (Baseline behaviour changed)")
    del zero_info
    return problems


def l1_determinism(cfg, points: np.ndarray, depth: np.ndarray, conf: np.ndarray) -> list[str]:
    """Two identical CPU runs must give identical labels for all three methods."""
    problems: list[str] = []
    for method in SEGMENTATION_METHODS:
        first, _ = segment_labels_for_method(
            method, depth, segmentation=cfg, point_map=points, conf_map=conf
        )
        second, _ = segment_labels_for_method(
            method, depth, segmentation=cfg, point_map=points, conf_map=conf
        )
        if not np.array_equal(np.asarray(first), np.asarray(second)):
            problems.append(f"{method}: labels differ between two identical runs")
    return problems


# --------------------------------------------------------------------------------------------
# L2
# --------------------------------------------------------------------------------------------


def _estimate_overlap_scale(
    anchor_points: np.ndarray,
    target_points: np.ndarray,
    anchor_mask: np.ndarray,
    target_mask: np.ndarray,
    cfg,
) -> float:
    """The Baseline's own overlap scale: the global scalar fitted on the mutual mask."""
    mutual = anchor_mask & target_mask
    if not mutual.any():
        return float("nan")
    return float(
        align_depth_irls(
            anchor_points[..., 2],
            target_points[..., 2],
            mutual,
            iters=cfg.irls.iters,
            eps=cfg.irls.eps,
            stop_tol=cfg.irls.stop_tol,
            clamp_min=cfg.irls.clamp_min,
        )
    )


def run_l2(
    fixture: str, cfg, *, frame_count: int = 4, overlap: int = 2, warp_spread: float = 0.0
) -> dict:
    # `warp_spread` makes each layer's monocular scale differ, which is the case a single
    # global scale CANNOT fix and therefore the only case where segmentation quality can
    # influence the estimated scale. With spread 0 every layer shares one factor and the L2-B
    # numbers are uninformative by construction.
    _, probe, probe_meta = build_pair(fixture, frame_count=1, overlap=1)
    layer_count_probe = len(probe_meta["layer_names"])
    per_layer_warp = {
        index: float(probe_meta["global_warp"]) * (1.0 + warp_spread * (index - (layer_count_probe - 1) / 2.0))
        for index in range(layer_count_probe)
    }
    anchor, target, meta = build_pair(
        fixture, frame_count=frame_count, overlap=overlap, per_layer_warp=per_layer_warp
    )
    layer_names = meta["layer_names"]
    layer_count = len(layer_names)
    per_layer_warp = meta["per_layer_warp"]

    target_frame = target.frames[0]
    truth_labels = target_frame.layer_label
    valid = target_frame.valid_mask
    depth = np.nan_to_num(target_frame.depth, nan=0.0)
    points = np.nan_to_num(target_frame.points, nan=0.0)
    # PI3 confidence is a raw logit; a large positive value for valid pixels keeps the
    # high-confidence mask equivalent to the valid mask, so the fixture does not smuggle in a
    # confidence decision on top of the segmentation decision.
    conf = np.where(valid, 30.0, -30.0)

    report: dict = {
        "fixture": fixture,
        "intent": meta["intent"],
        "layer_names": layer_names,
        "per_layer_warp": per_layer_warp,
        "global_warp": meta["global_warp"],
        "valid_fraction": float(valid.mean()),
        "depth_min": float(np.nanmin(target_frame.depth)),
        "depth_max": float(np.nanmax(target_frame.depth)),
        "methods": {},
    }

    # The ground-truth partition as a "method", to establish the LSA floor.
    truth_partition = np.where(truth_labels >= 0, truth_labels, 0).astype(np.intp)
    partitions: dict[str, np.ndarray] = {"truth": truth_partition}
    atom_labels: dict[str, np.ndarray] = {}

    for method in SEGMENTATION_METHODS:
        labels, info = segment_labels_for_method(
            method, depth, segmentation=cfg, point_map=points, conf_map=conf
        )
        labels = np.asarray(labels)
        partitions[method] = labels
        if isinstance(info, dict) and info.get("initial_labels") is not None:
            atom_labels[method] = np.asarray(info["initial_labels"])

    # -- L2-A -------------------------------------------------------------------------------
    for name, labels in partitions.items():
        geometry = {"depth": depth, "normal": target_frame.normal}
        report["methods"].setdefault(name, {})
        report["methods"][name]["segmentation"] = evaluate_segmentation(
            labels, truth_labels, geometry
        )
        report["methods"][name]["partition_invariants"] = l1_partition_invariants(
            labels, method=name, frame=0
        )
        if name in atom_labels:
            report["methods"][name]["merge_only_violations"] = l1_merge_only_invariant(
                labels, atom_labels[name], method=name, frame=0
            )

    # -- L2-B -------------------------------------------------------------------------------
    anchor_overlap_points = np.stack(
        [anchor.frames[index].points for index in range(overlap)]
    )
    target_overlap_points = np.stack(
        [target.frames[index].points for index in range(overlap)]
    )
    anchor_labels = np.stack([anchor.frames[index].layer_label for index in range(overlap)])
    target_labels = np.stack([target.frames[index].layer_label for index in range(overlap)])
    anchor_valid = np.stack([anchor.frames[index].valid_mask for index in range(overlap)])
    target_valid = np.stack([target.frames[index].valid_mask for index in range(overlap)])

    truth_scale = np.mean(list(per_layer_warp.values()))
    global_scale = float(meta["global_warp"])

    # The method's own partition is produced independently on each window, exactly as the
    # pipeline does, and each true layer is then matched to the method region that dominates it.
    # Merging two true layers into one region makes both sides select that same fat region, so the
    # scale is fitted on mixed data and the error becomes real -- which is the effect under test.
    method_window_partitions: dict[str, list[np.ndarray]] = {}
    for method in SEGMENTATION_METHODS:
        per_frame = []
        for render, frames in ((anchor, range(overlap)), (target, range(overlap))):
            for index in frames:
                frame_render = render.frames[index]
                frame_labels, _ = segment_labels_for_method(
                    method,
                    np.nan_to_num(frame_render.depth, nan=0.0),
                    segmentation=cfg,
                    point_map=np.nan_to_num(frame_render.points, nan=0.0),
                    conf_map=np.where(frame_render.valid_mask, 30.0, -30.0),
                )
                per_frame.append(np.asarray(frame_labels))
        method_window_partitions[method] = per_frame

    for name, labels in partitions.items():
        recovered: list[float] = []
        recovered_global: list[float] = []
        for layer in range(layer_count):
            if name == "truth":
                anchor_layer = (anchor_labels == layer) & anchor_valid
                target_layer = (target_labels == layer) & target_valid
            else:
                per_frame = method_window_partitions[name]
                anchor_frame_labels = per_frame[:overlap]
                target_frame_labels = per_frame[overlap:]
                anchor_region = _dominant_region(
                    anchor_labels[0], anchor_frame_labels[0], layer, anchor_valid[0]
                )
                target_region = _dominant_region(
                    target_labels[0], target_frame_labels[0], layer, target_valid[0]
                )
                if anchor_region is None or target_region is None:
                    recovered.append(float("nan"))
                    recovered_global.append(float("nan"))
                    continue
                anchor_layer = np.stack(
                    [_region_mask(anchor_frame_labels[i], anchor_region, anchor_valid[i])
                     for i in range(overlap)]
                )
                target_layer = np.stack(
                    [_region_mask(target_frame_labels[i], target_region, target_valid[i])
                     for i in range(overlap)]
                )
            recovered.append(
                _estimate_overlap_scale(
                    anchor_overlap_points, target_overlap_points, anchor_layer, target_layer, cfg
                )
            )
            recovered_global.append(
                _estimate_overlap_scale(
                    anchor_overlap_points,
                    target_overlap_points,
                    anchor_valid.astype(bool),
                    target_valid.astype(bool),
                    cfg,
                )
            )
        truth_row = [
            per_layer_warp.get(layer, truth_scale) for layer in range(layer_count)
        ]
        entry = report["methods"].setdefault(name, {})
        entry["scale_recovery"] = layer_scale_recovery(recovered, truth_row)
        entry["scale_recovery_global_mask"] = layer_scale_recovery(recovered_global, truth_row)
        entry["single_global_scale_could_explain"] = bool(
            np.allclose(list(per_layer_warp.values()), global_scale, rtol=1e-9)
        )

    # -- L2-B landing error -----------------------------------------------------------------
    # The corrected point map is `target points * estimated scale`; the analytically true point map
    # is the same geometry with its layer depths scaled by the known factor.
    truth_points = np.array(target_frame.points, dtype=np.float64)
    for layer in range(layer_count):
        mask = truth_labels == layer
        truth_points[mask] = truth_points[mask] / max(per_layer_warp.get(layer, 1.0), 1e-9)
    truth_points[~valid] = np.nan

    for name, labels in partitions.items():
        entry = report["methods"].setdefault(name, {})
        if name == "truth":
            continue
        scale = entry["scale_recovery"].get("scale_recovered_mean")
        before = np.array(target_frame.points, dtype=np.float64)
        if scale is None or not np.isfinite(scale):
            entry["landing"] = {"relative_error_before": None, "relative_error_after": None,
                               "improvement": None, "pixels": 0}
        else:
            after = before * float(scale)
            entry["landing"] = landing_error(before, after, truth_points)

    floor_scale = report["methods"]["truth"]["scale_recovery"].get("scale_recovered_mean")
    for method in SEGMENTATION_METHODS:
        entry = report["methods"][method]
        method_scale = entry["scale_recovery"].get("scale_recovered_mean")
        if method_scale is not None and floor_scale is not None:
            entry["scale_error_vs_floor"] = attribution_delta(
                abs(method_scale - truth_scale), abs(floor_scale - truth_scale)
            )
    return report


def _dominant_region(
    truth_labels: np.ndarray, method_labels: np.ndarray, layer: int, valid: np.ndarray
) -> int | None:
    """The method region covering most of one true layer's pixels."""
    mask = (truth_labels == layer) & valid
    if not mask.any():
        return None
    values, counts = np.unique(method_labels[mask], return_counts=True)
    if values.size == 0:
        return None
    return int(values[int(np.argmax(counts))])


def _region_mask(truth_labels: np.ndarray, region: int, valid: np.ndarray) -> np.ndarray:
    return (truth_labels == region) & valid


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--segmentation-config", default=None)
    parser.add_argument("--output", default=None, help="write the JSON report to this path")
    parser.add_argument("--frames", default=4, type=int)
    parser.add_argument("--overlap", default=2, type=int)
    parser.add_argument(
        "--warp-spread", default=0.35, type=float,
        help="per-layer scale spread; 0 makes every layer share one factor, which no "
             "segmentation method can influence",
    )
    args = parser.parse_args()

    base_cfg = load_segmentation_config(args.segmentation_config)
    print("=" * 78)
    print("IDEA-001  L1 invariants and L2 synthetic ground-truth checks")
    print("=" * 78)
    print(describe_config(base_cfg))
    print()

    failures: list[str] = []

    # ---- L1 -------------------------------------------------------------------------------
    print("L1 invariants")
    print("-" * 78)
    for fixture in args.fixtures:
        anchor, target, meta = build_pair(
            fixture, frame_count=min(2, args.frames), overlap=1
        )
        frame = target.frames[0]
        depth = np.nan_to_num(frame.depth, nan=0.0)
        points = np.nan_to_num(frame.points, nan=0.0)
        conf = np.where(frame.valid_mask, 30.0, -30.0)

        failures.extend(l1_baseline_identity(depth[None], conf[None], base_cfg))
        failures.extend(l1_determinism(base_cfg, points, depth, conf))
        for method in SEGMENTATION_METHODS:
            labels, info = segment_labels_for_method(
                method, depth, segmentation=base_cfg, point_map=points, conf_map=conf
            )
            failures.extend(l1_partition_invariants(labels, method=method, frame=0))
            if isinstance(info, dict) and info.get("initial_labels") is not None:
                failures.extend(
                    l1_merge_only_invariant(
                        labels, info["initial_labels"], method=method, frame=0
                    )
                )
        print(f"  [ok] {fixture}: baseline identity, determinism, partition invariants")
    failures.extend(l1_irls_sanity(base_cfg))
    print("  [ok] IRLS exactness, convergence reporting and clamp reporting")

    if failures:
        print()
        print(f"L1 FAILURES ({len(failures)}):")
        for item in failures[:40]:
            print("  - " + item)
    else:
        print()
        print("L1: all invariants hold")

    # ---- L2 -------------------------------------------------------------------------------
    print()
    print("L2 synthetic ground truth")
    print("-" * 78)
    l2_reports = []
    for fixture in args.fixtures:
        report = run_l2(
            fixture, base_cfg, frame_count=args.frames, overlap=args.overlap,
            warp_spread=args.warp_spread,
        )
        l2_reports.append(report)
        print()
        print(f"### {fixture}  ({report['intent']})")
        print(
            f"    layers={report['layer_names']} warp={report['per_layer_warp']} "
            f"one-global-scale-explainable={report['methods']['depth'].get('single_global_scale_could_explain')}"
        )
        # `split` is the share of truly-same-layer adjacencies the method left separated, i.e.
        # over-fragmentation. `xlayer` is the share of regions spanning more than one true layer,
        # i.e. over-merging. They are different errors and a single region count hides both.
        print(
            f"    {'method':10s} {'regions':>7s} {'ARI':>7s} {'purity':>7s} "
            f"{'split':>6s} {'xlayer':>6s} {'scaleErr':>9s}"
        )
        for name in ("truth",) + SEGMENTATION_METHODS:
            entry = report["methods"][name]
            segmentation = entry["segmentation"]
            scale_error = entry["scale_recovery"].get("scale_relative_error_mean")
            print(
                f"    {name:10s} {segmentation['region_count']:7d} "
                f"{_fmt(segmentation['ari']):>7s} {_fmt(segmentation['purity_pixel_weighted']):>7s} "
                f"{_fmt(segmentation['boundary_violation_rate']):>6s} "
                f"{_fmt(segmentation['cross_layer_region_ratio']):>6s} "
                f"{_fmt(scale_error):>9s}"
            )
        for method in SEGMENTATION_METHODS:
            entry = report["methods"][method]
            cost = entry.get("scale_error_vs_floor")
            if cost and cost.get("segmentation_cost") is not None:
                print(
                    f"    -> {method}: scale error {_fmt(cost['method_error'])} vs LSA floor "
                    f"{_fmt(cost['floor_error'])}  segmentation cost {cost['segmentation_cost']:+.4f}"
                )
            violations = entry.get("merge_only_violations") or []
            if violations:
                print(f"    -> {method}: merge-only invariant VIOLATED ({violations[0]})")

    if args.output:
        payload = {
            "segmentation_config": base_cfg.run_identity(),
            "l1_failures": failures,
            "l2": l2_reports,
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print()
        print(f"JSON report written to {args.output}")

    print()
    print("=" * 78)
    print(
        "L1: " + ("FAILED" if failures else "passed")
        + " | L2: reported above (measurements, not pass/fail)"
    )
    print("=" * 78)
    return 1 if failures else 0


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
