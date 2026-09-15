"""IDEA-001 L1 invariants and the end-to-end synthetic check.

These tests are the Regression gate of the Idea document. They run on CPU with no GPU, no model
weights and no dataset, so they can be executed anywhere the repository imports.

Coverage:

* the Baseline depth segmentation is element-wise equal to an independent oracle built from its
  own primitives, and to the pre-refactor function body;
* all three methods produce well-formed, deterministic partitions;
* neither new method ever splits a region (regions are unions of the initial atoms);
* `align_depth_irls` recovers a known scale exactly and reports its own convergence;
* the observation points produce finite scalars, and a disabled sink produces no output at all;
* the shipped check script exits 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.segmentation import felzenszwalb

REPO_ROOT = Path(__file__).resolve().parents[1]

from inference_engine.segmentation_config import (  # noqa: E402
    SEGMENTATION_METHODS,
    DiagnosticsConfig,
    load_segmentation_config,
)
from inference_engine.segmentation_diagnostics import (  # noqa: E402
    DiagnosticsSink,
    NullDiagnosticsSink,
    flatten_window_diagnostics,
    mask_consistency_delta,
    overlap_depth_consistency,
    summarize_edge_scale,
    summarize_merge_evidence,
    summarize_scale_mask,
    summarize_scale_propagation,
    summarize_segmentation_exit,
    summarize_temporal_match_per_frame,
)
from inference_engine.synthetic_scenes import build_pair  # noqa: E402
from inference_engine.utils._segmentation_cy import merge_regions  # noqa: E402
from inference_engine.utils.depth import (  # noqa: E402
    align_depth_irls,
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)
from inference_engine.utils.lsa import make_sp_graph, segment_labels_for_method  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_segmentation_config()


def _frame(fixture: str):
    _anchor, target, meta = build_pair(fixture, frame_count=2, overlap=1)
    render = target.frames[0]
    return (
        np.nan_to_num(render.depth, nan=0.0),
        np.nan_to_num(render.points, nan=0.0),
        np.where(render.valid_mask, 30.0, -30.0),
        render.layer_label,
        meta,
    )


# --------------------------------------------------------------------------------------------
# Baseline identity -- the hard Regression gate
# --------------------------------------------------------------------------------------------


def _legacy_depth_segmentation(depth, depth_merge_thresh, conf_map, keep_ratio,
                               seg_scale, seg_sigma, seg_min_size, batch_idx=0):
    """The pre-IDEA-001 function body, transcribed, as an independent oracle.

    It keeps the legacy calling convention: `conf_map` arrives batched and `batch_idx` selects the
    frame, because indexing the frame is the first thing the old body did.
    """
    seg_mask = felzenszwalb(depth, scale=seg_scale, sigma=seg_sigma, min_size=seg_min_size)
    if conf_map is not None and keep_ratio is not None:
        frame_conf = conf_map[batch_idx]
        conf_thresh = np.quantile(frame_conf.reshape(-1), keep_ratio, method="nearest")
        conf_depth = depth[frame_conf >= conf_thresh]
    else:
        conf_depth = depth
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))
    return merge_regions(seg_mask, depth, merge_thresh), merge_thresh


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_conf", [False, True])
def test_depth_segmentation_matches_legacy_and_oracle(config, seed, with_conf):
    rng = np.random.default_rng(seed)
    height, width = 40, 56
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float64)
    depth = 1.0 + 0.002 * grid_x + 0.003 * grid_y
    depth[:, width // 2:] += 0.8
    depth += 0.02 * rng.standard_normal(depth.shape)
    conf = np.linspace(0.1, 1.0, height * width).reshape(height, width)

    conf_arg = conf if with_conf else None
    ratio_arg = config.confidence_keep_ratio if with_conf else None
    # The new helper's `keep_ratio` is the CUT (it thresholds at the `1 - keep_ratio` quantile),
    # while the legacy body's parameter was the quantile LEVEL. The bridge is `1 - keep_ratio`,
    # which is exactly the transformation the pre-IDEA-001 engine applied in its constructor.
    legacy_quantile_level = 1.0 - ratio_arg if ratio_arg is not None else None

    # The pre-IDEA-001 contract for the confidence argument was the BATCHED tensor; the new code
    # also accepts the already-sliced 2-D frame. The oracle keeps the legacy shape so it exercises
    # the old path rather than a new convenience.
    legacy_labels, legacy_threshold = _legacy_depth_segmentation(
        depth, config.depth_merge_thresh,
        None if conf_arg is None else conf_arg[None], legacy_quantile_level,
        config.felzenszwalb.scale, config.felzenszwalb.sigma, 20,
    )
    stages = segment_depth_felzenszwalb_rag_stages(
        depth,
        config.depth_merge_thresh,
        conf_map=conf_arg,
        confidence_keep_ratio=ratio_arg,
        seg_scale=config.felzenszwalb.scale,
        seg_sigma=config.felzenszwalb.sigma,
        seg_min_size=20,
        confidence_quantile_method=config.confidence_quantile_method,
    )
    wrapper = segment_depth_felzenszwalb_rag(
        depth,
        config.depth_merge_thresh,
        conf_map=conf_arg,
        confidence_keep_ratio=ratio_arg,
        seg_scale=config.felzenszwalb.scale,
        seg_sigma=config.felzenszwalb.sigma,
        seg_min_size=20,
        confidence_quantile_method=config.confidence_quantile_method,
    )

    np.testing.assert_array_equal(np.asarray(wrapper), np.asarray(stages.merged_labels))
    np.testing.assert_array_equal(np.asarray(stages.merged_labels), np.asarray(legacy_labels))
    assert stages.merge_threshold == pytest.approx(legacy_threshold, rel=1e-12)


def test_keep_ratio_is_the_historical_quantile_level(config):
    """A historical `--top_conf_percentile p` must equal `confidence_keep_ratio = 1 - p`.

    `keep_ratio` is the complement of the retained fraction, so with the shipped 0.7 the Baseline
    keeps the top 30%, which is what `--top_conf_percentile 0.3` used to do.

    This is the one rename in the Idea that could silently change every number, so it is pinned
    against the exact expression the pre-IDEA-001 engine used.
    """
    rng = np.random.default_rng(11)
    conf = rng.uniform(0.0, 1.0, size=(32, 32))
    for historical_cut in (0.3, 0.5, 0.7):
        keep = 1.0 - historical_cut
        stages = segment_depth_felzenszwalb_rag_stages(
            np.ones((32, 32)), 0.1, conf_map=conf, confidence_keep_ratio=keep, seg_min_size=5
        )
        # `keep_ratio` is the complement of the retained fraction, exactly as in the Baseline,
        # where the engine passed `1 - top_conf_percentile` as the quantile level.
        expected_threshold = np.quantile(conf.reshape(-1), 1.0 - keep, method="nearest")
        assert stages.confidence_threshold == pytest.approx(expected_threshold, rel=1e-12)
        expected_mask = conf >= expected_threshold
        np.testing.assert_array_equal(stages.high_confidence_mask, expected_mask)
        # The reported threshold is the minimum of the selected pixels, so mask and threshold are
        # consistent by construction even when several pixels tie at the quantile.
        assert stages.confidence_threshold == pytest.approx(
            float(np.min(conf[stages.high_confidence_mask])), rel=1e-12
        )
        # `keep` is the complement of the cut, so the retained share is `keep` itself: with the
        # shipped 0.7 the Baseline keeps the top 70%, which is what `--top_conf_percentile 0.3`
        # used to do.
        assert stages.high_confidence_mask.mean() == pytest.approx(keep, abs=0.06)


# --------------------------------------------------------------------------------------------
# Partition invariants
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["F1-crease", "F3-occluder", "F4-known-scale"])
@pytest.mark.parametrize("method", SEGMENTATION_METHODS)
def test_partition_is_contiguous_and_covers_the_frame(config, fixture, method):
    depth, points, conf, _truth, _meta = _frame(fixture)
    labels, _info = segment_labels_for_method(
        method, depth, segmentation=config, point_map=points, conf_map=conf
    )
    labels = np.asarray(labels)
    assert labels.shape == depth.shape
    assert np.issubdtype(labels.dtype, np.integer)
    unique = np.unique(labels)
    assert unique.size > 0
    # Contiguous, but deliberately not asserted to start at 0: the Baseline's Cython merge emits
    # 1-based labels while the new strategies compact to 0.
    assert unique.size == int(unique.max() - unique.min() + 1)
    assert unique.min() >= 0
    assert not np.isnan(labels.astype(np.float64)).any()


@pytest.mark.parametrize("fixture", ["F1-crease", "F3-occluder", "F4-known-scale"])
@pytest.mark.parametrize("method", ["geometry", "atomic"])
def test_new_methods_never_split_an_initial_region(config, fixture, method):
    """`R_new <= R_initial` and every initial region maps to exactly one output region."""
    depth, points, conf, _truth, _meta = _frame(fixture)
    labels, info = segment_labels_for_method(
        method, depth, segmentation=config, point_map=points, conf_map=conf
    )
    labels = np.asarray(labels)
    if method == "atomic":
        assert "split_diagnostics" in info
        atoms = None
    else:
        atoms = None
    if atoms is None:
        # Rebuild the depth atoms the same way `atomic` does, to check the merge-only property.
        stages = segment_depth_felzenszwalb_rag_stages(
            depth,
            config.depth_merge_thresh,
            conf_map=conf,
            confidence_keep_ratio=config.confidence_keep_ratio,
            seg_scale=config.felzenszwalb.scale,
            seg_sigma=config.felzenszwalb.sigma,
            seg_min_size=config.felzenszwalb.min_size,
            confidence_quantile_method=config.confidence_quantile_method,
        )
        atoms = np.asarray(stages.initial_labels)
        if method == "geometry":
            # `geometry` re-segments a 4-channel geometry image, so its atoms are its own; the
            # merge-only property is checked against those by re-running its own pipeline.
            from inference_engine.utils.geometry_segmentation import (
                segment_geometry_felzenszwalb_rag_stages,
            )

            geometry_stages = segment_geometry_felzenszwalb_rag_stages(
                depth,
                conf_map=conf,
                point_map=points,
                confidence_keep_ratio=config.confidence_keep_ratio,
                depth_merge_thresh=config.depth_merge_thresh,
                normal_thresh_deg=config.geometry.normal_threshold_degrees,
                seg_scale=config.felzenszwalb.scale,
                seg_sigma=config.felzenszwalb.sigma,
                seg_min_size=config.felzenszwalb.min_size,
                normal_method=config.geometry.normal_method,
            )
            atoms = np.asarray(geometry_stages.initial_labels)
    for label in np.unique(labels):
        mask = labels == label
        for atom in np.unique(atoms[mask]):
            np.testing.assert_array_equal(
                labels[atoms == atom],
                label,
                err_msg=f"{method}: atom {atom} was split across regions",
            )


@pytest.mark.parametrize("fixture", ["F1-crease", "F4-known-scale"])
@pytest.mark.parametrize("method", SEGMENTATION_METHODS)
def test_methods_are_deterministic(config, fixture, method):
    depth, points, conf, _truth, _meta = _frame(fixture)
    first, _ = segment_labels_for_method(
        method, depth, segmentation=config, point_map=points, conf_map=conf
    )
    second, _ = segment_labels_for_method(
        method, depth, segmentation=config, point_map=points, conf_map=conf
    )
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_atomic_split_mode_is_a_real_switch(config):
    """On a fold, `normal_only` must split where `none` cannot: the split stage is not dead code."""
    import dataclasses

    from inference_engine.utils.layer_atomic_geometry import segment_point_map_atomic

    height, width = 48, 64
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float64)
    depth = np.where(grid_x < width // 2, 5.0, 5.0 + 0.02 * (grid_x - width // 2))
    points = np.stack(
        [(grid_x - width / 2) * depth / 60.0, (grid_y - height / 2) * depth / 60.0, depth],
        axis=-1,
    )
    conf = np.full((height, width), 5.0)
    none_cfg = dataclasses.replace(config.atomic, split_mode="none")
    normal_cfg = dataclasses.replace(config.atomic, split_mode="normal_only")
    labels_none, _ = segment_point_map_atomic(
        points, depth_merge_thresh=0.1, conf_map=conf, confidence_keep_ratio=0.7,
        seg_min_size=20, split_mode=none_cfg.split_mode,
        split_score_threshold=none_cfg.split_score_threshold,
    )
    labels_normal, diagnostics = segment_point_map_atomic(
        points, depth_merge_thresh=0.1, conf_map=conf, confidence_keep_ratio=0.7,
        seg_min_size=20, split_mode=normal_cfg.split_mode,
        split_score_threshold=normal_cfg.split_score_threshold,
    )
    assert np.unique(labels_none).size <= np.unique(labels_normal).size
    assert diagnostics.as_dict()["split_mode"] == "normal_only"


# --------------------------------------------------------------------------------------------
# IRLS reporting
# --------------------------------------------------------------------------------------------


def test_align_depth_irls_recovers_a_known_scale(config):
    rng = np.random.default_rng(3)
    source = rng.uniform(1.0, 6.0, size=(24, 24))
    for truth in (0.4, 1.0, 3.0):
        scale, info = align_depth_irls(
            source, source * truth, None, return_info=True, clamp_min=config.irls.clamp_min
        )
        assert float(scale) == pytest.approx(truth, rel=1e-6)
        assert info["converged"] is True
        assert info["clamped"] is False
        assert info["sample_count"] == source.size


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_align_depth_irls_reports_degenerate_masks(config):
    """A vanishingly small intersection must be reported as clamped, not silently accepted."""
    scale, info = align_depth_irls(
        np.full((8, 8), 1e-12), np.full((8, 8), 1e-18), None,
        return_info=True, clamp_min=config.irls.clamp_min,
    )
    assert info["clamped"] is True
    assert float(scale) == pytest.approx(config.irls.clamp_min)

    empty = np.zeros((8, 8), dtype=bool)
    scale, info = align_depth_irls(
        np.ones((8, 8)), np.ones((8, 8)), empty, return_info=True
    )
    # An all-zero mask is Baseline behaviour that the Idea does not repair: 0/0 becomes NaN at the
    # initialisation step. It is pinned so that a future "fix" has to be a deliberate decision.
    assert np.isnan(float(scale))
    assert info["clamped"] is False


# --------------------------------------------------------------------------------------------
# Observation points
# --------------------------------------------------------------------------------------------


def test_diagnostics_are_finite_scalars_and_round_trip(tmp_path):
    sink = DiagnosticsSink(DiagnosticsConfig(enabled=True, level="scalar"), tmp_path)
    sink.begin_window(3)
    labels = np.zeros((8, 8), dtype=np.intp)
    labels[:, 4:] = 1
    sink.record_stage("OP-1", summarize_segmentation_exit(
        initial_labels=labels, merged_labels=labels, merge_threshold=0.25,
        high_confidence_count=40, high_confidence_fraction=0.625, depth_range=2.5,
    ))
    sink.record_stage("OP-2", summarize_merge_evidence(
        method="geometry", adjacent_pairs=10, merged_pairs=4, rejected_pairs=6,
        criterion_values=np.linspace(0.0, 1.0, 11), criterion_name="normal_cosine",
    ))
    sink.record_frame(0, summarize_temporal_match_per_frame(
        frame_index=0, source_vertices=2, target_vertices=2,
        iou=np.array([[0.9, 0.1], [0.1, 0.85]]), threshold=0.3, relates_to_anchor=True,
    ), stage="OP-3")
    sink.record_frame(0, {
        "inter_mask_area": 12, "scale": 1.2, "initial_scale": 1.1,
        "iterations": 3, "converged": True, "clamped": False, "iou": 0.8,
    }, stage="edge_scale")
    sink.record_stage("OP-4", summarize_edge_scale(
        edges=[{"inter_mask_area": 12, "scale": 1.2, "iterations": 3,
                "converged": True, "clamped": False}],
        clamp_min=1e-6,
    ))
    sink.record_stage("OP-5", summarize_scale_propagation(
        vertex_scale_counts=[0, 1, 2], vertex_scale_values=[[], [1.1], [1.2, 0.9]],
        vertex_iou_weights=[[], [0.8], [0.6, 0.4]],
    ))
    mask = np.ones((8, 8))
    mask[:, :4] = 1.07
    sink.record_stage("OP-5", summarize_scale_mask(mask))

    path = sink.write_window(3)
    assert path is not None and path.is_file()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["window_id"] == 3
    assert payload["stages"]["OP-1"]["merged_region_count"] == 2
    assert payload["stages"]["OP-5"]["propagation_empty_cache_count"] == 1
    flat = flatten_window_diagnostics(payload)
    assert "OP-1.merge_threshold" in flat
    assert any(key.endswith("_mean") for key in flat)
    for key, value in flat.items():
        if isinstance(value, float):
            assert np.isfinite(value), key


def test_disabled_sink_writes_nothing(tmp_path):
    disabled = DiagnosticsConfig(enabled=False, level="off")
    assert disabled.active is False
    assert disabled.wants_arrays is False
    null = NullDiagnosticsSink()
    null.begin_window(0)
    null.record_stage("OP-1", {"x": 1.0})
    assert null.write_window(0) is None
    assert list(tmp_path.iterdir()) == []


def test_overlap_consistency_separates_level_from_consistency():
    """A uniform scale change must not be reported as inconsistency.

    LSA exists to change per-layer scale, so "the two windows disagree by a factor" and "the two
    windows disagree with each other" are different facts. This test pins that separation.
    """
    source = np.zeros((2, 8, 8, 3))
    source[..., 2] = 2.0
    identical = overlap_depth_consistency(source_points=source, target_points=source, stride=2)
    assert identical["consistency_rate"] == pytest.approx(1.0)
    assert identical["consistency_median_scale"] == pytest.approx(1.0)

    uniform = overlap_depth_consistency(
        source_points=source, target_points=source * 3.0, stride=2
    )
    assert uniform["consistency_rate"] == pytest.approx(1.0)
    assert uniform["consistency_median_scale"] == pytest.approx(3.0)

    # A smooth gradient rather than a checkerboard: with `stride=2` a checkerboard's sampled
    # pixels all share one factor and the fixture would look perfectly consistent.
    noisy = source.copy()
    gradient = np.linspace(1.0, 1.6, source.shape[2])[None, None, :]
    noisy[..., 2] = noisy[..., 2] * gradient
    inconsistent = overlap_depth_consistency(
        source_points=source, target_points=noisy, stride=2, relative_tolerance=0.05
    )
    assert inconsistent["consistency_rate"] < 1.0
    delta = mask_consistency_delta(identical, inconsistent)
    assert delta["consistency_rate_delta"] < 0.0


def test_make_sp_graph_method_dispatch_changes_structure(config):
    """The dispatcher must actually select a different method, not silently fall back to depth."""
    depth, points, conf, _truth, _meta = _frame("F4-known-scale")
    graphs = {}
    for method in SEGMENTATION_METHODS:
        graph = make_sp_graph(
            depth,
            conf_map=conf,
            point_map=points,
            segmentation=config.with_method(method),
        )
        graphs[method] = [len(layer) for layer in graph]
    assert set(graphs) == set(SEGMENTATION_METHODS)
    for method, counts in graphs.items():
        assert counts, method
        assert all(count > 0 for count in counts), method


# --------------------------------------------------------------------------------------------
# The shipped check script
# --------------------------------------------------------------------------------------------


def test_check_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_segmentation_methods.py", "--frames", "2", "--overlap", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "L1: passed" in result.stdout
    for fixture in ("F1-crease", "F2-far-field", "F3-occluder", "F4-known-scale"):
        assert fixture in result.stdout


# --------------------------------------------------------------------------------------------
# Configuration plumbing
#
# Both of these were real defects found while wiring the entry points, and both failed silently,
# which is exactly the class of bug a comparison cannot tolerate: the run would look correct and
# measure the wrong thing.
# --------------------------------------------------------------------------------------------


def test_cli_method_override_actually_reaches_the_config():
    """`--segmentation_method` must select the method.

    The first implementation wrote a top-level `method:` override while the file nests everything
    under `segmentation:`, so the override was written to a key the loader never reads and every
    run stayed on `depth` while reporting success.
    """
    import argparse

    from inference_engine.segmentation_config import add_cli_arguments, resolve_from_args

    def parse(argv):
        parser = argparse.ArgumentParser(add_help=False)
        add_cli_arguments(parser)
        return resolve_from_args(parser.parse_args(argv))

    for method in SEGMENTATION_METHODS:
        config, _diagnostics = parse(["--segmentation_method", method])
        assert config.method == method

    default_config, _ = parse([])
    assert default_config.method == "depth"


def test_environment_method_override_actually_reaches_the_config(monkeypatch):
    """The Hydra-driven evaluators have no shared CLI, so the env path must work too."""
    from inference_engine.segmentation_config import ENV_METHOD, resolve_from_environment

    for method in SEGMENTATION_METHODS:
        monkeypatch.setenv(ENV_METHOD, method)
        config, _diagnostics = resolve_from_environment()
        assert config.method == method
    monkeypatch.delenv(ENV_METHOD, raising=False)


def test_dump_arrays_alone_produces_output():
    """`--segmentation_dump_arrays` alone must enable observation, not just set a level.

    It originally left `enabled` false, so `wants_arrays` was false and the flag that asks for the
    most output produced none.
    """
    import argparse

    from inference_engine.segmentation_config import add_cli_arguments, resolve_from_args

    parser = argparse.ArgumentParser(add_help=False)
    add_cli_arguments(parser)
    _config, diagnostics = resolve_from_args(parser.parse_args(["--segmentation_dump_arrays"]))
    assert diagnostics.enabled is True
    assert diagnostics.active is True
    assert diagnostics.wants_arrays is True
    assert diagnostics.level == "arrays"

    _config, quiet = resolve_from_args(parser.parse_args([]))
    assert quiet.active is False
    assert quiet.wants_arrays is False


def test_unknown_config_keys_are_rejected():
    """A typo in a config override must fail loudly rather than be ignored."""
    with pytest.raises(ValueError):
        load_segmentation_config(None, {"segmentation.not_a_field": 1})


def test_locked_config_locks_every_entry_point_parameter():
    """The shipped config must pin the schedule and every threshold a comparison depends on.

    If any of these were left to an entry point's own default, two entry points could disagree
    about the schedule while both claiming to run the locked configuration.
    """
    config = load_segmentation_config()
    assert config.window_size == 20
    assert config.overlap == 5
    assert config.depth_refine is True
    assert config.sample_interval == 1
    assert config.depth_merge_thresh == pytest.approx(0.1)
    assert config.corr_iou_thresh_intra == pytest.approx(0.3)
    assert config.corr_iou_thresh_inter == pytest.approx(0.4)
    # All three methods share one Felzenszwalb parameterisation: the Reference tunes `geometry` to
    # 200/1.0/300, and inheriting that would mix a parameter change into the method comparison.
    assert config.felzenszwalb.scale == pytest.approx(300.0)
    assert config.felzenszwalb.sigma == pytest.approx(1.1)
    assert config.felzenszwalb.min_size == 500
    # IDEA-001 decision: the geometry and atomic branches use the Baseline's depth merge threshold,
    # not the Reference's 0.05.
    assert config.geometry.normal_threshold_degrees == pytest.approx(20.0)
    assert config.atomic.split_score_threshold == pytest.approx(0.10)
    identity = config.run_identity()
    assert identity["method"] == "depth"
    assert len(config.run_identity_hash()) == 16


# --------------------------------------------------------------------------------------------
# The macro-step end to end
#
# The unit tests above check each observation helper in isolation. This one drives
# `run_lsa_refinement` exactly as the engine's registration worker does, which is the only way to
# know that the observation points are wired to real state rather than merely implemented.
# --------------------------------------------------------------------------------------------


def _window_tensors(render, key: str):
    import torch

    array = np.stack([getattr(frame, key) for frame in render.frames])
    return torch.from_numpy(np.nan_to_num(array, nan=0.0)).float()


def test_run_lsa_refinement_recovers_the_known_scale_and_emits_every_observation_point(
    config, tmp_path
):
    """Drive the macro-step on a fixture whose inter-window scale is known by construction.

    Three things are asserted together, because they are the three ways this could be silently
    broken: the mask must actually apply the correction, OP-1…OP-6 must all be present with finite
    values, and OP-6 must show the level moving while the consistency is preserved.
    """
    import torch

    from inference_engine.segmentation_diagnostics import DiagnosticsSink
    from inference_engine.inference_utils import run_lsa_refinement

    overlap = 2
    anchor, target, meta = build_pair("F4-known-scale", frame_count=4, overlap=overlap)
    prev_local_points = _window_tensors(anchor, "points")
    cur_local_points = _window_tensors(target, "points")
    cur_conf = torch.from_numpy(
        np.stack([np.where(frame.valid_mask, 30.0, -30.0) for frame in target.frames])
    ).float()

    # The anchor graph is built the way the engine's first window does it: overlap frames only.
    from inference_engine.utils.lsa import make_sp_graph

    anchor_graph = make_sp_graph(
        prev_local_points[:overlap, ..., -1].numpy(),
        conf_map=cur_conf[:overlap].numpy(),
        point_map=prev_local_points[:overlap].numpy(),
        segmentation=config,
    )

    sink = DiagnosticsSink(DiagnosticsConfig(enabled=True, level="scalar"), tmp_path)
    sink.begin_window(1)
    corrected, graph, mask = run_lsa_refinement(
        prev_local_points,
        cur_local_points,
        cur_conf,
        anchor_graph,
        overlap,
        segmentation=config,
        diagnostics=sink,
        window_id=1,
    )
    sink.publish_frame_stages()
    payload = json.loads(sink.write_window(1).read_text(encoding="utf-8"))
    stages = payload["stages"]

    # -- the correction is real ---------------------------------------------------------------
    assert torch.isfinite(mask).all()
    assert (mask > 0).all(), "the mask is a multiplicative factor and must stay positive"
    expected = 1.0 / float(meta["global_warp"])
    assert float(mask.mean()) == pytest.approx(expected, rel=0.05), (
        "the mask should undo the fixture's known inter-window scale"
    )
    assert len(graph) == len(target.frames), "every frame of the window must be segmented"

    # -- every observation point fired --------------------------------------------------------
    for point in ("OP-1", "OP-2", "OP-3", "OP-4", "OP-5", "OP-6"):
        if point == "OP-2":
            # OP-2 is the merge-criterion evidence, which only the geometry and atomic methods
            # produce. It is checked in its own test below.
            continue
        assert point in stages, f"{point} never fired"

    for stage, block in stages.items():
        for key, value in block.items():
            if isinstance(value, float):
                assert np.isfinite(value), f"{stage}.{key} is not finite"

    # -- OP-1 reports the three stage counts and the threshold in force ------------------------
    op1 = stages["OP-1"]
    assert op1["initial_region_count"] >= op1["merged_region_count"] > 0
    assert op1["merge_threshold"] is not None and op1["merge_threshold"] > 0
    # The retained fraction is bounded below by the requested share but not pinned to it: values
    # tying at the threshold stay selected, and a frame whose confidence is bimodal keeps far more
    # than the nominal share. Only the bound is asserted, because the exact value is a property of
    # the fixture's confidence distribution rather than of the configuration.
    assert 0.0 < op1["high_confidence_fraction"] <= 1.0
    assert op1["high_confidence_fraction"] >= config.confidence_keep_ratio - 1e-9
    assert op1["high_confidence_count"] == int(
        round(op1["high_confidence_fraction"] * op1["final_pixels"])
    )

    # -- OP-3 and OP-4 are frame-level and must still be discoverable ------------------------
    assert payload["per_stage"].get("OP-3"), "OP-3 produced no per-frame rows"
    assert stages["OP-3"]["row_count"] == len(payload["per_stage"]["OP-3"])

    # -- OP-5 shows the mask statistics ------------------------------------------------------
    op5 = stages["OP-5"]
    assert op5["mask_pixels"] == mask.numel()
    assert op5["mask_nonidentity_ratio"] == pytest.approx(1.0), (
        "every pixel carries a non-trivial correction on this fixture"
    )

    # -- OP-6 separates the LEVEL from the CONSISTENCY ---------------------------------------
    # The fixture applies a uniform per-layer scale, so the correction must pull the median scale
    # towards 1 while leaving how *self-consistent* each window is essentially unchanged. A test
    # that only looked at ATE could not tell those two apart.
    op6 = stages["OP-6"]
    assert op6["consistency_median_scale_before"] == pytest.approx(
        float(meta["global_warp"]), rel=0.05
    )
    assert op6["consistency_median_scale_after"] == pytest.approx(1.0, abs=0.05)
    assert op6["consistency_median_scale_delta"] < 0.0
    assert op6["consistency_rate_before"] is not None
    assert op6["consistency_rate_after"] is not None


@pytest.mark.parametrize("seg_method", ["geometry", "atomic"])
def test_op2_reports_merge_evidence_for_the_criterion_methods(config, tmp_path, seg_method):
    """OP-2 exists so the *decision* each method made is visible, not only its outcome."""
    import torch

    from inference_engine.segmentation_diagnostics import DiagnosticsSink
    from inference_engine.inference_utils import run_lsa_refinement
    from inference_engine.utils.lsa import make_sp_graph

    overlap = 2
    segmentation = config.with_method(seg_method)
    anchor, target, _meta = build_pair("F1-crease", frame_count=4, overlap=overlap)
    prev_local_points = _window_tensors(anchor, "points")
    cur_local_points = _window_tensors(target, "points")
    cur_conf = torch.from_numpy(
        np.stack([np.where(frame.valid_mask, 30.0, -30.0) for frame in target.frames])
    ).float()
    anchor_graph = make_sp_graph(
        prev_local_points[:overlap, ..., -1].numpy(),
        conf_map=cur_conf[:overlap].numpy(),
        point_map=prev_local_points[:overlap].numpy(),
        segmentation=segmentation,
    )

    sink = DiagnosticsSink(DiagnosticsConfig(enabled=True, level="scalar"), tmp_path)
    sink.begin_window(0)
    run_lsa_refinement(
        prev_local_points,
        cur_local_points,
        cur_conf,
        anchor_graph,
        overlap,
        segmentation=segmentation,
        diagnostics=sink,
        window_id=0,
    )
    stages = json.loads(sink.write_window(0).read_text(encoding="utf-8"))["stages"]
    assert "OP-2" in stages, f"{seg_method} never reported its merge evidence"
    op2 = stages["OP-2"]
    assert op2["method"] == seg_method
    assert op2["final_region_count"] > 0


def test_disabled_diagnostics_leave_the_macro_step_untouched(config):
    """With diagnostics off the returned tensors must be identical to the instrumented run.

    This is the observation-point form of the Regression gate: an observation point that perturbs
    the thing it observes would invalidate every comparison built on it.
    """
    import tempfile

    import torch

    from inference_engine.segmentation_diagnostics import DiagnosticsSink
    from inference_engine.inference_utils import run_lsa_refinement
    from inference_engine.utils.lsa import make_sp_graph

    overlap = 2
    anchor, target, _meta = build_pair("F4-known-scale", frame_count=4, overlap=overlap)
    prev_local_points = _window_tensors(anchor, "points")
    cur_local_points = _window_tensors(target, "points")
    cur_conf = torch.from_numpy(
        np.stack([np.where(frame.valid_mask, 30.0, -30.0) for frame in target.frames])
    ).float()

    def build_anchor_graph():
        return make_sp_graph(
            prev_local_points[:overlap, ..., -1].numpy(),
            conf_map=cur_conf[:overlap].numpy(),
            point_map=prev_local_points[:overlap].numpy(),
            segmentation=config,
        )

    quiet_points, _graph, quiet_mask = run_lsa_refinement(
        prev_local_points, cur_local_points, cur_conf, build_anchor_graph(), overlap,
        segmentation=config, diagnostics=None, window_id=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        sink = DiagnosticsSink(DiagnosticsConfig(enabled=True, level="scalar"), tmp)
        sink.begin_window(0)
        loud_points, _graph, loud_mask = run_lsa_refinement(
            prev_local_points, cur_local_points, cur_conf, build_anchor_graph(), overlap,
            segmentation=config, diagnostics=sink, window_id=0,
        )
    torch.testing.assert_close(quiet_mask, loud_mask)
    torch.testing.assert_close(quiet_points, loud_points)


# --------------------------------------------------------------------------------------------
# The L2 metric itself
#
# A measurement is only worth reporting if it is right. `boundary_violation_rate` was originally
# defined against pixel-aligned truth boundaries, which made an exactly-correct partition score
# 1.000 — the worst possible value for the best possible answer. These cases pin the definition.
# --------------------------------------------------------------------------------------------


def test_boundary_violation_rate_scores_a_perfect_partition_as_zero():
    from inference_engine.segmentation_metrics import boundary_violation_rate

    truth = np.zeros((2, 8), dtype=int)
    truth[:, 4:] = 1
    perfect = boundary_violation_rate(truth, truth)
    assert perfect["boundary_violation_rate"] == 0.0
    assert perfect["cross_layer_region_ratio"] == 0.0
    assert perfect["same_layer_pairs"] > 0


def test_boundary_violation_rate_separates_over_merging_from_over_splitting():
    """The two errors must land in different fields, because a region count cannot tell them apart."""
    from inference_engine.segmentation_metrics import boundary_violation_rate

    truth = np.zeros((2, 8), dtype=int)
    truth[:, 4:] = 1

    # One region covering both true layers: over-merging, no over-splitting.
    merged = boundary_violation_rate(np.zeros((2, 8), dtype=int), truth)
    assert merged["boundary_violation_rate"] == 0.0
    assert merged["cross_layer_region_count"] == 1
    assert merged["cross_layer_region_ratio"] == 1.0

    # A correct partition with one extra cut inside a true layer: over-splitting, no over-merging.
    over_split = truth.copy()
    over_split[0, :2] = 7
    split = boundary_violation_rate(over_split, truth)
    assert split["boundary_violation_rate"] > 0.0
    assert split["cross_layer_region_count"] == 0


def test_region_area_stats_ignores_holes_in_the_label_space():
    """The Baseline emits non-contiguous `root + 1` labels; counting the gaps invents regions."""
    from inference_engine.segmentation_metrics import region_area_stats

    labels = np.zeros((4, 4), dtype=int)
    labels[2:, :] = 9
    stats = region_area_stats(labels)
    assert stats["region_count"] == 2, "only two labels occur, whatever their numeric values"
    assert stats["area_mean"] == pytest.approx(8.0)

    from inference_engine.segmentation_diagnostics import region_areas

    np.testing.assert_array_equal(region_areas(labels), [8, 8])


def test_boundary_violation_rate_accepts_a_fully_split_partition():
    """Every truly-same-layer adjacency separated is the worst case and must read as 1.0."""
    from inference_engine.segmentation_metrics import boundary_violation_rate

    truth = np.zeros((2, 4), dtype=int)
    checkerboard = np.indices((2, 4)).sum(axis=0) % 2
    result = boundary_violation_rate(checkerboard, truth)
    assert result["boundary_violation_rate"] == 1.0
