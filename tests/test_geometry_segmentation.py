"""Behaviour specification for the ported geometry-aware segmentation module.

Ported from the reference suite ``test_geometry_segmentation.py``. Only the import
lines were rewritten for this repository:

* the reference imports the confidence helper from
  ``inference_engine.segmentation.confidence``; here it lives in
  ``inference_engine.utils.confidence`` (this repository has no
  ``inference_engine/segmentation`` package).

Every reference test in this file is kept: none of them depends on ``pipeline.*``
or ``inference_engine.segmentation.*``. The two ``select_numpy_top_confidence_mask``
tests at the bottom are ported from the reference ``test_segmentation_strategies.py``
(that file also exercises ``pipeline.config``/``inference_engine.segmentation`` and is
therefore not portable), so the confidence helper shipped here is still covered.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from inference_engine.utils import depth as depth_module
from inference_engine.utils.confidence import select_numpy_top_confidence_mask
from inference_engine.utils.geometry import build_geometry_info_np
from inference_engine.utils import geometry_segmentation as geom_seg
from inference_engine.utils.geometry_segmentation import merge_regions_geometry


def test_depth_module_does_not_export_geometry_segmentation_entrypoints():
    assert not hasattr(depth_module, "compute_region_geometry_descriptors")
    assert not hasattr(depth_module, "should_merge_geometry")
    assert not hasattr(depth_module, "merge_regions_geometry")
    assert not hasattr(depth_module, "segment_geometry_felzenszwalb_rag")


def test_merge_regions_geometry_merges_adjacent_regions_with_similar_depth_and_normals():
    labels = np.array(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ],
        dtype=np.int32,
    )
    depth = np.full(labels.shape, 2.0, dtype=np.float32)
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    geometry_info = {
        "normal": normals,
        "valid_mask": np.ones(labels.shape, dtype=bool),
    }

    merged = merge_regions_geometry(
        labels,
        depth,
        geometry_info,
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    assert merged.shape == labels.shape
    assert np.unique(merged).tolist() == [0]


def test_merge_regions_geometry_keeps_depth_discontinuity_separate_and_relabels():
    labels = np.array(
        [
            [3, 3, 9, 9],
            [3, 3, 9, 9],
            [3, 3, 9, 9],
            [3, 3, 9, 9],
        ],
        dtype=np.int32,
    )
    depth = np.array(
        [
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    geometry_info = {
        "normal": normals,
        "valid_mask": np.ones(labels.shape, dtype=bool),
    }

    merged = merge_regions_geometry(
        labels,
        depth,
        geometry_info,
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    assert np.unique(merged).tolist() == [0, 1]
    assert np.all(merged[:, :2] == 0)
    assert np.all(merged[:, 2:] == 1)


def test_build_geometry_info_supports_sobel_normals():
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 2.0],
            [0.0, 100.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    geometry_info = build_geometry_info_np(
        depth=depth,
        intrinsic=intrinsic,
        normal_method="sobel",
    )

    assert geometry_info["normal"].shape == (5, 5, 3)
    assert np.isfinite(geometry_info["normal"]).all()
    assert geometry_info["normal_edge"].shape == (5, 5)


def test_geometry_segmentation_selects_batched_auxiliary_inputs(monkeypatch):
    calls = {}
    labels = np.array([[0, 1], [0, 1]], dtype=np.int32)

    def fake_build_geometry_info_np(depth, conf=None, intrinsic=None, points=None, normal_method="cross"):
        calls["depth"] = depth
        calls["conf"] = conf
        calls["intrinsic"] = intrinsic
        calls["points"] = points
        calls["normal_method"] = normal_method
        normals = np.zeros((*depth.shape, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        return {"normal": normals}

    def fake_felzenszwalb(image, **kwargs):
        calls["felzenszwalb_image_shape"] = image.shape
        return labels

    def fake_merge_regions_geometry(labels_arg, depth, geometry_info, conf=None, **kwargs):
        calls["merge_conf"] = conf
        return labels_arg

    monkeypatch.setattr(geom_seg, "build_geometry_info_np", fake_build_geometry_info_np)
    monkeypatch.setattr(geom_seg, "felzenszwalb", fake_felzenszwalb)
    monkeypatch.setattr(geom_seg, "merge_regions_geometry", fake_merge_regions_geometry)

    depth = np.ones((2, 2), dtype=np.float32)
    conf = np.stack(
        [
            np.full((2, 2), 0.1, dtype=np.float32),
            np.full((2, 2), 0.9, dtype=np.float32),
        ]
    )
    point_map = np.stack(
        [
            np.zeros((2, 2, 3), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32),
        ]
    )
    intrinsic = np.stack([np.eye(3), np.eye(3) * 2], axis=0).astype(np.float32)

    result = geom_seg.segment_geometry_felzenszwalb_rag(
        depth,
        conf_map=conf,
        point_map=point_map,
        intrinsic=intrinsic,
        confidence_keep_ratio=0.5,
        normal_method="sobel",
        batch_idx=1,
    )

    np.testing.assert_array_equal(result, labels)
    np.testing.assert_array_equal(calls["conf"], conf[1])
    np.testing.assert_array_equal(calls["merge_conf"], conf[1])
    np.testing.assert_array_equal(calls["points"], point_map[1])
    np.testing.assert_array_equal(calls["intrinsic"], intrinsic[1])
    assert calls["normal_method"] == "sobel"
    assert calls["felzenszwalb_image_shape"] == (2, 2, 4)


def test_geometry_baseline_params_stages_delegates_with_baseline_parameters(monkeypatch):
    calls = {}
    stages = SimpleNamespace(merged_labels=np.array([[0, 1]], dtype=np.intp))

    def fake_stages(depth_map, **kwargs):
        calls["depth_map"] = depth_map
        calls.update(kwargs)
        return stages

    monkeypatch.setattr(
        geom_seg,
        "segment_geometry_felzenszwalb_rag_stages",
        fake_stages,
    )

    depth = np.ones((2, 2), dtype=np.float32)
    conf = np.full((2, 2), 0.8, dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    point_map = np.zeros((2, 2, 3), dtype=np.float32)

    result = geom_seg.segment_geometry_felzenszwalb_rag_baseline_params_stages(
        depth,
        conf_map=conf,
        intrinsic=intrinsic,
        point_map=point_map,
        confidence_keep_ratio=0.4,
        depth_merge_thresh=0.2,
        normal_thresh_deg=15.0,
        normal_method="sobel",
        batch_idx=1,
    )

    assert result is stages
    assert calls["depth_map"] is depth
    assert calls["conf_map"] is conf
    assert calls["intrinsic"] is intrinsic
    assert calls["point_map"] is point_map
    assert calls["confidence_keep_ratio"] == 0.4
    assert calls["depth_merge_thresh"] == 0.2
    assert calls["normal_thresh_deg"] == 15.0
    assert calls["seg_scale"] == 300
    assert calls["seg_sigma"] == 1.1
    assert calls["seg_min_size"] == 500
    assert calls["normal_method"] == "sobel"
    assert calls["batch_idx"] == 1


def test_geometry_baseline_params_labels_returns_delegated_merged_labels(monkeypatch):
    calls = {}
    merged_labels = np.array([[1, 0]], dtype=np.intp)
    stages = SimpleNamespace(merged_labels=merged_labels)

    def fake_stages(depth_map, **kwargs):
        calls["depth_map"] = depth_map
        calls.update(kwargs)
        return stages

    monkeypatch.setattr(
        geom_seg,
        "segment_geometry_felzenszwalb_rag_stages",
        fake_stages,
    )

    depth = np.ones((2, 2), dtype=np.float32)
    result = geom_seg.segment_geometry_felzenszwalb_rag_baseline_params(
        depth,
        normal_method="sobel",
    )

    assert result is merged_labels
    assert calls["depth_map"] is depth
    assert calls["seg_scale"] == 300
    assert calls["seg_sigma"] == 1.1
    assert calls["seg_min_size"] == 500
    assert calls["normal_method"] == "sobel"


def test_geometry_baseline_params_wrappers_forward_parameter_overrides(monkeypatch):
    calls = []
    merged_labels = np.array([[1, 0]], dtype=np.intp)
    stages = SimpleNamespace(merged_labels=merged_labels)

    def fake_stages(depth_map, **kwargs):
        calls.append((depth_map, kwargs))
        return stages

    monkeypatch.setattr(
        geom_seg,
        "segment_geometry_felzenszwalb_rag_stages",
        fake_stages,
    )

    depth = np.ones((2, 2), dtype=np.float32)
    overrides = {
        "seg_scale": 120,
        "seg_sigma": 0.7,
        "seg_min_size": 80,
    }

    stages_result = geom_seg.segment_geometry_felzenszwalb_rag_baseline_params_stages(
        depth,
        **overrides,
    )
    labels_result = geom_seg.segment_geometry_felzenszwalb_rag_baseline_params(
        depth,
        **overrides,
    )

    assert stages_result is stages
    assert labels_result is merged_labels
    assert len(calls) == 2
    for delegated_depth, delegated_kwargs in calls:
        assert delegated_depth is depth
        assert delegated_kwargs["seg_scale"] == 120
        assert delegated_kwargs["seg_sigma"] == 0.7
        assert delegated_kwargs["seg_min_size"] == 80


# --- ported from the reference test_segmentation_strategies.py ------------------


def test_numpy_confidence_keep_ratio_is_the_complement_of_the_retained_fraction():
    """`keep_ratio` is the CUT, so 0.3 keeps the top 70%.

    This is the Reference implementation's convention and the Baseline's: the streaming engine has
    always passed `1 - top_conf_percentile` as the quantile level, so a historical
    `--top_conf_percentile 0.3` is `keep_ratio = 0.7`. See `inference_engine/utils/confidence.py`.
    """
    confidence = np.arange(10, dtype=np.float32)
    # keep_ratio=0.3 cuts at the 0.7 quantile (= 6.0), so everything from 6 upward is retained.
    mask = select_numpy_top_confidence_mask(confidence, 0.3)
    assert set(confidence[mask].tolist()) == {6.0, 7.0, 8.0, 9.0}
    # keep_ratio=0.7 cuts at the 0.3 quantile (= 2.0), retaining the top 70%.
    mask = select_numpy_top_confidence_mask(confidence, 0.7)
    assert set(confidence[mask].tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}


def test_numpy_confidence_ignores_nonfinite_values():
    """Non-finite entries never enter the quantile and never pass the mask."""
    confidence = np.array([0.0, 1.0, np.nan, np.inf], dtype=np.float32)
    # `method` is stated explicitly: the Baseline's rule is "nearest" and the Reference's is
    # "higher", and at a 50% cut on two finite values they straddle the boundary. This test is
    # about non-finite handling, not about the interpolation rule, so it must not depend on the
    # default.
    mask = select_numpy_top_confidence_mask(confidence, 0.5, method="higher")
    np.testing.assert_array_equal(mask, [False, True, False, False])
    mask = select_numpy_top_confidence_mask(confidence, 0.5, method="nearest")
    np.testing.assert_array_equal(mask, [True, True, False, False])
    # A frame with a single finite value still resolves rather than raising, because the finite
    # subset is non-empty.
    sparse = np.array([np.nan, 4.0, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(
        select_numpy_top_confidence_mask(sparse, 0.9), [False, True, False]
    )


def test_numpy_confidence_rejects_invalid_method_and_keep_ratio():
    confidence = np.arange(4, dtype=np.float32)
    with pytest.raises(ValueError):
        select_numpy_top_confidence_mask(confidence, 0.5, method="linear")
    with pytest.raises(ValueError):
        select_numpy_top_confidence_mask(confidence, 0.0)
    with pytest.raises(ValueError):
        select_numpy_top_confidence_mask(confidence, 1.5)
    with pytest.raises(ValueError):
        select_numpy_top_confidence_mask(
            np.full((2, 2), np.nan, dtype=np.float32),
            0.5,
        )
