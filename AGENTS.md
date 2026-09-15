# Project Purpose

LASER — *Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction* — converts
an offline feed-forward reconstruction model into a streaming 4D reconstruction system without
retraining, by aligning the predictions of consecutive overlapping temporal windows. Baseline
paper: arXiv:2512.13680 (CVPR 2026); project page https://neu-vi.github.io/LASER/. This checkout
is the upstream reference implementation (`neu-vi/LASER`, branch `main`); research here builds
*on* that baseline rather than reproducing it.

`docs/dsh/PROJECT_MODEL.md` is the detailed knowledge source (module map, data flow,
paper-to-code mapping, observations, unknowns). This file is the short durable rule set.

# Canonical Pipeline

Frames -> `demo.py` (frozen backbone `Pi3.from_pretrained("yyfz233/Pi3")`) ->
`StreamingWindowEngine` (`inference_engine/streaming_window_engine.py`) cuts the sequence into
overlapping windows (schedule from `configs/segmentation_config.yaml`; `--sample_interval` per
entry point) -> the frozen backbone predicts per-window `local_points` (H,W,3), `camera_poses`
(4,4), `conf` (H,W) -> a registration worker aligns each new window to the previous window's
globally-registered cache:

1. confidence-quantile mutual mask (`confidence_keep_ratio`, AND over the shared overlap frames);
2. global Sim(3): scale by IRLS on masked point correspondences (`align_cam_pts_irls`), then
   R,t by Kabsch on scaled camera anchors (`register_camera_poses_kabsch_pytorch`), applied with
   `apply_sim3_to_pose`;
3. optional depth refinement (LSA, enabled by `configs/segmentation_config.yaml::depth_refine`):
   build a per-frame region graph (`make_sp_graph`, dispatching on `segmentation.method`), estimate
   inter-window per-layer scales by IRLS (`align_depth_irls`), propagate them over the layer graph,
   and multiply the resulting per-pixel scale mask into `local_points`. The whole step is one
   function, `inference_utils.run_lsa_refinement`, shared by both streaming engine variants.

Windows are cached as `window_cache_<id>.pt`; `parse_inference_cache_summary` drops the overlap
frames of every non-first window, concatenates, derives world `points`
(`camera_poses ⊗ homogenize(local_points)`), and `eval/save_func.save_for_viser` writes
depth, confidence, poses and RGB into `viser_results/<scene>/`.

**Segmentation methods.** `segmentation.method` selects `depth` (Baseline: Felzenszwalb on the
depth map plus an absolute mean-depth merge), `geometry` (4-channel depth+normal image, merge gated
by depth, normal angle and confidence), or `atomic` (depth atoms and coarse layers as a prior, then
a scale-normalised 3D boundary-gap merge with an optional split). All three share everything from
`match_segmentation_seq` onward.

**One parameter source.** `configs/segmentation_config.yaml` +
`inference_engine/segmentation_config.py` fix the window schedule, confidence, `depth_refine` and
every threshold for *all* entry points, so that a comparison varies one thing. The per-entry-point
`--window_size`, `--overlap`, `--depth_refine` flags were removed. `--segmentation_method` is the
intended independent variable; the Hydra-driven `mv_recon` evaluators read
`LASER_SEGMENTATION_METHOD` instead.

Loop-closure variant: `StreamingWindowEngineLC` + `loop_closure/LoopClosureEngine` (SALAD VPR
detection plus L-M optimisation over the Sim(3) chain); entry point `demo_lc.py`.

# Repository Map

- `demo.py`, `demo_lc.py` — inference entry points for one sequence.
- `inference_engine/` — the baseline core: `vanilla_engine.py` (raw backbone wrapper),
  `inference_utils.py` (windowing, pseudo-intrinsics, adjacent-window registration,
  `run_lsa_refinement`), `utils/{lsa,depth,geometry,segmentation,batch_threading}.py`,
  `utils/*.pyx` (Cython), `sliding_window_engine.py` and `streaming_window_engine_lc.py`
  (variants).
- `inference_engine/segmentation_config.py` + `configs/segmentation_config.yaml` — the single
  parameter source for every entry point (schedule, confidence, LSA switch, all thresholds).
- `inference_engine/segmentation_diagnostics.py` — the OP-1…OP-6 observation points and their
  per-window JSON writer; inert unless a `DiagnosticsConfig` enables it.
- `inference_engine/segmentation_metrics.py`, `inference_engine/synthetic_scenes.py`,
  `scripts/check_segmentation_methods.py`, `tests/` — the analytically exact scenes and the
  CPU-only L1/L2 checks they support.
- `inference_engine/utils/{confidence,geometry_segmentation,layer_atomic_geometry,post_merge_split,segmentation_trace}.py`
  — the region-selection material behind the `geometry` and `atomic` methods.
- `pi3/` — frozen π³ backbone (`pi3/models/pi3.py`); `pi3/utils/graph.py::Vertex` is the layer
  graph node type reused by LSA.
- `vggt/` — vendored VGGT model and utilities (pose encoding, geometry, point maps).
- `loop_closure/` — VPR loop detection, Sim(3) optimisation, vendored backbones/aggregators.
- `eval/`, `eval_launch.py`, `post_eval_launch.py`, `depth_metric.py` — pose and depth evaluation.
- `mv_recon/` — multi-view point-map evaluation; `datasets/` — dataset readers plus frozen
  key-frame maps under `datasets/seq-id-maps/`.
- `configs/` — Hydra configs for point-map evaluation and `loop_config.yaml` for loop closure.
- `viser/` — visualization submodule (`viser/visualizer_monst3r.py`), not initialized here.

# Canonical Baselines

- Streaming baseline: `StreamingWindowEngine`. The schedule is locked in
  `configs/segmentation_config.yaml` (20 / 5), so every entry point runs the same one; the
  per-entry-point values that used to differ (demo 10/5, LC 75/30, outdoor 60/30) survive only as
  comments in the call sites.
- Offline baseline: `VanillaEngine`, i.e. the whole sequence through π³ at once (`--model=pi3`).
- LSA is the paper's core switch and is now a single locked field: it used to default to `True` in
  `StreamingWindowEngine`, `False` in `StreamingWindowEngineLC`, and was reachable in `demo.py`
  only through `--depth_refine`, which meant the LC path silently never ran LSA. All entry points
  now take it from the config.
- Evaluation harnesses are part of the baseline: `eval_launch.py` + `eval/*.py` (pose/depth),
  `depth_metric.py` (depth metrics), `mv_recon/eval.py` (7-Scenes/NRGBD), `mv_recon/eval_outdoor.py`.

# Research Guardrails / Invariants

- When a research variable is explicitly changed, change only that variable and preserve
  unrelated baseline behaviour, evaluation, datasets and interfaces.
- Keep the training-free contract: do not fine-tune or retrain the frozen backbone to make a
  streaming experiment work.
- Keep the window contract: `window_size`, `overlap`, and the overlap-frame trimming in
  `parse_cache_file` define the temporal semantics; a change belongs to a declared variant.
- Keep the LSA switch explicit and recorded for every run (`depth_refine` on/off plus the
  confidence- and IoU-threshold values actually used). For `depth` on synthetic fixtures the
  Baseline is pinned element-wise against an independent oracle, so a change there is a defect,
  not a tuning result.
- `confidence_keep_ratio` is the **complement** of the retained fraction: it thresholds at
  `quantile(conf, 1 - keep_ratio)`, so the shipped 0.7 keeps the top 30% and reproduces the
  historical `--top_conf_percentile 0.3`. Reading it as "keep 70%" inverts every comparison.
- A comparison varies exactly one field: `segmentation.method`. `window_size`, `overlap`,
  `depth_refine` and every threshold come from `configs/segmentation_config.yaml`, and a run must
  record them (`SegmentationConfig.run_identity()`).
- Do not alter evaluation datasets, metrics or their defaults to improve numbers; the protocol is
  part of the baseline.
- Prefer adding a variant (engine subclass, new config) over editing the canonical path, and
  reuse `register_adjacent_windows`, `align_cam_pts_irls` and the `Vertex` graph instead of
  re-implementing alignment.

# Environment Boundary

- Running the baseline needs CUDA, the Cython build (`python setup.py build_ext --inplace`),
  π³ weights (HF `yyfz233/Pi3` or `--model_ckpt`), loop-closure weights
  (`scripts/download_weights.sh`), and datasets under `data/`.
- Importing `inference_engine` requires the compiled `inference_engine/utils/*.pyx` extensions;
  the pure-Python `inference_engine/utils/segmentation.py` is not wired into that path.
- The `viser` submodule is not initialized, so the visualization entry point is unavailable until
  it is fetched.
- Do not install dependencies, build extensions, download weights or datasets, or launch GPU jobs
  as part of baseline/onboarding work; ask before touching the environment.

# Verification Principles

- Label material claims `VERIFIED` (source/config), `INFERRED` (partial trace) or `UNKNOWN`;
  a paper claim never becomes repository behaviour without source evidence.
- Verify behaviour by tracing a path end to end (entry point -> engine -> util) and cite
  repository-relative paths and symbols.
- Record observed issues as observed issue + evidence + possible impact; do not fix them
  silently inside unrelated work.
- Keep `docs/dsh/PROJECT_MODEL.md` in sync with any structural change.

# Sources of Truth

- `README.md` — installation, inference, evaluation commands, paper link, citation.
- `demo.py`, `eval_launch.py`, `mv_recon/eval.py` — executable definitions of the pipeline.
- `configs/` — evaluation and loop-closure parameters, plus `segmentation_config.yaml`, the locked
  parameter source for every entry point.
- `docs/dsh/PROJECT_MODEL.md` — consolidated baseline model, paper mapping, unknowns. It describes
  the pre-research upstream checkout; where it still documents `--window_size` / `--depth_refine`
  entry-point flags or lists `utils/lsa.py::make_sp_graph` as taking only depth arguments, the
  source and this file are authoritative.
- `docs/dsh/ideas/` — attempted changes, one file per Idea; `INDEX.md` is the navigation table.
- `scripts/check_segmentation_methods.py` — the CPU-only L1/L2 gate, runnable without GPU, weights
  or datasets.
