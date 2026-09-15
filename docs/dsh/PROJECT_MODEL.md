# Baseline Overview

**Problem.** Feed-forward reconstruction models (VGGT, π³) reconstruct well but cannot process
streaming video: attention over all frames has quadratic memory cost. Learned streaming
alternatives require retraining and drift over long sequences. LASER instead converts a frozen
offline model into a streaming system by aligning consecutive temporal windows, and identifies
*layer depth misalignment* (monocular scale ambiguity makes relative depth scales of different
scene layers vary between windows) as the reason plain Sim(3) alignment fails.

**Claimed contributions (paper claims, not repository behaviour).**
1. A training-free framework that converts offline reconstruction models into streaming systems.
2. Identification of layer depth misalignment plus *Layer-wise Scale Alignment* (LSA): segment
   depth predictions into layers, compute per-layer scale factors, propagate them across windows
   and timestamps.
3. State-of-the-art streaming pose/reconstruction (paper reports -68.6% ATE on Sintel pose,
   -63.9% Acc on 7-Scenes) at 14 FPS and 6 GB peak memory on one RTX A6000.

**Provenance of this checkout.**
- Remote: `https://github.com/neu-vi/LASER.git`; branch `main`;
  commit `7adbb7d5c1558f0446398310f31ee92fb4bc2de1` ("fix mutual conf mask"), acquired as a
  depth-1 shallow clone with submodules excluded, so `viser/` is empty.
- Paper: arXiv:2512.13680 (v3, 2026-03-28), CVPR 2026, CC BY 4.0, uniquely identified from
  `README.md` (title, arXiv badge, BibTeX, project page `https://neu-vi.github.io/LASER/`).
- Scale: ~277 tracked files, 145 Python files (~24k lines); license in `LICENSE`.
- Domain (INFERRED): computer-vision 3D/4D reconstruction — streaming monocular video depth,
  camera pose estimation and multi-view point-map estimation.

**Repository composition.** Baseline-specific code lives in `inference_engine/`, `loop_closure/`,
`eval/`, `mv_recon/`, `configs/` and the top-level entry points. `pi3/` and `vggt/` are vendored
backbone/model code; `pi3/models/dinov2/` and `loop_closure/{backbones,aggregators}/` are
vendored third-party networks. There is no training entry point: the repository contains only
inference, evaluation and dataset-reading code (VERIFIED: no train script, no optimizer use
outside `eval/misc.py` training leftovers).

# Canonical Pipeline

VERIFIED unless labelled otherwise.

1. **Frame list.** `demo.py::run_dynamic_scene` globs a directory with `os.listdir`, filters
   image suffixes and applies `[::sample_interval]`. `eval/pose_eval.py` instead builds a sorted
   `filelist` strided by `--pose_eval_stride`; `loop_closure/loop_closure.py::run` uses
   `sorted(glob(*.jpg) + glob(*.png))[::sample_interval]`.
2. **Preprocessing.** `utils/load_fn.py::load_and_preprocess_images` resizes each image to width
   518 (mode `crop`, height rounded to a multiple of 14, center-cropped at 518), stacks to
   `(N, 3, H, W)` in `[0,1]`; `mv_recon`/`utils/interfaces.py` use the same helper.
3. **Windowing.** `StreamingWindowEngine.img_sliding_window` -> `sliding_window_l` (paths) or
   `sliding_window_t` (tensor), step = `window_size - overlap`; a truncated final window is kept
   only if longer than `overlap`.
4. **Backbone inference.** Two worker threads started by `begin()`:
   `_model_inference_worker` runs `self.delegate(sample)` under `torch.autocast(dtype)` with no
   gradients and moves predictions to `process_device` (default `cpu`);
   `_registration_worker` consumes them from `registration_queue`. `forward()` only enqueues.
   The backbone wrapper is `inference_engine/vanilla_engine.py::VanillaEngine`; for π³,
   `Pi3.forward` returns `{points, local_points, conf, camera_poses, images}` with batch dim.
5. **Per-window post-processing.** `VanillaEngine._post_process_pred` adds `depth`, `intrinsic`
   (median-based pseudo-intrinsics) and aliases `depth_conf = conf`, `extrinsic = camera_poses`.
6. **First window.** `estimate_pseudo_depth_and_intrinsics` derives translation-invariant
   intrinsics from the predicted point map; `unproject_depth_to_local_points` rebuilds
   `local_points` from the z-channel. The resulting `ref_intrinsic` is reused for every later
   window ("fixed intrinsic enforce").
7. **Adjacent-window registration.** For each later window: mutual confidence mask over the
   `overlap` shared frames -> `inference_utils.register_adjacent_windows`
   -> `align_cam_pts_irls` yields the scalar scale `s_d` (initialised as
   `sum(src*tgt)/sum(src^2)`, then <=10 IRLS reweightings with weights `1/(|residual|+eps)`,
   `stop_tol=0.05`, clamped at `1e-6`)
   -> `utils/geometry.register_camera_poses_kabsch_pytorch` solves `R, t` by SVD on three
   correspondences per camera (position, position + scaled view direction, position + up
   direction) -> `apply_sim3_to_pose` and `s_d * local_points` move the window into world space.
8. **LSA depth refinement** (only when `depth_refine`): see `Paper-to-Code Mapping`; the per-pixel
   scale mask returned by `lsa.refine_depth_segments` multiplies the full `local_points` tensor.
9. **Cache and aggregation.** Each registered window is written to
   `<cache_root>/<tempdir>/window_cache_<id>.pt`; `parse_inference_cache_summary` reads them in
   order, drops the first `overlap` frames of every window after the first
   (`parse_cache_file`), concatenates to per-sequence tensors and derives world `points` via
   `einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize(local_points))`.
10. **Outputs.** `demo.py` squeezes the batch dim, converts to numpy and calls
    `eval/save_func.save_for_viser` (`inverse_extrinsic=False`), writing
    `viser_results/<scene>/`: `pred_intrinsics.txt`, `pred_traj.txt` (TUM), `frame_XXXX.npy`,
    `conf_i.npy`, `frame_XXXX.png`. `demo.py` also prints wall-clock inference time and
    `torch.cuda.max_memory_allocated()`; `StreamingWindowEngine.end()` prints per-window
    latencies with a 2-window warm-up. These are the in-repo paths behind the paper's
    14 FPS / 6 GB claim (numbers not reproduced in this session).

# Paper-to-Code Mapping

| Paper component | Paper statement | Code path / symbol | Evidence | Discrepancy / notes |
|---|---|---|---|---|
| Sliding-window streaming wrapper (Sec. 3.1) | Process frames in overlapping windows `W_i` with `a_{i+1} = a_i + L - O`, frozen reconstructor per window | `inference_engine/streaming_window_engine.py::StreamingWindowEngine` (`img_sliding_window`, `_model_inference_worker`, `_registration_worker`); `inference_utils.sliding_window_t/_l` | VERIFIED | L/O are runtime arguments (20/5 eval + dense mv-recon, 30/10 README example, 75/30 LC, 60/30 outdoor); the code additionally runs inference and registration in two threads and stages to CPU |
| Frozen offline reconstructor f(.) producing (P, T, C) (Sec. 3.1, Eq. 1) | VGGT and π³ backbones | `pi3/models/pi3.py::Pi3.forward` -> `local_points`, `camera_poses`, `conf`; wrapped by `VanillaEngine` | VERIFIED for π³ | No entry point instantiates a VGGT-backed streaming engine; `vggt/` is present but only its geometry/pose utilities are imported (VGGT streaming path UNKNOWN) |
| Incremental registration into world space, Sim(3) in `Sim(3)` (Sec. 3.1) | Global scale by robust IRLS, then `(R,t)` by Kabsch on scaled camera anchors | `inference_utils.register_adjacent_windows`, `align_cam_pts_irls`, `utils/geometry.register_camera_poses_kabsch_pytorch`, `apply_sim3_to_pose` | VERIFIED | The reference frame is the previous window's globally-transformed cache (its overlap part of `G_{i-1}`), matching the paper's intent; code also enforces one shared pseudo-intrinsic for all windows, which the retrieved paper sections do not describe (INFERRED) |
| Confident correspondences from pixel-wise confidence `C` (Sec. 3.1/3.2) | Confidence scores form mutually confident correspondences | `torch.quantile(conf, top_conf_percentile, interpolation='nearest')` masks AND-ed across the overlap | VERIFIED | Constructor stores `1 - top_conf_percentile` (keeps the top fraction); values differ per entry point: 0.3 in `demo.py`, LC and outdoor (0.6 in `eval_outdoor.py`), 0.5 in `eval_launch.py --model=streaming_pi3` and `mv_recon/eval.py` |
| Depth layer extraction (Sec. 3.2) | Segment each depth map into depth-ordered coherent layers with an efficient segmentation algorithm [11] | `inference_engine/utils/depth.py::segment_depth_felzenszwalb_rag` = `skimage.segmentation.felzenszwalb(depth, scale=300, sigma=1.1, min_size=500)` + `merge_regions` (mean-depth DSU merging) | VERIFIED (code) | Citation [11] itself is UNKNOWN (paper HTML truncated before references); merge threshold is `0.1 * (max-min)` of the top-confidence depth range; an alternative Cython segmentation path (`fast_seg`, `segment_depth_graph_fast`) exists but is never called |
| Layer graph with inter-window (`E_inter`) and intra-window (`E_intra`) edges, IoU > tau, tau = 0.3 (Sec. 3.2) | Directed graph over layer vertices | `inference_engine/utils/lsa.py::make_sp_graph` -> `depth.match_segmentation_seq` / `connect_bipartite_sp_graphs`; nodes are `pi3/utils/graph.py::Vertex`; inter-window edges added in `assign_overlap_window_depth_scale` | VERIFIED | Code uses 0.3 for intra-window edges (`make_sp_graph(corr_iou_thresh=0.3)`) but 0.4 for inter-window edges (`refine_depth_segments(corr_iou_thresh=0.4)`); the paper states one tau = 0.3. Comparisons in code are `>=` |
| Layer-wise scale estimation by IRLS over inter-window edges (Sec. 3.2) | Solve per-layer scale from depth correspondences in the intersection of two layers | `depth.align_depth_irls` called by `_edge_scale_worker` from `assign_overlap_window_depth_scale` with `inter_mask = src_mask & tgt_mask` | VERIFIED (code) | The paper's exact objective is UNKNOWN (truncated fetch); code: init `s = mean(tgt)/mean(src)`, <=10 iterations, weights `1/(|residual|+eps)`, `stop_tol=0.05`, clamp `1e-6`. The scale maps the current window's depth onto the previous (world) window's depth |
| Scale propagation over the layer graph (Sec. 3.2) | Corrected scales are propagated and aggregated across `E_inter` and `E_intra` | `Vertex.propagate_data_once(_propagate_scale_cache)` per frame graph in `lsa.align_adjacent_windows_depth_segments`; `_get_scale_mask` aggregates cached scales by IoU-weighted mean (1.0 when empty) | VERIFIED | The previous window's overlap vertices have all edges cleared before reconnection, so propagation is recomputed per registration step rather than accumulated |
| Applying layer scales | Correct foreground/background over- or under-scaling | `lsa.refine_depth_segments` returns `tgt_scale_mask[..., None]`; `StreamingWindowEngine._registration_worker` multiplies it into `local_points` | VERIFIED | The correction multiplies all three point coordinates (x,y,z) of the layer, not the z/depth component alone (INFERRED interpretation of "scale along depth") |
| Global map update `G_i = G_{i-1} ∪ {T_t^w P_t^(i)}` (Sec. 3.1) | Progressive global map | Per-window `window_cache_*.pt` + `parse_inference_cache_summary` -> `aggregate_caches` -> world `points` | VERIFIED | The global map is materialised from cache files at the end of a run; during streaming the alignment reference is `prev_window_cache` |
| Efficiency claim 14 FPS / 6 GB (Abstract, Sec. 4.6) | Peak memory and throughput on RTX A6000 | `StreamingWindowEngine.latencies` + `end()` summary; `demo.py` prints seconds and `torch.cuda.max_memory_allocated()` | INFERRED (measurement path only) | Claimed numbers UNKNOWN here (nothing was run; Environment = NONE) |
| Loop closure (README update 2026-03-12) | Not covered by the retrieved method sections | `StreamingWindowEngineLC`, `loop_closure/LoopClosureEngine`, `LoopDetector`, `Sim3LoopOptimizer`, `demo_lc.py` | VERIFIED (code) | Treat as a released extension, not as a paper-verified component |

# Detailed Data Flow

**A. Demo inference (`demo.py`).**
`get_args_parser` -> `load_model` (`Pi3()` + `--model_ckpt`, or `Pi3.from_pretrained("yyfz233/Pi3")`)
-> `StreamingWindowEngine(model, inference_device, dtype, window_size, overlap, cache_root,
depth_refine, top_conf_percentile=0.3)` -> `run_dynamic_scene` builds the image list ->
`run_model`: `img_sliding_window` -> `begin()` -> per window
`load_and_preprocess_images(sample).to(device)` and `model(imgs)` (enqueue) -> `end()` ->
`parse_inference_cache_summary()` -> numpy squeeze -> `save_for_viser(..., inverse_extrinsic=False)`
-> latency/memory summary print. State: `prev_window_cache`, `anchor_sp_graph`, `cache_id`,
`ref_intrinsic` (worker-local), `latencies`.

**B. Pose / depth evaluation (`eval_launch.py`).**
`--model` selects the engine: `pi3` -> `VanillaEngine`; `streaming_pi3` ->
`partial(StreamingWindowEngine, dtype, inference_device, window_size=20, overlap=5,
top_conf_percentile=0.5)`; `streaming_pi3_lc` -> `partial(StreamingWindowEngineLC, dtype,
inference_device, window_size=75, overlap=30, top_conf_percentile=0.3)`.
`pi3_main` builds the engine around `Pi3.from_pretrained("yyfz233/Pi3")`, then
`--mode=eval_pose` -> `eval/pose_eval.py::eval_pose_estimation` with
`eval_launch.py::inference_streaming_model` (or `_lc`) as the inference callable. For each
sequence: `metadata['dir_path_func']` -> sorted filelist strided by `pose_eval_stride` ->
`load_and_preprocess_images` -> engine -> per-frame depth `.npy` and `.png` plus
`save_for_viser` outputs -> `get_tum_poses` -> `eval/vo_eval.py::eval_metrics`
(evo APE/RPE) -> per-sequence `_eval_metric.txt` and trajectory plot -> `process_directory` +
`calculate_averages` -> JSON summary in `output_dir`.
`--mode=eval_depth` routes to `eval/depth_eval.py::eval_mono_depth_estimation`.

**C. Depth metrics (`depth_metric.py`).** Reads `result_dir/*/frame_*.npy` produced by the pose
driver, groups them by sequence directory (`group_by_directory`), resizes predictions to ground
truth with cubic interpolation, and calls `eval/depth_eval.py::depth_evaluation`
(`max_depth=70`, `align_with_scale=True`, `use_gpu=True`, `post_clip_max=70` for Sintel/Bonn;
`max_depth=None` for KITTI) to obtain Abs Rel, Sq Rel, RMSE, log RMSE and `delta < 1.25^k`, then
writes `<dataset>_depth.json` with valid-pixel-weighted averages.

**D. Multi-view point maps (`mv_recon/eval.py`).** Hydra config `configs/eval_mv_recon_dense.yaml`
-> dataset instantiation from `configs/data/mv_recon_dense.yaml` (7scenes-dense, NRGBD-dense) ->
pre-sampled `seq_id_map` JSON -> `dataset.get_data(sequence_name, ids)` ->
`utils/interfaces.py::infer_streaming_mv_pointclouds(filelist, engine, cfg, data_size)` (same
engine start/stop/cache-aggregate cycle, then bilinear interpolation of `points` to ground-truth
size) -> optional CUT3R-style 224x224 center crop -> `umeyama` coarse Sim(3) to ground truth ->
Open3D point-to-point ICP refinement -> normal estimation -> `accuracy`/`completion` (mean and
median, plus normal-consistency) -> per-sample CSV and averaged metric CSV.
`mv_recon/eval_outdoor.py` mirrors this flow for Waymo with a 60/30 window, `depth_refine=False`
and confidence quantile filtering.

**E. Loop-closure variant (`demo_lc.py`, `eval_launch.py --model=streaming_pi3_lc`).**
`StreamingWindowEngineLC._registration_worker` stores each window's relative Sim(3) under
`working_window['sim3']` instead of applying it, and caches raw points.
`LoopClosureEngine.run` builds `img_list` (`sorted(glob(*.jpg/*.png))[::sample_interval]`), runs
`LoopDetector` (DINOv2 + SALAD descriptors, FAISS inner-product search, similarity and NMS
filtering) -> `process_loop_list` maps each loop pair to chunk index pairs and frame ranges ->
`process_single_chunk` re-runs π³ on the two frame ranges -> two `register_adjacent_windows`
calls -> `compute_sim3_ab` builds a loop constraint -> `Sim3LoopOptimizer.optimize` (pypose
Sim(3), Levenberg-Marquardt via `loop_closure/fastloop/solve_python.py` or the optional C++
`sim3solve`) -> corrected per-window Sim(3) list -> `StreamingWindowEngineLC.aggregate_caches`
accumulates `sim3` per window (`accumulate_sim3`) and applies it to `local_points`/`camera_poses`.

# Core Modules

| Module | Role | Key symbols |
|---|---|---|
| `inference_engine/streaming_window_engine.py` | Canonical streaming engine; threaded inference + registration, caching, aggregation | `StreamingWindowEngine`, `_model_inference_worker`, `_registration_worker`, `begin/end/forward`, `parse_cache_file`, `aggregate_caches`, `parse_inference_cache_summary` |
| `inference_engine/inference_utils.py` | Windowing, pseudo-intrinsics, metric/Sim(3) alignment, shared by all engines | `sliding_window_t/_l`, `aggregate_windows`, `estimate_pseudo_depth_and_intrinsics`, `unproject_depth_to_local_points`, `register_adjacent_windows`, `align_cam_pts_irls` |
| `inference_engine/utils/lsa.py` | LSA orchestration: layer graph, propagation, per-pixel scale mask | `make_sp_graph`, `refine_depth_segments`, `align_adjacent_windows_depth_segments` |
| `inference_engine/utils/depth.py` | Layer segmentation, IoU matching, IRLS depth alignment | `segment_depth_felzenszwalb_rag`, `align_depth_irls`, `pairwise_iou`, `match_segmentation_seq`, `connect_bipartite_sp_graphs`, `assign_overlap_window_depth_scale` |
| `inference_engine/utils/geometry.py` | Sim(3)/SE(3) helpers used by the engines and LC | `homogenize_points`, `register_camera_poses_kabsch_pytorch`, `apply_sim3_to_pose`, `accumulate_sim3`, `closed_form_inverse_sim3` |
| `pi3/utils/graph.py` | Generic layer-graph node with IoU edge weights and per-node cache | `Vertex.add_edge`, `remove_all_edges`, `propagate_data_once`, `data_cache_op` |
| `inference_engine/streaming_window_engine_lc.py` | LC engine: defers Sim(3) to aggregation | `StreamingWindowEngineLC.aggregate_caches` |
| `loop_closure/` | VPR loop detection and Sim(3) pose-graph optimisation | `LoopClosureEngine`, `LoopDetector`, `VPRModel`, `Sim3LoopOptimizer`, `process_loop_list`, `compute_sim3_ab` |
| `eval/` | Pose and depth evaluation | `eval_pose_estimation`, `eval_metrics`, `depth_evaluation`, `save_for_viser`, `dataset_metadata` |
| `mv_recon/` | Point-map evaluation | `eval.py::main`, `eval_utils.umeyama/accuracy/completion` |
| `utils/` | Shared IO and geometry | `load_and_preprocess_images`, `infer_streaming_mv_pointclouds`, `closed_form_inverse_se3`, `write_csv` |
| `datasets/` | Dataset readers + frozen sampling maps | `SevenScenes`, `NRGBD`, `Waymo`, `get_data`, `get_seq_framenum` |

# Key Classes and Functions

- `StreamingWindowEngine.__init__(delegate, inference_device, dtype, intermediate_device='cuda',
  process_device='cpu', top_conf_percentile=0.5, window_size=20, overlap=5, depth_refine=True,
  cache_root='./cache', benchmark_latency=True)` — stores `1 - top_conf_percentile`; creates the
  cache root, both queues and the running flag.
- `StreamingWindowEngine.forward(sample)` — enqueues an image window only; all work happens on the
  two daemon threads (`begin()` starts them, `end()` sends `STOP_SIGNAL`, joins, prints latency
  statistics and resets state).
- `_registration_worker` — the algorithmic heart: squeezes the batch dim, computes the mutual
  confidence mask, calls `register_adjacent_windows`, applies `s_d`/`apply_sim3_to_pose`, runs the
  LSA branch, updates `prev_window_cache`/`anchor_sp_graph` and saves the window cache.
- `parse_inference_cache_summary(remove_cache=True)` — sorts `window_cache_*.pt` by numeric id,
  concatenates with overlap trimming for all but the first window, post-processes and deletes the
  temporary cache directory.
- `aggregate_caches(parsed_caches)` — concatenates all keys except `points`, then recomputes
  `points = einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize(local_points))`.
- `register_adjacent_windows(src_pcd, tgt_pcd, src_cam, tgt_cam, mask)` -> `(s_d, R, t)` —
  `s_d` from `align_cam_pts_irls`, then `R, t` from the Kabsch solver with `scale=s_d`.
- `register_camera_poses_kabsch_pytorch(src_cam_poses, tgt_cam_poses, scale)` — SVD-based
  `Sim(3)` fit over 3 correspondences per camera; reflection corrected by flipping the last
  singular vector.
- `make_sp_graph(depth, depth_merge_thresh=0.1, conf_map=None, top_conf_percentile=None,
  corr_iou_thresh=0.3)` -> list of per-frame vertex lists; thread-parallel segmentation through
  `batched_image_op_wrapper`.
- `refine_depth_segments(src_pcd, tgt_pcd, src_sp_graphs, tgt_sp_graphs, overlap,
  corr_iou_thresh=0.4)` -> `(N,H,W,1)` multiplicative scale mask.
- `align_depth_irls(src_depth, tgt_depth, mask, iters=10, eps=1e-8, stop_tol=0.05,
  clamp_min=1e-6)` -> scalar depth scale for one layer intersection.
- `align_cam_pts_irls(src_pts, tgt_pts, mask, iters=10, ...)` -> scalar point-map scale.
- `Pi3.forward(raw_imgs)` -> `{points, local_points, conf, camera_poses, images}`; it exponentiates
  the predicted z, forms `local_points = [xy*z, z]`, and lifts them with `camera_poses`.
- `eval_metrics(pred_traj, gt_traj, seq, filename)` — evo APE (translation part, `align=True`,
  `correct_scale=True`, rmse) and RPE (rotation degrees and translation, `delta=1` frame,
  `all_pairs=True`, `correct_scale=True`), averaged over sequences by `calculate_averages`.
- `depth_evaluation(...)` — alignment variants (median, lstsq, LAD, IRLS "scale") followed by
  Abs Rel / Sq Rel / RMSE / log RMSE / `delta < 1.25^k` over pixels with `0 < gt < max_depth`.
- `infer_streaming_mv_pointclouds(filelist, engine, cfg, data_size)` — engine cycle then resize of
  world `points` to the ground-truth resolution, returning `(points, conf)` as numpy.

# Baseline Variants

- `VanillaEngine` (`inference_engine/vanilla_engine.py`) — the offline baseline: one forward pass
  over the whole sequence, no windowing or alignment (`--model=pi3`).
- `StreamingWindowEngine` — the canonical LASER path (canonical with `depth_refine=True`; with
  `depth_refine=False` it is the plain Sim(3) streaming ablation).
- `StreamingWindowEngineLC` (`..._lc.py`) — loop-closure variant that defers Sim(3) application to
  aggregation; used by `demo_lc.py` and `--model=streaming_pi3_lc`.
- `SlidingWindowEngine` (`inference_engine/sliding_window_engine.py`) — non-streaming, one-shot
  windowed variant using `register_extrinsic_windows`; exported by `inference_engine/__init__.py`
  but never instantiated in any entry point (VERIFIED by import scan).
- `post_eval_launch.py` + `eval/post_pose_eval.py` — re-evaluation of externally stored poses for
  `kitti_odometry`/`vbr` from `<pose_dir>/<seq>/camera_poses.txt` (replica format).
- Unreferenced dataset readers: `datasets/{dtu,eth3d,co3d_v2,re10k}.py` and
  `datasets/base_dataset.py` are imported by nothing and appear in no config (VERIFIED); the
  configs only instantiate `SevenScenes`, `NRGBD` and `Waymo`.
- Unused helpers: `inference_engine/utils/{segmentation.py,fast_seg.pyx,segment_depth_graph_fast}`
  (the pure-Python `merge_regions` is shadowed by the Cython `_segmentation_cy.merge_regions`),
  `loop_closure/utils/{loop_refinement,visual_util}.py`, `loop_closure/helper.py`.

# Evaluation Paths

- **Camera pose** (`eval_launch.py --mode=eval_pose --model=streaming_pi3`): datasets
  `sintel` (14 sequences, `camdata_left`), `scannet` (`color_90`/`pose_90.txt`, replica format),
  `tum` (`rgb_90`/`groundtruth_90.txt`), `bonn` (`rgb_110`/`groundtruth_110.txt`, 5 sequences),
  `kitti`; metrics ATE (APE translation rmse, scale-corrected) plus RPE translation/rotation
  rmse with one-frame deltas; per-sequence `_eval_metric.txt`, trajectory plot, JSON summary.
- **Video depth** (README): the same `--mode=eval_pose` driver with `--model=streaming_pi3` and
  `--no_crop` (KITTI additionally `--flow_loss_weight 0 --translation_weight 1e-3`) writes
  `frame_*.npy` depth maps, then `depth_metric.py` computes Abs Rel / `delta < 1.25^k` per dataset
  (`sintel`, `bonn`, `kitti`) into `<dataset>_depth.json`.
- **Multi-view point maps**: `mv_recon/eval.py` (dense: `7scenes-dense`, `NRGBD-dense`,
  key frames every 10; kf15 config: `NRGBD-kf15`; all-frames config:
  `configs/data/mv_recon_all.yaml`) and `mv_recon/eval_outdoor.py` (Waymo) report
  Acc/Comp mean and median plus normal consistency, averaged over sequences.
- **Qualitative**: `save_for_viser` output directory consumed by the `viser` submodule
  (`python viser/visualizer_monst3r.py --data viser_results/<scene>`), unavailable here because
  the submodule is not initialized (SOURCE NOT AVAILABLE).
- Datasets are not vendored: `README.md` points to MonST3R/MonST3R-style preparation and expects
  everything under `data/` (7-Scenes, NRGBD, Sintel, ScanNet, TUM, Bonn, KITTI, Waymo caches);
  `datasets/preprocess/` only contains 7-Scenes/ETH3D/Re10k preparation scripts.

# Architecture Boundaries

- **Baseline boundary**: `inference_engine/` + `pi3/` (backbone) define the streaming method;
  everything that changes `window_size`, `overlap`, `depth_refine`, `top_conf_percentile`,
  layer-merge or IoU thresholds is a research variable.
- **Variant boundary**: `StreamingWindowEngineLC`/`loop_closure/` form post-hoc, non-causal
  correction on top of the streaming cache (it re-runs the backbone on loop chunks and
  re-optimises the Sim(3) chain); they do not change the canonical streaming path.
- **Shared path**: `inference_utils`, `utils/geometry`, `utils/load_fn`, `pi3/utils/graph.Vertex`
  are shared by the canonical path, the variants and the LC engine — edits there affect every
  experiment; `Vertex` in particular is mutated in place across windows.
- **Independent path**: the evaluation harness (`eval/`, `depth_metric.py`, `mv_recon/`) reads
  cached predictions only; it must stay untouched as the measuring instrument.
- **Third-party boundary**: `pi3/models/dinov2/`, `vggt/`, `loop_closure/{backbones,aggregators}`,
  `loop_closure/fastloop/solve.cpp` are vendored models/implementations; import-only dependencies
  for the flows above.
- **Data boundary**: `datasets/` + `datasets/seq-id-maps/*.json` fix the exact frames evaluated;
  `configs/` fixes the evaluation parameters. Both are part of the protocol, not tuning knobs.

# Idea-Relevant Baseline Areas

No Initial Idea was provided in this session, so no Idea-specific baseline areas were derived.
Stated objectively, the areas any future Idea would have to touch are: the windowing/alignment
stage (`inference_engine/streaming_window_engine.py::_registration_worker`), the LSA stage
(`inference_engine/utils/lsa.py` + `utils/depth.py`), the confidence/threshold policy
(`top_conf_percentile`, `corr_iou_thresh`, `depth_merge_thresh`), the backbone adapter
(`VanillaEngine` + `Pi3.forward` output contract), and the evaluation harness (`eval_launch.py`,
`depth_metric.py`, `mv_recon/eval*.py`). This list is descriptive only and endorses no method.

# Source-Level Observations

Observed issues only: each entry states the observation, its evidence and the possible impact.
No fixes are proposed or applied.

1. **Outdoor point-map evaluation imports a non-existent module.**
   `mv_recon/eval_outdoor.py:15` does `from mv_recon.utils import umeyama, accuracy, completion`,
   but the package contains `mv_recon/eval_utils.py` (which `mv_recon/eval.py:13` imports) and no
   `mv_recon/utils.py`. Possible impact: `eval_outdoor.py` cannot start (ImportError), so the
   Waymo/outdoor path cannot be executed as shipped.
2. **`--mode=eval_depth` is not usable with the Pi3 engines.**
   `eval/depth_eval.py:36-46` iterates a single path string as if it were a list of image paths
   (the file itself carries a TODO saying so) and reads `predictions["pose_enc"]`, while
   `optimizer`-free Pi3 outputs (`Pi3.forward`, `VanillaEngine._post_process_pred`) contain no
   `pose_enc` key. Possible impact: the documented README depth workflow works only because it
   routes through `--mode=eval_pose` plus `depth_metric.py`; the `eval_depth` code path fails.
3. **CUDA capability is queried at import time in ~10 modules.**
   e.g. `demo.py:14`, `demo_lc.py:19`, `eval_launch.py:92`, `eval/depth_eval.py` chains,
   `mv_recon/eval.py:21`, `mv_recon/eval_outdoor.py:25`, `utils/interfaces.py`,
   `loop_closure/loop_closure.py:15` call `torch.cuda.get_device_capability()` unconditionally,
   immediately after computing `device = "cuda" if torch.cuda.is_available() else "cpu"`.
   Possible impact: the CPU fallback is nominal; importing these entry points on a CUDA-less host
   fails before any device check (INFERRED impact; not executed here).
4. **Demo frame ordering is not deterministic.**
   `demo.py:101-103` and `demo_lc.py:115-117` build the frame list from `os.listdir` and then
   stride it, whereas the evaluation paths use `sorted(glob(...))`/`filelist.sort()`
   (`eval/pose_eval.py:86-88`, `loop_closure/loop_closure.py:204-205`,
   `eval/eval_metadata.py` process functions). Possible impact: on filesystems returning
   directory entries out of lexical order, demo runs feed the sliding window a wrong temporal
   order while evaluation runs do not.
5. **LC aggregation can apply a stale accumulated scale.**
   `inference_engine/streaming_window_engine_lc.py:147-154` computes
   `s_d, R, t = accumulate_sim3(ref_sim3, cache_sim3)`, then in the `scale_mask` branch scales
   `local_points` by `ref_sim3[0]` (the value from the previous iteration) instead of `s_d`.
   Possible impact: when the LC variant is run with `depth_refine=True`, the depth-refined points
   are scaled by the previous window's accumulated scale; current entry points do not enable
   `depth_refine` for the LC engine (class default `False`), so the branch is unexercised by
   default.
6. **A 7-Scenes key-frame map referenced by config is absent.**
   `configs/data/mv_recon_kf15.yaml:12` points to
   `datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf15.json`, but the directory ships only
   `-all`, `-kf10`, `-kf40`, `-kf200` for 7-Scenes (`NRGBD` ships `-kf15`).
   Possible impact: enabling the commented-out `7scenes-kf15` in
   `configs/evaluation/mv_recon_kf15.yaml` fails when `open(dataset_info.seq_id_map)` runs.
7. **Sampling regeneration path has unmet prerequisites.**
   `mv_recon/sampling.py:12` imports `rootutils` (not in `requirements.txt`) and uses
   `@hydra.main(config_name="eval")` while `configs/` contains only
   `eval_mv_recon_dense.yaml`/`eval_mv_recon_outdoor.yaml` (no `eval.yaml`).
   Possible impact: new `seq-id-map` files cannot be regenerated with the shipped environment;
   the checked-in maps still allow evaluation to run.
8. **Importing `inference_engine` requires the compiled Cython extensions.**
   `inference_engine/utils/depth.py:6-7` imports `fast_seg` and `_segmentation_cy` at module
   level, so `inference_engine/__init__.py` -> `inference_utils` -> `utils.lsa` -> `utils.depth`
   pulls both extensions in; `setup.py` cythonizes `inference_engine/utils/*.pyx`. The
   pure-Python `inference_engine/utils/segmentation.py` is imported nowhere, and
   `fast_graph_segmentation` is only called by the never-used `segment_depth_graph_fast`.
   Possible impact: without `python setup.py build_ext --inplace`, every entry point fails to
   import even when `depth_refine` is disabled.
9. **Saved pose convention differs between demo and evaluation paths.**
   The Pi3 engines store cam-to-world poses in `predictions['extrinsic']`
   (`VanillaEngine._post_process_pred`: `pred['extrinsic'] = pred['camera_poses']`), `demo.py:82`
   calls `save_for_viser(..., inverse_extrinsic=False)`, while `eval/pose_eval.py:128` calls
   `save_for_viser(predictions, seq, save_dir)` with the default `inverse_extrinsic=True`, which
   replaces `extrinsic`; `eval_launch.py` separately passes `inverse_extrinsic=False` for the
   trajectory used by the metrics. Possible impact: the `pred_traj.txt` written under the
   evaluation output directory uses the inverse pose convention relative to demo output, so
   qualitative comparisons between the two paths are not directly comparable (INFERRED impact;
   not executed here).
10. **LC loop detection ignores the evaluation stride.**
    `eval_launch.py:117-124` builds `LoopClosureEngine` without `sample_interval` (default 1)
    although `eval/pose_eval.py:88` strides the filelist by `--pose_eval_stride`.
    Possible impact: with a stride other than 1, detected loop indices no longer match the window
    cache sequence, so LC correction would be applied to the wrong windows (INFERRED).

# Unknowns

- **VGGT backbone integration.** The paper claims both VGGT and π³; the repository wires only
  `Pi3.from_pretrained("yyfz233/Pi3")` in `demo.py`, `eval_launch.py`, `mv_recon/eval.py` and
  `demo_lc.py`. Whether a VGGT-backed streaming engine is supported by construction is UNKNOWN
  (`vggt/` only supplies geometry/pose utilities to the executed paths).
- **Paper details lost to a truncated fetch.** The exact IRLS objective for layer-wise scales
  (Sec. 3.2), the bibliography entry behind the "efficient segmentation algorithm [11]", the
  implementation-details appendix, the ablation table and the formal efficiency benchmark
  protocol were not retrievable in this session (arXiv HTML truncated mid-Sec. 3.2) -> UNKNOWN;
  no attempt was made to reconstruct them from code.
- **Exact evaluation protocol values.** Which sequence subsets and frame counts the paper used per
  dataset, and the exact preprocessing ordering, are UNKNOWN beyond what the README commands,
  `eval/eval_metadata.py` and the configs state.
- **Uninitialized submodule.** `viser/` is empty (submodule not fetched), so the visualization
  entry point `viser/visualizer_monst3r.py` referenced by `README.md` is SOURCE NOT AVAILABLE in
  this checkout.
- **Number claims.** 14 FPS, 6 GB peak memory, -68.6% ATE and -63.9% Acc are paper claims; no
  run, benchmark or measurement was performed in this session (Environment = NONE), so they
  remain UNKNOWN as repository behaviour.
- **Data and weights.** pi³ weights (HF `yyfz233/Pi3`), loop-closure weights
  (`weights/dino_salad.ckpt`, `weights/dinov2_vitb14_pretrain.pth`) and all datasets under `data/`
  are absent; whether the shipped defaults reproduce the paper numbers is UNKNOWN.
- **Dead-code intent.** Whether `SlidingWindowEngine`, `fast_seg.pyx`, `segment_depth_graph_fast`,
  `loop_closure/utils/loop_refinement.py`, `visual_util.py`, `helper.py` and the unused dataset
  readers are abandoned experiments or intended extension points is UNKNOWN; they are only
  observed to be unreferenced.
