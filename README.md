<div align="center">
<h1>LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction</h1>
<a href="http://arxiv.org/abs/2512.13680"><img src="https://img.shields.io/badge/arXiv-2512.13680-b31b1b" alt="arXiv"></a>
<a href="https://neu-vi.github.io/LASER/"><img src="https://img.shields.io/badge/Project-Website-orange" alt="Project Page"></a>

[Tianye Ding<sup>1*</sup>](https://jerrygcding.github.io/), 
[Yiming Xie<sup>1*</sup>](https://ymingxie.github.io/), 
[Yiqing Liang<sup>2*</sup>](https://lynl7130.github.io/), 
[Moitreya Chatterjee<sup>3</sup>](https://sites.google.com/site/metrosmiles/), 
[Pedro Miraldo<sup>3</sup>](https://pmiraldo.github.io/), 
[Huaizu Jiang<sup>1</sup>](https://jianghz.me/)\
<sup>1</sup> Northeastern University, <sup>2</sup> Independent Researcher, <sup>3</sup> Mitsubishi Electric Research Laboratories\
<sup>*</sup> Equal Contribution
</div>

## 📢 Updates
* **[2026-03-12]** Loop-closure module released with robustness fix.
* **[2026-02-21]** Paper accepted by CVPR 2026.
* **[2025-12-15]** ArXiv preprint released.

## 📝 To-Do List

- [x] Release framework codebase
- [x] Release inference code
- [x] Add data preparation instruction
- [x] Release evaluation code
- [x] Add Viser integration
- [x] Release loop-closure demo

## 💡 Abstract
We propose LASER, a training-free framework that converts an offline reconstruction model into a streaming system by aligning predictions across consecutive temporal windows. 
We observe that simple similarity transformation (Sim(3)) alignment fails due to layer depth misalignment: monocular scale ambiguity causes relative depth scales of different scene layers to vary inconsistently between windows. 
To address this, we introduce layer-wise scale alignment, which segments depth predictions into discrete layers, computes per-layer scale factors, and propagates them across both adjacent windows and timestamps.

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone --recursive git@github.com:neu-vi/LASER.git
cd LASER

# 2. Create environment
conda create -n laser -y python=3.11
conda activate laser

# 3. Install dependencies
pip install -r requirements.txt

# 4. Compile cython modules
python setup.py build_ext --inplace

# 5. Install Viser
pip install -e viser
```

(Optional) Download checkpoints needed for loop-closure inference

```bash
bash ./scripts/download_weights.sh
```

## 🚀 Usage

### Inference
To run the inference code, you can use the following command:
```bash
export PYTHONPATH="./":$PYTHONPATH

python demo.py \
    --data_path DATA_PATH \
    --output_path "./viser_results" \
    --cache_path "./cache" \
    --sample_interval SAMPLE_INTERVAL \
    --segmentation_method depth

# example inference script
python demo.py \
    --data_path "examples/titanic" \
    --output_path "./viser_results" \
    --cache_path "./cache" \
    --sample_interval 1 \
    --segmentation_method depth
```
The results will be saved in the `viser_results/SEQ_NAME`directory for future visualization.

> **Window schedule, confidence and `depth_refine` now come from
> `configs/segmentation_config.yaml`**, not from per-entry-point flags. They are locked there so
> that `demo.py`, `demo_lc.py`, `eval_launch.py` and the `mv_recon` evaluators all run the same
> schedule and their numbers can be compared. The old `--window_size`, `--overlap` and
> `--depth_refine` flags were removed rather than left to silently disagree; edit the config, or
> point at another one with `--segmentation_config`.

### Initial segmentation methods

Three initial segmentation strategies are available and share everything downstream of the
per-frame regions:

| `--segmentation_method` | Mechanism |
|---|---|
| `depth` | Baseline. Felzenszwalb on the depth map, then a DSU merge gated only by the absolute difference of region mean depths. |
| `geometry` | Felzenszwalb on a 4-channel `[normalized depth, nx, ny, nz]` image, then a merge gated by depth difference **and** region-mean normal angle **and** confidence. |
| `atomic` | Keeps the depth atoms and coarse layers as a prior, then merges across a scale-normalised 3D boundary gap `d / sqrt(s_A * s_B)`, with an optional split stage. |

```bash
# the same run with each method, everything else held fixed
python eval_launch.py --mode=eval_pose --eval_dataset sintel --model=streaming_pi3 \
    --segmentation_method depth
python eval_launch.py --mode=eval_pose --eval_dataset sintel --model=streaming_pi3 \
    --segmentation_method geometry
python eval_launch.py --mode=eval_pose --eval_dataset sintel --model=streaming_pi3 \
    --segmentation_method atomic
```

The Hydra-driven point-map evaluators take the same choice through the environment:

```bash
LASER_SEGMENTATION_METHOD=geometry python mv_recon/eval.py
```

### Observing the segmentation → LSA chain

ATE alone cannot say *why* one method is better: between the per-frame regions and the per-pixel
scale mask the pipeline used to expose nothing. Adding `--segmentation_diagnostics` writes one
scalar JSON per window, covering the observation points OP-1…OP-6:

| Point | What it records |
|---|---|
| OP-1 | atom / coarse-layer / final region counts, area distribution, the merge threshold and the confidence cut actually in force |
| OP-2 | how many adjacent pairs each merge criterion accepted or rejected, and the criterion's value distribution |
| OP-3 | cross-window IoU matching: edge count, vertex degree, and the share of regions that found any correspondence |
| OP-4 | per-edge IRLS: intersection-mask area, recovered scale, iteration count, convergence, and whether the estimate hit the clamp |
| OP-5 | the `Vertex` scale cache that the mask consumes (including how often the `1.0` no-op fallback fired) and the statistics of the final per-pixel mask |
| OP-6 | overlap depth consistency **before and after** the mask, which falsifies "the gain travels through LSA" without needing ATE |

```bash
python demo.py --data_path DATA_PATH --segmentation_method geometry \
    --segmentation_diagnostics
```

Without diagnostics the observation branches are inert: no extra work runs and the returned tensors
are identical to a build with the instrumentation removed.

### Local checks without a GPU

The synthetic checks need no GPU, no model weights and no dataset:

```bash
python scripts/check_segmentation_methods.py
python -m pytest tests/ -q
```

`check_segmentation_methods.py` asserts the L1 invariants (Baseline element-wise equality,
well-formed and deterministic partitions, merge-only behaviour of the new methods, IRLS exactness)
and reports the L2 measurements on analytically exact scenes, where the per-pixel depth-layer
partition and the true inter-window per-layer scale are known by construction. Among those numbers,
`segmentation cost` is the one that attributes an error to the segmentation rather than to LSA:
it is the gap between the scale error obtained from a method's own regions and the error obtained
from the ground-truth partition.

The design, its decisions and its open questions are recorded in
`docs/dsh/ideas/IDEA-001-segmentation-method-attribution.md`.

### Loop-closure inference
Loop-closure requires additional dependencies for package `faiss` can be installed through:
```bash
pip install faiss-gpu-cu12 numpy==1.26.4
```
Run loop-closure inference for kilometer-scale sequence with the following command:
```bash
python demo_lc.py \
    --config_path "configs/loop_config.yaml" \
    --data_path DATA_PATH \
    --output_path "./viser_results" \
    --cache_path "./cache" \
    --sample_interval SAMPLE_INTERVAL \
    --segmentation_method depth

rm -r cache/
```
`demo_lc.py` takes its window schedule and `depth_refine` from the same locked config as
everything else, so the loop-closure path and the canonical streaming path can be compared
directly. Note that the LC engine applies the LSA scale mask at aggregation time and never to the
first window — an existing behaviour this repository preserves.

### Visualization
To visualize the interactive 4D results, you can use the following command:
```bash
python viser/visualizer_monst3r.py --data viser_results/SEQ_NAME

# example visualization script
python viser/visualizer_monst3r.py --data viser_results/titanic
```

## Evaluation
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/prepare_training.md#dataset-setup) for dataset setup details.

Put all datasets in `data/`.

### Video Depth

Sintel
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=sintel \
    --output_dir="outputs/video_depth/sintel_depth" \
    --full_seq \
    --no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
    --eval_dataset=sintel \
    --result_dir="outputs/video_depth/sintel_depth" \
    --output_dir="outputs/video_depth"
```

Bonn
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=bonn \
    --output_dir="outputs/video_depth/bonn_depth" \
    --no_crop

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
    --eval_dataset=bonn \
    --result_dir="outputs/video_depth/bonn_depth" \
    --output_dir="outputs/video_depth"
```

KITTI
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=kitti \
    --output_dir="outputs/video_depth/kitti_depth" \
    --no_crop \
    --flow_loss_weight 0 \
    --translation_weight 1e-3

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
    --eval_dataset=kitti \
    --result_dir="outputs/video_depth/kitti_depth" \
    --output_dir="outputs/video_depth"
```

### Camera Pose

Sintel
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=sintel \
    --output_dir="outputs/cam_pose/sintel_pose"
```

ScanNet
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=scannet \
    --output_dir="outputs/cam_pose/scannet_pose"
```

TUM
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
    --mode=eval_pose \
    --model=streaming_pi3 \
    --eval_dataset=tum \
    --output_dir="outputs/cam_pose/tum_pose"
```
<!-- 
KITTI Odometry
```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
--mode=eval_pose \
--model=streaming_pi3_lc \
--eval_dataset=kitti_odometry \
--output_dir="outputs/cam_pose/kitti_odometry_pose"
```

### MV Recon
```bash
export PYTHONPATH="./":$PYTHONPATH
python mv_recon/eval.py
``` -->

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@article{ding2025laser,
  title={LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction},
  author={Ding, Tianye and Xie, Yiming and Liang, Yiqing and Chatterjee, Moitreya and Miraldo, Pedro and Jiang, Huaizu},
  year={2025}
}
```

## Acknowledgements
We would like to thank the authors for the following excellent open source projects:
[VGGT](https://github.com/facebookresearch/vggt/tree/main), 
[&pi;<sup>3</sup>](https://github.com/yyfz/Pi3),
[MonST3R](https://github.com/Junyi42/monst3r),
[CUT3R](https://github.com/CUT3R/CUT3R),
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long/tree/main)
and many other inspiring works in the community.
