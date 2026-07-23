# SAM-Assisted Lightweight Waste Classification (RealWaste)

Comparative study of **Original vs Bounding-Box-Cropped vs SAM-Segmented** images,
plus two ways of **fusing Original + Bounding-Box-Cropped together**, for
lightweight waste classification (**MobileNetV3-Large** & **EfficientNet-B0**)
on the **RealWaste** dataset (no ground-truth annotations — a single pass of SAM's
automatic mask generator over the raw images produces both the segmentation masks
and the bounding boxes; bbox_cropped and sam_segmented come from that one SAM pass,
not from YOLO).

**5 training variants** in total:
| Variant | Input | How it's combined |
|---|---|---|
| `original` | full raw image | — |
| `bbox_cropped` | SAM-bbox crop | — |
| `sam_segmented` | SAM mask crop (background blanked) | — |
| `fused` | original + bbox_cropped | **late fusion**: two backbones, one per image, pooled features concatenated before the classifier |
| `early_fused` | original + bbox_cropped | **early/pixel fusion**: images channel-stacked into one 6-channel input, single backbone (first conv inflated from 3→6 channels) |

## 1. Setup (VS Code / terminal)

```bash
# Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

> **Kaggle auth**: `download_dataset.py` uses `kagglehub`, which will prompt you to
> authenticate (browser login or `~/.kaggle/kaggle.json`) on first run.

> **GPU**: Everything auto-detects CUDA and falls back to CPU. Training 10 runs
> (2 models × 5 variants) x 30 epochs is realistically **GPU-only** in a few hours;
> on CPU, drop `training.epochs` in `configs/config.yaml` to something smaller.

## 2. Run everything (one command)

```bash
python run_pipeline.py
```

This runs, in order: dataset download → split → EDA → SAM checkpoint download →
SAM auto-segmentation (masks + bboxes, one pass) → build 3 preprocessing variants →
train + evaluate + confusion-matrix + prediction-samples + efficiency-benchmark +
training-curves for **all 10 (model × variant) runs** (`fused` and `early_fused`
need no extra preprocessing - both reuse the `original` and `bbox_cropped`
folders directly at train/eval time) → final aggregated comparison.

## 3. Run stage-by-stage (recommended if you're short on time / debugging)

```bash
# --- Data & annotation pipeline (run once) ---
python -m src.data_prep.download_dataset
python -m src.data_prep.clean_and_split
python -m src.data_prep.dataset_stats
python -m src.annotation_generation.download_sam_checkpoint
python -m src.annotation_generation.sam_auto_segment      # SAM automatic mask generator on raw images;
                                                            # feeds BOTH bbox_cropped (via the picked
                                                            # mask's own bbox) AND sam_segmented (via
                                                            # the mask itself). No YOLO involved.
python -m src.preprocessing.generate_variants

# --- Per (model, variant) — repeat for both models × all 5 variants ---
python -m src.training.train              --model mobilenet_v3 --variant original
python -m src.evaluation.evaluate         --model mobilenet_v3 --variant original
python -m src.evaluation.confusion_matrix --model mobilenet_v3 --variant original
python -m src.evaluation.prediction_visualizer --model mobilenet_v3 --variant original
python -m src.evaluation.efficiency_benchmark  --model mobilenet_v3 --variant original
python -m src.evaluation.training_curves       --model mobilenet_v3 --variant original

# ... repeat for: mobilenet_v3/bbox_cropped, mobilenet_v3/sam_segmented,
#                 mobilenet_v3/fused, mobilenet_v3/early_fused,
#                 efficientnet_b0/original, efficientnet_b0/bbox_cropped,
#                 efficientnet_b0/sam_segmented, efficientnet_b0/fused,
#                 efficientnet_b0/early_fused

# --- Final comparison across all 10 runs ---
python -m src.evaluation.aggregate_results
```

Or run one stage across ALL model/variant combos automatically:
```bash
python run_pipeline.py --only train
python run_pipeline.py --only evaluate
python run_pipeline.py --only confusion_matrix
python run_pipeline.py --only prediction_visualizer
python run_pipeline.py --only efficiency_benchmark
python run_pipeline.py --only training_curves
python run_pipeline.py --only aggregate_results
```

## 4. What gets generated

```
experiments/runs/<model>_<variant>/
    checkpoints/best.pt          - best model weights
    history.csv                  - per-epoch train/val loss & accuracy
    training_val_curves.png      - loss + accuracy plots
    overall_metrics.json         - accuracy, macro/weighted precision-recall-F1
    per_class_metrics.csv        - precision, recall, F1, support PER CLASS
    predictions.csv              - every test image's true/predicted label
    confusion_matrix.png / .csv
    efficiency_metrics.json      - Params(M), FLOPs(G), Size(MB), Latency(ms), FPS, Memory(MB)
    prediction_samples/
        correct/, incorrect/     - individual sample images
        grid_correct.png, grid_incorrect.png

results/
    tables/final_metrics_summary.csv       - all 6 runs, all metrics, one table
    tables/efficiency_summary.csv          - all 6 runs' efficiency numbers
    tables/per_class_metrics_all_runs.csv
    figures/accuracy_comparison_all_runs.png
    figures/per_class_f1_heatmap.png
    figures/efficiency_vs_accuracy_scatter.png
    figures/class_distribution.png         - EDA
```

## 5. Design notes (for your report)

- **Why RealWaste has no annotations → both generated variants now come from one SAM pass**:
  SAM's **automatic mask generator** runs directly on the **raw image** (no YOLO
  anywhere in the pipeline) and proposes many candidate masks; the best candidate
  is chosen by area + centeredness heuristics (see `pick_best_mask()` in
  `sam_auto_segment.py`).
  - `bbox_cropped`: uses that candidate's own bounding box (as reported by SAM's
    mask generator, in `bbox`), cropped from the **unmasked** raw image. Images
    with no valid SAM candidate fall back to a margin-based full-image box.
  - `sam_segmented`: uses the mask itself — background outside the mask is
    filled (white/black), and the crop box is instead re-derived from the mask
    pixels (`cv2.boundingRect`, padded 5%), which is typically tighter than SAM's
    raw candidate box.
  - `bbox_cropped` and `sam_segmented` therefore share the *same* underlying
    SAM detection per image; they differ only in whether pixels outside the
    object are kept (`bbox_cropped`) or blanked out (`sam_segmented`) — which is
    exactly the comparison RQ2 ("does masking out the background help, beyond
    just cropping to it?") is asking.
  - `detect_bboxes_yolo.py` still exists under `src/annotation_generation/` if
    you want to reproduce the old YOLO-based `bbox_cropped` as an extra
    comparison point, but it isn't run by `run_pipeline.py` by default anymore.
- **`fused` vs `early_fused`** (both combine `original` + `bbox_cropped`, no extra
  preprocessing step — they reuse the folders `generate_variants.py` already built):
  - `fused` (**late fusion**): each image gets its own pretrained backbone; the two
    pooled feature vectors are concatenated and passed through one new classifier
    head trained from scratch. Chosen as the default two-stream approach because
    `original` and `bbox_cropped` aren't spatially aligned (one is the full frame,
    one is a sub-region crop), so there's no pixel-level correspondence for early
    conv filters to exploit.
  - `early_fused` (**early/pixel fusion**): the two images are channel-stacked into
    one 6-channel input and passed through a *single* backbone, whose first conv
    layer is "inflated" from 3→6 input channels (pretrained 3-channel filters are
    tiled across the extra channels and rescaled, so training doesn't start from a
    randomly-initialized first layer). Cheaper (one backbone instead of two) and a
    useful ablation against `fused` — it tests whether the model can extract useful
    joint signal from the two images despite the spatial-misalignment concern above.
  - See `FusedTwoStreamModel` / `build_early_fused_model` in
    `src/models/model_factory.py` and `FusedWasteDataset` /
    `EarlyFusedWasteDataset` in `src/training/dataset_loader.py`.
- **Augmentation**: applied only to RealWaste's **training split**, on-the-fly via
  `torchvision.transforms` (flip, rotation, color jitter, random-resized-crop) — not
  pre-baked into files, so val/test stay clean and identical across all 3 variants.
- **Fair comparison**: all 5 variants (including `fused` and `early_fused`, which
  reuse `original` + `bbox_cropped`) and both models share **identical**
  train/val/test splits (`data/splits/realwaste_*.csv`), same seed, same hyperparameters.
- **Efficiency metrics** use `thop` for Params/FLOPs, checkpoint file size for Model
  Size, and timed forward passes (with warmup) for Latency/FPS/Memory — matching what's
  typically reported in edge-deployment papers.

## 6. Troubleshooting

- `kagglehub` download fails → manual fallback instructions print automatically
  (Kaggle: `joebeachcapital/realwaste`, or UCI: dataset #908).
- SAM checkpoint download slow/blocked → download `sam_vit_b_01ec64.pth` manually from
  the URL in `configs/config.yaml` (`sam.checkpoint_url`) and place it at
  `checkpoints/sam_vit_b_01ec64.pth`.
- `sam_auto_segment.py` is slower per image than box-prompted SAM (it proposes many
  candidate masks per image via a grid of points). If it's too slow, lower
  `sam.automatic_mask_generator.points_per_side` in `configs/config.yaml` (e.g. 16)
  — fewer candidate masks per image, faster but slightly less precise.
- Out of time → reduce `training.epochs` in `configs/config.yaml`, or run only
  `original` + `sam_segmented` variants (skip `bbox_cropped`) to cut runs from 6 to 4.
