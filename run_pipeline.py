"""
Master orchestrator - runs the entire SAM-Assisted waste-classification pipeline end-to-end
(dataset configured in configs/config.yaml, currently TrashNet).

Stages:
    1. download_dataset          - fetch RealWaste (no annotations)
    2. clean_and_split           - stratified train/val/test split
    3. dataset_stats             - EDA report
    4. download_sam_checkpoint   - fetch SAM ViT-B weights
    5. sam_auto_segment          - single SAM pass over raw images that produces BOTH:
                                    (a) segmentation masks (used ONLY by sam_segmented), and
                                    (b) bounding boxes derived from those masks (used ONLY by
                                    bbox_cropped). YOLO is no longer used anywhere in the
                                    default pipeline.
    6. generate_variants         - build original / bbox_cropped / sam_segmented datasets
                                    (`fused` and `early_fused` need no separate build step -
                                    they're training-time variants that reuse original +
                                    bbox_cropped directly)
    7. train                     - train MobileNetV3 & EfficientNet-B0 on all 5 variants
                                    (original, bbox_cropped, sam_segmented, fused,
                                    early_fused = 10 runs)
    8. evaluate                  - test-set metrics for all 10 runs
    9. confusion_matrix          - confusion matrices for all 10 runs
   10. prediction_visualizer     - sample prediction images for all 10 runs
   11. efficiency_benchmark      - Params/FLOPs/Size/Latency/FPS/Memory for all 10 runs
   12. training_curves           - loss/accuracy plots for all 10 runs
   13. aggregate_results         - final comparison tables + figures

(detect_bboxes_yolo.py still exists in src/annotation_generation/ if you ever
want to reproduce the old YOLO-based bbox_cropped variant for comparison, but
it is not part of this pipeline by default.)

Usage:
    python run_pipeline.py                     # run everything
    python run_pipeline.py --skip-data          # skip stages 1-6 (already generated)
    python run_pipeline.py --only train         # run only the training stage
    python run_pipeline.py --start-from train   # resume from a given stage
"""
import argparse
import subprocess
import sys

import yaml

MODELS = ["mobilenet_v3", "efficientnet_b0"]
# `fused` (two-stream late-fusion) and `early_fused` (single-backbone,
# 6-channel early-fusion) both combine original + bbox_cropped - they are
# TRAINING-time variants only, not preprocessing/image variants, so neither
# is in configs/config.yaml's `variants:` list (which drives generate_variants.py).
VARIANTS = ["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"]

DATA_STAGES = [
    ("download_dataset", ["python", "-m", "src.data_prep.download_dataset"]),
    ("clean_and_split", ["python", "-m", "src.data_prep.clean_and_split"]),
    ("dataset_stats", ["python", "-m", "src.data_prep.dataset_stats"]),
    ("download_sam_checkpoint", ["python", "-m", "src.annotation_generation.download_sam_checkpoint"]),
    ("sam_auto_segment", ["python", "-m", "src.annotation_generation.sam_auto_segment"]),
    ("generate_variants", ["python", "-m", "src.preprocessing.generate_variants"]),
]


def run(cmd, label):
    print(f"\n{'=' * 70}\n>>> STAGE: {label}\n{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAILED] Stage '{label}' exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip dataset/annotation/preprocessing stages (1-7)")
    parser.add_argument("--only", default=None,
                        help="Run only this stage name, e.g. 'train', 'evaluate'")
    args = parser.parse_args()

    if args.only:
        if args.only in dict(DATA_STAGES):
            run(dict(DATA_STAGES)[args.only], args.only)
        elif args.only in ["train", "evaluate", "confusion_matrix",
                           "prediction_visualizer", "efficiency_benchmark", "training_curves"]:
            module_map = {
                "train": "src.training.train",
                "evaluate": "src.evaluation.evaluate",
                "confusion_matrix": "src.evaluation.confusion_matrix",
                "prediction_visualizer": "src.evaluation.prediction_visualizer",
                "efficiency_benchmark": "src.evaluation.efficiency_benchmark",
                "training_curves": "src.evaluation.training_curves",
            }
            for model in MODELS:
                for variant in VARIANTS:
                    run(["python", "-m", module_map[args.only],
                        "--model", model, "--variant", variant],
                        f"{args.only} [{model} / {variant}]")
        elif args.only == "aggregate_results":
            run(["python", "-m", "src.evaluation.aggregate_results"], "aggregate_results")
        else:
            print(f"Unknown stage: {args.only}")
            sys.exit(1)
        return

    # ---- Full pipeline ----
    if not args.skip_data:
        for label, cmd in DATA_STAGES:
            run(cmd, label)

    for model in MODELS:
        for variant in VARIANTS:
            run(["python", "-m", "src.training.train", "--model", model, "--variant", variant],
                f"train [{model} / {variant}]")
            run(["python", "-m", "src.evaluation.evaluate", "--model", model, "--variant", variant],
                f"evaluate [{model} / {variant}]")
            run(["python", "-m", "src.evaluation.confusion_matrix", "--model", model, "--variant", variant],
                f"confusion_matrix [{model} / {variant}]")
            run(["python", "-m", "src.evaluation.prediction_visualizer", "--model", model, "--variant", variant],
                f"prediction_visualizer [{model} / {variant}]")
            run(["python", "-m", "src.evaluation.efficiency_benchmark", "--model", model, "--variant", variant],
                f"efficiency_benchmark [{model} / {variant}]")
            run(["python", "-m", "src.evaluation.training_curves", "--model", model, "--variant", variant],
                f"training_curves [{model} / {variant}]")

    run(["python", "-m", "src.evaluation.aggregate_results"], "aggregate_results")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE. Check results/ for final tables & figures,")
    print("and experiments/runs/<model>_<variant>/ for per-run outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
