"""
Collects overall_metrics.json + efficiency_metrics.json from EVERY run and
builds the final comparison tables/figures for the report:

    results/tables/final_metrics_summary.csv
    results/tables/efficiency_summary.csv
    results/figures/accuracy_comparison_all_runs.png
    results/figures/per_class_f1_heatmap.png
    results/figures/efficiency_vs_accuracy_scatter.png

Usage (after all 10 runs are trained + evaluated + benchmarked):
    python -m src.evaluation.aggregate_results
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("aggregate_results")


def main():
    cfg = load_config()
    exp_dir = resolve_path(cfg["paths"]["experiments_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    overall_rows, efficiency_rows, per_class_rows = [], [], []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        om = run_dir / "overall_metrics.json"
        em = run_dir / "efficiency_metrics.json"
        pcm = run_dir / "per_class_metrics.csv"

        if om.exists():
            with open(om) as f:
                overall_rows.append(json.load(f))
        if em.exists():
            with open(em) as f:
                efficiency_rows.append(json.load(f))
        if pcm.exists():
            df = pd.read_csv(pcm)
            df["run_name"] = run_dir.name
            per_class_rows.append(df)

    if not overall_rows:
        logger.error("No overall_metrics.json found under %s. "
                      "Run train.py + evaluate.py for each (model, variant) first.", exp_dir)
        return

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(results_dir / "tables" / "final_metrics_summary.csv", index=False)

    if efficiency_rows:
        eff_df = pd.DataFrame(efficiency_rows)
        eff_df.to_csv(results_dir / "tables" / "efficiency_summary.csv", index=False)
    else:
        eff_df = pd.DataFrame()

    # ---- Figure 1: Accuracy comparison across all runs ----
    plt.figure(figsize=(10, 5))
    order = overall_df.sort_values("accuracy", ascending=False)
    sns.barplot(data=order, x="run_name", y="accuracy", hue="variant", dodge=False)
    plt.title("Test Accuracy Comparison Across All Runs")
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "accuracy_comparison_all_runs.png", dpi=150)
    plt.close()

    # ---- Figure 2: Per-class F1 heatmap (rows=class, cols=run) ----
    if per_class_rows:
        pc_all = pd.concat(per_class_rows, ignore_index=True)
        pivot = pc_all.pivot(index="class", columns="run_name", values="f1_score")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Per-Class F1-Score Across All Runs")
        plt.tight_layout()
        plt.savefig(results_dir / "figures" / "per_class_f1_heatmap.png", dpi=150)
        plt.close()
        pc_all.to_csv(results_dir / "tables" / "per_class_metrics_all_runs.csv", index=False)

    # ---- Figure 3: Efficiency vs accuracy tradeoff ----
    if not eff_df.empty:
        merged = overall_df.merge(eff_df, on=["run_name", "model", "variant"], suffixes=("", "_eff"))
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=merged, x="inference_time_ms", y="accuracy",
                        hue="model", style="variant", s=150)
        for _, row in merged.iterrows():
            plt.annotate(row["variant"], (row["inference_time_ms"], row["accuracy"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        plt.title("Accuracy vs Inference Latency (bubble = model/variant)")
        plt.xlabel("Inference Time (ms)")
        plt.ylabel("Test Accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "figures" / "efficiency_vs_accuracy_scatter.png", dpi=150)
        plt.close()
        merged.to_csv(results_dir / "tables" / "accuracy_efficiency_merged.csv", index=False)

    logger.info("Aggregated %d runs.", len(overall_df))
    logger.info("\n%s", overall_df[["run_name", "accuracy", "macro_f1", "weighted_f1"]]
                .sort_values("accuracy", ascending=False).to_string(index=False))
    logger.info("Saved comparison tables/figures to %s", results_dir)


if __name__ == "__main__":
    main()
