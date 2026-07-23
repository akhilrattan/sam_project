"""
Benchmarks a trained model's deployment efficiency:
    - Parameters (M)
    - FLOPs (G)
    - Model size on disk (MB)
    - Inference time (ms) per image
    - FPS (frames per second)
    - Memory usage (MB) during inference

Saves: experiments/runs/<run>/efficiency_metrics.json

Usage:
    python -m src.evaluation.efficiency_benchmark --model mobilenet_v3 --variant original
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger
from src.models.model_factory import build_model, build_fused_model, build_early_fused_model

logger = get_logger("efficiency_benchmark")


def get_params_flops(model, input_size):
    from thop import profile
    dummy = torch.randn(1, *input_size)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops_g = (2 * macs) / 1e9   # MACs -> FLOPs (approx: 1 MAC = 2 FLOPs)
    params_m = params / 1e6
    return round(params_m, 3), round(flops_g, 3)


def benchmark_latency(model, input_size, device, n_runs, warmup):
    model.eval().to(device)
    dummy = torch.randn(1, *input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time_s = end - start
    avg_ms = (total_time_s / n_runs) * 1000
    fps = n_runs / total_time_s
    return round(avg_ms, 3), round(fps, 2)


def measure_memory(model, input_size, device):
    dummy = torch.randn(1, *input_size).to(device)
    model.eval().to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return round(peak_mb, 2)
    else:
        import psutil, os
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)
        with torch.no_grad():
            _ = model(dummy)
        mem_after = process.memory_info().rss / (1024 ** 2)
        return round(max(mem_after - mem_before, 0.0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--variant", required=True,
                        choices=["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"])
    args = parser.parse_args()

    cfg = load_config()
    run_name = f"{args.model}_{args.variant}"
    run_dir = resolve_path(cfg["paths"]["experiments_dir"]) / run_name
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s. Train this run first.", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = len(ckpt["classes"])

    if args.variant == "fused":
        model = build_fused_model(args.model, num_classes)
    elif args.variant == "early_fused":
        model = build_early_fused_model(args.model, num_classes)
    else:
        model = build_model(args.model, num_classes)
    model.load_state_dict(ckpt["state_dict"])

    img_size = cfg["dataset"]["image_size"]
    # fused takes a stacked (2, C, H, W) input per sample - see FusedWasteDataset.
    # early_fused takes a single channel-concatenated (6, H, W) input - see
    # EarlyFusedWasteDataset.
    if args.variant == "fused":
        input_size = (2, 3, img_size, img_size)
    elif args.variant == "early_fused":
        input_size = (6, img_size, img_size)
    else:
        input_size = (3, img_size, img_size)

    if args.variant == "fused":
        build_fn = build_fused_model
    elif args.variant == "early_fused":
        build_fn = build_early_fused_model
    else:
        build_fn = build_model
    params_m, flops_g = get_params_flops(build_fn(args.model, num_classes), input_size)
    model_size_mb = round(ckpt_path.stat().st_size / (1024 ** 2), 2)
    avg_ms, fps = benchmark_latency(
        model, input_size, device,
        n_runs=cfg["evaluation"]["benchmark_runs"],
        warmup=cfg["evaluation"]["benchmark_warmup"],
    )
    memory_mb = measure_memory(model, input_size, device)

    metrics = {
        "run_name": run_name, "model": args.model, "variant": args.variant,
        "device": str(device),
        "params_M": params_m,
        "flops_G": flops_g,
        "model_size_MB": model_size_mb,
        "inference_time_ms": avg_ms,
        "fps": fps,
        "memory_usage_MB": memory_mb,
        "input_size": list(input_size),
    }
    with open(run_dir / "efficiency_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("[%s] Params: %.2fM | FLOPs: %.2fG | Size: %.2fMB | Latency: %.2fms | "
                "FPS: %.1f | Memory: %.2fMB",
                run_name, params_m, flops_g, model_size_mb, avg_ms, fps, memory_mb)


if __name__ == "__main__":
    main()
