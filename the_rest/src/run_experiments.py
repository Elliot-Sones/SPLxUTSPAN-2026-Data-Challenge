"""
Experiment runner for SPLxUTSPAN 2026 Data Challenge.

Runs multiple experiments and compares results to find the best approach.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np


OUTPUT_DIR = Path(__file__).parent.parent / "output"


def run_experiment(name: str, command: str) -> dict:
    """Run a single experiment and return results."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.time() - start_time

        return {
            "name": name,
            "command": command,
            "status": "success" if result.returncode == 0 else "failed",
            "elapsed_seconds": elapsed,
            "stdout": result.stdout[-5000:],  # Last 5000 chars
            "stderr": result.stderr[-2000:] if result.returncode != 0 else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "command": command,
            "status": "timeout",
            "elapsed_seconds": 3600,
            "stdout": "",
            "stderr": "Experiment timed out after 1 hour",
        }
    except Exception as e:
        return {
            "name": name,
            "command": command,
            "status": "error",
            "elapsed_seconds": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
        }


def parse_mse_from_output(output: str) -> dict:
    """Extract MSE values from training output."""
    import re

    result = {
        "mean_mse": None,
        "angle_mse": None,
        "depth_mse": None,
        "left_right_mse": None,
        "scaled_mse": None,
    }

    # Look for patterns like "Mean MSE: 40.8844"
    mean_match = re.search(r"Mean MSE:\s*([\d.]+)", output)
    if mean_match:
        result["mean_mse"] = float(mean_match.group(1))

    # Look for scaled MSE
    scaled_match = re.search(r"Scaled MSE.*?:\s*([\d.]+)", output)
    if scaled_match:
        result["scaled_mse"] = float(scaled_match.group(1))

    # Look for per-target MSE
    angle_match = re.search(r"angle.*?:\s*([\d.]+)", output, re.IGNORECASE)
    if angle_match:
        result["angle_mse"] = float(angle_match.group(1))

    depth_match = re.search(r"depth.*?:\s*([\d.]+)", output, re.IGNORECASE)
    if depth_match:
        result["depth_mse"] = float(depth_match.group(1))

    lr_match = re.search(r"left_right.*?:\s*([\d.]+)", output, re.IGNORECASE)
    if lr_match:
        result["left_right_mse"] = float(lr_match.group(1))

    return result


def main():
    parser = argparse.ArgumentParser(description="Run experiments for SPLxUTSPAN 2026")
    parser.add_argument("--quick", action="store_true", help="Run quick experiments only")
    parser.add_argument("--full", action="store_true", help="Run full experiment suite")
    args = parser.parse_args()

    experiments = []

    if args.quick:
        # Quick experiments for testing
        experiments = [
            ("XGBoost GPU (default)", "python src/train_gpu.py --model xgboost_gpu"),
            ("LightGBM GPU (default)", "python src/train_gpu.py --model lightgbm_gpu"),
        ]
    elif args.full:
        # Full experiment suite
        experiments = [
            # Gradient boosting experiments
            ("XGBoost GPU (default)", "python src/train_gpu.py --model xgboost_gpu"),
            ("XGBoost GPU (tuned)", "python src/train_gpu.py --model xgboost_gpu --tune --trials 50"),
            ("LightGBM GPU (default)", "python src/train_gpu.py --model lightgbm_gpu"),
            ("LightGBM GPU (tuned)", "python src/train_gpu.py --model lightgbm_gpu --tune --trials 50"),

            # Deep learning experiments
            ("CNN-LSTM", "python src/train_gpu.py --model cnn_lstm"),
            ("Transformer", "python src/train_gpu.py --model transformer"),
        ]
    else:
        print("Specify --quick or --full to run experiments")
        print("  --quick: Run XGBoost and LightGBM with default params")
        print("  --full:  Run all models including tuning and deep learning")
        return

    # Run experiments
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Running {len(experiments)} experiments...")

    for name, command in experiments:
        result = run_experiment(name, command)
        result["mse"] = parse_mse_from_output(result["stdout"])
        all_results.append(result)

        # Print quick summary
        if result["status"] == "success":
            mse = result["mse"]["mean_mse"]
            scaled = result["mse"]["scaled_mse"]
            print(f"  -> MSE: {mse}, Scaled: {scaled}")
        else:
            print(f"  -> {result['status']}: {result['stderr'][:100]}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Status':<10} {'MSE':<10} {'Scaled MSE':<12} {'Time':<10}")
    print("-" * 70)

    leader_score = 0.008381
    for r in all_results:
        status = r["status"]
        mse = r["mse"]["mean_mse"]
        scaled = r["mse"]["scaled_mse"]
        elapsed = r["elapsed_seconds"]

        mse_str = f"{mse:.4f}" if mse else "N/A"
        scaled_str = f"{scaled:.6f}" if scaled else "N/A"
        time_str = f"{elapsed/60:.1f}m"

        print(f"{r['name']:<30} {status:<10} {mse_str:<10} {scaled_str:<12} {time_str:<10}")

    # Best result
    valid_results = [r for r in all_results if r["mse"]["scaled_mse"] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x["mse"]["scaled_mse"])
        print(f"\nBest: {best['name']} with scaled MSE = {best['mse']['scaled_mse']:.6f}")
        print(f"Gap to leader ({leader_score}): {best['mse']['scaled_mse']/leader_score:.2f}x")

    # Save results
    results_file = OUTPUT_DIR / f"experiments_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
