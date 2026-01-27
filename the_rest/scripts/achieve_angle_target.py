#!/usr/bin/env python3
"""
Main orchestration script to achieve angle MSE < 0.01 (RMSE < 0.1 degrees).

Runs all phases sequentially:
1. Baseline: Current multi-output model
2. Phase 1: Angle-specific model with domain features
3. Phase 2: Add trajectory features
4. Phase 3: Ensemble of diverse models
5. Phase 4: Hyperparameter optimization (optional)

Tracks progress toward target and reports results.
"""

import numpy as np
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TARGET_MSE = 0.01
TARGET_RMSE = np.sqrt(TARGET_MSE)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_results(phase: str, mse: float, improvement: float = None, baseline_mse: float = None):
    """Print phase results."""
    rmse = np.sqrt(mse)

    print(f"\n{phase} Results:")
    print(f"  Val MSE:  {mse:.6f} (RMSE: {rmse:.4f}°)")

    if baseline_mse is not None:
        improvement_pct = (1 - mse / baseline_mse) * 100
        if mse > 0:
            improvement_x = baseline_mse / mse
            print(f"  vs Baseline: {improvement_pct:+.1f}% ({improvement_x:.2f}x improvement)")
        else:
            print(f"  vs Baseline: {improvement_pct:+.1f}% (perfect prediction!)")

    # Progress toward target
    if mse < TARGET_MSE:
        print(f"  TARGET ACHIEVED! (MSE < {TARGET_MSE})")
    else:
        needed_improvement = mse / TARGET_MSE
        print(f"  Target: MSE < {TARGET_MSE} (RMSE < {TARGET_RMSE:.4f}°)")
        print(f"  Need {needed_improvement:.1f}x further improvement")


def run_baseline():
    """Run baseline: quick test to get current performance."""
    print_header("BASELINE: Current Multi-Output Model")

    print("Running quick baseline test...")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "src/quick_test.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output for MSE
        output = result.stdout
        print(output)

        # Extract angle MSE (this is approximate)
        # Look for pattern like "angle: MSE=X.XX"
        for line in output.split("\n"):
            if "angle" in line.lower() and "mse" in line.lower():
                # Try to extract number
                import re
                match = re.search(r"mse[=:\s]+([0-9.]+)", line.lower())
                if match:
                    mse = float(match.group(1))
                    return mse

        # Fallback: assume ~4.70 from exploration
        print("Could not parse MSE from output. Using estimated baseline: 4.70")
        return 4.70

    except Exception as e:
        print(f"Baseline test failed: {e}")
        print("Using estimated baseline MSE: 4.70")
        return 4.70


def run_phase1():
    """Phase 1: Angle-specific model with all features."""
    print_header("PHASE 1: Angle-Specific Model")

    print("Training XGBoost angle-specific model with top 100 features...")

    try:
        result = subprocess.run(
            [
                "uv", "run", "python", "src/train_angle_model.py",
                "--model", "xgboost",
                "--n-features", "100",
                "--use-physics",
                "--use-engineering",
                "--use-angle-specific"
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # Parse validation MSE from output
        output = result.stdout
        for line in output.split("\n"):
            if "Mean Val MSE" in line:
                import re
                match = re.search(r"([0-9.]+)", line)
                if match:
                    return float(match.group(1))

        return None

    except Exception as e:
        print(f"Phase 1 failed: {e}")
        return None


def run_phase2():
    """Phase 2: Angle-specific model focusing on angle features."""
    print_header("PHASE 2: Focus on Angle-Specific Features")

    print("Training XGBoost with angle-specific features + physics...")

    try:
        result = subprocess.run(
            [
                "uv", "run", "python", "src/train_angle_model.py",
                "--model", "xgboost",
                "--n-features", "50",  # Fewer features, more focused
                "--use-physics",
                "--use-angle-specific"
                # No --use-engineering: focus on angle features
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # Parse MSE
        output = result.stdout
        for line in output.split("\n"):
            if "Mean Val MSE" in line:
                import re
                match = re.search(r"([0-9.]+)", line)
                if match:
                    return float(match.group(1))

        return None

    except Exception as e:
        print(f"Phase 2 failed: {e}")
        return None


def run_phase3():
    """Phase 3: Ensemble of diverse models."""
    print_header("PHASE 3: Ensemble of Diverse Models")

    print("Training ensemble (5 models)...")
    print("  1. XGBoost (top 100 features)")
    print("  2. LightGBM (top 100 features)")
    print("  3. XGBoost (angle-specific only)")
    print("  4. Ridge (polynomial)")
    print("  5. XGBoost (physics only)")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "src/ensemble_angle.py"],
            capture_output=True,
            text=True,
            timeout=1200  # 20 minutes
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # Parse ensemble MSE
        output = result.stdout
        for line in output.split("\n"):
            if "Ensemble Val MSE" in line:
                import re
                match = re.search(r"([0-9.]+)", line)
                if match:
                    return float(match.group(1))

        return None

    except Exception as e:
        print(f"Phase 3 failed: {e}")
        return None


def run_phase4_optimization():
    """Phase 4 (Optional): Hyperparameter optimization."""
    print_header("PHASE 4: Hyperparameter Optimization (Optional)")

    print("Running Bayesian optimization...")
    print("This may take 30-60 minutes for 100 trials.")

    response = input("Run optimization? (y/n): ")
    if response.lower() != "y":
        print("Skipping optimization.")
        return None

    try:
        result = subprocess.run(
            [
                "uv", "run", "python", "src/optimize_angle_model.py",
                "--model", "xgboost",
                "--method", "bayesian",
                "--n-trials", "100",
                "--n-features", "100"
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # After optimization, retrain with best params
        print("\nRetraining with optimized hyperparameters...")
        # TODO: Load best params and retrain

        return None

    except Exception as e:
        print(f"Phase 4 failed: {e}")
        return None


def main():
    """Main orchestration."""
    print_header("ACHIEVE ANGLE MSE < 0.01")
    print(f"\nTarget: Angle MSE < {TARGET_MSE} (RMSE < {TARGET_RMSE:.4f} degrees)")
    print("Current estimate: MSE ~4.70 (RMSE ~2.17 degrees)")
    print("Required improvement: ~470x reduction in MSE")

    results = {}

    # Baseline
    baseline_mse = run_baseline()
    results["baseline"] = baseline_mse
    print_results("Baseline", baseline_mse)

    # Phase 1
    phase1_mse = run_phase1()
    if phase1_mse is not None:
        results["phase1"] = phase1_mse
        print_results("Phase 1", phase1_mse, baseline_mse=baseline_mse)

        if phase1_mse < TARGET_MSE:
            print("\nTARGET ACHIEVED IN PHASE 1!")
            return
    else:
        print("Phase 1 failed. Stopping.")
        return

    # Phase 2
    phase2_mse = run_phase2()
    if phase2_mse is not None:
        results["phase2"] = phase2_mse
        print_results("Phase 2", phase2_mse, baseline_mse=baseline_mse)

        if phase2_mse < TARGET_MSE:
            print("\nTARGET ACHIEVED IN PHASE 2!")
            return

    # Phase 3
    phase3_mse = run_phase3()
    if phase3_mse is not None:
        results["phase3"] = phase3_mse
        print_results("Phase 3 (Ensemble)", phase3_mse, baseline_mse=baseline_mse)

        if phase3_mse < TARGET_MSE:
            print("\nTARGET ACHIEVED IN PHASE 3!")
            return

    # Phase 4 (optional)
    # run_phase4_optimization()

    # Final summary
    print_header("FINAL SUMMARY")
    print(f"\nTarget: MSE < {TARGET_MSE} (RMSE < {TARGET_RMSE:.4f}°)")
    print("\nResults:")
    for phase, mse in results.items():
        rmse = np.sqrt(mse)
        status = "ACHIEVED" if mse < TARGET_MSE else "not achieved"
        print(f"  {phase:15} MSE: {mse:.6f} (RMSE: {rmse:.4f}°) - {status}")

    best_phase = min(results, key=results.get)
    best_mse = results[best_phase]

    print(f"\nBest result: {best_phase} with MSE = {best_mse:.6f}")

    if best_mse < TARGET_MSE:
        print("\nSUCCESS! Target achieved.")
    else:
        improvement_needed = best_mse / TARGET_MSE
        print(f"\nTarget not achieved. Need {improvement_needed:.1f}x further improvement.")
        print("\nNext steps:")
        print("  1. Run hyperparameter optimization (Phase 4)")
        print("  2. Add data augmentation")
        print("  3. Try per-participant calibration")
        print("  4. Investigate systematic errors in predictions")


if __name__ == "__main__":
    main()
