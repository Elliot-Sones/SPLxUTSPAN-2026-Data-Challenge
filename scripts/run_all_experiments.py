"""
Master experiment runner for comprehensive model strategy testing.

Runs all experiments from the plan and records results to Research/EXPERIMENT_RESULTS.md
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
RESEARCH_DIR = PROJECT_DIR / "Research"

# Experiments to run in order
EXPERIMENTS = [
    # Batch 1: Foundation + Quick Wins
    ("stability_feature_selection.py", "1.1 Stability-Adjusted Feature Selection"),
    ("multitask_model.py", "1.2 Multi-Task Learning"),
    ("polynomial_model.py", "2.2 Polynomial Interactions"),

    # Batch 2: Sequence and Uncertainty Models
    ("fourier_phase.py", "2.5 Fourier Phase Features"),
    ("gp_model.py", "2.1 Gaussian Processes"),

    # Batch 3: Ensemble and Representation
    ("subspace_ensemble.py", "1.3 Feature Subspace Ensemble"),

    # Batch 4: Moonshots (Neural-based)
    ("lstm_model.py", "2.3 LSTM Sequence Modeling"),
    ("contrastive_learning.py", "2.4 Contrastive Learning"),
    ("physics_constrained_nn.py", "3.1 Physics-Constrained NN"),
    ("maml_model.py", "3.2 Meta-Learning MAML"),
]


def run_experiment(script_name: str, description: str) -> dict:
    """Run a single experiment and capture results."""
    script_path = Path(__file__).parent / script_name

    result = {
        "script": script_name,
        "description": description,
        "status": "pending",
        "output": "",
        "error": "",
        "duration": 0,
        "best_cv": None,
        "profile_pass": None,
    }

    if not script_path.exists():
        result["status"] = "missing"
        result["error"] = f"Script not found: {script_path}"
        return result

    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        proc = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=PROJECT_DIR
        )

        result["output"] = proc.stdout
        result["error"] = proc.stderr
        result["status"] = "success" if proc.returncode == 0 else "failed"

        # Try to extract best CV from output
        for line in proc.stdout.split("\n"):
            if "LOPO MSE:" in line or "lopo_mse" in line.lower():
                try:
                    # Extract number
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = float(parts[-1].strip().split()[0])
                        if result["best_cv"] is None or val < result["best_cv"]:
                            result["best_cv"] = val
                except:
                    pass

            if "Profile check: PASS" in line:
                result["profile_pass"] = True
            elif "Profile check: FAIL" in line:
                result["profile_pass"] = False

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment timed out after 10 minutes"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    result["duration"] = time.time() - start_time
    return result


def write_results_markdown(results: list, output_path: Path):
    """Write results to markdown file."""
    with open(output_path, "w") as f:
        f.write("# Comprehensive Model Strategy Experiments\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Baseline**: LB 0.007809 (Sub 133 blend)\n")
        f.write("**Target**: LB 0.007\n\n")

        f.write("---\n\n")
        f.write("## Summary Table\n\n")
        f.write("| # | Experiment | Status | Best LOPO CV | Profile | Duration |\n")
        f.write("|---|------------|--------|--------------|---------|----------|\n")

        for i, r in enumerate(results, 1):
            cv_str = f"{r['best_cv']:.6f}" if r['best_cv'] else "N/A"
            profile_str = "PASS" if r['profile_pass'] else ("FAIL" if r['profile_pass'] is False else "N/A")
            dur_str = f"{r['duration']:.1f}s"
            f.write(f"| {i} | {r['description']} | {r['status']} | {cv_str} | {profile_str} | {dur_str} |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Results\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"### {i}. {r['description']}\n\n")
            f.write(f"**Script**: `{r['script']}`\n\n")
            f.write(f"**Status**: {r['status']}\n\n")
            f.write(f"**Duration**: {r['duration']:.1f}s\n\n")

            if r['best_cv']:
                f.write(f"**Best LOPO CV**: {r['best_cv']:.6f}\n\n")

            if r['profile_pass'] is not None:
                f.write(f"**Profile Check**: {'PASS' if r['profile_pass'] else 'FAIL'}\n\n")

            if r['error'] and r['status'] != 'success':
                f.write("**Error**:\n```\n")
                f.write(r['error'][:500])  # Truncate long errors
                f.write("\n```\n\n")

            # Key output lines
            if r['output']:
                f.write("**Key Output**:\n```\n")
                # Extract important lines
                key_lines = []
                for line in r['output'].split("\n"):
                    if any(kw in line.lower() for kw in ["best", "mse", "improvement", "profile", "summary"]):
                        key_lines.append(line)
                f.write("\n".join(key_lines[-20:]))  # Last 20 key lines
                f.write("\n```\n\n")

            f.write("---\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        successful = [r for r in results if r['status'] == 'success' and r['best_cv']]
        if successful:
            best = min(successful, key=lambda x: x['best_cv'])
            f.write(f"**Best performing experiment**: {best['description']}\n")
            f.write(f"**Best LOPO CV**: {best['best_cv']:.6f}\n\n")

            passing = [r for r in successful if r['profile_pass']]
            if passing:
                best_passing = min(passing, key=lambda x: x['best_cv'])
                f.write(f"**Best with passing profile**: {best_passing['description']}\n")
                f.write(f"**LOPO CV**: {best_passing['best_cv']:.6f}\n\n")

        failed = [r for r in results if r['status'] in ['failed', 'error', 'timeout']]
        if failed:
            f.write(f"**Failed experiments**: {len(failed)}\n")
            for r in failed:
                f.write(f"- {r['description']}: {r['status']}\n")


def main():
    print("=" * 60)
    print("COMPREHENSIVE MODEL STRATEGY EXPERIMENTS")
    print("=" * 60)
    print(f"Running {len(EXPERIMENTS)} experiments...")
    print(f"Results will be saved to: {RESEARCH_DIR / 'EXPERIMENT_RESULTS.md'}")

    results = []

    for script_name, description in EXPERIMENTS:
        result = run_experiment(script_name, description)
        results.append(result)

        print(f"\n{description}:")
        print(f"  Status: {result['status']}")
        if result['best_cv']:
            print(f"  Best CV: {result['best_cv']:.6f}")
        if result['profile_pass'] is not None:
            print(f"  Profile: {'PASS' if result['profile_pass'] else 'FAIL'}")
        print(f"  Duration: {result['duration']:.1f}s")

    # Write markdown report
    write_results_markdown(results, RESEARCH_DIR / "EXPERIMENT_RESULTS.md")
    print(f"\n\nResults saved to: {RESEARCH_DIR / 'EXPERIMENT_RESULTS.md'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r['status'] == 'success']
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        best = min([r for r in successful if r['best_cv']], key=lambda x: x['best_cv'], default=None)
        if best:
            print(f"Best CV: {best['best_cv']:.6f} ({best['description']})")


if __name__ == "__main__":
    main()
