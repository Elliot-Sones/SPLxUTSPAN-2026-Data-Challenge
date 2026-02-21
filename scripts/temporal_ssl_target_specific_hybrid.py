"""
Target-specific hybrid submissions:
- Angle and depth blended between base and temporal SSL submissions
- Left-right anchored to base submission (weight 0 on SSL by default)

This script creates reproducible submission files and writes exact run artifacts.
"""

import argparse
import fcntl
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create target-specific temporal SSL hybrid submissions")
    p.add_argument("--base-submission", type=int, default=2503)
    p.add_argument("--ssl-submission", type=int, default=2591)
    p.add_argument(
        "--angle-weights",
        type=str,
        default="0.10,0.20,0.30,0.50,1.00",
        help="Comma-separated SSL weights for angle",
    )
    p.add_argument(
        "--depth-weights",
        type=str,
        default="0.10,0.20,0.30,0.50,1.00",
        help="Comma-separated SSL weights for depth",
    )
    p.add_argument(
        "--lr-weight",
        type=float,
        default=0.0,
        help="SSL weight for left_right (default 0.0 keeps base left_right)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    angle_weights = parse_float_list(args.angle_weights)
    depth_weights = parse_float_list(args.depth_weights)
    if len(angle_weights) != len(depth_weights):
        raise ValueError("angle-weights and depth-weights must have the same length.")

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    ssl_path = SUBMISSION_DIR / f"submission_{args.ssl_submission}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_path}")
    if not ssl_path.exists():
        raise FileNotFoundError(f"SSL submission not found: {ssl_path}")

    base = pd.read_csv(base_path)
    ssl = pd.read_csv(ssl_path)

    created = []
    for wa, wd in zip(angle_weights, depth_weights):
        out = base.copy()
        out["scaled_angle"] = np.clip(
            wa * ssl["scaled_angle"].values + (1.0 - wa) * base["scaled_angle"].values,
            0.0,
            1.0,
        )
        out["scaled_depth"] = np.clip(
            wd * ssl["scaled_depth"].values + (1.0 - wd) * base["scaled_depth"].values,
            0.0,
            1.0,
        )
        out["scaled_left_right"] = np.clip(
            args.lr_weight * ssl["scaled_left_right"].values
            + (1.0 - args.lr_weight) * base["scaled_left_right"].values,
            0.0,
            1.0,
        )

        sub_num = get_next_submission_number()
        out_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        out.to_csv(out_path, index=False)

        # Diagnostics vs base
        corr = {}
        mae = {}
        for col in TARGET_COLS:
            corr[col] = float(np.corrcoef(base[col].values, out[col].values)[0, 1])
            mae[col] = float(np.mean(np.abs(base[col].values - out[col].values)))

        created.append(
            {
                "submission_number": sub_num,
                "path": str(out_path),
                "weights": {
                    "angle_ssl": float(wa),
                    "depth_ssl": float(wd),
                    "left_right_ssl": float(args.lr_weight),
                    "base_submission": int(args.base_submission),
                    "ssl_submission": int(args.ssl_submission),
                },
                "vs_base": {"corr": corr, "mae": mae},
            }
        )
        print(
            f"Created submission_{sub_num}.csv - "
            f"w_angle={wa:.6f}, w_depth={wd:.6f}, w_lr={args.lr_weight:.6f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"temporal_ssl_target_hybrid_run_{ts}.json"
    run_md = OUTPUT_DIR / f"temporal_ssl_target_hybrid_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "base_submission": args.base_submission,
            "ssl_submission": args.ssl_submission,
            "angle_weights": angle_weights,
            "depth_weights": depth_weights,
            "lr_weight": args.lr_weight,
        },
        "created_submissions": created,
    }
    run_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Temporal SSL Target-specific Hybrid Run",
        "",
        f"- Timestamp: `{ts}`",
        f"- Command: `{payload['command']}`",
        "",
        "## Config",
        f"- base_submission: `{args.base_submission}`",
        f"- ssl_submission: `{args.ssl_submission}`",
        f"- angle_weights: `{angle_weights}`",
        f"- depth_weights: `{depth_weights}`",
        f"- lr_weight: `{args.lr_weight}`",
        "",
        "## Created Submissions",
    ]
    for row in created:
        lines.append(f"- {row}")
    run_md.write_text("\n".join(lines) + "\n")

    print(f"Run JSON: {run_json}")
    print(f"Run details: {run_md}")


if __name__ == "__main__":
    main()
