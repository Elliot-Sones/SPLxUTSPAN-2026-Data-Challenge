"""
Create row-surgery submission candidates from Sub 2503 confidence diagnostics.

Policy:
- Start from submission_2503.
- Only modify rows flagged as uncertain / high risk.
- Use conservative fallback experts (2475, 2506) per target.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub2503 row surgery candidate generator")
    parser.add_argument("--base-sub", type=int, default=2503)
    parser.add_argument("--fallback-sub", type=int, default=2475)
    parser.add_argument("--angle-sub", type=int, default=2506)
    parser.add_argument(
        "--risk-csv",
        type=Path,
        default=OUTPUT_DIR / "sub2503_row_confidence_sub2503_prob_row_gate_20260215.csv",
    )
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--run-tag", type=str, default="")
    return parser.parse_args()


def load_sub(sub_num: int) -> pd.DataFrame:
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    df = pd.read_csv(path)
    need = {"id", *TARGET_COLS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df[["id", *TARGET_COLS]].copy()


def get_next_submission_number(submission_dir: Path) -> int:
    lock_path = submission_dir / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            existing = list(submission_dir.glob("submission_*.csv"))
            nums = [
                int(path.stem.split("_")[1])
                for path in existing
                if path.stem.split("_")[1].isdigit()
            ]
            next_num = max(nums) + 1 if nums else 1
            (submission_dir / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def apply_policy(
    *,
    name: str,
    base: pd.DataFrame,
    fb: pd.DataFrame,
    angle_sub: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    out = base.copy()

    uncertain = risk["confidence"] <= 0.20
    high_risk = risk["confidence"] <= 0.10
    ood = risk["risk_reasons"].fillna("").str.contains("ood_motion_pattern")
    hard_disagree = risk["risk_reasons"].fillna("").str.contains("hard_model_disagrees_with_base")
    p5 = risk["participant_id"] == 5

    idx_uncertain = risk.loc[uncertain, "id"]
    idx_high = risk.loc[high_risk, "id"]
    idx_ood = risk.loc[ood, "id"]
    idx_p5_risky = risk.loc[(uncertain & p5), "id"]
    idx_hard = risk.loc[(hard_disagree & uncertain), "id"]

    if name == "C1_uncertain_lr_fallback":
        out.loc[out["id"].isin(idx_uncertain), "scaled_left_right"] = fb.loc[
            fb["id"].isin(idx_uncertain), "scaled_left_right"
        ].values
        return out

    if name == "C2_uncertain_depth_lr_fallback":
        mask = out["id"].isin(idx_uncertain)
        out.loc[mask, "scaled_depth"] = fb.loc[mask, "scaled_depth"].values
        out.loc[mask, "scaled_left_right"] = fb.loc[mask, "scaled_left_right"].values
        return out

    if name == "C3_uncertain_angle2506_lr2475":
        mask = out["id"].isin(idx_uncertain)
        out.loc[mask, "scaled_angle"] = angle_sub.loc[mask, "scaled_angle"].values
        out.loc[mask, "scaled_left_right"] = fb.loc[mask, "scaled_left_right"].values
        return out

    if name == "C4_highrisk_all_fallback":
        mask = out["id"].isin(idx_high)
        for col in TARGET_COLS:
            out.loc[mask, col] = fb.loc[mask, col].values
        return out

    if name == "C5_p5_uncertain_depth_lr_fallback_plus_harddisagree_lr":
        mask_p5 = out["id"].isin(idx_p5_risky)
        out.loc[mask_p5, "scaled_depth"] = fb.loc[mask_p5, "scaled_depth"].values
        out.loc[mask_p5, "scaled_left_right"] = fb.loc[mask_p5, "scaled_left_right"].values

        mask_hard = out["id"].isin(idx_hard)
        out.loc[mask_hard, "scaled_left_right"] = fb.loc[mask_hard, "scaled_left_right"].values
        return out

    if name == "C6_ood_all_fallback_uncertain_angle2506":
        mask_ood = out["id"].isin(idx_ood)
        for col in TARGET_COLS:
            out.loc[mask_ood, col] = fb.loc[mask_ood, col].values

        mask_uncertain = out["id"].isin(idx_uncertain)
        out.loc[mask_uncertain, "scaled_angle"] = angle_sub.loc[mask_uncertain, "scaled_angle"].values
        return out

    raise ValueError(f"Unknown policy: {name}")


def summarize_delta(base: pd.DataFrame, cand: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in TARGET_COLS:
        out[f"mean_abs_delta_{col}"] = float(np.mean(np.abs(cand[col].values - base[col].values)))
        out[f"max_abs_delta_{col}"] = float(np.max(np.abs(cand[col].values - base[col].values)))
        out[f"num_changed_{col}"] = int(np.sum(cand[col].values != base[col].values))
    out["num_any_row_changed"] = int(
        np.sum(
            (cand[TARGET_COLS].values != base[TARGET_COLS].values).any(axis=1)
        )
    )
    return out


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag.strip() or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    base = load_sub(args.base_sub)
    fb = load_sub(args.fallback_sub)
    angle_sub = load_sub(args.angle_sub)

    if not np.array_equal(base["id"].values, fb["id"].values):
        raise ValueError("ID mismatch between base and fallback submissions")
    if not np.array_equal(base["id"].values, angle_sub["id"].values):
        raise ValueError("ID mismatch between base and angle submissions")

    risk = pd.read_csv(args.risk_csv)
    if "id" not in risk.columns:
        raise ValueError("risk csv missing id column")
    risk = risk.sort_values("id").reset_index(drop=True)
    base_sorted = base.sort_values("id").reset_index(drop=True)
    if not np.array_equal(risk["id"].values, base_sorted["id"].values):
        raise ValueError("risk csv id order/content mismatch with base submission ids")
    # restore original order
    risk = risk.set_index("id").loc[base["id"]].reset_index()

    policies = [
        "C1_uncertain_lr_fallback",
        "C2_uncertain_depth_lr_fallback",
        "C3_uncertain_angle2506_lr2475",
        "C4_highrisk_all_fallback",
        "C5_p5_uncertain_depth_lr_fallback_plus_harddisagree_lr",
        "C6_ood_all_fallback_uncertain_angle2506",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created: List[Dict[str, object]] = []

    for policy in policies:
        cand = apply_policy(
            name=policy,
            base=base,
            fb=fb,
            angle_sub=angle_sub,
            risk=risk,
        )

        sub_num = get_next_submission_number(SUBMISSION_DIR)
        out_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        cand.to_csv(out_path, index=False)

        stats = summarize_delta(base, cand)
        created.append(
            {
                "policy": policy,
                "submission": sub_num,
                "path": str(out_path),
                **stats,
            }
        )
        print(
            f"{policy}: submission_{sub_num}.csv "
            f"rows_changed={stats['num_any_row_changed']} "
            f"mean_abs_delta_angle={stats['mean_abs_delta_scaled_angle']:.12f} "
            f"mean_abs_delta_depth={stats['mean_abs_delta_scaled_depth']:.12f} "
            f"mean_abs_delta_left_right={stats['mean_abs_delta_scaled_left_right']:.12f}"
        )

    run_payload = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_sub": args.base_sub,
        "fallback_sub": args.fallback_sub,
        "angle_sub": args.angle_sub,
        "risk_csv": str(args.risk_csv),
        "policies": created,
    }
    run_json = OUTPUT_DIR / f"sub2503_row_surgery_run_{run_tag}.json"
    with run_json.open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    print(f"saved_run_json={run_json}")


if __name__ == "__main__":
    main()

