import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_DIR / "submission"
RESEARCH_SUMMARY = PROJECT_DIR / "Research" / "COMPLETE_RESEARCH_SUMMARY.md"
DATA_DIR = PROJECT_DIR / "data"


SCALED_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


@dataclass(frozen=True)
class SummaryRow:
    sub: int
    lb_score: float | None
    angle_mean: float
    angle_std: float
    angle_min: float
    angle_max: float
    depth_mean: float
    depth_std: float
    depth_min: float
    depth_max: float
    lr_mean: float
    lr_std: float
    lr_min: float
    lr_max: float


def _parse_lb_scores_from_research_summary(path: Path) -> dict[int, float]:
    """
    Parse LB scores from Research/COMPLETE_RESEARCH_SUMMARY.md.

    Expected table fragment:
    | 25 | **0.008305** | ...
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lb_scores: dict[int, float] = {}

    # Robust parsing: capture the first "All Submissions and LB Scores" table rows.
    in_table = False
    for line in text.splitlines():
        if line.strip() == "## All Submissions and LB Scores":
            in_table = False
            continue
        if "## All Submissions and LB Scores" in line:
            in_table = False
            continue
        if line.startswith("| Sub | LB Score"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 2:
                continue
            sub_str, score_str = parts[0], parts[1]
            try:
                sub = int(re.sub(r"[^\d]", "", sub_str))
            except ValueError:
                continue
            score_str = score_str.replace("**", "").strip()
            try:
                score = float(score_str)
            except ValueError:
                continue
            lb_scores[sub] = score

    return lb_scores


def _stats_for_submission(df: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    for col in SCALED_COLS:
        x = df[col].to_numpy(dtype=np.float64)
        stats[f"{col}_mean"] = float(np.mean(x))
        stats[f"{col}_std"] = float(np.std(x))
        stats[f"{col}_min"] = float(np.min(x))
        stats[f"{col}_max"] = float(np.max(x))
    return stats


def _ensure_scaled_columns(df: pd.DataFrame, scalers: dict[str, object] | None) -> pd.DataFrame:
    if all(c in df.columns for c in SCALED_COLS):
        return df

    raw_cols = ["angle", "depth", "left_right"]
    if not all(c in df.columns for c in raw_cols):
        raise ValueError(f"Missing both scaled cols {SCALED_COLS} and raw cols {raw_cols}")
    if scalers is None:
        raise ValueError("Raw (unscaled) submission requires scalers, but scalers were not loaded")

    out = df.copy()
    out["scaled_angle"] = scalers["angle"].transform(out["angle"].to_numpy().reshape(-1, 1)).ravel()
    out["scaled_depth"] = scalers["depth"].transform(out["depth"].to_numpy().reshape(-1, 1)).ravel()
    out["scaled_left_right"] = scalers["left_right"].transform(out["left_right"].to_numpy().reshape(-1, 1)).ravel()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions-dir", type=Path, default=SUBMISSION_DIR)
    ap.add_argument("--research-summary", type=Path, default=RESEARCH_SUMMARY)
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "output" / "submission_profile.csv")
    ap.add_argument("--include-unscaled", action="store_true", help="Include submissions with raw (unscaled) cols")
    args = ap.parse_args()

    subs = sorted(args.submissions_dir.glob("submission_*.csv"))
    if not subs:
        raise SystemExit(f"No submissions found in {args.submissions_dir}")

    lb_scores = {}
    if args.research_summary.exists():
        lb_scores = _parse_lb_scores_from_research_summary(args.research_summary)

    scalers = None
    if args.include_unscaled:
        import joblib
        import warnings

        scalers = {}
        for t in ["angle", "depth", "left_right"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scalers[t] = joblib.load(DATA_DIR / f"scaler_{t}.pkl")

    rows: list[SummaryRow] = []
    for path in subs:
        m = re.match(r"submission_(\d+)\.csv$", path.name)
        if not m:
            continue
        sub = int(m.group(1))
        df = pd.read_csv(path)
        if "id" not in df.columns:
            raise SystemExit(f"{path} missing required column: id")
        try:
            df_scaled = _ensure_scaled_columns(df, scalers if args.include_unscaled else None)
        except ValueError as e:
            print(f"Skipping {path.name}: {e}")
            continue

        s = _stats_for_submission(df_scaled)
        rows.append(
            SummaryRow(
                sub=sub,
                lb_score=lb_scores.get(sub),
                angle_mean=s["scaled_angle_mean"],
                angle_std=s["scaled_angle_std"],
                angle_min=s["scaled_angle_min"],
                angle_max=s["scaled_angle_max"],
                depth_mean=s["scaled_depth_mean"],
                depth_std=s["scaled_depth_std"],
                depth_min=s["scaled_depth_min"],
                depth_max=s["scaled_depth_max"],
                lr_mean=s["scaled_left_right_mean"],
                lr_std=s["scaled_left_right_std"],
                lr_min=s["scaled_left_right_min"],
                lr_max=s["scaled_left_right_max"],
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("sub").reset_index(drop=True)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    print(f"Rows: {len(out_df)}")

    known = out_df.dropna(subset=["lb_score"]).copy()
    if len(known) >= 3:
        cols = [
            "angle_std",
            "depth_mean",
            "depth_max",
            "lr_std",
            "lr_mean",
        ]
        print("\nPearson correlations vs LB (known scores only):")
        for col in cols:
            r = float(np.corrcoef(known[col].to_numpy(), known["lb_score"].to_numpy())[0, 1])
            print(f"  {col}: r={r:.6f} (n={len(known)})")
    else:
        print("\nNot enough known LB scores parsed to compute correlations.")

    # Show best LB if available.
    if len(known) > 0:
        best = known.sort_values("lb_score").iloc[0]
        print(
            "\nBest known LB:\n"
            f"  sub={int(best['sub'])} lb={best['lb_score']:.6f} "
            f"angle_std={best['angle_std']:.6f} depth_mean={best['depth_mean']:.6f} depth_max={best['depth_max']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
