"""
Channel Strength vs Player Performance Analysis

Tests:
1. Does max|r| (channel strength) correlate with outcome variance?
2. Motor Determinism Hypothesis: Higher max|r| = lower residual variance?
3. Cross-player x target analysis (15 data points)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ---------- Load data ----------
train = pd.read_csv(DATA_DIR / "train.csv")

# Targets are angle, depth, left_right (raw values in [0,1] range already scaled)
target_cols = {"angle": "angle", "depth": "depth", "left_right": "left_right"}

players = sorted(train["participant_id"].unique())
print(f"Players: {players}")
print(f"Counts: {train['participant_id'].value_counts().sort_index().to_dict()}")

# ---------- Max |r| from rigorous channel analysis ----------
max_r = {
    "left_right": {1: 0.696, 2: 0.788, 3: 0.692, 4: 0.619, 5: 0.611},
    "depth":      {1: 0.684, 2: 0.409, 3: 0.544, 4: 0.687, 5: 0.860},
    "angle":      {1: 0.400, 2: 0.377, 3: 0.350, 4: 0.522, 5: 0.525},
}

# ---------- Test 1: Channel strength vs outcome stats ----------
print("\n" + "="*80)
print("TEST 1: Channel Strength vs Outcome Statistics")
print("="*80)

results = []

for target in ["angle", "depth", "left_right"]:
    col = target_cols[target]
    print(f"\n--- {target} ---")
    print(f"{'Player':>8} {'N':>4} {'Mean':>8} {'Std':>8} {'Var':>10} {'MSE_mean':>10} {'max|r|':>8}")

    for p in players:
        mask = train["participant_id"] == p
        vals = train.loc[mask, col].values
        n = len(vals)
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        var_val = np.var(vals, ddof=1)
        mse_mean = np.mean((vals - mean_val)**2)  # MSE from mean prediction (=var with ddof=0)
        mr = max_r[target][p]

        print(f"{p:>8} {n:>4} {mean_val:>8.4f} {std_val:>8.4f} {var_val:>10.6f} {mse_mean:>10.6f} {mr:>8.3f}")

        results.append({
            "player": p, "target": target,
            "n": n, "mean": mean_val, "std": std_val,
            "variance": var_val, "mse_mean": mse_mean,
            "max_r": mr
        })

df = pd.DataFrame(results)

# ---------- Test 2: Correlations across 15 data points ----------
print("\n" + "="*80)
print("TEST 2: Cross-player x target correlations (N=15)")
print("="*80)

r_vs_var, p_vs_var = stats.pearsonr(df["max_r"], df["variance"])
r_vs_std, p_vs_std = stats.pearsonr(df["max_r"], df["std"])
r_vs_mse, p_vs_mse = stats.pearsonr(df["max_r"], df["mse_mean"])

print(f"max|r| vs outcome_variance:  r={r_vs_var:.4f}, p={p_vs_var:.4f}")
print(f"max|r| vs outcome_std:       r={r_vs_std:.4f}, p={p_vs_std:.4f}")
print(f"max|r| vs MSE_from_mean:     r={r_vs_mse:.4f}, p={p_vs_mse:.4f}")

# Per-target correlations
print("\nPer-target correlations (N=5 each, low power):")
for target in ["angle", "depth", "left_right"]:
    sub = df[df["target"] == target]
    r, p = stats.pearsonr(sub["max_r"], sub["variance"])
    rho, prho = stats.spearmanr(sub["max_r"], sub["variance"])
    print(f"  {target:>12}: Pearson r={r:.4f} (p={p:.4f}), Spearman rho={rho:.4f} (p={prho:.4f})")

# ---------- Test 3: Motor Determinism Hypothesis ----------
print("\n" + "="*80)
print("TEST 3: Motor Determinism Hypothesis")
print("="*80)
print("R^2 from max|r| tells fraction of outcome variance explained by ONE feature.")
print("Residual variance = total_var * (1 - r^2) = unexplained / random component.")
print()

det_results = []
for _, row in df.iterrows():
    total_var = row["variance"]
    r_sq = row["max_r"]**2
    explained_var = total_var * r_sq
    residual_var = total_var * (1 - r_sq)

    det_results.append({
        "player": int(row["player"]), "target": row["target"],
        "total_var": total_var, "max_r": row["max_r"],
        "r_squared": r_sq, "explained_var": explained_var,
        "residual_var": residual_var,
    })

det_df = pd.DataFrame(det_results)

print(f"{'Player':>8} {'Target':>12} {'TotalVar':>10} {'max|r|':>8} {'R^2':>6} {'ExplVar':>10} {'ResidVar':>10} {'%Expl':>8}")
for _, row in det_df.iterrows():
    print(f"{row['player']:>8} {row['target']:>12} {row['total_var']:>10.6f} {row['max_r']:>8.3f} "
          f"{row['r_squared']:>6.3f} {row['explained_var']:>10.6f} {row['residual_var']:>10.6f} "
          f"{row['r_squared']*100:>7.1f}%")

# Correlate residual variance with total variance
r_resid_total, p_resid_total = stats.pearsonr(det_df["residual_var"], det_df["total_var"])
print(f"\nresidual_var vs total_var: r={r_resid_total:.4f} (p={p_resid_total:.4f})")

# Key: does max|r| predict residual variance beyond what total variance predicts?
# Partial correlation: residual_var ~ max_r | total_var
from sklearn.linear_model import LinearRegression
# Regress both on total_var, correlate residuals
lr1 = LinearRegression().fit(det_df[["total_var"]], det_df["residual_var"])
lr2 = LinearRegression().fit(det_df[["total_var"]], det_df["max_r"])
resid1 = det_df["residual_var"].values - lr1.predict(det_df[["total_var"]])
resid2 = det_df["max_r"].values - lr2.predict(det_df[["total_var"]])
partial_r, partial_p = stats.pearsonr(resid1, resid2)
print(f"Partial corr (residual_var ~ max_r | total_var): r={partial_r:.4f} (p={partial_p:.4f})")
print("Interpretation: negative partial r = higher channel strength reduces residual")
print("                variance BEYOND what total variance level predicts.")

# Per-player aggregated determinism
print("\n--- Per-player aggregated determinism (mean across 3 targets) ---")
print(f"{'Player':>8} {'Mean_max|r|':>12} {'Mean_R^2':>10} {'Mean_TotalVar':>14} {'Mean_ResidVar':>14} {'Mean_%Expl':>12}")
for p in players:
    sub = det_df[det_df["player"] == p]
    print(f"{p:>8} {sub['max_r'].mean():>12.4f} {sub['r_squared'].mean():>10.4f} "
          f"{sub['total_var'].mean():>14.6f} {sub['residual_var'].mean():>14.6f} "
          f"{sub['r_squared'].mean()*100:>11.1f}%")

# ---------- Test 4: Shooting consistency ----------
print("\n" + "="*80)
print("TEST 4: Channel Strength vs Shooting Consistency")
print("="*80)
print("Consistency = low std of outcome. If strong channels = consistent shooting,")
print("expect negative correlation between max|r| and std.")

# Per-player: average max|r| across targets vs average std across targets
player_stats = []
for p in players:
    sub = df[df["player"] == p]
    avg_mr = sub["max_r"].mean()
    avg_std = sub["std"].mean()
    avg_var = sub["variance"].mean()
    player_stats.append({"player": p, "avg_max_r": avg_mr, "avg_std": avg_std, "avg_var": avg_var})

ps_df = pd.DataFrame(player_stats)
print(f"\n{'Player':>8} {'Avg_max|r|':>12} {'Avg_Std':>10} {'Avg_Var':>10}")
for _, row in ps_df.iterrows():
    print(f"{int(row['player']):>8} {row['avg_max_r']:>12.4f} {row['avg_std']:>10.4f} {row['avg_var']:>10.6f}")

r_agg, p_agg = stats.pearsonr(ps_df["avg_max_r"], ps_df["avg_std"])
rho_agg, prho_agg = stats.spearmanr(ps_df["avg_max_r"], ps_df["avg_std"])
print(f"\nPlayer-level (N=5): avg_max|r| vs avg_std: Pearson r={r_agg:.4f} (p={p_agg:.4f}), Spearman rho={rho_agg:.4f} (p={prho_agg:.4f})")

# ---------- Test 5: Spearman rank correlations (robust) ----------
print("\n" + "="*80)
print("TEST 5: Spearman Rank Correlations (robust, N=15)")
print("="*80)

rho_var, pv = stats.spearmanr(df["max_r"], df["variance"])
rho_std, ps2 = stats.spearmanr(df["max_r"], df["std"])
rho_mse, pm = stats.spearmanr(df["max_r"], df["mse_mean"])
print(f"max|r| vs variance (Spearman): rho={rho_var:.4f}, p={pv:.4f}")
print(f"max|r| vs std (Spearman):      rho={rho_std:.4f}, p={ps2:.4f}")
print(f"max|r| vs MSE_mean (Spearman): rho={rho_mse:.4f}, p={pm:.4f}")

# ---------- SUMMARY ----------
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"N=15 data points (5 players x 3 targets)")
print(f"")
print(f"Q1: Does channel strength correlate with outcome variance?")
print(f"    Pearson  r = {r_vs_var:+.4f} (p={p_vs_var:.4f})")
print(f"    Spearman rho = {rho_var:+.4f} (p={pv:.4f})")
if abs(r_vs_var) < 0.3:
    q1 = "WEAK/NO correlation"
elif r_vs_var > 0:
    q1 = "POSITIVE: stronger channels on MORE variable targets"
else:
    q1 = "NEGATIVE: stronger channels on LESS variable targets"
print(f"    --> {q1}")

print(f"")
print(f"Q2: Motor Determinism - does max|r| reduce residual variance?")
print(f"    Partial corr (residual ~ max|r| | total_var): r = {partial_r:+.4f} (p={partial_p:.4f})")
if partial_r < -0.3:
    q2 = "YES: higher max|r| reduces unexplained variance beyond baseline"
elif partial_r > 0.3:
    q2 = "NO (REVERSED): higher max|r| associates with MORE unexplained variance"
else:
    q2 = "INCONCLUSIVE: weak partial correlation"
print(f"    --> {q2}")

print(f"")
print(f"Q3: Player-level consistency (N=5)?")
print(f"    avg_max|r| vs avg_std: r={r_agg:+.4f} (p={p_agg:.4f})")
if r_agg < -0.5:
    q3 = "Supports: players with stronger channels shoot more consistently"
elif r_agg > 0.5:
    q3 = "Contradicts: players with stronger channels are MORE variable"
else:
    q3 = "No clear relationship at player level"
print(f"    --> {q3}")

# Rank players by determinism
print(f"")
mean_r2_by_player = det_df.groupby("player")["r_squared"].mean()
ranked = mean_r2_by_player.sort_values(ascending=False)
print(f"Player determinism ranking (mean R^2 across targets):")
for p, r2 in ranked.items():
    print(f"    P{p}: mean R^2 = {r2:.4f} ({r2*100:.1f}% of variance explained by top feature)")
