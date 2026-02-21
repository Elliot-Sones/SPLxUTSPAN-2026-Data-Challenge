# Almost-Perfect Score Strategy - 2026-02-10

## Objective
Find the highest-confidence path to improve beyond the current best known leaderboard score (`Sub 1640`, LB `0.006698`) using only evidence already present in this repository.

## Ground-Truth Inputs
- Competition artifacts:
  - `submission/submission_*.csv`
  - `scripts/angle_fix.py`
  - `scripts/limiting_factors_fixes.py`
  - `scripts/optimal_blend_search.py`
- Leaderboard evidence extracted from research docs:
  - Table-parsed anchors in `output/lb_table_parse_summary_2026-02-10.csv`
  - Manual anchors added (not in LB tables):
    - `Sub 784 = 0.007224`
    - `Sub 1109 = 0.007223`
    - `Sub 1640 = 0.006698`

## Exact Commands Run

1. Parse LB mentions from all research markdown (broad regex, conflict audit)
```bash
python3 - <<'PY'
# parser over Research/*.md, outputs:
# output/lb_mentions_extracted_2026-02-10.csv
# output/lb_mentions_summary_2026-02-10.csv
PY
```

2. Parse only markdown tables with explicit `Sub` and `LB` columns (high-confidence extraction)
```bash
python3 - <<'PY'
# markdown table parser, outputs:
# output/lb_table_parse_rows_2026-02-10.csv
# output/lb_table_parse_summary_2026-02-10.csv
PY
```

3. Build consensus surrogate on curated 16-anchor set (legacy + modern) and constrained synthetic search around Sub 1640
```bash
python3 - <<'PY'
# outputs:
# output/surrogate_consensus_existing_2026-02-10.csv
# output/surrogate_consensus_synthetic_2026-02-10.csv
# output/surrogate_consensus_anchor_table_2026-02-10.csv
# output/surrogate_consensus_loocv_2026-02-10.csv
PY
```

4. Sensitivity analysis across anchor subsets (robust-weighted)
```bash
python3 - <<'PY'
# outputs:
# output/surrogate_sensitivity_top20_robust_2026-02-10.csv
# output/surrogate_sensitivity_stability_robust_2026-02-10.csv
PY
```

5. High-confidence anchor-rich surrogate (38 anchors from parsed LB tables + manual 784/1109/1640)
```bash
python3 - <<'PY'
# outputs:
# output/surrogate_hc_anchors_existing_2026-02-10.csv
# output/surrogate_hc_anchors_table_2026-02-10.csv
PY
```

6. Candidate delta diagnostics versus Sub 1640
```bash
python3 - <<'PY'
# output:
# output/candidate_delta_stats_2026-02-10.csv
PY
```

7. Best `aw` per angle-source family under high-confidence model
```bash
python3 - <<'PY'
# output:
# output/angle_source_best_aw_hc_2026-02-10.csv
PY
```

## Exact Results

### A) Curated 16-anchor consensus model
- Anchors used:
  `[133, 183, 219, 676, 739, 752, 754, 784, 1109, 1350, 1354, 1421, 1430, 1455, 1492, 1640]`
- Feature dimension: `54`
- LOOCV best alpha: `0.0001`
- LOOCV MAE: `0.00022598330950142717`
- LOOCV RMSE: `0.0004449746971174245`

Top predicted existing submissions (by `pred_mean`):
- `1828`: `pred_mean=0.006688`, `pred_std=0.000314`, `p_better_1640=0.3618`
- `1816`: `pred_mean=0.006690`, `pred_std=0.000333`, `p_better_1640=0.3534`
- `1729`: `pred_mean=0.006692`, `pred_std=0.000163`, `p_better_1640=0.3602`
- `1822`: `pred_mean=0.006692`, `pred_std=0.000313`, `p_better_1640=0.3516`
- `1804`: `pred_mean=0.006693`, `pred_std=0.000228`, `p_better_1640=0.3314`

### B) High-confidence 38-anchor model
- Anchor count: `38`
- Top predicted existing submissions:
  - `1803`: `pred_mean=0.006641`, `pred_std=0.000166`, `p_better_1640=0.65450`
  - `1779`: `pred_mean=0.006642`, `pred_std=0.000173`, `p_better_1640=0.64225`
  - `1820`: `pred_mean=0.006648`, `pred_std=0.000156`, `p_better_1640=0.64075`
  - `1778`: `pred_mean=0.006649`, `pred_std=0.000142`, `p_better_1640=0.66050`
  - `1826`: `pred_mean=0.006649`, `pred_std=0.000155`, `p_better_1640=0.63850`

### C) Family-level optimum (`angle_fix` mapped)
From `output/angle_source_best_aw_hc_2026-02-10.csv`:
- `global_ridge_a500`: best at `aw=0.3` (`Sub 1803`, `pred_mean=0.006641`)
- `extreme_bw080_a200`: best at `aw=0.3` (`Sub 1779`, `pred_mean=0.006642`)
- `avg_all`: best at `aw=0.2` (`Sub 1820`, `pred_mean=0.006648`)
- `safe_average`: best at `aw=0.2` (`Sub 1826`, `pred_mean=0.006649`)

### D) Delta vs Sub 1640 (exact)
From `output/candidate_delta_stats_2026-02-10.csv`:
- `Sub 1803`:
  - angle corr vs 1640: `0.998834219002467`
  - angle MSE vs 1640: `5.639464424598413e-05`
  - angle MAE vs 1640: `0.005723487550624284`
- `Sub 1826`:
  - angle corr vs 1640: `0.9989829144896195`
  - angle MSE vs 1640: `4.768827248657387e-05`
  - angle MAE vs 1640: `0.005111425248911122`
- `Sub 1828`:
  - angle corr vs 1640: `0.9937851133070799`
  - angle MSE vs 1640: `0.00029805170304108734`
  - angle MAE vs 1640: `0.012778565622276267`
- `Sub 1829`:
  - angle corr vs 1640: `0.9880418319718935`
  - angle MSE vs 1640: `0.0005841813379605306`
  - angle MAE vs 1640: `0.017889991871186465`

## Final Technical Conclusion

1. The highest-signal lever is still angle-only modification on top of Sub 1640 (depth and left_right stay unchanged).
2. Across uncertainty-aware models, the stable optimum is moderate angle weight (`aw ~ 0.2-0.3`) rather than aggressive (`aw ~ 0.7`).
3. The strongest high-confidence candidates are:
   - `Sub 1803` (global_ridge_a500, `aw=0.3`)
   - `Sub 1779` (extreme_bw080_a200, `aw=0.3`)
   - `Sub 1820` (avg_all, `aw=0.2`)
   - `Sub 1826` (safe_average, `aw=0.2`)
4. Aggressive candidates like `Sub 1829` have upside in some models but materially higher uncertainty and larger angle perturbation.

## Submission Playbook (Recommended Order)
If testing now on leaderboard, submit in this order:
1. `submission/submission_1803.csv` - best high-confidence expected gain
2. `submission/submission_1826.csv` - similar gain, lower perturbation than aggressive variants
3. `submission/submission_1779.csv` - complementary angle source at same effective weight regime
4. `submission/submission_1820.csv` - ensemble-style moderate angle variant
5. `submission/submission_1828.csv` - higher-upside, higher-risk variant

Fallback baseline: `submission/submission_1640.csv`

## Confidence Statement
Given current repository evidence and surrogate stability checks, the most defensible route to an almost-perfect score is **moderate angle regularization on top of Sub 1640**, not major architecture changes or heavy feature expansion.
