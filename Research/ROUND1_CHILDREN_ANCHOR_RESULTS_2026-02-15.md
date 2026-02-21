# Round 1 - Anchor Children Optimization (2026-02-15)

## Objective
- Generate high-probability child submissions near proven anchor family to maximize chance of leaderboard improvement below best known LB `0.006596`.

## Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import runpy, sys
sys.argv = [
    'scripts/probabilistic_gate_ensemble.py',
    '--scale', '20',
    '--seed', '20260215',
    '--top-k', '3',
    '--best-lb', '0.006596',
    '--base-subs', '2243,2255,2253,2063',
    '--injector-subs', '1506,1546,1570,1589,1828,2020,2244,2245,2246,2247,2253,2255',
    '--reference-subs', '784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2253,2255,1506',
    '--extra-anchors', '2243=0.006596,2255=0.006596,2253=0.006597,2405=0.008098',
    '--run-tag', 'round1_children_anchor_20260215',
]
runpy.run_path('scripts/probabilistic_gate_ensemble.py', run_name='__main__')
PY
```

## Data
- Prediction sources: `submission/submission_*.csv` for all `base`, `injector`, `reference`, and `anchor` IDs used by the run.
- Anchor supervision included exact modern LBs:
  - `2020=0.006619`
  - `2063=0.006603`
  - `2151=0.006605`
  - `2152=0.006604`
  - `2243=0.006596`
  - `2255=0.006596`
  - `2253=0.006597`
  - `2405=0.008098`

## Model/config used
- Script: `scripts/probabilistic_gate_ensemble.py`
- Search scale: `20`
- Seed: `20260215`
- Top-k exported: `3`
- Candidate count: `12964`
- Bootstrap samples: `1600`
- Surrogate model: bootstrap ridge with LOOCV alpha search
- Surrogate calibration:
  - `surrogate_best_alpha_loocv=3.5111917342151275`
  - `surrogate_loocv_mae=0.00015064600325750477`
  - `surrogate_loocv_rmse=0.00031366869340427884`

## Exported submissions (exact)
- `submission/submission_2437.csv`
  - template: `pair_inject`
  - weights: `{"1570": 0.018930171022882147, "1828": 0.029056040038060123, "2253": 0.9520137889390577}`
  - `pred_mean=0.006646634678677741`
  - `pred_std=0.00022429233654249454`
  - `pred_q10=0.006459300971925355`
  - `pred_q50=0.006586022597449023`
  - `pred_q90=0.006929226010334258`
  - `p_better_best=0.539375`
  - `gate_score=0.00675762430389164`

- `submission/submission_2438.csv`
  - template: `single_inject`
  - weights: `{"1589": 0.14255230125523013, "2253": 0.8574476987447699}`
  - `pred_mean=0.006648404432415902`
  - `pred_std=0.0002272647316268286`
  - `pred_q10=0.0064553224427073164`
  - `pred_q50=0.006589616120900086`
  - `pred_q90=0.006935561860712479`
  - `p_better_best=0.530625`
  - `gate_score=0.0067625889908062824`

- `submission/submission_2439.csv`
  - template: `single_inject`
  - weights: `{"1506": 0.020041841004184102, "2063": 0.979958158995816}`
  - `pred_mean=0.006660129818007885`
  - `pred_std=0.000217488041431524`
  - `pred_q10=0.00648415375385746`
  - `pred_q50=0.006599597903813924`
  - `pred_q90=0.0069308382337477815`
  - `p_better_best=0.483125`
  - `gate_score=0.0067652180687808525`

## Artifacts
- `output/probabilistic_gate_ensemble_candidates_round1_children_anchor_20260215.csv`
- `output/probabilistic_gate_ensemble_selected_round1_children_anchor_20260215.csv`
- `output/probabilistic_gate_ensemble_run_round1_children_anchor_20260215.json`
- `output/probabilistic_gate_ensemble_submission_details_round1_children_anchor_20260215.md`

## Runtime
- `elapsed_seconds=24.308234930038452`
