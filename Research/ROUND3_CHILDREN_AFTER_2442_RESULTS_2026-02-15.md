# Round 3 - Anchor Children After Sub 2442 (2026-02-15)

## Objective
- Continue local leaderboard optimization after `submission_2442.csv = 0.006597`.
- Search for one child prediction that can beat `0.006596`.

## Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import runpy, sys
sys.argv = [
    'scripts/probabilistic_gate_ensemble.py',
    '--scale', '35',
    '--seed', '20260217',
    '--top-k', '3',
    '--best-lb', '0.006596',
    '--base-subs', '2442,2437,2243,2255,2253,2063',
    '--injector-subs', '1506,1546,1570,1589,1828,2020,2244,2245,2246,2247,2253,2255,2438,2439,2443,2444',
    '--reference-subs', '784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2253,2255,2437,2438,2439,2442,2443,2444,1506',
    '--extra-anchors', '2243=0.006596,2255=0.006596,2253=0.006597,2437=0.006598,2442=0.006597,2405=0.008098',
    '--run-tag', 'round3_children_after2442_20260215',
]
runpy.run_path('scripts/probabilistic_gate_ensemble.py', run_name='__main__')
PY
```

## Data
- Inputs: `submission/submission_*.csv` for all base/injector/reference/anchor IDs in command.
- Anchors include exact modern values plus manual exact anchors:
  - `2243=0.006596`
  - `2255=0.006596`
  - `2253=0.006597`
  - `2437=0.006598`
  - `2442=0.006597`
  - `2405=0.008098`
  - `2020=0.006619`
  - `2063=0.006603`
  - `2151=0.006605`
  - `2152=0.006604`

## Model/config
- Script: `scripts/probabilistic_gate_ensemble.py`
- `scale=35`
- `seed=20260217`
- `top_k=3`
- `n_candidates=44246`
- `n_bootstrap=2800`
- Surrogate:
  - `surrogate_best_alpha_loocv=3.5111917342151275`
  - `surrogate_loocv_mae=0.00012827056347332084`
  - `surrogate_loocv_rmse=0.0002392576982800711`

## Exported submissions (exact)
- `submission/submission_2446.csv`
  - template: `single_inject`
  - weights: `{"2253": 0.75, "2443": 0.25}`
  - `pred_mean=0.006648150935175101`
  - `pred_std=0.00022714060648555413`
  - `pred_q10=0.006455392831724878`
  - `pred_q50=0.006592290669005824`
  - `pred_q90=0.006914627189151359`
  - `p_better_best=0.5128571428571429`
  - `gate_score=0.006753458929078591`

- `submission/submission_2447.csv`
  - template: `pair_inject`
  - weights: `{"1506": 0.006214026007193182, "1589": 0.15236788770883827, "2437": 0.8414180862839685}`
  - `pred_mean=0.006651333733926497`
  - `pred_std=0.0002306412823879462`
  - `pred_q10=0.006449680551985826`
  - `pred_q50=0.006598068555762697`
  - `pred_q90=0.006920167009252458`
  - `p_better_best=0.4907142857142857`
  - `gate_score=0.006759117782507577`

- `submission/submission_2448.csv`
  - template: `pair_inject`
  - weights: `{"1506": 0.020192563715464327, "1828": 0.10330660167882219, "2063": 0.8765008346057135}`
  - `pred_mean=0.006654646711343362`
  - `pred_std=0.00022131610149272115`
  - `pred_q10=0.006474110431147578`
  - `pred_q50=0.0065986848252758985`
  - `pred_q90=0.0069206756967571306`
  - `p_better_best=0.4882142857142857`
  - `gate_score=0.0067596802610165145`

## Artifacts
- `output/probabilistic_gate_ensemble_candidates_round3_children_after2442_20260215.csv`
- `output/probabilistic_gate_ensemble_selected_round3_children_after2442_20260215.csv`
- `output/probabilistic_gate_ensemble_run_round3_children_after2442_20260215.json`
- `output/probabilistic_gate_ensemble_submission_details_round3_children_after2442_20260215.md`

## Runtime
- `elapsed_seconds=138.77451419830322`
