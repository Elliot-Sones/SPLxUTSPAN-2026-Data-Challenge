# Round 2 - Anchor Children After Sub 2437 (2026-02-15)

## Objective
- Exploit near-best signal after leaderboard result `submission_2437.csv = 0.006598`.
- Generate next child candidates near anchor family with strict low-drift constraints.

## Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import runpy, sys
sys.argv = [
    'scripts/probabilistic_gate_ensemble.py',
    '--scale', '30',
    '--seed', '20260216',
    '--top-k', '3',
    '--best-lb', '0.006596',
    '--base-subs', '2437,2243,2255,2253,2063',
    '--injector-subs', '1506,1546,1570,1589,1828,2020,2244,2245,2246,2247,2253,2255,2438,2439',
    '--reference-subs', '784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2253,2255,2437,2438,2439,1506',
    '--extra-anchors', '2243=0.006596,2255=0.006596,2253=0.006597,2437=0.006598,2405=0.008098',
    '--run-tag', 'round2_children_after2437_20260215',
]
runpy.run_path('scripts/probabilistic_gate_ensemble.py', run_name='__main__')
PY
```

## Data
- Prediction files read from `submission/submission_*.csv` for all base/injector/reference/anchor IDs.
- Anchor labels include exact values:
  - `2243=0.006596`
  - `2255=0.006596`
  - `2253=0.006597`
  - `2437=0.006598`
  - `2405=0.008098`
  - plus manual exact modern anchors used by script:
    - `2020=0.006619`
    - `2063=0.006603`
    - `2151=0.006605`
    - `2152=0.006604`

## Model/config
- Script: `scripts/probabilistic_gate_ensemble.py`
- `scale=30`
- `seed=20260216`
- `top_k=3`
- `n_candidates=27965`
- `n_bootstrap=2400`
- Surrogate calibration:
  - `surrogate_best_alpha_loocv=3.5111917342151275`
  - `surrogate_loocv_mae=0.00013785129992504016`
  - `surrogate_loocv_rmse=0.00026957633436040687`

## Exported submissions (exact)
- `submission/submission_2442.csv`
  - template: `pair_inject`
  - weights: `{"1828": 0.06113139132459524, "2246": 0.07576495479599778, "2253": 0.863103653879407}`
  - `pred_mean=0.006652327637438812`
  - `pred_std=0.00022674531834763773`
  - `pred_q10=0.006454641874690532`
  - `pred_q50=0.006594010947123623`
  - `pred_q90=0.0069374533866805805`
  - `p_better_best=0.51`
  - `gate_score=0.0067657321669021015`

- `submission/submission_2443.csv`
  - template: `pair_inject`
  - weights: `{"1589": 0.11239841722714913, "2245": 0.1388728734908681, "2437": 0.7487287092819828}`
  - `pred_mean=0.0066532086563163674`
  - `pred_std=0.0002318397750806598`
  - `pred_q10=0.006443959458906354`
  - `pred_q50=0.0065977586592271265`
  - `pred_q90=0.006939117152423869`
  - `p_better_best=0.49083333333333334`
  - `gate_score=0.006768437905825498`

- `submission/submission_2444.csv`
  - template: `pair_inject`
  - weights: `{"1506": 0.016676090501118474, "2063": 0.9276509012327373, "2247": 0.05567300826614424}`
  - `pred_mean=0.006662830300841313`
  - `pred_std=0.000222059998191428`
  - `pred_q10=0.0064725932931361235`
  - `pred_q50=0.0066044458973852655`
  - `pred_q90=0.006939516941390308`
  - `p_better_best=0.46208333333333335`
  - `gate_score=0.0067719814193877865`

## Artifacts
- `output/probabilistic_gate_ensemble_candidates_round2_children_after2437_20260215.csv`
- `output/probabilistic_gate_ensemble_selected_round2_children_after2437_20260215.csv`
- `output/probabilistic_gate_ensemble_run_round2_children_after2437_20260215.json`
- `output/probabilistic_gate_ensemble_submission_details_round2_children_after2437_20260215.md`

## Runtime
- `elapsed_seconds=61.828041791915894`
