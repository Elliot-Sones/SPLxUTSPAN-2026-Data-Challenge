# Probabilistic Gate Ensemble Results - 2026-02-14

## Objective
Search uncertainty-aware blend candidates around the current best family (`2063/2151/2152`) and export reproducible submission files.

## Script
- `scripts/probabilistic_gate_ensemble.py`

## Data used
- Base submissions: `submission/submission_2063.csv`, `submission/submission_2151.csv`, `submission/submission_2152.csv`
- Injector submissions: `submission/submission_1506.csv`, `submission/submission_1546.csv`, `submission/submission_1570.csv`, `submission/submission_1589.csv`, `submission/submission_2020.csv`, `submission/submission_1828.csv`
- Surrogate anchor table: `output/lb_table_parse_summary_2026-02-10.csv`
- Manual exact modern anchors:
  - `2020 = 0.006619`
  - `2063 = 0.006603`
  - `2151 = 0.006605`
  - `2152 = 0.006604`

## Model config
- Feature families:
  - per-target mean/std
  - per-target corr/MSE/MAE to references `784,1109,1350,1640,1828,2020,2063,2151,2152,1506`
- Surrogate:
  - ridge with bootstrap
  - alpha LOOCV search on `np.logspace(-4, 1, 12)`
  - best alpha: `3.511191734215128`
- Bootstrap count: `80 * scale`
- Candidate count grows with `scale`
- Selection objective:
  - `gate_score = q50 + 0.5 * (q90 - q50)`
  - greedy diversity filter across selected candidates

## Exact commands run
1. Compile check:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile scripts/probabilistic_gate_ensemble.py
```

2. Pilot run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/probabilistic_gate_ensemble.py --scale 1 --seed 20260214 --top-k 5
```

3. Full run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/probabilistic_gate_ensemble.py --scale 20 --seed 20260214 --top-k 5
```

## Pilot result (post-fix)
- run_tag: `20260214_194109`
- scale: `1`
- anchors: `39`
- candidates: `295`
- bootstrap: `80`
- surrogate_loocv_mae: `0.000177014658996`
- surrogate_loocv_rmse: `0.000354233616653`
- artifacts:
  - `output/probabilistic_gate_ensemble_candidates_20260214_194109.csv`
  - `output/probabilistic_gate_ensemble_selected_20260214_194109.csv`
  - `output/probabilistic_gate_ensemble_run_20260214_194109.json`
  - `output/probabilistic_gate_ensemble_submission_details_20260214_194109.md`
- submissions:
  - `submission/submission_2238.csv`
  - `submission/submission_2239.csv`
  - `submission/submission_2240.csv`
  - `submission/submission_2241.csv`
  - `submission/submission_2242.csv`

## Full result (recommended)
- run_tag: `20260214_194145`
- scale: `20`
- anchors: `39`
- candidates: `5843`
- bootstrap: `1600`
- surrogate_loocv_mae: `0.000177014658996`
- surrogate_loocv_rmse: `0.000354233616653`
- best predicted mean in candidate pool: `0.006643813586736`
- best `p_better_best` in candidate pool: `0.560625`
- artifacts:
  - `output/probabilistic_gate_ensemble_candidates_20260214_194145.csv`
  - `output/probabilistic_gate_ensemble_selected_20260214_194145.csv`
  - `output/probabilistic_gate_ensemble_run_20260214_194145.json`
  - `output/probabilistic_gate_ensemble_submission_details_20260214_194145.md`

## Recommended submission files and exact details

1. `submission/submission_2243.csv`
- weights: `{"1506": 0.019210707294233114, "1570": 0.02423981308469898, "2151": 0.956549479621068}`
- pred_mean: `0.006644681666825141`
- pred_std: `0.00024003182615918954`
- pred_q10/q50/q90: `0.006437825165855978 / 0.006581752558055343 / 0.006916402686245411`
- p_better_best: `0.555625`
- gate_score: `0.006749077622150377`

2. `submission/submission_2244.csv`
- weights: `{"1506": 0.015744184373653974, "1589": 0.18283632471572508, "2151": 0.8014194909106209}`
- pred_mean: `0.006647133962441738`
- pred_std: `0.00024187070077992217`
- pred_q10/q50/q90: `0.0064342941501296935 / 0.006585127921246776 / 0.006921927979543654`
- p_better_best: `0.549375`
- gate_score: `0.006753527950395215`

3. `submission/submission_2245.csv`
- weights: `{"1506": 0.04896719243769998, "1589": 0.08226984666696624, "2151": 0.8687629608953338}`
- pred_mean: `0.006647422755164723`
- pred_std: `0.00024192325774577872`
- pred_q10/q50/q90: `0.006430764063270109 / 0.006584271702535468 / 0.006924123390459124`
- p_better_best: `0.5475`
- gate_score: `0.006754197546497296`

4. `submission/submission_2246.csv`
- weights: `{"1506": 0.039121338912133895, "2063": 0.9608786610878661}`
- pred_mean: `0.006656136312628966`
- pred_std: `0.0002364658141464384`
- pred_q10/q50/q90: `0.006451064108144678 / 0.0065944376990535875 / 0.006917635359903809`
- p_better_best: `0.524375`
- gate_score: `0.006756036529478699`

5. `submission/submission_2247.csv`
- weights: `{"1570": 0.13344039662083937, "1828": 0.1187826252786253, "2151": 0.7477769781005353}`
- pred_mean: `0.006650337042978936`
- pred_std: `0.0002411204826433186`
- pred_q10/q50/q90: `0.006439230293762911 / 0.006589096111622185 / 0.006925971965341462`
- p_better_best: `0.541875`
- gate_score: `0.006757534038481823`

## Integrity validation (exact)
After fixing export indexing, each generated file was reconstructed from its recorded weights.

Max absolute reconstruction error by file:
- `2243`: `5.551115123125783e-16`
- `2244`: `5.551115123125783e-16`
- `2245`: `5.551115123125783e-16`
- `2246`: `4.996003610813204e-16`
- `2247`: `5.551115123125783e-16`

## Notes
- Deprecated outputs:
  - Runs before `20260214_194109` were produced before export-index fix.
  - Do not submit files `2223` to `2237`.
- These runs are surrogate-ranked and still require leaderboard validation.
- Highest recurring signal in this sweep: dominant `2151/2152` base with small `1506` injection.

## Post-LB update after Sub 2243
- New measured leaderboard anchor:
  - `submission/submission_2243.csv -> LB 0.006596`

### Exact commands run
1. Pilot around new anchor:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/probabilistic_gate_ensemble.py --scale 1 --seed 20260214 --top-k 5 --best-lb 0.006596 --base-subs 2243,2151,2152,2063 --injector-subs 1506,1546,1570,1589,2020,1828,2244,2245,2247 --reference-subs 784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2244,2245,2247,1506 --extra-anchors 2243=0.006596
```

2. Full run around new anchor:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/probabilistic_gate_ensemble.py --scale 20 --seed 20260214 --top-k 5 --best-lb 0.006596 --base-subs 2243,2151,2152,2063 --injector-subs 1506,1546,1570,1589,2020,1828,2244,2245,2247 --reference-subs 784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2244,2245,2247,1506 --extra-anchors 2243=0.006596
```

### Exact results
- Pilot run_tag: `20260214_195629`
  - anchors: `40`
  - candidates: `532`
  - bootstrap: `80`
  - surrogate_loocv_mae: `0.000151454772234`
  - surrogate_loocv_rmse: `0.000267433514013`
  - artifacts:
    - `output/probabilistic_gate_ensemble_candidates_20260214_195629.csv`
    - `output/probabilistic_gate_ensemble_selected_20260214_195629.csv`
    - `output/probabilistic_gate_ensemble_run_20260214_195629.json`
    - `output/probabilistic_gate_ensemble_submission_details_20260214_195629.md`

- Full run_tag: `20260214_195806`
  - anchors: `40`
  - candidates: `10564`
  - bootstrap: `1600`
  - surrogate_loocv_mae: `0.000151454772234`
  - surrogate_loocv_rmse: `0.000267433514013`
  - best predicted mean in pool: `0.006626873020708`
  - best `p_better_best` in pool: `0.5825`
  - artifacts:
    - `output/probabilistic_gate_ensemble_candidates_20260214_195806.csv`
    - `output/probabilistic_gate_ensemble_selected_20260214_195806.csv`
    - `output/probabilistic_gate_ensemble_run_20260214_195806.json`
    - `output/probabilistic_gate_ensemble_submission_details_20260214_195806.md`

### New submission files (next test batch)
1. `submission/submission_2253.csv`
- weights: `{"1828": 0.16056134466893193, "2243": 0.7281049440238738, "2245": 0.1113337113071943}`
- pred_mean: `0.006627854274282343`
- pred_std: `0.00023976224486092206`
- p_better_best: `0.581875`
- gate_score: `0.006733280219837352`

2. `submission/submission_2254.csv`
- weights: `{"1506": 0.03293201086497385, "1589": 0.05833699265422571, "2243": 0.9087309964808005}`
- pred_mean: `0.006631229576491897`
- pred_std: `0.00024436402514264237`
- p_better_best: `0.5575`
- gate_score: `0.006737347852267389`

3. `submission/submission_2255.csv`
- weights: `{"1589": 0.14757322175732218, "2243": 0.8524267782426778}`
- pred_mean: `0.006629098607848054`
- pred_std: `0.00024734672908782913`
- p_better_best: `0.571875`
- gate_score: `0.0067374607284540755`

4. `submission/submission_2256.csv`
- weights: `{"1506": 0.02954665783874701, "2063": 0.8591451172768254, "2245": 0.11130822488442756}`
- pred_mean: `0.006638971089901499`
- pred_std: `0.00023432219103876648`
- p_better_best: `0.545`
- gate_score: `0.006740934771312091`

5. `submission/submission_2257.csv`
- weights: `{"1570": 0.14757322175732218, "2243": 0.8524267782426778}`
- pred_mean: `0.006633470247519138`
- pred_std: `0.0002474719466162246`
- p_better_best: `0.55625`
- gate_score: `0.006741240757455349`

### Integrity validation
Each exported file exactly matches its logged blend weights.

Max absolute reconstruction error:
- `2253`: `5.551115123125783e-16`
- `2254`: `5.551115123125783e-16`
- `2255`: `4.996003610813204e-16`
- `2256`: `5.551115123125783e-16`
- `2257`: `4.996003610813204e-16`

## Post-LB update after Sub 2255
- New measured leaderboard anchor:
  - `submission/submission_2255.csv -> LB 0.006596`

### Exact command run
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/probabilistic_gate_ensemble.py --scale 20 --seed 20260214 --top-k 5 --best-lb 0.006596 --base-subs 2243,2255,2151,2152 --injector-subs 1506,1546,1570,1589,2020,1828,2244,2245,2246,2247,2253,2254,2256,2257 --reference-subs 784,1109,1350,1640,1828,2020,2063,2151,2152,2243,2244,2245,2246,2247,2253,2254,2255,2256,2257,1506 --extra-anchors 2243=0.006596,2255=0.006596,2253=0.006597
```

### Exact results
- run_tag: `20260214_203513`
- anchors: `42`
- candidates: `15364`
- bootstrap: `1600`
- surrogate_loocv_mae: `0.000126610543003`
- surrogate_loocv_rmse: `0.000223002917389`
- artifacts:
  - `output/probabilistic_gate_ensemble_candidates_20260214_203513.csv`
  - `output/probabilistic_gate_ensemble_selected_20260214_203513.csv`
  - `output/probabilistic_gate_ensemble_run_20260214_203513.json`
  - `output/probabilistic_gate_ensemble_submission_details_20260214_203513.md`

### New submission files
1. `submission/submission_2258.csv`
- weights: `{"1506": 0.032092050209205025, "2151": 0.967907949790795}`
- pred_mean: `0.006644032492040042`
- pred_std: `0.00024931304787149613`
- p_better_best: `0.556875`
- gate_score: `0.006783599519869766`

2. `submission/submission_2259.csv`
- weights: `{"1506": 0.06523012552301255, "2151": 0.9347698744769875}`
- pred_mean: `0.006652584514360338`
- pred_std: `0.0002496609153085277`
- p_better_best: `0.523125`
- gate_score: `0.006790736078307606`

3. `submission/submission_2260.csv`
- weights: `{"1828": 0.5022012578616353, "2063": 0.4977987421383647}`
- pred_mean: `0.0066557439696275835`
- pred_std: `0.0002490929946084511`
- p_better_best: `0.516875`
- gate_score: `0.006794261745408727`

4. `submission/submission_2261.csv`
- weights: `{"1506": 0.029837887003488845, "2255": 0.9164755453012615, "2257": 0.05368656769524964}`
- pred_mean: `0.006647535075386295`
- pred_std: `0.00025915175332112293`
- p_better_best: `0.53625`
- gate_score: `0.006795181802956622`

5. `submission/submission_2262.csv`
- weights: `{"1828": 0.2269037656903766, "2255": 0.7730962343096234}`
- pred_mean: `0.0066499212207472265`
- pred_std: `0.00025790688890785663`
- p_better_best: `0.531875`
- gate_score: `0.00679853356053366`

### Integrity validation
Max absolute reconstruction error:
- `2258`: `4.996003610813204e-16`
- `2259`: `4.996003610813204e-16`
- `2260`: `4.996003610813204e-16`
- `2261`: `5.551115123125783e-16`
- `2262`: `5.551115123125783e-16`
