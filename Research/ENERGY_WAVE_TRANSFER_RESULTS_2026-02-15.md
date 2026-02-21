# Energy Wave Transfer Results - 2026-02-15

## Context
We implemented an "Energy Wave" transfer feature pipeline to add dynamic biomechanical sequencing signals to the current hoop-relative per-player locally weighted Ridge baseline.

Key implementation file:
- `scripts/energy_wave_transfer.py`

This implements:
1. External pro-template wave from NBA kinematic traces (`external_data/spacejam/path_detail.csv`)
2. Per-shot 3D velocity energy wave extraction from SPL pose time-series
3. `Pro_Form_Match_Score` and related wave-shape features
4. Honest per-player LOO benchmark vs baseline
5. Optional submission generation against base submission 2503

Important data limitation:
- No raw NBA videos are present in this workspace.
- The external transfer source is kinematic traces, not optical-flow pixels.

---

## Environment and Data
- Project: `/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge`
- Python execution: `uv run`
- Train data: `data/train.csv` (345 rows)
- Test data: `data/test.csv` (113 rows)
- External transfer trace data: `external_data/spacejam/path_detail.csv` (66 sequences)
- Target scaling: `data/scaler_angle.pkl`, `data/scaler_depth.pkl`, `data/scaler_left_right.pkl`

---

## Model/Feature Configuration (Final Improved Version)
- Base pipeline:
  - Hoop-relative frame features (207 features)
  - Per-player locally weighted Ridge
  - `bw_quantile=0.45`
  - `alpha=10.0`
  - Honest per-player LOO
- Energy Wave branch:
  - Pro-template built from top `scale * sequences_per_scale` sequences sorted by available rows
  - `scale=8`, `sequences_per_scale=8` -> `used_sequences=64`
  - Wave length: `128`
  - Energy feature set: `core`
  - `include_raw_energy=False` (PLS-only injection)
  - `n_pls_energy_grid=[0,2,4,6]`

Selected core energy features:
- `pro_form_match_score`
- `pro_corr_distance`
- `wave_peak_lag_norm`
- `prox_to_distal_peak_lag_norm`
- `double_peak_ratio`
- `num_local_peaks`
- `flat_peak_fraction`
- `shooting_side_right`

---

## Exact Commands Run

### 1. Compile check
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile scripts/energy_wave_transfer.py
```

### 2. Initial implementation test (full feature set, default then at that time)
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 1 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 8 --skip-submissions
```

### 3. Refined implementation test (core feature set + PLS-only)
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 1 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 8 --skip-submissions
```

### 4. Submission generation with improved config
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 8
```

---

## Exact Results

### A. Initial implementation (before refinement) - not competitive

Pilot (`--scale 1`, artifact: `output/energy_wave_transfer_run_20260215_202847.json`):
- Baseline mean: `0.011689061596872836`
- Best Energy Wave mean: `0.011948043357594704` (`hoop_plus_energy_4pls`)
- Delta vs baseline: `+2.2155906919949224%` (worse)

Full (`--scale 8`, artifact: `output/energy_wave_transfer_run_20260215_203044.json`):
- Baseline mean: `0.011689061596872836`
- Best Energy Wave mean: `0.012006222860317833` (`hoop_plus_energy_2pls`)
- Delta vs baseline: `+2.7133167262104805%` (worse)

### B. Refined implementation (core features + PLS-only) - improved

Pilot (`--scale 1`, artifact: `output/energy_wave_transfer_run_20260215_203313.json`):
- Baseline mean: `0.011689061596872836`
- Best Energy Wave mean: `0.011626707917145426` (`hoop_plus_energy_2pls`)
- Delta vs baseline: `-0.5334361434234027%` (better)
- Per-target best config (`n_pls_energy=2`):
  - angle: `0.012173267701040143`
  - depth: `0.012379402670941293`
  - left_right: `0.010327453379454843`

Full (`--scale 8`, artifact: `output/energy_wave_transfer_run_20260215_203715.json`):
- Baseline mean: `0.011689061596872836`
- Best Energy Wave mean: `0.011621248489001852` (`hoop_plus_energy_2pls`)
- Delta vs baseline: `-0.5801415905715356%` (better)
- Per-target best config (`n_pls_energy=2`):
  - angle: `0.01218193910560451`
  - depth: `0.012357576777680408`
  - left_right: `0.010324229583720635`

### C. Submission generation run (improved config)

Command:
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/energy_wave_transfer.py --scale 8`

Artifact:
- `output/energy_wave_transfer_run_20260215_211338.json`
- `output/energy_wave_transfer_details_20260215_211338.md`

Generated submissions:
- `submission/submission_2602.csv` - standalone Energy Wave model (`hoop_plus_energy_2pls`)
- `submission/submission_2603.csv` - 5% Energy Wave + 95% Sub2503
- `submission/submission_2604.csv` - 10% Energy Wave + 90% Sub2503
- `submission/submission_2605.csv` - 15% Energy Wave + 85% Sub2503
- `submission/submission_2606.csv` - 20% Energy Wave + 80% Sub2503
- `submission/submission_2607.csv` - 30% Energy Wave + 70% Sub2503

---

## Interpretation
1. Injecting raw Energy Wave features overfit and hurt CV.
2. Restricting to compact dynamic descriptors and using PLS-only transfer produced consistent pilot and full gains.
3. The best current config from this branch is:
   - `energy_feature_set=core`
   - `include_raw_energy=False`
   - `n_pls_energy=2`
   - `scale=8`

Given CV-LB gap history, blended submissions (`2603` to `2607`) are safer first LB probes than the standalone (`2602`).

---

## Submission Recommendation Findings (Documented)

### Goal
Choose first LB candidate among `2602-2607` by balancing:
1. CV improvement evidence from Energy Wave branch
2. Distance from strong base submission `2503` (risk control)

### Exact command run
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import pandas as pd
from pathlib import Path
PROJECT=Path('/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge')
sub_dir=PROJECT/'submission'
out_dir=PROJECT/'output'
out_dir.mkdir(parents=True, exist_ok=True)
base_num=2503
candidates=[2602,2603,2604,2605,2606,2607]
base_df=pd.read_csv(sub_dir/f'submission_{base_num}.csv')
rows=[]
for n in candidates:
    df=pd.read_csv(sub_dir/f'submission_{n}.csv')
    rec={'submission_number':n}
    for c in ['scaled_angle','scaled_depth','scaled_left_right']:
        d=(df[c]-base_df[c]).to_numpy()
        rec[f'{c}_mean_abs_delta']=float(abs(d).mean())
        rec[f'{c}_max_abs_delta']=float(abs(d).max())
        rec[f'{c}_std_delta']=float(d.std())
    all_d=(df[['scaled_angle','scaled_depth','scaled_left_right']].to_numpy()-
           base_df[['scaled_angle','scaled_depth','scaled_left_right']].to_numpy())
    rec['overall_mean_abs_delta']=float(abs(all_d).mean())
    rec['overall_max_abs_delta']=float(abs(all_d).max())
    rows.append(rec)
res=pd.DataFrame(rows).sort_values('submission_number')
out_csv=out_dir/'energy_wave_submission_delta_vs_2503_20260215.csv'
res.to_csv(out_csv, index=False)
print(out_csv)
print(res.to_string(index=False))
PY
```

### Output artifact
- `output/energy_wave_submission_delta_vs_2503_20260215.csv`

### Exact measured deltas vs Sub2503
- `2602`: `overall_mean_abs_delta=0.063146`, `overall_max_abs_delta=0.688973`
- `2603`: `overall_mean_abs_delta=0.003157`, `overall_max_abs_delta=0.034449`
- `2604`: `overall_mean_abs_delta=0.006315`, `overall_max_abs_delta=0.068897`
- `2605`: `overall_mean_abs_delta=0.009472`, `overall_max_abs_delta=0.103346`
- `2606`: `overall_mean_abs_delta=0.012629`, `overall_max_abs_delta=0.137795`
- `2607`: `overall_mean_abs_delta=0.018944`, `overall_max_abs_delta=0.206692`

### Decision
- Recommended first submit: `submission/submission_2604.csv`

Why:
1. It keeps a conservative distance from base (much safer than `2602`, still meaningful change vs `2603`).
2. The Energy Wave branch had full-scale CV gain (`-0.5801415905715356%`), so a moderate blend is the risk-balanced first LB probe.
3. `2602` is the highest-upside but highest-risk option due to very large max row deltas.
