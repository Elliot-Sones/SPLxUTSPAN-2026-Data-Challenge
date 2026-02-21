# Pulse Features Transfer Results - 2026-02-15

## Context
We implemented a new transfer pipeline to test "The Pulse" hypothesis:
- Convert motion sequences into 1D pulse signals (motion-energy over time).
- Build SHOT7M2 free-throw-like pulse templates from weighted shooting frames.
- Add pulse-shape features to target-frame hoop-relative features.
- Evaluate with honest per-player leave-one-out (LOO), then generate conservative blends with `submission_2503`.

This run is meant to test whether pulse-shape transfer carries real signal in our current competition setup.

## Script
- `scripts/pulse_features_transfer.py`

## Data Used
- Competition train: `data/train.csv`
- Competition test: `data/test.csv`
- Target scalers:
  - `data/scaler_angle.pkl`
  - `data/scaler_depth.pkl`
  - `data/scaler_left_right.pkl`
- SHOT7M2:
  - `external_data/shot7m2_sample/train/train_dictionary_poses.npy`
  - `external_data/shot7m2_sample/train/train_dictionary_actions.npy`
- Base submission for blending: `submission/submission_2503.csv`

## Model and Method Details
- Input skeleton for pulse transfer: 14 mapped joints (`mid_hip`, hips/knees/ankles, shoulders/elbows/wrists, `neck`).
- Pulse definition per frame: mean joint speed magnitude.
- Segment pulses: `full`, `arm`, `leg`, `trunk`.
- SHOT7M2 template weighting:
  - Keep frames with `action_Shoot > 0.3`
  - Free-throw-like weight = shoot confidence * exp(-mobility / temperature), with mobility from dribble/move/sprint labels.
- Feature block per segment and target frame:
  - template correlation, template MSE, cosine similarity
  - peak magnitude
  - pulse width at half max
  - rise time and decay time
  - follow-through jitter (tail diff std)
  - window area
  - peak offset from target frame
  - pre/post local energy statistics
- Final features per target:
  - Hoop block: position, velocity, acceleration at target frame from 14 joints (`126` dims)
  - Pulse block: engineered pulse features (`57` dims)
- Regressor:
  - Per-player locally weighted ridge
  - PLS refit inside LOO loop (honest)
  - Config: `n_pls_hoop=15`, `n_pls_pulse in {1,3,5}`, `bw_quantile=0.45`, `alpha=10.0`

## Run A - Pilot (scale=1, no submissions)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/pulse_features_transfer.py --scale 1 --skip-submissions
```

### Exact artifact
- `output/pulse_features_transfer_run_20260215_211613.json`
- `output/pulse_features_transfer_details_20260215_211613.md`

### Exact SHOT7M2 template stats
- `total_shooting_frames`: `11845`
- `kept_frames`: `1500`
- `kept_ratio`: `0.1266357112705783`
- `kept_conf_mean`: `0.9258537184993426`
- `kept_mobility_mean`: `0.5211505378792526`
- `template_length`: `41`

### Exact honest LOO results
- `baseline_hoop_only`
  - angle: `0.01801338099218552`
  - depth: `0.022642753783280487`
  - left_right: `0.02129589667413526`
  - mean: `0.020650677149867087`
- `hoop_plus_pulse_1pls`
  - angle: `0.018204949401818167`
  - depth: `0.022370828694183748`
  - left_right: `0.018869379883314222`
  - mean: `0.019815052659772046`
  - delta vs baseline: `-4.046475009176246%`
- `hoop_plus_pulse_3pls`
  - angle: `0.017738656054988377`
  - depth: `0.022075532804782798`
  - left_right: `0.01876636739651002`
  - mean: `0.019526852085427062`
  - delta vs baseline: `-5.442073672858994%`
- `hoop_plus_pulse_5pls` (best)
  - angle: `0.01761581040678867`
  - depth: `0.021421817046809188`
  - left_right: `0.01881085728247464`
  - mean: `0.0192828282453575`
  - delta vs baseline: `-6.623748434895223%`

## Run B - Full validation (scale=8, no submissions)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/pulse_features_transfer.py --scale 8 --skip-submissions
```

### Exact artifact
- `output/pulse_features_transfer_run_20260215_212142.json`
- `output/pulse_features_transfer_details_20260215_212142.md`

### Exact SHOT7M2 template stats
- `total_shooting_frames`: `11845`
- `kept_frames`: `11845`
- `kept_ratio`: `1.0`
- `kept_conf_mean`: `0.788137340216055`
- `kept_mobility_mean`: `1.0869365113018348`
- `template_length`: `41`

### Exact honest LOO results
- `baseline_hoop_only`
  - angle: `0.01801338099218552`
  - depth: `0.022642753783280487`
  - left_right: `0.02129589667413526`
  - mean: `0.020650677149867087`
- `hoop_plus_pulse_1pls`
  - angle: `0.018197640876528268`
  - depth: `0.022342105544573818`
  - left_right: `0.018851928416695297`
  - mean: `0.01979722494593246`
  - delta vs baseline: `-4.132804932937128%`
- `hoop_plus_pulse_3pls`
  - angle: `0.01775042537659743`
  - depth: `0.022049707883268602`
  - left_right: `0.019007542420159583`
  - mean: `0.01960255856000854`
  - delta vs baseline: `-5.075468384170125%`
- `hoop_plus_pulse_5pls` (best)
  - angle: `0.017475090534395964`
  - depth: `0.021221850043511608`
  - left_right: `0.019172911841836577`
  - mean: `0.01928995080658138`
  - delta vs baseline: `-6.589257743998303%`

## Run C - Full with submissions
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/pulse_features_transfer.py --scale 8
```

### Exact artifact
- `output/pulse_features_transfer_run_20260215_212207.json`
- `output/pulse_features_transfer_details_20260215_212207.md`

### Best config selected
- `best_config_name`: `hoop_plus_pulse_5pls`
- `best_n_pls_pulse`: `5`

### Submission files created
1. `submission/submission_2608.csv`
- Type: standalone pulse transfer model
- Model details:
  - Per-target, per-player locally weighted ridge on hoop + pulse features
  - PLS settings: hoop=15, pulse=5
  - Bandwidth quantile: `0.45`
  - Ridge alpha: `10.0`
- Blend weights:
  - pulse model: `1.0`
  - base (2503): `0.0`

2. `submission/submission_2609.csv`
- Type: conservative blend with base
- Construction:
  - `0.01 * submission_2608 + 0.99 * submission_2503`
- Blend weights:
  - pulse model: `0.01`
  - base (2503): `0.99`

3. `submission/submission_2610.csv`
- Type: conservative blend with base
- Construction:
  - `0.02 * submission_2608 + 0.98 * submission_2503`
- Blend weights:
  - pulse model: `0.02`
  - base (2503): `0.98`

## Findings
1. The Pulse feature family shows consistent offline improvement in this controlled pipeline at both pilot and full scale.
2. Improvement is strongest on `left_right` and `depth` in this setup, with smaller angle gains.
3. The scale transition (`1 -> 8`) preserved direction and magnitude of gain, which is a positive stability sign.
4. Transfer remains unproven on LB until submitted scores are returned.

## Important Caveat
- The baseline in this script is the script's own 14-joint hoop block baseline, not the full historical `sub2503` feature stack. The `2503`-anchored blends (`2609`, `2610`) are the safe LB probes.

## One-Submission Decision (Most Promising)
Context:
- We had to select a single submission for LB, so we measured exact perturbation size versus `submission_2503.csv`.
- Objective was maximizing chance of preserving baseline quality while injecting a small amount of Pulse signal.

### Exact comparison command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import pandas as pd, numpy as np; base=pd.read_csv('submission/submission_2503.csv');\
for n in [2608,2609,2610]:\
 d=pd.read_csv(f'submission/submission_{n}.csv');\
 print('submission',n);\
 for c in ['scaled_angle','scaled_depth','scaled_left_right']:\
  x=base[c].to_numpy(); y=d[c].to_numpy();\
  mae=np.mean(np.abs(y-x)); mx=np.max(np.abs(y-x)); corr=np.corrcoef(x,y)[0,1];\
  print(c,'mae',format(mae,'.15f'),'max',format(mx,'.15f'),'corr',format(corr,'.15f'));\
 print('')"
```

### Exact perturbation results
`submission_2608` (standalone pulse):
- `scaled_angle`: `mae=0.075838109687820`, `max=0.615135466529479`, `corr=0.792033225937845`
- `scaled_depth`: `mae=0.088129491333183`, `max=0.436232738022718`, `corr=0.502012813868767`
- `scaled_left_right`: `mae=0.079485595420271`, `max=0.344835254057329`, `corr=0.680248884137917`

`submission_2609` (1% pulse blend):
- `scaled_angle`: `mae=0.000758381096878`, `max=0.006151354665295`, `corr=0.999973546197419`
- `scaled_depth`: `mae=0.000881294913332`, `max=0.004362327380227`, `corr=0.999939246744887`
- `scaled_left_right`: `mae=0.000794855954203`, `max=0.003448352540573`, `corr=0.999935232695484`

`submission_2610` (2% pulse blend):
- `scaled_angle`: `mae=0.001516762193756`, `max=0.012302709330590`, `corr=0.999894076996719`
- `scaled_depth`: `mae=0.001762589826664`, `max=0.008724654760454`, `corr=0.999755276809647`
- `scaled_left_right`: `mae=0.001589711908405`, `max=0.006896705081147`, `corr=0.999741300491586`

### Decision
- Most promising single submission: `submission/submission_2609.csv`
- Why: smallest controlled perturbation from `2503` while still injecting Pulse signal.
