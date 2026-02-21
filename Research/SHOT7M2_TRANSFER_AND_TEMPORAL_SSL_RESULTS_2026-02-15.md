# SHOT7M2 Transfer and Temporal SSL Results - 2026-02-15

## Context
We tested whether SHOT7M2 can improve leaderboard performance through three branches:
1. Weighted SHOT7M2 transfer features
2. Temporal SSL pretraining + transfer
3. SHOT7M2 release-timing prototype + gated residual correction

All tests were run with pilot first, then full scale with only `--scale` changed.

---

## Environment and Data
- Project: `/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge`
- Python execution: `uv run`
- Competition data:
  - `data/train.csv` (345 rows)
  - `data/test.csv` (113 rows)
- SHOT7M2 sample:
  - `external_data/shot7m2_sample/train/train_dictionary_poses.npy`
  - `external_data/shot7m2_sample/train/train_dictionary_actions.npy`
- SHOT7M2 shooting frames used (`action_Shoot > 0.3`): `11845`

---

## Experiment A - Weighted SHOT7M2 Transfer (frame-level)

### Script
- `scripts/shot7m2_weighted_free_throw_transfer.py`

### Exact commands
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_weighted_free_throw_transfer.py --scale 1 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_weighted_free_throw_transfer.py --scale 8 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_weighted_free_throw_transfer.py --scale 8
```

### Model/config highlights
- Axis alignment: `axis_perm=(0,2,1)`, `axis_signs=(1,1,1)`
- Weighted SHOT7M2 frame selection by `action_Dribble`, `action_Move`, `action_Sprint`
- Per-target model:
  - Hoop features + SHOT7M2 features
  - `PLS(hoop)=15`, `PLS(shot) in {1,3,5}`
  - Locally weighted Ridge (`bw_quantile=0.45`, `alpha=10.0`)
- Honest per-player LOO

### Exact results

#### Pilot (`--scale 1`)
- Baseline mean: `0.011689061596872836`
- Best config: `hoop_plus_weighted_s7_5pls`
- Best mean: `0.01114455402929357`
- Delta: `-4.658265874960%`

#### Full (`--scale 8`)
- Baseline mean: `0.011689061596872836`
- Best config: `hoop_plus_weighted_s7_5pls`
- Best mean: `0.011245713429067328`
- Delta: `-3.792846535466262%`
- Per-target (best full):
  - angle: `0.011795464348100086`
  - depth: `0.012415776537564567`
  - left_right: `0.009525899401537327`

### Generated submissions
- Standalone: `submission/submission_2585.csv`
- Blends vs Sub2503:
  - `submission/submission_2586.csv` (5%)
  - `submission/submission_2587.csv` (10%)
  - `submission/submission_2588.csv` (15%)
  - `submission/submission_2589.csv` (20%)
  - `submission/submission_2590.csv` (30%)

### Artifacts
- `output/shot7m2_weighted_ft_transfer_run_20260215_171308.json`
- `output/shot7m2_weighted_ft_transfer_run_20260215_171620.json`
- `output/shot7m2_weighted_ft_transfer_run_20260215_171808.json`
- Matching `..._details_*.md` files

---

## Experiment B - Temporal SSL Transfer

### Script
- `scripts/shot7m2_temporal_ssl_transfer.py`

### Exact commands
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile scripts/shot7m2_temporal_ssl_transfer.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_temporal_ssl_transfer.py --scale 1 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_temporal_ssl_transfer.py --scale 8 --skip-submissions
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_temporal_ssl_transfer.py --scale 8
```

### Model/config highlights
- Temporal masked autoencoder:
  - seq_len: `33` (`window_radius=16`)
  - latent_dim: `64`
  - hidden_dim: `128`
  - epochs: `8`
  - batch_size: `256`
  - mask_ratio: `0.35`
  - velocity_loss_weight: `0.25`
- SSL pretrain windows:
  - SHOT7M2 weighted windows + competition unlabeled windows
- Final per-target model:
  - Hoop + SSL embeddings
  - `PLS(hoop)=15`, `PLS(ssl) in {1,3,5}`
  - Locally weighted Ridge (`bw_quantile=0.45`, `alpha=10.0`)
- Honest per-player LOO

### Exact results

#### Pilot (`--scale 1`)
- Baseline mean: `0.011689061596872836`
- Best config: `hoop_plus_temporal_ssl_5pls`
- Best mean: `0.011203283952`
- Delta: `-4.155830991248%`

#### Full (`--scale 8`)
- Baseline mean: `0.011689061596872836`
- Best config: `hoop_plus_temporal_ssl_5pls`
- Best mean: `0.011281361249734312`
- Delta: `-3.487879191667504%`
- Per-target delta vs baseline (best full):
  - angle: `-5.159796043048099%`
  - depth: `-7.184939846796611%`
  - left_right: `+2.843489122885336%` (worse)

### Generated submissions
- Standalone SSL: `submission/submission_2591.csv`
- Blends vs Sub2503:
  - `submission/submission_2592.csv` (5%)
  - `submission/submission_2593.csv` (10%)
  - `submission/submission_2594.csv` (15%)
  - `submission/submission_2595.csv` (20%)
  - `submission/submission_2596.csv` (30%)

### Artifacts
- `output/shot7m2_temporal_ssl_transfer_run_20260215_181723.json`
- `output/shot7m2_temporal_ssl_transfer_run_20260215_182706.json`
- `output/shot7m2_temporal_ssl_transfer_run_20260215_182926.json`
- Matching `..._details_*.md` files

---

## Experiment C - Target-specific Hybrid (use SSL only for angle/depth)

### Script
- `scripts/temporal_ssl_target_specific_hybrid.py`

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile scripts/temporal_ssl_target_specific_hybrid.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/temporal_ssl_target_specific_hybrid.py --base-submission 2503 --ssl-submission 2591 --angle-weights 0.10,0.20,0.30,0.50,1.00 --depth-weights 0.10,0.20,0.30,0.50,1.00 --lr-weight 0.0
```

### Generated submissions
- `submission/submission_2597.csv` - angle/depth 10%, LR 0%
- `submission/submission_2598.csv` - angle/depth 20%, LR 0%
- `submission/submission_2599.csv` - angle/depth 30%, LR 0%
- `submission/submission_2600.csv` - angle/depth 50%, LR 0%
- `submission/submission_2601.csv` - angle/depth 100%, LR 0%

### Verification
- Left-right exactly equals base in all hybrids:
  - `left_right_exact_base_max_abs = 0.0`

### LB outcome (reported)
- `submission_2601.csv`: `0.010100` (reported by user)
- This is substantially worse than best known `0.006471`.

### Artifacts
- `output/temporal_ssl_target_hybrid_run_20260215_185812.json`
- `output/temporal_ssl_target_hybrid_details_20260215_185812.md`

---

## Experiment D - SHOT7M2 Release Timing + Gated Residual

### Script
- `scripts/shot7m2_release_timing_gated_residual.py`

### Exact commands
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile scripts/shot7m2_release_timing_gated_residual.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_release_timing_gated_residual.py --scale 1
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/shot7m2_release_timing_gated_residual.py --scale 8
```

### Model/config highlights
- SHOT7M2 release-pose prototype from shoot segments
- Per-shot blended release frame:
  - `rf_blend = 0.7 * rf_phys + 0.3 * rf_shot`
- Residual correction:
  - `final = base + lambda * gate * correction`
  - lambda grid: `[0.0, 0.02, 0.05, 0.08, 0.1]`

### Exact results
- Pilot and full both selected:
  - `best_lambda = 0.0` for angle, depth, left_right
- Full summary:
  - baseline mean: `0.011689061596872836`
  - best mean: `0.011689061596872836`
  - mean delta: `+0.000000000000%`
- Nonzero lambda worsened every target.

### Artifacts
- `output/shot7m2_release_timing_gated_residual_run_20260215_193259.json`
- `output/shot7m2_release_timing_gated_residual_run_20260215_193503.json`
- Matching `..._details_*.md` files

---

## Consolidated Findings
1. SHOT7M2-based offline gains were reproducible in weighted transfer and temporal SSL.
2. Those gains did not transfer reliably to LB when pushed aggressively (`submission_2601.csv` score `0.010100`).
3. Release-timing gated residual is a no-op under honest LOO (`lambda=0.0` best).
4. Current evidence indicates SHOT7M2 branches are not a reliable route to beat current best LB.

## Decision
- Treat SHOT7M2 transfer as paused/dead-end for immediate LB gains.
- Prioritize Sub2503-local conservative improvements for next submissions.
