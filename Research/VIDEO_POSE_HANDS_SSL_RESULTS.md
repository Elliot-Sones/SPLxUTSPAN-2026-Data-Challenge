# Video Pose+Hands SSL Results

## Objective
Build a fundamentally different new-data model by extracting pose + hand keypoints from external videos and pretraining in the same feature space size as challenge keypoints (`69 joints -> 207-d sequence channels`), then fine-tune for target regression.

## Implementation
- New script: `scripts/video_pose_hands_ssl_submission.py`
- Key change vs prior pose-only pipeline:
  - Uses Mediapipe PoseLandmarker + HandLandmarker tasks per frame.
  - Maps detections into challenge-style keypoint schema with finger chains.
  - Produces sequence dim `207` (`x,y,mask` for each of 69 joints).

## Data
- Challenge train: `data/train.csv` (345 labeled)
- Challenge test: `data/test.csv` (113 unlabeled)
- External videos: `Basketball_51 dataset/ft0/*.mp4` + `Basketball_51 dataset/ft1/*.mp4`

---

## Run A: Sanity pilot (10 external)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 10 \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 16 \
  --hidden-dim 32 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_pilot10 \
  --output-dir output/video_pose_hands_ssl_submission_pilot10_e1_f1
```

### Exact outputs
- `external_samples=10 external_dim=207`
- `pretrain_last_loss=0.965652869807`
- `cv_pretrained mse_total=0.036913873255`
- Submissions:
  - `submission_1095.csv` (direct)
  - `submission_1096.csv` (blend w=0.2)
  - `submission_1097.csv` (blend w=0.5)
  - `submission_1098.csv` (direct depth lock)
- Metrics file:
  - `output/video_pose_hands_ssl_submission_pilot10_e1_f1/run_metrics.json`

---

## Run B: Scale benchmark (200 external)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --hidden-dim 64 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3
```

### Exact outputs
- `external_samples=200 external_dim=207`
- `pretrain_epoch losses: 0.895424005290, 0.680891935767, 0.643524958189`
- `pretrain_last_loss=0.643524958189`
- `cv_pretrained mse_angle=0.019395189593`
- `cv_pretrained mse_depth=0.021987180179`
- `cv_pretrained mse_left_right=0.019742014632`
- `cv_pretrained mse_total=0.020374795515`
- Submissions:
  - `submission_1103.csv` (direct)
  - `submission_1104.csv` (blend w=0.2)
  - `submission_1105.csv` (blend w=0.35)
  - `submission_1106.csv` (blend w=0.5)
  - `submission_1107.csv` (direct depth lock)
- Metrics file:
  - `output/video_pose_hands_ssl_submission_scale200_e3_f3/run_metrics.json`

### Additional calibrations from Run B direct (`1103`)
Generated submissions:
- `submission_1119.csv` - `1103_angle_std_base`
- `submission_1120.csv` - `1103_lr_std_base`
- `submission_1121.csv` - `1103_profile_lock_base`
- `submission_1122.csv` - `1103_profile_lock_half`
- `submission_1123.csv` - `blend1053_new1103_w0.15`
- `submission_1124.csv` - `blend1053_new1103_w0.25`
- `submission_1125.csv` - `blend1053_new1103_w0.35`
- `submission_1126.csv` - `blend1053_new1103_w0.50`

---

## Run C: Full scale (1132 external)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 1132 \
  --pretrain-epochs 8 \
  --finetune-epochs 6 \
  --batch-size 32 \
  --hidden-dim 64 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_full1132 \
  --output-dir output/video_pose_hands_ssl_submission_full1132_e8_f6
```

### Exact outputs
- `external_samples=1132 external_dim=207`
- `pretrain_last_loss=0.700689506756`
- `cv_pretrained mse_angle=0.040021851286`
- `cv_pretrained mse_depth=0.022950088605`
- `cv_pretrained mse_left_right=0.022385407146`
- `cv_pretrained mse_total=0.028452449478`
- Direct submission profile:
  - `submission_1131.csv`
  - `angle_std=0.135198667645`
  - `depth_mean=0.514799833298`
  - corr vs `submission_771.csv`: `angle=0.953476968375`, `depth=0.255718226061`, `left_right=0.277215008671`
- Blend submissions:
  - `submission_1132.csv` (`w=0.2`)
  - `submission_1133.csv` (`w=0.35`)
  - `submission_1134.csv` (`w=0.5`)
  - `submission_1135.csv` (depth lock)
- Metrics file:
  - `output/video_pose_hands_ssl_submission_full1132_e8_f6/run_metrics.json`

### Additional blends using strong anchor (`submission_1053`) + full direct (`1131`)
Generated submissions:
- `submission_1136.csv` to `submission_1143.csv`: global blends `blend1053_1131_w{0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30}`
- `submission_1144.csv` to `submission_1149.csv`: target-specific blends (`ts_*`)
- `submission_1150.csv`: `1131_profile_lock_base`
- `submission_1151.csv` to `submission_1153.csv`: blends with calibrated `1131`

Blend grid artifact:
- `output/video_pose_hands_ssl_submission_full1132_e8_f6/blend_grid_from_1053_1131.csv`

---

## Notes
- This is a true "train on more data" approach with a new representation pipeline, not disagreement-mask-only tuning.
- Best CV in this family observed at intermediate external scale (`200`) rather than full `1132`.

---

## Run D: Scale-200 longer training (`8+6`)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 8 \
  --finetune-epochs 6 \
  --batch-size 32 \
  --hidden-dim 64 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e8_f6
```

### Exact outputs
- `external_samples=200 external_dim=207`
- `pretrain_last_loss=0.627470075903`
- `cv_pretrained mse_angle=0.039689185191`
- `cv_pretrained mse_depth=0.023462790623`
- `cv_pretrained mse_left_right=0.019523755275`
- `cv_pretrained mse_total=0.027558576502`
- Submissions:
  - `submission_1154.csv` (direct)
  - `submission_1155.csv` (`w=0.2`)
  - `submission_1156.csv` (`w=0.35`)
  - `submission_1157.csv` (`w=0.5`)
  - `submission_1158.csv` (depth lock)
- Metrics file:
  - `output/video_pose_hands_ssl_submission_scale200_e8_f6/run_metrics.json`

### Blends from strong anchor (`1053`) + direct (`1154`)
Generated submissions:
- `submission_1159.csv` to `submission_1166.csv`: `blend1053_1154_w*`
- `submission_1167.csv` to `submission_1172.csv`: target-specific blends (`ts_*`)
- `submission_1173.csv`: `1154_profile_lock_base`
- `submission_1174.csv` to `submission_1176.csv`: blends with calibrated `1154`

Grid artifact:
- `output/video_pose_hands_ssl_submission_scale200_e8_f6/blend_grid_from_1053_1154.csv`

---

## Run E: Seed diversification on best architecture (`max_external=200`, `3+3`)

### Seed 7
#### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --hidden-dim 64 \
  --seed 7 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed7
```

#### Exact outputs
- `cv_pretrained mse_total=0.020825911686`
- Submissions: `1177` (direct), `1178`, `1179`, `1180`, `1181`

### Seed 133
#### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --hidden-dim 64 \
  --seed 133 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed133
```

#### Exact outputs
- `cv_pretrained mse_total=0.023544855975`
- Submissions: `1182` (direct), `1183`, `1184`, `1185`, `1186`

---

## Run F: Seed ensemble model blending
Constructed direct ensemble from seed-42/7/133 direct submissions (`1103`, `1177`, `1182`) and blended against strong anchor `1053`.

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
# Builds seed ensemble and writes submission_1187..submission_1214
# Also saves grid to output/video_pose_hands_ssl_submission_scale200_e3_f3_seed_ensemble_grid.csv
PY
```

### Key artifacts
- Grid file: `output/video_pose_hands_ssl_submission_scale200_e3_f3_seed_ensemble_grid.csv`
- Generated submissions: `1187` to `1214`
- Weighted ensemble weights (inverse CV):
  - seed42: `0.3516545949348434`
  - seed7: `0.34403730178708636`
  - seed133: `0.30430810327807023`

---

## Run G: Additional seed diversification (`seed=99`, `seed=2026`)

### Seed 99
#### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --hidden-dim 64 \
  --seed 99 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed99
```
- `cv_pretrained mse_total=0.024656140432`
- Submissions: `1232` to `1236`

### Seed 2026
#### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with mediapipe python scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --hidden-dim 64 \
  --seed 2026 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed2026
```
- `cv_pretrained mse_total=0.021589862928`
- Submissions: `1237` to `1241`

---

## Run H: 5-seed ensemble and blend grid
Seed direct sources:
- `1103` (seed 42)
- `1177` (seed 7)
- `1182` (seed 133)
- `1232` (seed 99)
- `1237` (seed 2026)

Inverse-CV weights:
- seed42: `0.21674212204603344`
- seed7: `0.21204720747681677`
- seed133: `0.18756013716389294`
- seed99: `0.17910655677657053`
- seed2026: `0.20454397653668627`

Generated submissions:
- `1242` to `1281`

Grid artifacts:
- `output/video_pose_hands_ssl_submission_scale200_e3_f3_seed5_ensemble_grid.csv`
- `output/video_pose_hands_ssl_submission_scale200_e3_f3_seed_ensemble_grid.csv` (previous 3-seed grid)

---

## Run I: New pretrained objective wiring (`mae_bce`) + pilot validation

Code updates in `scripts/video_pose_hands_ssl_submission.py`:
- Added `SequenceMAEWithClassifier` and joint `pretrain_external_mae_bce(...)`
- Added CLI args:
  - `--pretrain-mode {mae,mae_bce}`
  - `--pretrain-cls-weight`
  - `--pretrain-cls-schedule {constant,linear_decay}`
- Added mode-specific pretrain metrics into `run_metrics.json`:
  - `train_acc`, `cls_weight`, `cls_schedule`, `cls_losses`, `cls_epoch_weights`

### Syntax validation
```bash
uv run python -m py_compile ../scripts/video_pose_hands_ssl_submission.py
```
- Exit code: `0`

### Pilot command (exact)
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 40 \
  --pretrain-mode mae_bce \
  --pretrain-cls-weight 0.35 \
  --pretrain-epochs 1 \
  --finetune-epochs 1 \
  --batch-size 32 \
  --seed 42 \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --base-submission 771 \
  --blend-weights 0.35 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_pilot40_mae_bce_seed42
```

### Pilot outputs (exact)
- `external_samples=40 external_dim=207`
- `external_positive_rate=0.500000000000`
- `pretrain_corpus_samples=498`
- `pretrain_epoch=1 mae_loss=0.965856807538 cls_loss=0.700580491598`
- `pretrain_mode=mae_bce`
- `pretrain_last_loss=0.965856807538`
- `pretrain_last_cls_loss=0.700580491598`
- `pretrain_train_acc=0.500000000000`
- `cv_pretrained mse_angle=0.055023588054`
- `cv_pretrained mse_depth=0.040192774311`
- `cv_pretrained mse_left_right=0.021982597560`
- `cv_pretrained mse_total=0.039066321962`
- Submissions: `1296` (direct), `1297` (w=0.35), `1298` (depth lock)
- Metrics file: `output/video_pose_hands_ssl_submission_pilot40_mae_bce_seed42/run_metrics.json`

---

## Run J: Controlled objective sweep at fixed scale (`max_external=200`, `pretrain=3`, `finetune=3`, `seed=42`)

All commands below used:
- `--include-challenge-train-unlabeled`
- `--include-challenge-test-unlabeled`
- `--cache-dir output/pose_hands_cache_mediapipe_scale200`
- `--base-submission 771`
- `--blend-weights 0.2 0.35 0.5`

Reference baseline (MAE only, already logged earlier):
- `cv_pretrained mse_total=0.020374795515`

### J1: `mae_bce`, `cls_weight=0.2`
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.2 \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed42_mae_bce_w0p2
```
- `pretrain_last_loss=0.653377883340`
- `pretrain_last_cls_loss=0.685254288662`
- `pretrain_train_acc=0.560000000000`
- `cv_pretrained mse_total=0.020311657339`
- Submissions: `1299` to `1303`

### J2: `mae_bce`, `cls_weight=0.35`
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.35 \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed42_mae_bce_w0p35
```
- `pretrain_last_loss=0.653217972231`
- `pretrain_last_cls_loss=0.682876370598`
- `pretrain_train_acc=0.560000000000`
- `cv_pretrained mse_total=0.019801482931`
- Submissions: `1304` to `1308`

### J3: `mae_bce`, `cls_weight=0.5` (best in sweep)
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed42_mae_bce_w0p5
```
- `pretrain_last_loss=0.653581597341`
- `pretrain_last_cls_loss=0.682018789296`
- `pretrain_train_acc=0.555000000000`
- `cv_pretrained mse_total=0.019596663304`
- Submissions: `1309` to `1313`

### J4: `mae_bce`, `cls_weight=0.8`
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.8 \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed42_mae_bce_w0p8
```
- `pretrain_last_loss=0.653806814307`
- `pretrain_last_cls_loss=0.680592177487`
- `pretrain_train_acc=0.550000000000`
- `cv_pretrained mse_total=0.020149014704`
- Submissions: `1314` to `1318`

---

## Run K: Depth/overfitting checks for `mae_bce` (`cls_weight=0.5`, `seed=42`)

### K1: Longer training (`pretrain=8`, `finetune=6`, constant cls weight)
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e8_f6_seed42_mae_bce_w0p5
```
- `pretrain_last_loss=0.635744714085`
- `pretrain_last_cls_loss=0.556038420976`
- `pretrain_train_acc=0.780000000000`
- `cv_pretrained mse_total=0.021348189376`
- Submissions: `1319` to `1323`

### K2: Longer training with linear cls decay (`pretrain=8`, `finetune=6`)
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule linear_decay \
  --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e8_f6_seed42_mae_bce_w0p5_decay
```
- `pretrain_last_loss=0.634725058332`
- `pretrain_last_cls_loss=0.639813585303`
- `pretrain_train_acc=0.640000000000`
- `cv_pretrained mse_total=0.021247281693`
- Submissions: `1324` to `1328`

### K3: Intermediate depth (`pretrain=5`, `finetune=4`)
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 5 --finetune-epochs 4 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e5_f4_seed42_mae_bce_w0p5
```
- `pretrain_last_loss=0.638577339135`
- `pretrain_last_cls_loss=0.660991700587`
- `pretrain_train_acc=0.670000000000`
- `cv_pretrained mse_total=0.022405932099`
- Submissions: `1329` to `1333`

Conclusion: for this objective family, `3+3` outperformed deeper schedules.

---

## Run L: Seed robustness for best `mae_bce` setting (`max_external=200`, `3+3`, `cls_weight=0.5`)

### Seed 7
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 7 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed7_mae_bce_w0p5
```
- `pretrain_last_loss=0.671337489752`
- `pretrain_last_cls_loss=0.686016564674`
- `pretrain_train_acc=0.565000000000`
- `cv_pretrained mse_total=0.020373049472`
- Submissions: `1334` to `1338`

### Seed 2026
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 200 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 2026 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale200 \
  --output-dir output/video_pose_hands_ssl_submission_scale200_e3_f3_seed2026_mae_bce_w0p5
```
- `pretrain_last_loss=0.692827215220`
- `pretrain_last_cls_loss=0.685171341099`
- `pretrain_train_acc=0.525000000000`
- `cv_pretrained mse_total=0.023370897025`
- Submissions: `1339` to `1343`

---

## Run M: True data scaling with fixed best-pretrain recipe (`mae_bce`, `cls_weight=0.5`, `3+3`, `seed=42`)

### M1: Scale to 400 external videos
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 400 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale400 \
  --output-dir output/video_pose_hands_ssl_submission_scale400_e3_f3_seed42_mae_bce_w0p5
```
- `external_samples=400 external_dim=207`
- `pretrain_corpus_samples=858`
- `pretrain_last_loss=0.707275799869`
- `pretrain_last_cls_loss=0.682537081080`
- `pretrain_train_acc=0.587500000000`
- `cv_pretrained mse_angle=0.019703889918`
- `cv_pretrained mse_depth=0.020733421249`
- `cv_pretrained mse_left_right=0.017236065492`
- `cv_pretrained mse_total=0.019224458467`
- Submissions: `1344` (direct), `1345` (`w=0.2`), `1346` (`w=0.35`), `1347` (`w=0.5`), `1348` (depth lock)
- Metrics: `output/video_pose_hands_ssl_submission_scale400_e3_f3_seed42_mae_bce_w0p5/run_metrics.json`

### M2: Scale to 600 external videos
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 600 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 42 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale400 \
  --output-dir output/video_pose_hands_ssl_submission_scale600_e3_f3_seed42_mae_bce_w0p5
```
- `external_samples=600 external_dim=207`
- `pretrain_corpus_samples=1058`
- `pretrain_last_loss=0.721431270242`
- `pretrain_last_cls_loss=0.685271078900`
- `pretrain_train_acc=0.535000000000`
- `cv_pretrained mse_angle=0.022287231963`
- `cv_pretrained mse_depth=0.019120884221`
- `cv_pretrained mse_left_right=0.017433306575`
- `cv_pretrained mse_total=0.019613807928`
- Submissions: `1368` (direct), `1369` (`w=0.2`), `1370` (`w=0.35`), `1371` (`w=0.5`), `1372` (depth lock)
- Metrics: `output/video_pose_hands_ssl_submission_scale600_e3_f3_seed42_mae_bce_w0p5/run_metrics.json`

Scaling takeaway in this objective family:
- `200 -> 400` improved (`0.019596663304 -> 0.019224458467`)
- `400 -> 600` regressed (`0.019224458467 -> 0.019613807928`)
- Current best pretraining CV from these runs: `0.019224458467` at `max_external=400`, `3+3`, `seed=42`, `mae_bce`, `cls_weight=0.5`

### M3: Scale-400 robustness check with `seed=7`
```bash
uv run --with mediapipe python ../scripts/video_pose_hands_ssl_submission.py \
  --max-external 400 --pretrain-mode mae_bce --pretrain-cls-weight 0.5 \
  --pretrain-cls-schedule constant \
  --pretrain-epochs 3 --finetune-epochs 3 --batch-size 32 --seed 7 \
  --include-challenge-train-unlabeled --include-challenge-test-unlabeled \
  --base-submission 771 --blend-weights 0.2 0.35 0.5 \
  --cache-dir output/pose_hands_cache_mediapipe_scale400 \
  --output-dir output/video_pose_hands_ssl_submission_scale400_e3_f3_seed7_mae_bce_w0p5
```
- `external_samples=400 external_dim=207`
- `pretrain_corpus_samples=858`
- `pretrain_last_loss=0.721299187843`
- `pretrain_last_cls_loss=0.671793271195`
- `pretrain_train_acc=0.587500000000`
- `cv_pretrained mse_angle=0.040225584712`
- `cv_pretrained mse_depth=0.017664004583`
- `cv_pretrained mse_left_right=0.020316998847`
- `cv_pretrained mse_total=0.026068861410`
- Submissions: `1416` (direct), `1417` (`w=0.2`), `1418` (`w=0.35`), `1419` (`w=0.5`), `1420` (depth lock)
- Metrics: `output/video_pose_hands_ssl_submission_scale400_e3_f3_seed7_mae_bce_w0p5/run_metrics.json`
