# Video SSL + Selective Amplification Submissions

Date: 2026-02-05

## Objective
Inject external-video pretrained embeddings into the existing strong disagreement-mask workflow by anchoring on `submission_771.csv` and applying target-specific per-player selective moves toward SSL predictions.

## Script
- `scripts/video_ssl_selective_amp.py`

## Data Used
- Challenge train: `data/train.csv` (345 shots)
- Challenge test: `data/test.csv` (113 shots)
- External videos:
  - Source: `Basketball_51 dataset/ft0/*.mp4`, `Basketball_51 dataset/ft1/*.mp4`
  - Requested: `--max-external 1200`
  - Successfully decoded and used: `1166`
  - Positive class rate (`ft1`): `0.5145797729492188`

## Full-Scale Run - Exact Command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_full
```

## Model/Config Details
- External pretraining task: binary classification (`ft0` vs `ft1`)
- External pretraining model:
  - Encoder: `LayerNorm -> Linear(64->32) -> ReLU -> BiGRU(hidden=32) -> mean pool -> Dropout(0.1)`
  - Head: `Linear(64->1)` with `BCEWithLogitsLoss`
  - Epochs: `1`
  - Batch size: `16`
  - LR: `0.001`
  - Device: `cpu`
- Challenge fine-tuning model:
  - Same encoder + regression head `Linear(64->3)`
  - Targets: scaled `angle`, `depth`, `left_right`
  - Epochs: `3`
  - Batch size: `16`
  - LR: `0.001`
  - CV: `GroupKFold(n_splits=5)` by `participant_id`

## CV Results (Pretrained Regressor)
- `mse_angle`: `0.036003199033`
- `mse_depth`: `0.018051747512`
- `mse_left_right`: `0.016532764956`
- `mse_total`: `0.023529236671`

## Diversity vs Anchor (`submission_771.csv`)
- Correlation with base:
  - `scaled_angle`: `0.7012775714184699`
  - `scaled_depth`: `0.10993624020029116`
  - `scaled_left_right`: `0.09229133179849243`

This is why it was used as disagreement signal only, not as direct replacement.

## Submissions Created

Base anchor profile (`submission_771.csv`):
- `angle_std`: `0.14676362392929707`
- `depth_mean`: `0.5139000000000001`
- `left_right_std`: `0.06774320883550582`

### Submission 862
- File: `submission/submission_862.csv`
- Config name: `ssl_safe_depth`
- Mask params:
  - angle: `pctl=95.0`, `alpha=0.0`
  - depth: `pctl=90.0`, `alpha=0.1`
  - left_right: `pctl=95.0`, `alpha=0.0`
- Selected samples:
  - angle: `0` (`0.0`)
  - depth: `15` (`0.13274336283185842`)
  - left_right: `0` (`0.0`)
- Profile:
  - `angle_std`: `0.14676362392929707`
  - `depth_mean`: `0.5139000000000001`
  - `left_right_std`: `0.06774320883550582`
- Correlation with base:
  - angle: `1.0`
  - depth: `0.9981404426191369`
  - left_right: `1.0`

### Submission 863
- File: `submission/submission_863.csv`
- Config name: `ssl_safe_depth_lr`
- Mask params:
  - angle: `pctl=95.0`, `alpha=0.0`
  - depth: `pctl=90.0`, `alpha=0.1`
  - left_right: `pctl=90.0`, `alpha=0.1`
- Selected samples:
  - angle: `0` (`0.0`)
  - depth: `15` (`0.13274336283185842`)
  - left_right: `15` (`0.13274336283185842`)
- Profile:
  - `angle_std`: `0.14676362392929707`
  - `depth_mean`: `0.5139000000000001`
  - `left_right_std`: `0.06513892577391729`
- Correlation with base:
  - angle: `1.0`
  - depth: `0.9981404426191369`
  - left_right: `0.9980155203246149`

### Submission 864
- File: `submission/submission_864.csv`
- Config name: `ssl_medium_depth_lr`
- Mask params:
  - angle: `pctl=95.0`, `alpha=0.0`
  - depth: `pctl=85.0`, `alpha=0.18`
  - left_right: `pctl=85.0`, `alpha=0.18`
- Selected samples:
  - angle: `0` (`0.0`)
  - depth: `20` (`0.17699115044247787`)
  - left_right: `20` (`0.17699115044247787`)
- Profile:
  - `angle_std`: `0.14676362392929707`
  - `depth_mean`: `0.5139000000000001`
  - `left_right_std`: `0.06214445577955254`
- Correlation with base:
  - angle: `1.0`
  - depth: `0.9933165646838706`
  - left_right: `0.9923637747559609`

### Submission 865
- File: `submission/submission_865.csv`
- Config name: `ssl_lr_aggressive`
- Mask params:
  - angle: `pctl=95.0`, `alpha=0.0`
  - depth: `pctl=88.0`, `alpha=0.15`
  - left_right: `pctl=80.0`, `alpha=0.25`
- Selected samples:
  - angle: `0` (`0.0`)
  - depth: `15` (`0.13274336283185842`)
  - left_right: `25` (`0.22123893805309736`)
- Profile:
  - `angle_std`: `0.14676362392929707`
  - `depth_mean`: `0.5139000000000001`
  - `left_right_std`: `0.058952157878972455`
- Correlation with base:
  - angle: `1.0`
  - depth: `0.9956087987176211`
  - left_right: `0.9829379651440756`

### Submission 866
- File: `submission/submission_866.csv`
- Config name: `ssl_all_targets_low`
- Mask params:
  - angle: `pctl=92.0`, `alpha=0.05`
  - depth: `pctl=88.0`, `alpha=0.15`
  - left_right: `pctl=85.0`, `alpha=0.2`
- Selected samples:
  - angle: `10` (`0.08849557522123894`)
  - depth: `15` (`0.13274336283185842`)
  - left_right: `20` (`0.17699115044247787`)
- Profile:
  - `angle_std`: `0.14525526234453293`
  - `depth_mean`: `0.5139000000000001`
  - `left_right_std`: `0.06158684858050697`
- Correlation with base:
  - angle: `0.9997588585853978`
  - depth: `0.9956087987176211`
  - left_right: `0.9903915653579654`

## Run Metadata Artifact
- `output/video_ssl_selective_amp_full/run_metrics.json`

## Pilot Validation Run
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 80 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_pilot
```

Pilot outputs:
- Submissions: `submission/submission_857.csv` to `submission/submission_861.csv`
- Metrics: `output/video_ssl_selective_amp_pilot/run_metrics.json`

## Calibration Fix - V2

Issue found after `submission_862.csv` leaderboard result:
- Depth recentering in initial implementation shifted all 113 rows.
- This was too global for a disagreement-mask method.

Code fix in `scripts/video_ssl_selective_amp.py`:
- Depth correction now applies only to selected depth-mask rows.
- Non-selected rows stay unchanged.

### V2 Full Run Command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --pretrain-mode bce --base-submission 771 --output-dir output/video_ssl_selective_amp_full_v2
```

V2 outputs:
- Submissions: `submission/submission_873.csv` to `submission/submission_878.csv`
- Metrics: `output/video_ssl_selective_amp_full_v2/run_metrics.json`

Exact row-change counts vs `submission_771.csv`:
- `873`: angle `0`, depth `10`, left_right `0`
- `874`: angle `0`, depth `15`, left_right `0`
- `875`: angle `0`, depth `15`, left_right `15`
- `876`: angle `0`, depth `20`, left_right `20`
- `877`: angle `0`, depth `15`, left_right `25`
- `878`: angle `10`, depth `15`, left_right `20`

## Micro-Variant Sweep - V3

Goal:
- Add ultra-conservative depth-only micro moves around V2 safest profile.
- Minimize distance to `submission_771.csv` while preserving new signal.

### V3 Full Run Command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --pretrain-mode bce --base-submission 771 --output-dir output/video_ssl_selective_amp_full_v3
```

V3 outputs:
- Submissions: `submission/submission_879.csv` to `submission/submission_887.csv`
- Metrics: `output/video_ssl_selective_amp_full_v3/run_metrics.json`

Closest candidates to `submission_771.csv` (L2 distance over all targets):
1. `submission_879.csv`:
   - changed rows: angle `0`, depth `5`, left_right `0`
   - L2 distance: `0.0206600179429188`
   - depth mean: `0.5137057967911672`
2. `submission_880.csv`:
   - changed rows: angle `0`, depth `5`, left_right `0`
   - L2 distance: `0.02582502242864858`
   - depth mean: `0.5136572459889591`
3. `submission_882.csv`:
   - changed rows: angle `0`, depth `10`, left_right `0`
   - L2 distance: `0.04007304589694508`
   - depth mean: `0.5133980287019223`

More aggressive variants:
- `submission_883.csv` to `submission_887.csv`

## MAE Full-Corpus Runs - 2026-02-06

Objective:
- Move from short BCE pretraining to heavy MAE pretraining with substantially more unlabeled data, then inject only disagreement masks relative to `submission_771.csv`.

### Robustness fix applied before reruns
- File: `scripts/video_ssl_transfer.py`
- Function: `load_external_dataset`
- Change:
  - Before: labels were truncated by length (`labels[:len(seqs)]`) after decode loop.
  - After: labels are collected only for successfully decoded videos (`used_labels`).
- Note:
  - For the current Basketball_51 corpus there were no decode failures in the sampled settings used below, so metrics were unchanged by this fix in these exact runs.

### 1. Scale validation run - exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 120 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 1 --finetune-epochs 1 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_mae_pilot_e1_f1
```

Exact results:
- `external_samples=120`
- `external_positive_rate=0.500000000000`
- `pretrain_corpus_samples=578`
- `cv_pretrained mse_total=0.042583927140`
- `corr_with_base`: angle `0.509041954550`, depth `0.011261831521`, left_right `-0.065191220808`

Submission files created:
- `submission/submission_905.csv` to `submission/submission_913.csv`
- Metadata: `output/video_ssl_selective_amp_mae_pilot_e1_f1/run_metrics.json`

### 2. Heavy balanced run - exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_mae_full_balanced1200_e8_f6
```

Exact training data/model details:
- External source: `Basketball_51 dataset/ft0/*.mp4`, `Basketball_51 dataset/ft1/*.mp4`
- Requested external max: `1200`
- External used: `1166`
- External positive rate: `0.5145797729492188`
- MAE pretrain corpus:
  - External: `1166`
  - Challenge-train unlabeled: `345`
  - Challenge-test unlabeled: `113`
  - Total: `1624`
- Model:
  - Encoder: `LayerNorm -> Linear(64->64) -> ReLU -> BiGRU(hidden=64, bidirectional) -> mean pool -> Dropout(0.1)`
  - MAE decoder head: `Linear(128->128) -> ReLU -> Linear(128->24*64)`
  - Regression head: `Linear(128->3)`

Exact results:
- Pretrain losses by epoch:
  - `0.908082697485`
  - `0.684033028304`
  - `0.558555979887`
  - `0.506989392594`
  - `0.486440350269`
  - `0.475563126538`
  - `0.463006305724`
  - `0.454738297780`
- `cv_pretrained mse_angle=0.038167845923`
- `cv_pretrained mse_depth=0.020124563202`
- `cv_pretrained mse_left_right=0.019078565203`
- `cv_pretrained mse_total=0.025790324435`
- `corr_with_base`: angle `0.778942776063`, depth `0.079273847528`, left_right `0.225693715024`

Submission files created:
- `submission/submission_914.csv` (`ssl_micro_depth_p98_a004`)
- `submission/submission_915.csv` (`ssl_micro_depth_p96_a005`)
- `submission/submission_916.csv` (`ssl_micro_depth_p95_a008`)
- `submission/submission_917.csv` (`ssl_ultra_safe_depth`)
- `submission/submission_918.csv` (`ssl_safe_depth`)
- `submission/submission_919.csv` (`ssl_safe_depth_lr`)
- `submission/submission_920.csv` (`ssl_medium_depth_lr`)
- `submission/submission_921.csv` (`ssl_lr_aggressive`)
- `submission/submission_922.csv` (`ssl_all_targets_low`)
- Metadata: `output/video_ssl_selective_amp_mae_full_balanced1200_e8_f6/run_metrics.json`

### 3. Heavy all-external run - exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 5000 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_mae_full_all2290_e8_f6
```

Exact training data/model details:
- External corpus size in folder:
  - `ft0=566`
  - `ft1=1724`
  - Total `2290`
- External used: `2290`
- External positive rate: `0.7528384327888489`
- MAE pretrain corpus total: `2748` (`2290 + 345 + 113`)

Exact results:
- Pretrain losses by epoch:
  - `0.868874179658`
  - `0.631761737526`
  - `0.564975429831`
  - `0.539864954036`
  - `0.519651742828`
  - `0.505315332086`
  - `0.496922815297`
  - `0.484610110304`
- `cv_pretrained mse_angle=0.044585892186`
- `cv_pretrained mse_depth=0.021508417092`
- `cv_pretrained mse_left_right=0.019114978239`
- `cv_pretrained mse_total=0.028403095901`
- `corr_with_base`: angle `0.799935479622`, depth `0.284508514959`, left_right `0.452000654223`

Submission files created:
- `submission/submission_923.csv` (`ssl_micro_depth_p98_a004`)
- `submission/submission_924.csv` (`ssl_micro_depth_p96_a005`)
- `submission/submission_925.csv` (`ssl_micro_depth_p95_a008`)
- `submission/submission_926.csv` (`ssl_ultra_safe_depth`)
- `submission/submission_927.csv` (`ssl_safe_depth`)
- `submission/submission_928.csv` (`ssl_safe_depth_lr`)
- `submission/submission_929.csv` (`ssl_medium_depth_lr`)
- `submission/submission_930.csv` (`ssl_lr_aggressive`)
- `submission/submission_931.csv` (`ssl_all_targets_low`)
- Metadata: `output/video_ssl_selective_amp_mae_full_all2290_e8_f6/run_metrics.json`

### 4. Post-fix reproducibility rerun
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_mae_full_balanced1200_e8_f6_fixlabels
```

Exact result:
- Run metrics were numerically identical to `output/video_ssl_selective_amp_mae_full_balanced1200_e8_f6/run_metrics.json`.
- Duplicate submission files:
  - `submission_941.csv` to `submission_949.csv`
  - Exact byte-level matches:
    - `914==941`, `915==942`, `916==943`, `917==944`, `918==945`, `919==946`, `920==947`, `921==948`, `922==949`

### 5. Ranked candidates by perturbation distance to `submission_771.csv`

Balanced family (`914` to `922`) had better CV than all-external family:
- Balanced CV total MSE: `0.025790324435`
- All-external CV total MSE: `0.028403095901`

Top low-risk candidates:
1. `submission_914.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.021620447215519`
2. `submission_915.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.027025559019398`
3. `submission_917.csv` - changed rows: angle `0`, depth `10`, left_right `0`, L2 `0.041248837791059`
4. `submission_916.csv` - changed rows: angle `0`, depth `10`, left_right `0`, L2 `0.054998450388078`
5. `submission_918.csv` - changed rows: angle `0`, depth `15`, left_right `0`, L2 `0.077405885059673`

All-external low-risk counterparts:
1. `submission_923.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.022413160611647`
2. `submission_924.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.028016450764558`
3. `submission_926.csv` - changed rows: angle `0`, depth `10`, left_right `0`, L2 `0.040974839744919`

## Balanced-1132 Breakthrough Run - 2026-02-06

Reason for run:
- Heavy MAE setup was working, but `max_external=1200` still contained slight class imbalance (`566` misses vs `600` makes).
- This run enforces exact external balance (`566` + `566`) while keeping the heavy settings fixed.

### Transfer benchmark command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1132 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_balanced1132_mae_e8_f6
```

Exact transfer result:
- `external_samples=1132`
- `external_label_mean=0.500000000000`
- `pretrain_corpus_samples=1590`
- `pretrained_avg mse_total=0.025087740645`
- `baseline_avg mse_total=0.029110912420`
- `transfer_delta_total=0.004023171775`
- `transfer_relative_improvement=13.820150042082%`
- Metrics artifact: `output/video_ssl_transfer_balanced1132_mae_e8_f6/metrics.json`

### Submission generation command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1132 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --output-dir output/video_ssl_selective_amp_mae_full_balanced1132_e8_f6
```

Exact selective-amp run result:
- `cv_pretrained mse_total=0.025087740645`
- `corr_with_base`: angle `0.830505369440`, depth `0.294628226652`, left_right `0.303481361136`
- Submissions:
  - `submission/submission_950.csv` (`ssl_micro_depth_p98_a004`)
  - `submission/submission_951.csv` (`ssl_micro_depth_p96_a005`)
  - `submission/submission_952.csv` (`ssl_micro_depth_p95_a008`)
  - `submission/submission_953.csv` (`ssl_ultra_safe_depth`)
  - `submission/submission_954.csv` (`ssl_safe_depth`)
  - `submission/submission_955.csv` (`ssl_safe_depth_lr`)
  - `submission/submission_956.csv` (`ssl_medium_depth_lr`)
  - `submission/submission_957.csv` (`ssl_lr_aggressive`)
  - `submission/submission_958.csv` (`ssl_all_targets_low`)
- Metadata: `output/video_ssl_selective_amp_mae_full_balanced1132_e8_f6/run_metrics.json`

### Best-first submit order from new family
1. `submission_950.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.021475619860832`
2. `submission_951.csv` - changed rows: angle `0`, depth `5`, left_right `0`, L2 `0.026844524826040`
3. `submission_953.csv` - changed rows: angle `0`, depth `10`, left_right `0`, L2 `0.039604812699863`
4. `submission_952.csv` - changed rows: angle `0`, depth `10`, left_right `0`, L2 `0.052806416933151`
5. `submission_954.csv` - changed rows: angle `0`, depth `15`, left_right `0`, L2 `0.074569677615255`

Current conclusion:
- This is the strongest measured pretrained model in this repo to date.
- It is both better in CV and slightly closer-to-anchor than the previous best family (`914` to `922`).

## Direct-Model Run (No Mask) - 2026-02-05

Goal:
- Save the full model prediction directly to submission, without disagreement masking.
- This tests the strongest possible model signal instead of incremental anchor edits.

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selective_amp.py --max-external 1132 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --base-submission 771 --save-direct --output-dir output/video_ssl_selective_amp_mae_full_balanced1132_e8_f6_direct
```

Exact model result:
- `external_samples=1132`
- `external_positive_rate=0.500000000000`
- `pretrain_corpus_samples=1590`
- `cv_pretrained mse_total=0.025087740645`
- direct submission profile:
  - `angle_std=0.125901475549`
  - `depth_mean=0.524980902672`
  - `left_right_std=0.049361775580`
- correlation with base (`submission_771.csv`):
  - angle `0.830505369440`
  - depth `0.294628226652`
  - left_right `0.303481361136`

Submissions created:
- `submission/submission_973.csv` (`ssl_direct`)
- `submission/submission_974.csv` to `submission/submission_982.csv` (same candidate family as `950` to `958`)
- Metadata: `output/video_ssl_selective_amp_mae_full_balanced1132_e8_f6_direct/run_metrics.json`

## Direct Calibration Sweep - 2026-02-05

Goal:
- Keep full-model ranking signal from `submission_973.csv` while adjusting profile drift (mean/std) globally.

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd

project=Path('..').resolve()
sub_dir=project/'submission'
base=pd.read_csv(sub_dir/'submission_771.csv').sort_values('id').reset_index(drop=True)
direct=pd.read_csv(sub_dir/'submission_973.csv').sort_values('id').reset_index(drop=True)
cols=['scaled_angle','scaled_depth','scaled_left_right']

base_stats={c:{'mean':float(base[c].mean()),'std':float(base[c].std())} for c in cols}
dir_stats={c:{'mean':float(direct[c].mean()),'std':float(direct[c].std())} for c in cols}

def recenter(arr,target_mean):
    return arr + (target_mean - float(arr.mean()))

def restd(arr,target_std,target_mean=None):
    cur_std=float(arr.std())
    cur_mean=float(arr.mean()) if target_mean is None else float(target_mean)
    if cur_std<1e-12:
        out=np.full_like(arr,cur_mean)
    else:
        out=cur_mean + (arr-float(arr.mean()))*(target_std/cur_std)
    return out

cands=[]
x=direct.copy(); x['scaled_depth']=np.clip(recenter(x['scaled_depth'].values, base_stats['scaled_depth']['mean']),0,1); cands.append(('direct_depth_mean_base',x))
x=direct.copy(); tm=0.5*dir_stats['scaled_depth']['mean']+0.5*base_stats['scaled_depth']['mean']; x['scaled_depth']=np.clip(recenter(x['scaled_depth'].values, tm),0,1); cands.append(('direct_depth_mean_half',x))
x=direct.copy(); a=x['scaled_angle'].values; a=restd(a, base_stats['scaled_angle']['std'], target_mean=float(a.mean())); x['scaled_angle']=np.clip(a,0,1); cands.append(('direct_angle_std_base',x))
x=direct.copy(); l=x['scaled_left_right'].values; l=restd(l, base_stats['scaled_left_right']['std'], target_mean=float(l.mean())); x['scaled_left_right']=np.clip(l,0,1); cands.append(('direct_lr_std_base',x))
x=direct.copy(); a=x['scaled_angle'].values; l=x['scaled_left_right'].values; d=x['scaled_depth'].values; a=restd(a, base_stats['scaled_angle']['std'], target_mean=float(a.mean())); l=restd(l, base_stats['scaled_left_right']['std'], target_mean=float(l.mean())); d=recenter(d, base_stats['scaled_depth']['mean']); x['scaled_angle']=np.clip(a,0,1); x['scaled_left_right']=np.clip(l,0,1); x['scaled_depth']=np.clip(d,0,1); cands.append(('direct_profile_lock_base',x))
x=direct.copy(); a=x['scaled_angle'].values; l=x['scaled_left_right'].values; d=x['scaled_depth'].values; at_std=0.5*dir_stats['scaled_angle']['std']+0.5*base_stats['scaled_angle']['std']; lt_std=0.5*dir_stats['scaled_left_right']['std']+0.5*base_stats['scaled_left_right']['std']; dt_mean=0.5*dir_stats['scaled_depth']['mean']+0.5*base_stats['scaled_depth']['mean']; a=restd(a, at_std, target_mean=float(a.mean())); l=restd(l, lt_std, target_mean=float(l.mean())); d=recenter(d, dt_mean); x['scaled_angle']=np.clip(a,0,1); x['scaled_left_right']=np.clip(l,0,1); x['scaled_depth']=np.clip(d,0,1); cands.append(('direct_profile_lock_half',x))

lock=sub_dir/'.submission_lock'
lock.touch(exist_ok=True)
for _,df in cands:
    with lock.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        nums=[]
        for fp in sub_dir.glob('submission_*.csv'):
            p=fp.stem.split('_')
            if len(p)==2 and p[1].isdigit():
                nums.append(int(p[1]))
        n=max(nums)+1 if nums else 1
        out=sub_dir/f'submission_{n}.csv'
        out.touch(exist_ok=True)
        fcntl.flock(f, fcntl.LOCK_UN)
    df.to_csv(out,index=False)
PY
```

Submissions created:
- `submission_983.csv` - `direct_depth_mean_base`
- `submission_984.csv` - `direct_depth_mean_half`
- `submission_985.csv` - `direct_angle_std_base`
- `submission_986.csv` - `direct_lr_std_base`
- `submission_987.csv` - `direct_profile_lock_base`
- `submission_988.csv` - `direct_profile_lock_half`

Exact profiles:
- `983`: angle_std `0.125901469224`, depth_mean `0.513900000000`, lr_std `0.049361775580`
- `984`: angle_std `0.125901469224`, depth_mean `0.519440435531`, lr_std `0.049361775580`
- `985`: angle_std `0.147417362683`, depth_mean `0.524980871062`, lr_std `0.049361775580`
- `986`: angle_std `0.125901469224`, depth_mean `0.524980871062`, lr_std `0.068044961816`
- `987`: angle_std `0.147417362683`, depth_mean `0.513900000000`, lr_std `0.068044961816`
- `988`: angle_std `0.136939821505`, depth_mean `0.519440435531`, lr_std `0.058813306381`
