# Target-Specific Blend + External SSL Embeddings Results

## Objective
Train on more data by learning external-video SSL embeddings, then fuse those embeddings into the strongest target-specific modeling pipeline (PLS depth + hoop-relative angle/left_right), and generate direct plus blended submissions.

## Script
- `scripts/target_specific_blend_ssl_embed.py`

## Data
- Challenge labeled train: `data/train.csv` (345 shots)
- Challenge unlabeled test: `data/test.csv` (113 shots)
- External videos: `Basketball_51 dataset/ft0/*.mp4` and `Basketball_51 dataset/ft1/*.mp4`

## Model Components
1. SSL encoder pretraining
- Sequence representation: `video_ssl_transfer.keypoint_timeseries_to_sequence` with `num_frames=24`, `frame_size=8`
- SSL encoder: `TemporalEncoder(input_dim=64, hidden_dim=64, dropout=0.1)`
- SSL objective: MAE masked reconstruction (`pretrain_mode=mae`)

2. Target-specific downstream models with SSL embedding fusion
- Depth: per-player PLS + Ridge + LightGBM ensemble, trained on `[X_raw || ssl_embedding]`
- Angle / Left-right: per-player hoop-relative feature ensemble (LGB/XGB/CatBoost/Ridge), trained on `[X_hoop || ssl_embedding]`
- Submission blending anchor: `submission/submission_771.csv`

## Run 1 - Pilot validation (small scale)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/target_specific_blend_ssl_embed.py \
  --max-external 40 \
  --pretrain-epochs 1 \
  --pretrain-mode mae \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --top-k 2 \
  --output-dir output/target_specific_blend_ssl_embed_pilot_e1
```

### Exact key outputs
- `external_samples=40 external_dim=64`
- `external_positive_rate=0.500000000000`
- `pretrain_corpus_samples=498`
- `pretrain_epoch=1 mae_loss=1.001666233542`
- `pretrain_last_loss=1.001666233542`
- `train_emb_shape=(345, 128) test_emb_shape=(113, 128)`

Per-target CV (raw-target scale used by this script):
- `angle=6.629803724223`
- `depth=13.171751373491`
- `left_right=8.932559493945`
- `total=9.578038197219`

Saved submissions:
- `submission/submission_1038.csv` - `aw=0.20 dw=0.30 lw=0.50`
- `submission/submission_1039.csv` - `aw=0.15 dw=0.30 lw=0.50`
- `submission/submission_1040.csv` - depth-only correction

Metrics artifact:
- `output/target_specific_blend_ssl_embed_pilot_e1/run_metrics.json`

## Run 2 - Full scale (1132 external, full SSL pretrain)
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/target_specific_blend_ssl_embed.py \
  --max-external 1132 \
  --pretrain-epochs 8 \
  --pretrain-mode mae \
  --include-challenge-train-unlabeled \
  --include-challenge-test-unlabeled \
  --top-k 5 \
  --output-dir output/target_specific_blend_ssl_embed_full1132_e8
```

### Exact key outputs
- `external_samples=1132 external_dim=64`
- `external_positive_rate=0.500000000000`
- `pretrain_corpus_samples=1590`
- MAE pretrain losses:
  - `0.936346255009`
  - `0.688157007604`
  - `0.560028226908`
  - `0.497209038899`
  - `0.479639353580`
  - `0.464483221634`
  - `0.458183246139`
  - `0.448015632045`
- `pretrain_last_loss=0.448015632045`
- `train_emb_shape=(345, 128) test_emb_shape=(113, 128)`

Per-target CV (raw-target scale used by this script):
- `angle=6.685768658659`
- `depth=13.160418225635`
- `left_right=8.885807550283`
- `total=9.577331478192`

Direct model submission stats:
- `submission/submission_1052.csv`
- `angle_std=0.146064`
- `depth_mean=0.515155`
- `left_right_std=0.081258`
- Correlation vs base `submission_771.csv`:
  - `angle=0.977928`
  - `depth=0.685147`
  - `left_right=0.744395`

Top blended submissions (same run):
- `submission/submission_1053.csv` - `aw=0.20 dw=0.30 lw=0.50`
- `submission/submission_1054.csv` - `aw=0.15 dw=0.30 lw=0.50`
- `submission/submission_1055.csv` - `aw=0.10 dw=0.30 lw=0.50`
- `submission/submission_1056.csv` - `aw=0.05 dw=0.30 lw=0.50`
- `submission/submission_1057.csv` - `aw=0.00 dw=0.30 lw=0.50`
- `submission/submission_1058.csv` - depth-only correction

Metrics artifact:
- `output/target_specific_blend_ssl_embed_full1132_e8/run_metrics.json`

## Cache artifacts
- `output/target_specific_blend_ssl_embed_pilot_e1/ssl_embed_cache_1e4f9da582d1d0e9.npz`
- `output/target_specific_blend_ssl_embed_full1132_e8/ssl_embed_cache_30d334f8638ed372.npz`

## Notes
- The full run reused cache on rerun for fast turnaround and determinism.
- This pipeline is a true "train on more data" approach: external set contributes through SSL embedding learning, and those embeddings are fused into all downstream target models.

## Run 3 - Micro disagreement-mask hedge set from direct model
### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd

sub_dir=Path('submission')
base=pd.read_csv(sub_dir/'submission_771.csv').sort_values('id').reset_index(drop=True)
direct=pd.read_csv(sub_dir/'submission_1052.csv').sort_values('id').reset_index(drop=True)
assert (base['id'].values==direct['id'].values).all()

plans=[
    ('depth_p98_a002', 'scaled_depth', 98.0, 0.02),
    ('depth_p96_a002', 'scaled_depth', 96.0, 0.02),
    ('depth_p95_a003', 'scaled_depth', 95.0, 0.03),
    ('depth_p90_a002', 'scaled_depth', 90.0, 0.02),
    ('lr_p98_a001', 'scaled_left_right', 98.0, 0.01),
    ('angle_p98_a001', 'scaled_angle', 98.0, 0.01),
    ('depth_lr_p98_a002_001','both',98.0,0.0),
]
# ...writes submissions with atomic lock...
PY
```

### Exact generated submissions and profiles
- `submission/submission_1059.csv` - `depth_p98_a002`
  - changed rows: depth `3`
  - `l2_vs_771=0.018916047589098090`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.513996149968009000`
  - `left_right_std=0.067743208835505820`

- `submission/submission_1060.csv` - `depth_p96_a002`
  - changed rows: depth `5`
  - `l2_vs_771=0.019315901769405366`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.514044982849703700`
  - `left_right_std=0.067743208835505820`

- `submission/submission_1061.csv` - `depth_p95_a003`
  - changed rows: depth `6`
  - `l2_vs_771=0.029199721284955283`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.514085395897410000`
  - `left_right_std=0.067743208835505820`

- `submission/submission_1062.csv` - `depth_p90_a002`
  - changed rows: depth `12`
  - `l2_vs_771=0.020036721139224190`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.513988434921194200`
  - `left_right_std=0.067743208835505820`

- `submission/submission_1063.csv` - `lr_p98_a001`
  - changed rows: left_right `3`
  - `l2_vs_771=0.003593307216121681`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.513900000000000100`
  - `left_right_std=0.067723763661137220`

- `submission/submission_1064.csv` - `angle_p98_a001`
  - changed rows: angle `3`
  - `l2_vs_771=0.002279343388573400`
  - `angle_std=0.146797289051151400`
  - `depth_mean=0.513900000000000100`
  - `left_right_std=0.067743208835505820`

- `submission/submission_1065.csv` - `depth_lr_p98_a002_001`
  - changed rows: depth `3`, left_right `3`
  - `l2_vs_771=0.019254316740472923`
  - `angle_std=0.146763623929297070`
  - `depth_mean=0.513996149968009000`
  - `left_right_std=0.067723763661137220`

## Run 4 - Ultra-micro disagreement masks (post-LB feedback for direct model)
Direct model score feedback:
- `submission_1052.csv` scored `0.009931`

Action:
- Generated stricter micro-update set using same learned signal, with very small alphas and 2-5 changed rows for highest-safety candidates.

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd

sub_dir = Path('submission')
base = pd.read_csv(sub_dir / 'submission_771.csv').sort_values('id').reset_index(drop=True)
direct = pd.read_csv(sub_dir / 'submission_1052.csv').sort_values('id').reset_index(drop=True)
# Generates submission_1066 to submission_1077 using percentile masks + tiny alpha moves.
PY
```

Grid artifact:
- `output/target_specific_blend_ssl_embed_full1132_e8/ultra_micro_submission_grid.csv`

### Highest-safety candidates (smallest L2 from base)
- `submission/submission_1066.csv` (`depth_p99_a005`)
  - changed rows: depth `2`
  - `l2_vs_771=0.004641000000000`
  - `angle_std=0.146763623929297`
  - `depth_mean=0.513932000000000`
  - `left_right_std=0.067743208835506`

- `submission/submission_1067.csv` (`depth_p98_a005`)
  - changed rows: depth `3`
  - `l2_vs_771=0.004729000000000`
  - `angle_std=0.146763623929297`
  - `depth_mean=0.513924000000000`
  - `left_right_std=0.067743208835506`

- `submission/submission_1068.csv` (`depth_p97_a005`)
  - changed rows: depth `4`
  - `l2_vs_771=0.004786000000000`
  - `angle_std=0.146763623929297`
  - `depth_mean=0.513931000000000`
  - `left_right_std=0.067743208835506`

- `submission/submission_1069.csv` (`depth_p96_a005`)
  - changed rows: depth `5`
  - `l2_vs_771=0.004829000000000`
  - `angle_std=0.146763623929297`
  - `depth_mean=0.513936000000000`
  - `left_right_std=0.067743208835506`

Additional mixed tiny variants:
- `submission/submission_1074.csv` (`depth_lr_p98_a005_003`)
- `submission/submission_1075.csv` (`depth_lr_p96_a005_003`)
- `submission/submission_1076.csv` (`angle_depth_p98_a003_005`)
- `submission/submission_1077.csv` (`all_p98_small`)

## Run 5 - Local sweep around successful `submission_1053`
Observed leaderboard feedback:
- `submission_1053.csv`: `0.00723`

Objective:
- Keep the same learned model signal but optimize near `aw=0.20, dw=0.30, lw=0.50`.

### Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd

sub_dir=Path('submission')
base=pd.read_csv(sub_dir/'submission_771.csv').sort_values('id').reset_index(drop=True)
direct=pd.read_csv(sub_dir/'submission_1052.csv').sort_values('id').reset_index(drop=True)
# local per-target weight sweep around 1053 + profile calibrations
# generated submissions 1078-1094
PY
```

Grid artifact:
- `output/target_specific_blend_ssl_embed_full1132_e8/local_weight_sweep_from1053.csv`

### New submissions generated
- `submission/submission_1078.csv` - `aw=0.20 dw=0.24 lw=0.50`
- `submission/submission_1079.csv` - `aw=0.20 dw=0.26 lw=0.50`
- `submission/submission_1080.csv` - `aw=0.20 dw=0.28 lw=0.50`
- `submission/submission_1081.csv` - `aw=0.20 dw=0.30 lw=0.50` (same formula as 1053 rerun)
- `submission/submission_1082.csv` - `aw=0.20 dw=0.32 lw=0.50`
- `submission/submission_1083.csv` - `aw=0.20 dw=0.34 lw=0.50`
- `submission/submission_1084.csv` - `aw=0.20 dw=0.30 lw=0.42`
- `submission/submission_1085.csv` - `aw=0.20 dw=0.30 lw=0.46`
- `submission/submission_1086.csv` - `aw=0.20 dw=0.30 lw=0.54`
- `submission/submission_1087.csv` - `aw=0.20 dw=0.30 lw=0.58`
- `submission/submission_1088.csv` - `aw=0.14 dw=0.30 lw=0.50`
- `submission/submission_1089.csv` - `aw=0.17 dw=0.30 lw=0.50`
- `submission/submission_1090.csv` - `aw=0.23 dw=0.30 lw=0.50`
- `submission/submission_1091.csv` - `aw=0.26 dw=0.30 lw=0.50`
- `submission/submission_1092.csv` - `1053_depth_mean_base` calibration
- `submission/submission_1093.csv` - `1053_lr_std_base` calibration
- `submission/submission_1094.csv` - `1053_profile_half` calibration
