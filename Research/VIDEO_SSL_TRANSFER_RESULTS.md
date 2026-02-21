# Video SSL Transfer Results

Date: 2026-02-05

## Objective
Test whether external free-throw videos in `Basketball_51 dataset/` can improve challenge regression via pretraining, then fine-tuning on challenge labels.

## Implementation
Script: `scripts/video_ssl_transfer.py`

Pipeline:
1. External video pretraining task: binary `ft0` vs `ft1` classification.
2. Transfer task: challenge regression for `scaled_angle`, `scaled_depth`, `scaled_left_right`.
3. Evaluation: 5-fold `GroupKFold` by `participant_id`.
4. Baseline control: same regression architecture and training loop, but no external pretraining.

Common fixed config (unless explicitly changed):
- `external_root=Basketball_51 dataset`
- `num_frames=24`
- `frame_size=8`
- `hidden_dim=32`
- `dropout=0.1`
- `finetune_epochs=3`
- `batch_size=16`
- `lr_pretrain=0.001`
- `lr_finetune=0.001`
- `seed=42`
- `device=cpu`

## Exact commands and exact results

### Run 1 - Pilot validation
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 20 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_pilot
```

Data and model details:
- External videos used: `20`
- External positive rate (`ft1`): `0.5`
- Pretrain epochs: `1`
- Pretrain train accuracy: `0.55`

Results:
- Pretrained CV total MSE: `0.02392664337530732`
- Baseline CV total MSE: `0.025341469049453735`
- Transfer delta (`baseline - pretrained`): `0.0014148256741464138`
- Relative improvement: `0.05583045210936231` (5.583045210936231%)

Output:
- `output/video_ssl_transfer_pilot/metrics.json`

---

### Run 2 - Scale external data only
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 400 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_scale400
```

Data and model details:
- External videos used: `400`
- External positive rate (`ft1`): `0.5`
- Pretrain epochs: `1`
- Pretrain train accuracy: `0.5325`

Results:
- Pretrained CV total MSE: `0.02373409690335393`
- Baseline CV total MSE: `0.025341469049453735`
- Transfer delta (`baseline - pretrained`): `0.0016073721460998051`
- Relative improvement: `0.06342853064133841` (6.342853064133841%)

Output:
- `output/video_ssl_transfer_scale400/metrics.json`

---

### Run 3 - Larger scale external data only
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_scale1200
```

Data and model details:
- Requested external videos: `1200`
- Successfully decoded and used: `1166`
- External positive rate (`ft1`): `0.5145797729492188`
- Pretrain epochs: `1`
- Pretrain train accuracy: `0.5377358490566038`

Results:
- Pretrained CV total MSE: `0.023529236670583487`
- Baseline CV total MSE: `0.025341469049453735`
- Transfer delta (`baseline - pretrained`): `0.001812232378870248`
- Relative improvement: `0.0715125226297531` (7.15125226297531%)

Output:
- `output/video_ssl_transfer_scale1200/metrics.json`

---

### Run 4 - Same large scale, longer pretraining
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-epochs 3 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_scale1200_ep3
```

Data and model details:
- External videos used: `1166`
- External positive rate (`ft1`): `0.5145797729492188`
- Pretrain epochs: `3`
- Pretrain train accuracy: `0.6252144082332761`

Results:
- Pretrained CV total MSE: `0.024029639922082423`
- Baseline CV total MSE: `0.025341469049453735`
- Transfer delta (`baseline - pretrained`): `0.0013118291273713119`
- Relative improvement: `0.051766104199061413` (5.176610419906141%)

Output:
- `output/video_ssl_transfer_scale1200_ep3/metrics.json`

---

### Run 5 - Same large scale, temporal diff channels
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1200 --num-frames 24 --frame-size 8 --use-diff --pretrain-epochs 1 --finetune-epochs 3 --batch-size 16 --hidden-dim 32 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_scale1200_diff
```

Data and model details:
- External videos used: `1166`
- External positive rate (`ft1`): `0.5145797729492188`
- Input dim changed from `64` to `128` because `--use-diff` appends temporal difference channels
- Pretrain epochs: `1`
- Pretrain train accuracy: `0.5411663807890223`

Results:
- Pretrained CV total MSE: `0.02314303908497095`
- Baseline CV total MSE: `0.021221588272601365`
- Transfer delta (`baseline - pretrained`): `-0.0019214508123695864`
- Relative improvement: `-0.09054227175118279` (-9.054227175118279%)

Output:
- `output/video_ssl_transfer_scale1200_diff/metrics.json`

---

### Run 6 - MAE pretraining with challenge unlabeled augmentation (balanced external sample)
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1200 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_balanced1200_mae_e8_f6
```

Data and model details:
- External videos used: `1166`
- External positive rate (`ft1`): `0.5145797729492188`
- Additional unlabeled corpus included in MAE pretraining:
  - Challenge train sequences: `345`
  - Challenge test sequences: `113`
  - Total MAE pretrain corpus: `1624`
- Pretraining mode: `mae`
- Pretrain epochs: `8`
- Fine-tune epochs: `6`
- Model size: `hidden_dim=64`, `batch_size=32`

Results:
- Pretrained CV total MSE: `0.025790324434638023`
- Baseline CV total MSE: `0.029110912419855594`
- Transfer delta (`baseline - pretrained`): `0.0033205879852175706`
- Relative improvement: `0.11406677802900836` (11.406677802900836%)

Output:
- `output/video_ssl_transfer_balanced1200_mae_e8_f6/metrics.json`

---

### Run 7 - MAE pretraining with all external videos plus challenge unlabeled augmentation
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 5000 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_all_data_mae_e8_f6
```

Data and model details:
- External videos used: `2290` (all files available in `Basketball_51 dataset/`)
- Class counts in external corpus:
  - `ft0=566`
  - `ft1=1724`
- External positive rate (`ft1`): `0.7528384327888489`
- Additional unlabeled corpus included in MAE pretraining:
  - Challenge train sequences: `345`
  - Challenge test sequences: `113`
  - Total MAE pretrain corpus: `2748`
- Pretraining mode: `mae`
- Pretrain epochs: `8`
- Fine-tune epochs: `6`
- Model size: `hidden_dim=64`, `batch_size=32`

Results:
- Pretrained CV total MSE: `0.02840309590101242`
- Baseline CV total MSE: `0.029110912419855594`
- Transfer delta (`baseline - pretrained`): `0.0007078165188431733`
- Relative improvement: `0.02431447385209386` (2.431447385209386%)

Output:
- `output/video_ssl_transfer_all_data_mae_e8_f6/metrics.json`

---

### Run 8 - MAE pretraining with exactly balanced external corpus (full `ft0` coverage)
Command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_transfer.py --max-external 1132 --num-frames 24 --frame-size 8 --pretrain-mode mae --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pretrain-epochs 8 --finetune-epochs 6 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-finetune 1e-3 --seed 42 --output-dir output/video_ssl_transfer_balanced1132_mae_e8_f6
```

Data and model details:
- External videos used: `1132`
- External class balance: `ft0=566`, `ft1=566`
- External positive rate (`ft1`): `0.5`
- Additional unlabeled corpus included in MAE pretraining:
  - Challenge train sequences: `345`
  - Challenge test sequences: `113`
  - Total MAE pretrain corpus: `1590`
- Pretraining mode: `mae`
- Pretrain epochs: `8`
- Fine-tune epochs: `6`
- Model size: `hidden_dim=64`, `batch_size=32`

Results:
- Pretrained CV total MSE: `0.02508774064472854`
- Baseline CV total MSE: `0.029110912419855594`
- Transfer delta (`baseline - pretrained`): `0.004023171775127054`
- Relative improvement: `0.13820150042081789` (13.820150042081789%)

Output:
- `output/video_ssl_transfer_balanced1132_mae_e8_f6/metrics.json`

## Summary
- External video pretraining is feasible and reproducible in this repo with current environment and dependencies.
- Best measured run so far is MAE pretraining on an exactly balanced external corpus plus challenge unlabeled data:
  - `output/video_ssl_transfer_balanced1132_mae_e8_f6/metrics.json`
  - Transfer relative improvement over no-pretrain baseline: `13.820150042081789%`.
- Scaling to all external videos (`2290`) reduced transfer quality in this setup:
  - `output/video_ssl_transfer_all_data_mae_e8_f6/metrics.json`
  - Transfer relative improvement: `2.431447385209386%`.
- In this problem, external class-balance quality mattered more than raw external volume.

## Artifacts created
- Script: `scripts/video_ssl_transfer.py`
- Metrics:
  - `output/video_ssl_transfer_pilot/metrics.json`
  - `output/video_ssl_transfer_scale400/metrics.json`
  - `output/video_ssl_transfer_scale1200/metrics.json`
  - `output/video_ssl_transfer_scale1200_ep3/metrics.json`
  - `output/video_ssl_transfer_scale1200_diff/metrics.json`
  - `output/video_ssl_transfer_balanced1200_mae_e8_f6/metrics.json`
  - `output/video_ssl_transfer_all_data_mae_e8_f6/metrics.json`
  - `output/video_ssl_transfer_balanced1132_mae_e8_f6/metrics.json`
