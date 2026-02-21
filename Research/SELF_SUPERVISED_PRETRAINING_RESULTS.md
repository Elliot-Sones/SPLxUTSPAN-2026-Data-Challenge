# Self-Supervised Pretraining Results (CPU)

## Test 1 - Quick Pipeline Validation

**Date**: 2026-02-03

**Command**:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/self_supervised_pretrain.py \
  --subset 30 \
  --epochs 1 \
  --batch-size 8 \
  --frame-stride 8 \
  --hidden-dim 256 \
  --latent-dim 64 \
  --mask-ratio 0.15 \
  --lr 1e-3 \
  --seed 42 \
  --output-dir output/ssl_pretrain_quick
```

**Data**:
- Train: `data/train.csv` (random subset of 30 samples)
- Test: `data/test.csv` (random subset of 30 samples)

**Model**:
- Masked MLP autoencoder
- Input: flattened keypoints with frame stride = 8
- Mask ratio: 0.15
- Encoder: Linear(?,256) -> ReLU -> Linear(256,64) -> ReLU
- Decoder: Linear(64,256) -> ReLU -> Linear(256,?)

**Validation**:
- GroupKFold by participant_id
- Ridge regression (alpha=1.0) on encoder features

**Results**:
- epoch 1 loss = 1.018230
- cv_mse_angle = 59.700432
- cv_mse_depth = 40.680072
- cv_mse_left_right = 10.740811
- cv_mse_total = 37.040438

---

## Test 2 - Full Data, 1 Epoch (Scale-Up)

**Date**: 2026-02-03

**Command**:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/self_supervised_pretrain.py \
  --epochs 1 \
  --batch-size 8 \
  --frame-stride 8 \
  --hidden-dim 256 \
  --latent-dim 64 \
  --mask-ratio 0.15 \
  --lr 1e-3 \
  --seed 42 \
  --output-dir output/ssl_pretrain_full_ep1
```

**Data**:
- Train: `data/train.csv` (345 samples)
- Test: `data/test.csv` (113 samples)

**Model**:
- Same as Test 1

**Validation**:
- GroupKFold by participant_id
- Ridge regression (alpha=1.0) on encoder features

**Results**:
- epoch 1 loss = 0.716211
- cv_mse_angle = 24.688342
- cv_mse_depth = 393.225067
- cv_mse_left_right = 42.554415
- cv_mse_total = 153.489275

---

## Notes
- Outputs saved to `output/ssl_pretrain_quick/` and `output/ssl_pretrain_full_ep1/`.
- Checkpoints saved as `mae_checkpoint.pt` with config and normalization stats.
