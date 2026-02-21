# External Motion Pretraining Results

## Date: 2026-02-09

## Hypothesis
A temporal encoder pretrained on thousands of external motion sequences could learn
temporal dynamics that our 345-sample Ridge model cannot capture, providing complementary
features that improve prediction or enable better ensembling through diversity.

## Method

### Phase 1: External Motion Data
- Generated 5000 synthetic motion sequences (17 common body joints x 3 coords x 240 frames)
- 4 motion types: oscillatory (walking/running), ballistic (throwing), basketball-like (shooting), smooth random
- Also parsed 3 BVH files from CMU mocap (06_14_shoot, 06_15_shoot, 15_12_layup)
- Total: 5003 pretrain sequences
- Common joints: nose, neck, shoulders, elbows, wrists, hips, knees, ankles, mid_hip, toes (17 joints)

### Phase 2: Self-Supervised Pretraining
- Architecture: 1D CNN temporal encoder (5 conv layers with BN and ReLU)
- Pretext task: Masked frame prediction (15% of frames masked, predict original values)
- Training: 30 epochs, AdamW (lr=1e-3, wd=1e-4), cosine LR schedule, batch_size=128
- Device: MPS (Apple Silicon)
- Three embedding dimensions tested: 32, 64, 128

### Phase 3: Feature Extraction
- Passed 345 train + 113 test basketball shots through pretrained encoder
- Used only the 17 common body joints from our 69-keypoint data
- Extracted global average-pooled embeddings

### Phase 4: Evaluation
- Locally weighted Ridge regression (per-example, bandwidth=0.5 quantile, alpha=10)
- LOO CV within each player
- Compared: pretrained-only, HC+PLS+pretrained combined, and baseline HC+PLS

## Results

### Pretrain Loss
| embed_dim | Final Loss |
|-----------|-----------|
| 32        | 0.3338    |
| 64        | 0.2675    |
| 128       | 0.2318    |

### LOO CV MSE by Configuration

#### Angle (baseline: 0.002511)
| Config | MSE | Delta | r(Sub 1350) |
|--------|-----|-------|-------------|
| Baseline HC+PLS | 0.002511 | - | 0.9334 |
| PT only (32) | 0.007340 | +192.4% | 0.9782 |
| PT only (64) | 0.007447 | +196.6% | 0.9724 |
| PT only (128) | 0.008284 | +230.0% | 0.9698 |
| Combined (32) | 0.002586 | +3.0% | 0.9324 |
| Combined (64) | 0.002657 | +5.8% | 0.9316 |
| Combined (128) | 0.002960 | +17.9% | 0.9250 |

#### Depth (baseline: 0.004510)
| Config | MSE | Delta | r(Sub 1350) |
|--------|-----|-------|-------------|
| Baseline HC+PLS | 0.004510 | - | 0.9738 |
| PT only (32) | 0.015155 | +236.0% | 0.4453 |
| PT only (64) | 0.015279 | +238.8% | 0.4940 |
| PT only (128) | 0.015735 | +248.9% | 0.3104 |
| Combined (32) | 0.004634 | +2.7% | 0.9746 |
| Combined (64) | 0.004722 | +4.7% | 0.9725 |
| Combined (128) | 0.004865 | +7.9% | 0.9692 |

#### Left-Right (baseline: 0.004209)
| Config | MSE | Delta | r(Sub 1350) |
|--------|-----|-------|-------------|
| Baseline HC+PLS | 0.004209 | - | 0.9766 |
| PT only (32) | 0.013625 | +223.7% | 0.5449 |
| PT only (64) | 0.012757 | +203.1% | 0.6669 |
| PT only (128) | 0.012328 | +192.9% | 0.5698 |
| Combined (32) | 0.004251 | +1.0% | 0.9749 |
| Combined (64) | 0.004228 | +0.4% | 0.9715 |
| Combined (128) | 0.004391 | +4.3% | 0.9611 |

### Overall Mean MSE
- Baseline: 0.003743
- Best config: 0.003743 (baseline wins on all targets)
- Pretrained features add no improvement

### Diversity Analysis (Pretrained-Only)
| Target | embed_dim | r(Sub 1350) | r(Sub 784) |
|--------|-----------|-------------|------------|
| Angle | 128 | 0.9698 | 0.9698 |
| Depth | 128 | 0.3104 | 0.2867 |
| LR | 32 | 0.5449 | 0.6054 |

Depth and LR pretrained-only predictions are highly diverse (r=0.31, r=0.54) but this diversity
comes with 3-4x worse absolute performance, making it useless for ensembling.

## Submissions Generated
| Sub | Description |
|-----|-------------|
| 1612 | Best pretrained standalone (= baseline, since baseline won) |
| 1613 | Blend with Sub 784 (dw=0.30, lw=0.50) |
| 1614 | Blend with Sub 784 (dw=0.20, lw=0.30) |
| 1615 | Blend with Sub 784 (dw=0.15, lw=0.20) |
| 1616 | 10% pretrained + 90% Sub 1350 |
| 1617 | 20% pretrained + 80% Sub 1350 |
| 1618 | 30% pretrained + 70% Sub 1350 |
| 1619 | 10% diverse pretrained + 90% Sub 1350 |
| 1620 | 15% diverse pretrained + 85% Sub 1350 |

## Conclusions

### Why pretrained features don't help:

1. **Domain mismatch**: Synthetic/generic motions teach general temporal patterns (walking
   cycles, smooth trajectories). Basketball shot outcomes depend on subtle biomechanical
   details at specific frames (153/150/170) - not general dynamics.

2. **Information loss from joint reduction**: Our HC features use all 69 keypoints including
   detailed hand joints (critical for shooting). The pretrained encoder only sees 17 common
   body joints, missing all finger/hand data.

3. **Scale mismatch**: 5000 synthetic sequences is large relative to 345, but the synthetic
   data doesn't share the statistical properties of real basketball shots. A model pretrained
   on walking/throwing motions learns irrelevant temporal patterns.

4. **Adding noise to HC features**: When combined, the pretrained features (32-128 dims) add
   noise dimensions to the already-effective 213 HC+PLS features. The locally weighted Ridge
   uses distances in feature space, and extra noisy dimensions degrade neighbor quality.

5. **Pretrained diversity is unusable**: While pretrained-only predictions are diverse (depth
   r=0.31 with Sub 1350), they're 3-4x worse, so any blend would hurt rather than help.

### Key takeaway
External motion pretraining with synthetic data does NOT transfer to basketball shot outcome
prediction. The task-specific HC features (hoop-relative coordinates, arm mechanics, body
alignment at optimal frames) capture the relevant signal far better than any learned temporal
representation. This confirms that the bottleneck is NOT temporal modeling but rather the
limited number of training samples (345) for target-specific prediction.

## Reproduction
```bash
uv run python scripts/external_motion_pretrain.py
```
- Total runtime: 248.3s on Apple Silicon MPS
- Dependencies: torch, numpy, pandas, sklearn, scipy, joblib, lightgbm, xgboost, catboost
