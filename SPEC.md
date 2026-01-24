# SPLxUTSPAN 2026 Data Challenge - Implementation Specification

## Objective
Predict basketball free throw shot landing outcomes (angle, depth, left/right) from biomechanical motion capture data to win first place.

## Success Criteria
- **Quantitative (60%)**: Minimize MSE on scaled predictions - target top 1-3 on leaderboard
- **Qualitative (40%)**: Novel, well-documented, scientifically grounded methodology

---

## Architecture

### Data Pipeline
```
train.csv (324MB) -> Chunked Reader -> Per-Shot Processing -> Feature Cache
                                                           -> NumPy Arrays
```

### Feature Tiers

**Tier 1 - Essential** (~3000 features):
- Per-keypoint stats: mean, std, min, max, range, percentiles
- Velocity features: mean/max velocity, velocity at release
- Acceleration features: mean/max acceleration

**Tier 2 - Biomechanics** (~100 features):
- Joint angles: elbow, shoulder, knee, hip, trunk
- Release point detection and features
- Coordination metrics

**Tier 3 - Advanced** (~200 features):
- Phase-based features (4 phases x key joints)
- Frequency domain (FFT)
- Participant-relative normalization

### Models

1. **XGBoost Baseline**: Engineered features + GroupKFold
2. **LightGBM**: Alternative gradient boosting
3. **CNN-BiLSTM**: Raw time series input (240 x 207)
4. **Final Ensemble**: Weighted blend of all models

---

## Constraints

### Non-Negotiable
- Memory-efficient processing (324MB train file)
- GroupKFold by participant for validation
- Scale targets using provided scalers before submission

### Out of Scope
- Real-time inference optimization
- Model compression/deployment

---

## Implementation Strategy

### Phase 1: Data Pipeline
- Chunked CSV reading
- Per-shot feature extraction with caching
- Numpy float32 throughout

### Phase 2: Feature Engineering
- Implement Tier 1 features first
- Validate on small subset (10 shots)
- Add Tier 2/3 incrementally

### Phase 3: Baseline Models
- XGBoost with Tier 1 features
- Establish benchmark MSE
- Feature importance analysis

### Phase 4: Deep Learning
- CNN-BiLSTM on raw sequences
- Train on GPU (vast.ai)
- Compare to gradient boosting

### Phase 5: Ensemble
- Blend best models
- Optimize weights on validation

---

## File Structure
```
src/
  __init__.py
  data_loader.py        # Memory-efficient data loading
  feature_engineering.py # Feature extraction
  models/
    __init__.py
    baseline.py         # XGBoost/LightGBM
    deep_learning.py    # CNN-BiLSTM
  train.py              # Training with CV
  predict.py            # Generate submissions
  utils.py              # Helpers
```

---

## Validation Plan

1. **Data pipeline test**: Load 5 shots, verify shapes
2. **Feature test**: Extract features for 5 shots, check no NaNs
3. **Model test**: Train XGBoost on 80%, predict 20%
4. **CV test**: 5-fold GroupKFold, report per-fold MSE
5. **Submission test**: Generate submission.csv, verify format

---

## Open Questions
- Optimal release frame detection method: TBD (max wrist_z vs max velocity)
- Best ensemble weights: TBD (optimize on validation)
- Deep learning vs gradient boosting performance gap: TBD (benchmark)
