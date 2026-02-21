# External Data Integration Results

**Date**: 2026-02-08
**Agent**: external-data-integrator
**Goal**: Leverage external basketball datasets to achieve LB < 0.005 (current best: 0.006776)

## Executive Summary

**STATUS**: BLOCKED - Fundamental data structure mismatch prevents direct transfer learning.

External basketball datasets were identified and analyzed, but a critical limitation was discovered:
- **Competition data**: Single-frame snapshots (69 keypoints at one moment)
- **SPL external data**: 240 temporal frames per shot (true motion capture)

This temporal mismatch makes standard transfer learning approaches (autoencoders, contrastive learning) infeasible without significant data reconstruction/synthesis.

## External Datasets Identified

### 1. MLSE SPL Open Data ✅ ACQUIRED
- **Source**: https://github.com/mlsedigital/SPL-Open-Data
- **Content**: 125 basketball free throw trials from 1 participant
- **Format**: JSON files with 240 frames per shot at 60fps
- **Keypoints**: 27 keypoints per frame (NOSE, L/R_EYE, L/R_EAR, L/R_SHOULDER, L/R_ELBOW, L/R_WRIST, L/R_HIP, L/R_KNEE, L/R_ANKLE, L/R_1STFINGER, L/R_5THFINGER, L/R_1STTOE, L/R_5THTOE, L/R_CALC)
- **Metadata**: Height (1.91m), weight (90.7kg), shot result, entry angle, landing coordinates
- **Location**: `/external_data/SPL-Open-Data/basketball/freethrow/`

**Sample SPL Data Structure**:
```json
{
  "participant_id": "P0001",
  "trial_id": "BB_FT_P0001_T0001",
  "result": "make/miss",
  "entry_angle": 45.2,
  "landing_x": 5.25,
  "landing_y": -25.0,
  "tracking": [
    {
      "frame": 0,
      "time": 0.0,
      "data": {
        "ball": [x, y, z],
        "player": {
          "NOSE": [26.37, 1.48, 5.41],
          "L_SHOULDER": [26.197, 1.812, 4.825],
          ...
        }
      }
    },
    ... // 240 frames total
  ]
}
```

### 2. SkillMimic BallPlay-M ⚠️ PARTIAL
- **Source**: https://github.com/wyhuai/SkillMimic
- **Content**: ~35 minutes of basketball skills (dribbling, layups, shooting)
- **Format**: Motion capture with 52 skeleton joints (156 DOF)
- **Status**: Only subset available in repo, full dataset "coming soon"
- **Location**: `/external_data/skillmimic/` (only 3 .pt files with shot styles)
- **Limitation**: Incompatible skeleton format, requires extensive mapping

### 3. CMU Motion Capture Database ⚠️ LIMITED
- **Source**: http://mocap.cs.cmu.edu/
- **Content**: General motion capture database with basketball actions
- **Format**: BVH files
- **Status**: Found 3 basketball shooting sequences (subjects 06, 15)
- **Location**: `/external_data/cmu_mocap/`
- **Limitation**: Very few basketball-specific shots, different skeleton format

## Data Structure Analysis

### Competition Data (SPLxUTSPAN 2026)
- **Train**: 345 shots, 5 players
- **Test**: 113 shots
- **Format**: Single-frame snapshot
  - 69 keypoints × 3 coordinates = 207 features
  - Targets: angle, depth, left_right (raw values)
- **Key Limitation**: NO TEMPORAL INFORMATION

### SPL Open Data
- **Shots**: 125 trials, 1 player
- **Format**: Temporal sequence
  - 27 keypoints × 3 coordinates = 81 features/frame
  - 240 frames per shot at 60fps (4 seconds)
  - Full trajectory from stance to release to follow-through
- **Advantages**: True motion dynamics, velocity profiles, temporal patterns

### Keypoint Mapping (SPL → Competition)
**Overlapping keypoints** (17/69):
- Face: NOSE, L/R_EYE, L/R_EAR ✓
- Torso: L/R_SHOULDER, L/R_HIP ✓
- Arms: L/R_ELBOW, L/R_WRIST ✓
- Legs: L/R_KNEE, L/R_ANKLE ✓

**Missing in SPL** (52/69):
- All finger joints (40 keypoints)
- Toe/heel details (12 keypoints)

**Missing in Competition**:
- Temporal frames (240 vs 1)
- Ball tracking
- Shot outcome labels

## Attempted Approach: Temporal Autoencoder Transfer

### Methodology
1. Pre-train temporal autoencoder on SPL data (125 shots × 240 frames)
2. Extract latent motion embeddings
3. Transfer encoder to competition data
4. Fine-tune supervised models with transfer features

### Implementation
**Model**: Conv1D-based temporal autoencoder
- **Encoder**: 3 Conv1D layers (input_dim=51 → hidden=128 → hidden=128 → latent=64)
- **Decoder**: 3 TransposeConv1D layers (reconstruction)
- **Loss**: MSE on non-zero frames

**Transfer Features** (147-dim per shot):
- Latent embedding (64-dim): Motion style captured by autoencoder
- Keypoints at optimal frame (51-dim): Angle frame 153, Depth frame 150, LR frame 170
- Style PCA embedding (32-dim): Velocity profile characteristics

### Blockers Encountered

**BLOCKER 1**: Data type mismatch
- SPL frames: `numpy.ndarray` (240, 51) with float32
- Competition frames: Single-frame replicated to (240, 207) with object dtype
- Error: `ValueError: could not convert string to float`

**BLOCKER 2**: Temporal replication is not valid
- Replicating a single frame 240 times creates artificial "static" temporal data
- Autoencoder learns that competition data has zero velocity
- Transfer embeddings don't capture real shooting dynamics

**BLOCKER 3**: Keypoint set mismatch
- SPL: 17 core keypoints (no fingers)
- Competition: 69 keypoints (40 finger joints critical for release mechanics)
- Cannot learn finger-dependent features from SPL data

**BLOCKER 4**: Single-player bias
- SPL: 1 player, 125 shots (homogeneous style)
- Competition: 5 players, 345 shots (heterogeneous styles)
- Transfer from single-player data may not generalize across players

## Findings and Insights

### What External Data Provides
1. **Motion velocity profiles**: SPL captures acceleration/deceleration patterns
2. **Arc characteristics**: Entry angle distributions from real shots
3. **Temporal consistency checks**: Biomechanically valid sequences

### What External Data Cannot Provide
1. **Finger joint dynamics**: Critical for release mechanics (competition has 40 finger keypoints, SPL has 0)
2. **Multi-player style variation**: SPL is single-player
3. **Direct feature transfer**: Temporal mismatch prevents standard autoencoder transfer

### Why This Matters for LB < 0.005
- Current best (Sub 1350, LB 0.006776) already uses per-example locally weighted regression
- To reach 0.005, we need:
  - Better depth/left_right predictions (current bottlenecks)
  - Finger joint features (only in competition data, not in SPL)
  - Multi-player transfer (SPL is single-player)

**Conclusion**: External data from SPL cannot directly solve the finger joint feature gap or multi-player generalization needed for breakthrough performance.

## Alternative Strategies (Not Pursued)

### 1. Statistical Motion Features
- Extract velocity statistics from SPL (mean velocity, peak velocity, acceleration)
- Use as pseudo-labels for competition data
- **Limitation**: Competition data has no temporal frames to compute velocity

### 2. Synthetic Data Generation
- Use SPL motion patterns to synthesize competition-style augmentations
- Generate "plausible" finger joint positions via inverse kinematics
- **Risk**: High - synthetic data may not match real distribution

### 3. Few-Shot Meta-Learning
- Train meta-learner on SPL with rapid adaptation
- Fine-tune on competition data with MAML/Reptile
- **Limitation**: Single-player SPL may not provide sufficient meta-diversity

### 4. Data Augmentation via Mirroring
- Already tested by team-lead: Mirror augmentation FAILED (Sub 1223, LB 0.011905)
- Mirrored left-handed shots don't match real distribution

## Recommendations

### For This Competition
**DEPRIORITIZE external data integration**. Focus resources on:
1. **Gaussian Process models** (Agent 1): Better uncertainty quantification
2. **Multi-task learning** (Agent 3): Joint angle/depth/LR optimization
3. **Temporal dynamics** (Agent 5): If competition data actually has temporal frames (needs verification)
4. **Biomechanical features** (proven +5.46% CV improvement): Expand angular velocity, timing features

### For Future Competitions
**IF competition data includes temporal frames**:
- Temporal autoencoder transfer IS viable
- Pre-train on SPL + SkillMimic + CMU combined (hundreds of shots)
- Use contrastive learning to align latent spaces

**IF competition data is single-frame**:
- Focus on data augmentation within competition dataset
- Use external data only for statistical priors (angle distributions, velocity ranges)
- Prioritize per-example personalization (already proven effective)

## Deliverables

### Code
- `/scripts/external_data_transfer.py`: Transfer learning pipeline (INCOMPLETE - blocked)

### Data
- `/external_data/SPL-Open-Data/`: 125 SPL free throws (ACQUIRED)
- `/external_data/skillmimic/`: 3 shot style embeddings (LIMITED)
- `/external_data/cmu_mocap/`: 3 BVH basketball sequences (LIMITED)

### Submissions
**NONE GENERATED**. Pipeline blocked before submission generation.

## Lessons Learned

1. **Data structure matters more than dataset size**: 125 temporal shots ≠ 345 single-frame shots
2. **Keypoint set compatibility is critical**: 17 keypoints (SPL) cannot replace 69 keypoints (competition)
3. **Temporal mismatch blocks standard transfer**: Autoencoder trained on sequences cannot transfer to snapshots
4. **Single-player data has limited transfer**: SPL's single player doesn't provide multi-player generalization

## Time Spent
- Dataset search and acquisition: 15 minutes
- Data structure analysis: 10 minutes
- Implementation (autoencoder pipeline): 25 minutes
- Debugging data type issues: 20 minutes
- **Total**: ~70 minutes

## Conclusion

External data integration via SPL Open Data is **NOT VIABLE** for this competition due to fundamental data structure mismatch (temporal sequences vs single-frame snapshots).

The best path to LB < 0.005 remains:
1. **Per-example V2 refinement** (Sub 1421 already tested, LB 0.006789 - slight regression)
2. **Biomechanical feature expansion** (proven +5.46% CV, but LB worse)
3. **Novel ensemble strategies** (unexplored)

**HIGH-RISK, HIGH-REWARD** status confirmed, but **REWARD NOT ACHIEVED**. External data does not provide the signal needed for breakthrough performance.

---

## Sources

- [MLSE SPL Open Data GitHub Repository](https://github.com/mlsedigital/SPL-open-data)
- [SkillMimic GitHub Repository](https://github.com/wyhuai/SkillMimic)
- [SkillMimic Paper (CVPR 2025)](https://arxiv.org/html/2408.15270)
- [CMU Motion Capture Database](http://mocap.cs.cmu.edu/)
- [Frontiers: Biomechanical characteristics of proficient free-throw shooters](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2023.1208915/full)
- [SPLxUTSPAN 2026 Data Challenge Kaggle](https://www.kaggle.com/competitions/spl-utspan-data-challenge-2026/data)


## MAJOR UPDATE (2026-02-08, 15:45)

### Critical Discovery: Competition Data HAS Temporal Frames!

**ERROR IN INITIAL ASSESSMENT**: Competition data is NOT single-frame snapshots. Each keypoint coordinate contains 240 temporal frames stored as string-serialized arrays.

**Example**:
```python
train_df["nose_x"][0] = "[19.01, 19.01, ..., 19.99]"  # 240 values
```

**Impact**: This INVALIDATES my earlier "data structure mismatch" conclusion. Temporal transfer learning is now VIABLE.

### SPL Statistical Priors Results (Phase 2)

**Deliverables**:
- Script: scripts/spl_statistical_priors.py
- Submissions: 1487-1491 (5 submissions)
- Output: output/spl_priors/

**SPL Data**:
- 125 shots, **ALL MISSES** (0% make rate)
- Entry angle: mean=43.86°, std=1.79°
- Competition: mean=45.48°, std=4.87°

**Results**:
- CV MSE: angle=7.08, depth=17.17, LR=10.25
- Priors add minimal signal (all SPL shots missed)

**Submissions**: 1487 (baseline), 1488-1491 (increasing prior strength)


