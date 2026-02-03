# NBA Data Analysis for Basketball Shot Prediction

## Data Source
- Kaggle dataset: NBA basketball shooting data
- Downloaded to `external_data/` folder

## Data Structure

### player_metrics.csv (189 players)
| Column | Description | Mean | Std |
|--------|-------------|------|-----|
| rv | Release velocity (total) | 13.71 | 1.66 |
| rvx | Release velocity X (horizontal) | 10.45 | 1.62 |
| rvy | Release velocity Y (lateral) | 0.37 | 1.02 |
| rvz | Release velocity Z (vertical) | 12.28 | 1.58 |
| rx, ry, rz | Release position | varies | varies |
| plr | Path length ratio | ~1.0 | varies |
| hght | Player height | varies | varies |

### path_detail.csv (79,776 rows)
- Frame-by-frame ball trajectory data AFTER release
- Columns: player_id, frame, x, y, z

## Key Correlations with Shooting Quality

Using 1/plr (path length ratio) as quality proxy:
| Feature | Correlation |
|---------|-------------|
| rvx (horizontal velocity) | +0.320 |
| rz (release height) | +0.198 |
| rv (total velocity) | +0.087 |
| rvz (vertical velocity) | -0.058 |

## Fundamental Challenge

**The data modalities don't match:**

| Our Data | NBA Data |
|----------|----------|
| 33 body keypoints (x,y,z) | Ball position (x,y,z) |
| Body pose at each frame | Ball trajectory at each frame |
| BEFORE ball release | AFTER ball release |
| Motion capture | Ball tracking |

The mapping is:
```
Body mechanics -> Ball release parameters -> Ball trajectory -> Shot outcome
       ^                                            ^
   Our data                                    NBA data
```

There is a GAP between body pose and ball trajectory that we cannot directly bridge.

## Approaches Tested

### 1. Direct Transfer (scripts/transfer_from_nba.py)
- Method: Create pseudo-labels from NBA data, train on combined
- Result: Sub 252, LB 0.008240 (WORSE than Sub 219)
- Why it failed: Ball trajectory doesn't predict body pose

### 2. NBA-Guided Features (scripts/nba_guided_features.py)
- Method: Use NBA optimal patterns to create physics features
- Result: Sub 254-256, angle_std=0.135
- Why it likely failed: Added variance without adding signal

### 3. Physics-Based Model (scripts/physics_nba_model.py)
- Method: Estimate ball release from wrist velocity, compare to NBA
- Result: CV 0.033366 (poor), Sub 257-260
- Why it failed: Wrist velocity estimate is noisy

### 4. NBA Regularization (scripts/nba_regularized_model.py)
- Method: Penalize predictions with implausible release parameters
- Result: Plausibility scores only 0.33 (scale mismatch)
- Why it failed: Data scales are completely different

## Why NBA Data Cannot Help

1. **Different Modalities**: NBA tracks BALL, we track BODY
2. **Different Timing**: NBA is AFTER release, ours is BEFORE
3. **Different Scales**: NBA data in feet, ours in normalized coordinates
4. **Indirect Relationship**: Body pose -> ball release is many-to-one mapping
5. **Player-Specific**: Each player has unique mechanics; NBA patterns don't generalize

## What External Data WOULD Help

The ideal external data would be:
1. **Same modality**: Body pose data (motion capture)
2. **Same timing**: Pre-release and during release
3. **Same sport**: Basketball free throws
4. **With labels**: Shot outcome or deviation metrics
5. **Large scale**: Thousands of samples across many players

Such data does not appear to be publicly available.

## Conclusion

**NBA basketball data cannot meaningfully improve our predictions because:**
1. It describes ball physics, not body mechanics
2. The mapping from body pose to ball trajectory is noisy and player-specific
3. All attempts have made LB scores WORSE, not better

**Recommendation**: Focus on better using the data we have (motion capture) rather than trying to incorporate external ball trajectory data.

## Files Created
- scripts/transfer_from_nba.py (Sub 252)
- scripts/nba_guided_features.py (Sub 254-256)
- scripts/physics_nba_model.py (Sub 257-260)
- scripts/nba_regularized_model.py (Sub 261-264)

## Best Submission Remains
Sub 219: LB 0.007682 (no NBA data involved)
