# OpenBiomechanics Baseball Transfer Learning Results

## Summary

**Approach**: Use OpenBiomechanics baseball pitching data (411 pitches) to learn biomechanical patterns, then transfer to basketball shot prediction.

**Result**: Transfer learning did NOT improve predictions. The kinematic features from baseball do not transfer well to basketball shooting.

## Baseball Model Performance

### POI Metrics Only (transfer_from_baseball.py)
- Features: 16 summary kinematic metrics
- R2: 0.128 +/- 0.239 (very weak)
- Basketball CV MSE: 0.050989 (poor)

### Full Time-Series (baseball_timeseries_transfer.py)
- Features: 180 features extracted at ball release (BR_time)
- R2: 0.186 +/- 0.144 (still weak)
- Basketball angle CV: 0.0796 (poor)
- Basketball depth CV: 0.0332
- Basketball lr CV: 0.0171

### Top Baseball Features (predicting pitch speed)
| Feature | Importance |
|---------|------------|
| velo_shoulder_angle_z | 1.46 |
| mer_pelvis_angle_x | 1.30 |
| br_rear_knee_angle_x | 1.14 |
| br_elbow_angle_x | 1.12 |
| velo_rear_hip_angle_x | 0.99 |

## Basketball Feature Correlations

When applying similar angle-based features to basketball:

| Feature | Correlation with Angle |
|---------|----------------------|
| wrist_x_release | 0.5953 (best - simple position) |
| wrist_z_release | -0.4701 |
| change_mer_br_elbow_angle_x | -0.3107 |
| velo_elbow_angle_x | -0.3107 |
| br_pelvis_angle_z | -0.2730 |

**Key Observation**: The best basketball feature is a simple position (wrist_x_release), not a transferred kinematic pattern. The angular features learned from baseball have weak correlations (0.27-0.31).

## Why Transfer Learning Failed

1. **Different mechanics**: Baseball pitching is an overhand throw with extreme external rotation. Basketball shooting is an underhand arc motion with wrist snap.

2. **Different body positions**: Pitching involves forward momentum, leg drive, and torso rotation. Shooting is more stationary with vertical extension.

3. **Different target variables**: Pitch speed (linear velocity) vs shot outcome (angle/depth/left-right accuracy).

4. **Feature mismatch**: The most predictive baseball features (pelvis rotation, hip angles, rear knee) are not relevant to a standing basketball shot.

## Submissions Created

| Submission | Description | angle_std | Correlation with Sub 219 |
|------------|-------------|-----------|-------------------------|
| Sub 305 | POI metrics transfer | 0.129 | 0.91 |
| Sub 306-308 | Blends with Sub 219 | 0.132-0.135 | - |
| Sub 310 | Time-series transfer | 0.110 | 0.84 |
| Sub 311-314 | Blends with Sub 219 | 0.127-0.133 | - |

## Conclusion

**OpenBiomechanics baseball data does not help basketball shot prediction.** The fundamental mechanics are too different for meaningful transfer learning.

### Alternative External Data That Might Help
1. **Actual basketball motion capture** with known shot outcomes
2. **Shot tracking data** with release velocity and angle measurements
3. **NBA player shooting form analysis** with make/miss labels

### Lessons Learned
- Transfer learning requires similar tasks and mechanics
- Position features (wrist location) are more predictive than angular features for basketball
- The ball release dynamics in pitching vs shooting are fundamentally different
