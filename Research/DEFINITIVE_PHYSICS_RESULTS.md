# Definitive Physics Results

## Final Maximum Achievable R-squared

After exhaustive testing of all gaps (velocity windows 1-20, all body parts, interactions, multi-frame, jerk, etc.), here are the confirmed best results:

### Summary Table

| Target | Player | Best R-squared | Best Feature | Frame | Source Test |
|--------|--------|---------------|--------------|-------|-------------|
| **ANGLE** | 1 | 0.284 | combined (fingertip+wrist_snap+guide) | 165 | finger_deep_dive |
| **ANGLE** | 2 | 0.250 | left_knee pos+vel | 110 | exhaustive_search |
| **ANGLE** | 3 | 0.188 | left_fourth_finger_distal_vel_w5 | 105 | exhaustive_search |
| **ANGLE** | 4 | 0.311 | right_shoulder + right_knee | 160 | exhaustive_search |
| **ANGLE** | 5 | 0.378 | combined (fingertip+wrist_snap+guide) | 155 | finger_deep_dive |
| | | **AVG: 0.282** | | | |
| **DEPTH** | 1 | 0.632 | right_second_finger_distal 3frames | 150 | exhaustive_search |
| **DEPTH** | 2 | 0.569 | right_elbow_vel_w10 | 95 | comprehensive_groups |
| **DEPTH** | 3 | 0.519 | left_ankle_vel_w2 | 130 | gaps_test |
| **DEPTH** | 4 | 0.589 | arm_chain_full | 135 | comprehensive_groups |
| **DEPTH** | 5 | 0.728 | right_second_finger_mcp position | 145 | exhaustive_search |
| | | **AVG: 0.607** | | | |
| **LEFT_RIGHT** | 1 | 0.544 | guide_hand | 175 | comprehensive_groups |
| **LEFT_RIGHT** | 2 | 0.749 | right_wrist+left_wrist+right_hip | 150 | gaps_test |
| **LEFT_RIGHT** | 3 | 0.672 | body_alignment | 175 | comprehensive_groups |
| **LEFT_RIGHT** | 4 | 0.441 | left_elbow_vel_w10 | 150 | gaps_test |
| **LEFT_RIGHT** | 5 | 0.423 | right_ear_vel_w2 | 175 | gaps_test |
| | | **AVG: 0.566** | | | |

---

## Progression Through Testing

### ANGLE

| Test | Avg R-squared | Improvement |
|------|--------------|-------------|
| Initial Z-only | 0.052 | baseline |
| Comprehensive Groups | 0.148 | +185% |
| Finger Deep Dive | 0.202 | +36% |
| Exhaustive Search | 0.261 | +29% |
| **Best Combined** | **0.282** | +8% |

### DEPTH

| Test | Avg R-squared | Improvement |
|------|--------------|-------------|
| Initial | 0.387 | baseline |
| Comprehensive Groups | 0.590 | +52% |
| **Best Combined** | **0.607** | +3% |

### LEFT_RIGHT

| Test | Avg R-squared | Improvement |
|------|--------------|-------------|
| Initial | 0.226 | baseline |
| Comprehensive Groups | 0.517 | +129% |
| **Best Combined** | **0.566** | +9% |

---

## Key Findings

### 1. ANGLE is Explained by Multiple Body Parts

Surprising: Leg position helps predict angle for some players!

- Player 2: left_knee_pos+vel (R²=0.250)
- Player 4: right_shoulder + right_knee (R²=0.311)
- Player 5: left-right body asymmetry (R²=0.321)

This suggests angle is influenced by full-body mechanics, not just the arm.

### 2. Velocity Window Matters

Optimal velocity windows vary by feature:
- Fingertip velocity: window=1-3 frames (fast motion)
- Elbow velocity: window=10 frames
- Wrist velocity: window=3-5 frames
- Knee velocity: window=15-20 frames

### 3. Three-Part Combinations Beat Two-Part

For LEFT_RIGHT:
- Two parts: R²=0.606 (right_wrist + right_hip)
- Three parts: R²=0.749 (right_wrist + left_wrist + right_hip)

### 4. Unexpected Body Parts Help

- **Ears/Nose**: Help predict depth (head position indicates body lean)
- **Left fingers**: Help predict angle (guide hand stability)
- **Toes**: Help predict angle for some players (balance)

### 5. Player 3 Remains Hardest

- ANGLE: Max R²=0.188 (vs 0.378 for Player 5)
- Different shooting mechanics or more noise in data

---

## Feature Definitions

### ANGLE Features (by player)

**Player 1**: Combined fingertip features at frame 165
```python
features = [
    right_second_finger_distal_xyz,
    right_third_finger_distal_xyz,
    fingertip_velocity_xyz (window=3),
    wrist_snap (wrist_vel - elbow_vel),
    left_wrist_xyz,
]
```

**Player 4**: Shoulder + knee at frame 160
```python
features = [
    right_shoulder_xyz,
    right_knee_xyz,
]
```

**Player 5**: Wrists + hip + asymmetry at frame 155
```python
features = [
    right_wrist_xyz,
    left_wrist_xyz,
    right_hip_xyz,
    right_wrist - left_wrist (asymmetry),
    right_shoulder - left_shoulder (asymmetry),
    right_hip - left_hip (asymmetry),
]
```

### DEPTH Features (by player)

**Player 1**: Multi-frame fingertip at frames 150-160
```python
features = [
    right_second_finger_distal_xyz @ frame 150,
    right_second_finger_distal_xyz @ frame 155,
    right_second_finger_distal_xyz @ frame 160,
]
```

**Player 5**: Finger knuckle position at frame 145
```python
features = [
    right_second_finger_mcp_xyz,
]
```

### LEFT_RIGHT Features (by player)

**Player 2**: Both wrists + hip at frame 150
```python
features = [
    right_wrist_xyz,
    left_wrist_xyz,
    right_hip_xyz,
]
```

**Player 3**: Body alignment at frame 175
```python
features = [
    right_shoulder - left_shoulder,
    right_hip - left_hip,
    right_elbow - left_elbow,
]
```

---

## Confidence Level

**HIGH confidence** that these are near-optimal physics features:

1. Tested ALL 69 body parts
2. Tested ALL velocity windows 1-20
3. Tested position, velocity, acceleration, jerk
4. Tested 2-part and 3-part combinations
5. Tested feature interactions (products, ratios)
6. Tested multi-frame temporal patterns
7. Tested left-right asymmetry features
8. Tested fine frame granularity (every 2 frames)

**Remaining unexplored**:
- Non-linear transformations (polynomials)
- Neural network feature extraction
- More complex temporal patterns (RNN-style)

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/keypoint_exploration.py` | List all available keypoints |
| `scripts/comprehensive_physics_groups.py` | Test physics-based groups |
| `scripts/angle_finger_deep_dive.py` | Deep dive into angle with fingers |
| `scripts/exhaustive_physics_search.py` | Test all body parts |
| `scripts/exhaustive_gaps_test.py` | Test all remaining gaps |
| `scripts/final_best_features.py` | Combine best configs |
| `output/comprehensive_physics_groups.csv` | Full results |
| `output/angle_finger_deep_dive.csv` | Angle finger results |
| `output/exhaustive_physics_search.csv` | Exhaustive search results |
| `output/exhaustive_gaps_test.csv` | Gaps test results |
