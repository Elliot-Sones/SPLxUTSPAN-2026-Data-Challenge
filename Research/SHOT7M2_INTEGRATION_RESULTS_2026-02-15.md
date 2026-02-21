# SHOT7M2 Integration Results - 2026-02-15

## Dataset
- SHOT7M2: 7.2M frames of synthetic basketball motion (Neural State Machine)
- 26 joints, 1280 episodes x 1800 frames at 30fps
- 14 joints map to our 69-keypoint skeleton (body+limbs, NO hands)
- ~14,000 shooting frames (action_Shoot confidence > 0.3)
- Source: https://huggingface.co/datasets/amathislab/SHOT7M2

## Joint Mapping (14 joints)
| Our data | SHOT7M2 |
|----------|---------|
| mid_hip | center (0) |
| left/right_hip | l_hip (1) / r_hip (6) |
| left/right_knee | l_knee (2) / r_knee (7) |
| left/right_ankle | l_ankle (3) / r_ankle (8) |
| left/right_shoulder | l_shoulder (16) / r_shoulder (23) |
| left/right_elbow | l_elbow (17) / r_elbow (24) |
| left/right_wrist | l_wrist (18) / r_wrist (25) |
| neck | neck (19) |

Missing from SHOT7M2: all hand/finger joints (40 of our 69 keypoints), spine, feet details, head/face

## Features Extracted
1. PCA projections (10 components) from shooting pose PCA
2. PCA reconstruction error
3. Joint angle z-scores vs shooting distribution (10 angles)
4. Raw joint angles (10 angles)
5. Euclidean distance to shooting pose centroid
6. Arm configuration features (wrist heights, arm extension, asymmetry)

Total: ~37 raw features, compressed via PLS per player

## Normalization
- Center on pelvis (mid_hip)
- Scale by torso length (hip-to-neck distance)
- Makes poses body-size invariant

## Results (Honest LOO, PLS refit)

### Simple pipeline test (shot7m2_features.py)
| Config | mean LOO | vs baseline |
|--------|----------|-------------|
| SHOT7M2 only | 0.009796 | +0.60% |
| Hoop-relative only | 0.009738 | -- |
| Hoop + SHOT7M2 | 0.009560 | -1.82% |

### Integration test (shot7m2_integration.py)
| Config | angle | depth | LR | mean | vs baseline |
|--------|-------|-------|-----|------|-------------|
| hoop only | 0.012204 | 0.012400 | 0.010463 | 0.011689 | -- |
| hoop + S7 (1 PLS) | 0.012091 | 0.012530 | 0.010213 | 0.011611 | -0.66% |
| hoop + S7 (3 PLS) | 0.011953 | 0.012438 | 0.010216 | 0.011535 | -1.31% |
| **hoop + S7 (5 PLS)** | **0.011616** | **0.012432** | **0.010036** | **0.011362** | **-2.80%** |

Per-target: angle -4.82%, depth +0.26%, LR -4.08%

## Interpretation
- SHOT7M2 helps angle and LR (body posture contributes to these targets)
- SHOT7M2 does NOT help depth (needs hand/finger data which SHOT7M2 lacks)
- The -2.80% honest LOO improvement is modest but real
- This is on a SIMPLER pipeline than Sub 2503 (no KC-PLS, multi-frame, etc.)
- Improvement may shrink when added to the full pipeline (feature overlap)

## Submissions
- Sub 2573: Standalone hoop + SHOT7M2 (5 PLS)
- Sub 2574: 5% SHOT7M2 + 95% Sub2503
- Sub 2575: 10% SHOT7M2 + 90% Sub2503
- Sub 2576: 15% SHOT7M2 + 85% Sub2503
- Sub 2577: 20% SHOT7M2 + 80% Sub2503
- Sub 2578: 30% SHOT7M2 + 70% Sub2503

## Scripts
- scripts/shot7m2_features.py - initial feature extraction and simple pipeline test
- scripts/shot7m2_integration.py - integration with hoop-relative features

## LB Result
- Sub 2575 (10% SHOT7M2 + 90% Sub 2503): **LB 0.006502** (+0.48% vs Sub 2503's 0.006471)
- WORSE than Sub 2503 despite high diversity (r=0.63-0.69 on depth/LR)
- Diversity was real but accuracy too low - synthetic data quality gap too large
- CONCLUSION: SHOT7M2 is a DEAD END. Do not invest further.

## Risk Assessment (confirmed)
- Synthetic data mismatch: SHOT7M2 is game-engine output, not real mocap - CONFIRMED as fatal flaw
- Only 14 of 69 joints covered, missing the most predictive hand features
- -2.80% honest LOO in simple pipeline did NOT transfer to LB
- High diversity (r=0.63-0.69) was not enough to overcome low accuracy
