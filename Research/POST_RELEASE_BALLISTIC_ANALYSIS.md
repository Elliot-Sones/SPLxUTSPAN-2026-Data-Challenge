# Post-Release Ballistic Fitting Analysis

## Date: 2026-02-09

## Objective
Investigate whether fingertip keypoints briefly track the ball after release, enabling parabolic fitting to recover exact release velocity without differentiation noise.

## Hypothesis
After the ball leaves the hand, if fingertip keypoints track the ball for even 3-5 frames, we can fit z(t) = z0 + vz*t - 0.5*g*t^2 to recover velocity. This would avoid finite-difference noise entirely.

## Key Finding: FINGERTIPS DO NOT TRACK THE BALL

The evidence is conclusive: fingertip keypoints do NOT follow a ballistic trajectory after release. They follow the hand's follow-through motion instead.

### Evidence

**1. Fitted gravity is wrong (decisive test)**

If fingertips tracked the ball, the fitted gravitational acceleration should be ~32.174 ft/s^2. Instead:

| Keypoint | Mean g_fit | Within 20% of true g | Negative (going UP) |
|----------|-----------|----------------------|---------------------|
| Fingertip centroid | 13.92 | 18.5% | 25.5% |
| Right thumb distal | 18.12 | 16.3% | 22.7% |
| Right index distal | 14.13 | 19.9% | 25.2% |
| Right middle distal | 11.93 | 19.0% | 27.1% |
| Right ring distal | 12.21 | 19.6% | 27.3% |
| Right pinky distal | 14.10 | 19.1% | 25.3% |
| Right wrist | 21.84 | 26.6% | 8.2% |

- Mean fitted g = 13.9 ft/s^2 (less than half of true gravity)
- 25% of shots show NEGATIVE gravity (fingertips accelerating upward - follow-through)
- Only 18.5% of shots have g_fit within 20% of true value
- Standard deviation of g_fit = 25.2 (huge variability)

Conclusion: fingertips follow the hand's follow-through arc, not the ball's parabolic trajectory.

**2. R^2 values are high but misleading**

- Window=3: R^2 = 1.000 (any 3 points fit a parabola perfectly - no information)
- Window=5: R^2 = 0.975 (high, but this just means the follow-through is smooth)
- Window=7: R^2 = 0.974
- Window=10: R^2 = 0.934

High R^2 proves smoothness, not ballistic motion. The wrong gravity confirms this.

**3. Wrist has better parabolic fit than fingertips**

The wrist (R^2=0.995 at w=5, g_fit=21.8) fits better than fingertips (R^2=0.975, g_fit=13.9). This makes sense: the wrist is a more stable keypoint with less measurement noise. Neither tracks the ball.

**4. Ballistic velocity does NOT correlate with finite-difference velocity**

| Component | Correlation (r) |
|-----------|----------------|
| vx (lateral) | 0.852 |
| vy (toward hoop) | 0.422 |
| vz (vertical) | 0.275 |

The vz correlation is 0.275 - nearly random. This is because the "ballistic" fit captures follow-through deceleration, not ball velocity. The vy (most important for shot outcome) is only r=0.422.

**5. Fingertip spread INCREASES before release, then DECREASES after**

| Timing | Spread | Derivative |
|--------|--------|------------|
| Release-5 | 0.1358 | +0.00134 (opening) |
| Release-1 | - | +0.00278 (opening fast) |
| Release+0 | 0.1478 | +0.00161 |
| Release+1 | - | +0.00035 (nearly stopped) |
| Release+3 | - | -0.00026 (closing) |
| Release+5 | 0.1444 | -0.00232 (closing fast) |
| Release+10 | 0.1393 | -0.00053 |

Fingers open BEFORE release (pushing ball), then CLOSE after release (follow-through relaxation). This is classic shooting mechanics, NOT ball tracking. If fingers tracked the ball they would diverge as the ball moves away.

**6. Fingertip-wrist distance shows follow-through, not ball tracking**

- Fingertip-wrist distance peaks at release+2 (0.434) then decreases
- The divergence rate peaks at release-1 (+0.011), slows at release (+0.008), and reverses by release+5 (-0.004)
- This is the hand extending during release then relaxing - typical follow-through

## Forward Simulation Results

Forward simulation from ballistic-fit velocity is terrible:

| Approach | Angle r | Angle MSE (scaled) |
|----------|---------|-------------------|
| Centroid w=3 | 0.116 | 1.231 |
| Centroid w=5 | 0.018 | 1.459 |
| Centroid w=7 | -0.133 | 1.621 |
| Centroid w=10 | -0.262 | 1.733 |
| Wrist w=5 | -0.460 | 1.706 |

All much worse than random (MSE >> 0.25 for [0,1] targets). The "velocities" extracted from follow-through motion have no predictive value for actual ball trajectory.

## Ballistic Features as Predictive Features

Even without physics correctness, could the ballistic-fit parameters serve as useful features?

### Standalone (ballistic features only):
| Window | Angle MSE | Depth MSE | LR MSE |
|--------|-----------|-----------|--------|
| 3 | 0.013561 | 0.015503 | 0.012164 |
| 5 | 0.011603 | 0.015275 | 0.012695 |
| 7 | 0.010396 | 0.015354 | 0.012886 |

All much worse than Sub 1350 baseline (mean ~0.006776).

### Combined with static pose features:
| Target | Static Only | + Ballistic | Improvement |
|--------|------------|-------------|-------------|
| Angle | 0.007834 | 0.007796 | +0.5% |
| Depth | 0.011088 | 0.010679 | +3.7% |
| Left_right | 0.011847 | 0.011127 | +6.1% |

Marginal improvements: +0.5% for angle, +3.7% for depth, +6.1% for LR. These are tiny and likely to vanish on the leaderboard (LOO CV is systematically optimistic for this dataset).

## Wrist Transition Analysis

The wrist speed peak occurs ~10.7 frames BEFORE the wrist z-peak (release):
- Speed peak: mean frame 123.7
- Z-peak (release): mean frame 134.4

Deceleration rate (5 frames after speed peak): mean=0.291, std=0.081

This transition timing could be a useful feature (confirms previous finding that release_frame timing is predictive for depth).

## Conclusions

1. **Fingertips do NOT track the ball after release.** They follow the hand's follow-through motion. The fitted gravity (13.9 ft/s^2) is less than half of true gravity (32.2 ft/s^2), and 25% of shots show negative gravity (upward acceleration).

2. **Forward simulation from ballistic-fit velocity is useless.** Angle predictions have r=0.018 to -0.460 with targets.

3. **Ballistic-fit parameters as features add marginal value** (+0.5% to +6.1% in LOO CV), but this is unlikely to translate to LB improvement given the known LOO optimism.

4. **The fingertip spread pattern is interesting** - it captures the ball-release biomechanics (opening before, closing after) - but this information is already captured by the static pose features at the optimal extraction frames.

5. **This approach is a DEAD END for velocity recovery.** The keypoints track the body, not the ball. Any ball tracking signal is overwhelmed by the follow-through motion of the hand.

## Reproduction

Script: `scripts/post_release_ballistic.py`
```
uv run python scripts/post_release_ballistic.py
```

Data: `data/train.csv` (345 shots, 69 keypoints, 240 frames at 60fps)
Scalers: `data/scaler_{angle,depth,left_right}.pkl` (joblib format)
Release detection: right_wrist_z peak in frames 80-160
Ballistic fit: z(t) = z0 + vz*t - 0.5*g*t^2 via scipy curve_fit
