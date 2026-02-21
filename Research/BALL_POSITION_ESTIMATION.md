# Ball Position Estimation from Hand Keypoints

Date: 2026-02-09
Task: Research optimal methods for estimating basketball center position from mocap keypoints.

## Right Hand Keypoints Inventory

The dataset contains 23 right-hand joints (and 23 matching left-hand joints):

**Thumb (first finger):**
- right_first_finger_cmc (carpometacarpal)
- right_first_finger_mcp (metacarpophalangeal)
- right_first_finger_ip (interphalangeal)
- right_first_finger_distal (tip)

**Index (second finger):**
- right_second_finger_mcp
- right_second_finger_pip (proximal interphalangeal)
- right_second_finger_dip (distal interphalangeal)
- right_second_finger_distal (tip)

**Middle (third finger):**
- right_third_finger_mcp, pip, dip, distal

**Ring (fourth finger):**
- right_fourth_finger_mcp, pip, dip, distal

**Pinky (fifth finger):**
- right_fifth_finger_mcp, pip, dip, distal

**Other:**
- right_wrist
- right_thumb (duplicate/summary)
- right_pinky (duplicate/summary)

Total: 23 joints x 3 axes = 69 columns per hand.

---

## CRITICAL FINDING: Fingertip Resolution is Too Low for Sphere Fitting

**Ball diameter = 9.4 inches (0.783 ft)**

Measured fingertip spread at frame 120 (mid-release):
- Thumb-Index distance: 2.03 inches
- Thumb-Pinky distance: 3.16 inches
- Index-Pinky distance: 2.23 inches
- Max tip-to-centroid: 1.94 inches

**The fingertip spread (~2-3 inches) is MUCH smaller than expected when holding a basketball (~8-10 inches).** The keypoint system collapses hand landmarks into a tight cluster. Fingertips do NOT represent true positions on the ball surface.

This means:
1. Sphere fitting is INVALID (fingertips don't lie on the ball surface)
2. Fingertip-based methods add noise, not signal
3. Coarser joints (wrist, MCP) are actually more reliable

---

## Methods Tested

| Method | Description |
|--------|-------------|
| A: Wrist offset | Right wrist + ball_radius * direction(wrist -> mean fingertips) |
| B: Fingertip centroid | Centroid of 5 right fingertip keypoints |
| C: Sphere fit (both hands) | Least-squares sphere fit to all 10 fingertips |
| C2: Sphere fit (right only) | Least-squares sphere fit to 5 right fingertips |
| D: Palm normal | Palm center (MCP joints) + ball_radius * palm plane normal |
| E: Both hands centroid | Centroid of all 10 fingertip keypoints |
| F: Mid-hands | Midpoint of wrists + ball_radius * direction toward mean fingertips |
| G: Wrist only | Just right wrist position (baseline) |

---

## Smoothness Results (frames 100-150, all 345 shots)

Lower jitter = smoother trajectory = better position estimate.

| Method | Mean Jitter (ft) | SG Jitter | Mean Accel (ft/s^2) |
|--------|-------------------|-----------|---------------------|
| **G: Wrist only** | **0.039123** | - | - |
| F: Mid-hands | 0.041319 | 0.463 | 38.55 |
| E: Both centroid | 0.041966 | 0.469 | 40.35 |
| A: Wrist offset | 0.044150 | 0.524 | 45.85 |
| B: Fingertip centroid | 0.045084 | 0.503 | 47.53 |
| C: Sphere fit (both) | 0.072761 | 1.086 | 125.12 |
| D: Palm normal | 0.100962 | 1.342 | 183.30 |
| C2: Sphere fit (right) | 0.192302 | 2.949 | 623.25 |

**Wrist-only is the smoothest method by 9.8% over the best augmented method.**
Adding fingertip information INCREASES noise.

---

## Per-Player Jitter

| Player | Wrist Only | Mid-Hands (F) | Wrist Offset (A) |
|--------|-----------|---------------|-------------------|
| P1 (70 shots) | 0.026426 | 0.039257 | 0.039567 |
| P2 (66 shots) | 0.014809 | 0.018441 | 0.020605 |
| P3 (68 shots) | 0.056955 | 0.052777 | 0.057571 |
| P4 (67 shots) | 0.070070 | 0.068537 | 0.065809 |
| P5 (74 shots) | 0.028016 | 0.035562 | 0.037252 |

Wrist-only wins for Players 1, 2, 5 (62% of data). Mid-hands/Wrist-offset only slightly better for Players 3 and 4 (who have higher noise overall).

---

## Velocity-Target Correlations

Velocity at frame 120, SG smoothed, correlated with raw targets:

**Best correlations per method:**

| Method | Best angle r | Best depth r | Best LR r |
|--------|-------------|-------------|-----------|
| Wrist only | vz: +0.519 | vy: +0.249 | - (all <0.06) |
| Wrist offset | vx: -0.541 | vy: +0.321 | - |
| Fingertip centroid | vx: -0.552 | vy: +0.316 | - |
| Mid-hands | vz: +0.377 | vx: -0.176 | - |
| Both centroid | vz: +0.391 | vy: +0.160 | - |

Key observations:
1. **Wrist offset and fingertip centroid give highest angle correlation** (vx r=-0.55) but this is measuring hand direction, not true ball velocity
2. **Wrist-only vz has best angle correlation** (r=+0.52) - upward wrist speed predicts shot angle
3. **Wrist offset vy has best depth correlation** (r=+0.32) - forward velocity predicts depth
4. **No method produces useful LR correlation** (all |r| < 0.14) - lateral velocity is too noisy
5. **All velocities are ~5 ft/s, not the expected 20-25 ft/s** - we're measuring arm motion, not ball release velocity

---

## Optimal Frame for Velocity Extraction

Testing wrist-only elevation angle vs angle target:

| Frame | elev-angle r | speed-depth r | azim-LR r |
|-------|-------------|---------------|-----------|
| 115 | +0.317 | -0.153 | +0.016 |
| 118 | +0.257 | -0.053 | +0.007 |
| 120 | +0.231 | -0.002 | -0.048 |
| 122 | +0.092 | +0.023 | -0.050 |
| 125 | -0.009 | +0.073 | -0.068 |
| 128 | -0.053 | +0.112 | -0.029 |
| 130 | -0.104 | +0.099 | -0.052 |
| 135 | -0.399 | -0.231 | -0.043 |

- Angle: best at frame 115 (pre-release upward motion)
- Depth: weak everywhere (max r=0.112 at frame 128)
- LR: useless everywhere

---

## Inter-Method Position Agreement

Mean positions at frame 120 across all methods agree to within 1-4 inches. Two clusters emerge:
1. **Right-hand methods** (A, B, C2): centered around (18.26, -25.11, 6.22) ft
2. **Both-hand methods** (E, F, C): centered around (18.41, -25.41, 6.24) ft

The both-hand methods are ~4 inches more toward the basket (more negative Y), which makes sense as the left (guide) hand is further forward.

---

## Post-Release Analysis

After frame ~120-130, the ball is airborne and the wrist follows the hand (follow-through), NOT the ball. Wrist z-change from frame 120-160 varies wildly (-2.16 to +2.43 ft) depending on player follow-through style.

**Ball position post-release requires ballistic trajectory fitting, not keypoint tracking.**

---

## Phase-Based Noise Profile

Mean frame-to-frame wrist displacement by phase:

| Phase | Mean (ft) | Max (ft) |
|-------|-----------|----------|
| Setup (0-80) | 0.024 | 0.033 |
| Pre-release (80-110) | 0.047 | 0.088 |
| Release (110-130) | 0.085 | 0.106 |
| Post-release (130-160) | 0.056 | 0.073 |
| Flight (160-240) | 0.082 | 0.110 |

Highest noise during release phase (expected - fastest arm motion) and late flight phase (possibly hand gesture/clapping noise).

---

## Fingertip Spread Over Time (Hand Separation)

| Frame | R-hand spread | L-hand spread | Hand distance | Ball diameter |
|-------|--------------|--------------|---------------|---------------|
| 100 | 0.140 ft | 0.157 ft | 0.529 ft | 0.783 ft |
| 110 | 0.148 ft | 0.137 ft | 0.600 ft | 0.783 ft |
| 120 | 0.154 ft | 0.148 ft | 0.797 ft | 0.783 ft |
| 130 | 0.155 ft | 0.154 ft | 0.952 ft | 0.783 ft |
| 140 | 0.150 ft | 0.150 ft | 1.069 ft | 0.783 ft |
| 150 | 0.139 ft | 0.161 ft | 1.436 ft | 0.783 ft |

Hand separation crosses ball diameter around frame 120, confirming this is when the left hand releases the ball. By frame 130, hands are >1 ft apart - guide hand has clearly separated.

---

## Conclusions and Recommendations

### For Ball Position Estimation

1. **Use right wrist only** - it's the smoothest and most reliable proxy. Adding fingertip data adds 10-13% more noise because the keypoint system has insufficient hand resolution (~2-3 inch spread vs ~9 inch ball diameter).

2. **Sphere fitting is invalid** - fingertips are collapsed, not on ball surface. Methods C, C2, D produce 2-16x more jitter than wrist-only.

3. **Both-hands centroid** provides a slightly different viewpoint (~4 inches toward basket) but doesn't improve smoothness.

### For Velocity Estimation

4. **Wrist offset gives best target correlations** for angle (vx r=-0.55) and depth (vy r=+0.32), despite being noisier than wrist-only. The offset direction captures hand orientation information that pure wrist position lacks.

5. **No ball estimation method produces useful left_right velocity** (all |r| < 0.14). LR prediction must come from keypoint positions, not velocities.

6. **Measured velocities (~5 ft/s) are arm velocities, not ball velocities.** True ball release velocity (~20-25 ft/s) cannot be recovered from these keypoints because the ball is not directly tracked.

7. **Frame 115 gives best velocity-angle correlation** (r=+0.317), not frame 120. This suggests the upward arm acceleration (pre-release) is more informative than the release velocity itself.

### For the Physics Pipeline

8. **Wrist-only with SG smoothing** is the recommended ball position proxy for the Kalman filter. It has the lowest noise floor, making it the best starting point for physics-based velocity recovery.

9. **The wrist-offset direction** (wrist-to-fingertip-centroid) could be used as a supplementary orientation feature (hand direction), even though it shouldn't replace wrist position for trajectory smoothness.

10. **Post-release ball position must come from ballistic fitting** (parabolic trajectory from release point + velocity), not from keypoints.
