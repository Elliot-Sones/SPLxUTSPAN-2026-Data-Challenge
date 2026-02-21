# MuJoCo Inverse Kinematics Feasibility for Noise Reduction

## Research Date: 2026-02-09

## Executive Summary

Physics-based IK methods (MuJoCo, OpenSim) CAN reduce motion capture noise, but the achievable improvement is **insufficient to solve our core problem**. The fundamental bottleneck is that our data comes from markerless pose estimation (69 keypoints, ~20-40mm position error) rather than marker-based systems (~0.1mm error). Physics-based IK can reduce jitter by ~50-60% in acceleration space, but this translates to only ~2-3x noise reduction in velocity space - far short of the ~10-50x improvement needed to make velocity-based features competitive with our current per-example regression approach.

**Bottom line: Physics IK is a 2x improvement on a 50x problem. Not worth implementing.**

---

## 1. Noise Levels in Our Data vs. Lab-Grade Systems

### Marker-Based (Vicon/OptiTrack) - Gold Standard
- Position noise: 0.02-0.16 mm (sub-millimeter)
- Velocity derivable with <1% error after Butterworth filtering
- Frame rates: 120-370 Hz typical
- Source: [Vicon noise study (PMC6832304)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6832304/)

### Markerless Pose Estimation (MediaPipe, OpenPose, etc.) - Our Category
- Position error: **20-40 mm** for joint centers vs. marker-based ground truth
- High-frequency keypoint jitter is common, especially for low-confidence joints
- Systematic biases of 30-50 mm at hip/knee, 1-15 mm at ankle
- Source: [Pose estimation accuracy (Nature s41598-021-00212-x)](https://www.nature.com/articles/s41598-021-00212-x)

### The Gap
- Marker-based: ~0.1 mm noise
- Markerless: ~20-40 mm noise
- **Factor: 200-400x worse starting point**

Our 69-keypoint data at 60fps is firmly in the markerless category. Even if physics-based IK achieves perfect noise reduction on the markerless-specific jitter, the underlying position uncertainty of ~20-40mm per keypoint remains.

---

## 2. What Physics-Based IK Can Achieve

### 2.1 OpenSim Inverse Kinematics
- Standard IK produces joint angles with <2-4 cm marker tracking error (RMSE)
- AUKSMIKT (Kalman-smoothed IK) reduces angular acceleration errors by ~13.7%
- The IK process enforces skeletal constraints (bone lengths, joint limits) which removes physically impossible configurations
- **Key limitation**: IK output is joint angles, not marker positions. Converting back to Cartesian coordinates doesn't magically reduce position uncertainty.
- Source: [OpenSim IK docs](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090032/Getting+Started+with+Inverse+Kinematics), [AUKSMIKT (PMC12575601)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12575601/)

### 2.2 MuJoCo-Based Fitting
- MuJoCo supports mocap bodies with weld constraints for tracking
- Differentiable physics (MuJoCo MJX/Brax) enables gradient-based optimization of IK
- Bilevel optimization can jointly solve skeleton scaling + IK end-to-end
- mink library provides differential IK in Python on top of MuJoCo
- **Key advantage**: MuJoCo can enforce dynamics constraints (not just kinematics)
- Source: [Differentiable Biomechanics (arxiv 2402.17192)](https://arxiv.org/html/2402.17192v1), [mink (GitHub)](https://github.com/kevinzakka/mink)

### 2.3 Physics-Based Motion Capture (OSDCap - NeurIPS 2024)
- **Best published results for physics-based noise reduction**
- Uses neural Kalman filter + physics simulation (PD controller + contact forces)
- Acceleration jitter: reduced from 19.2 to 8.4 mm/s^2 (~56% reduction, ~2.3x factor)
- Global position error: 125.9 to 119.1 mm (~5% improvement)
- Joint error (MPJPE-G): 152.7 to 132.8 mm (~13% improvement)
- Source: [OSDCap (NeurIPS 2024)](https://arxiv.org/html/2410.07795v2)

### 2.4 Physical Inertial Poser (PIP - CVPR 2022)
- Physics-aware optimization on top of neural kinematics from 6 IMUs
- Improves temporal stability and physical correctness
- 60fps real-time capable
- Source: [PIP (CVPR 2022)](https://arxiv.org/abs/2203.08528)

### 2.5 Neural MoCap Solvers (Holden 2018)
- Deep denoising network maps corrupted marker data to joint transforms
- Achieves "precision within a few millimeters" for marker-based data
- Handles occlusion, mislabeling, jitter
- **Key caveat**: Trained on marker-based data, not directly applicable to markerless
- Source: [Holden (ACM ToG 2018)](https://dl.acm.org/doi/10.1145/3197517.3201302)

---

## 3. Noise Reduction Factors: Quantitative Summary

| Method | Metric | Before | After | Factor |
|--------|--------|--------|-------|--------|
| OSDCap physics | Acceleration jitter | 19.2 mm/s^2 | 8.4 mm/s^2 | 2.3x |
| OSDCap physics | Global position | 125.9 mm | 119.1 mm | 1.06x |
| AUKSMIKT (OpenSim) | Angular acceleration MAE | baseline | -13.7% | 1.16x |
| Kalman smoothing | CoM velocity error (OpenPose) | 0.943 m/s SD | 0.257 m/s SD | 3.7x |
| Butterworth filter | Velocity estimation error | ~16% | ~9% | 1.8x |
| Pose2Sim (OpenSim IK) | Joint angle error | - | 3-4 degrees | N/A |

**Key takeaway**: Physics-based methods achieve **1.5-3.7x** noise reduction, with the best results (~3.7x) from Kalman smoothing of extremely noisy markerless data. This is consistent across multiple studies and methods.

---

## 4. The Fundamental Problem for Our Use Case

### 4.1 Velocity Estimation from Noisy Positions
- Numerical differentiation amplifies noise linearly (velocity) and quadratically (acceleration)
- With 60fps and ~20mm position noise: velocity noise ~ 20mm * 60Hz = ~1.2 m/s
- Basketball release velocity: ~7 m/s with SD of 0.05-0.13 m/s between shots
- **Signal-to-noise ratio**: 7.0 / 1.2 = ~6:1 (terrible for discrimination)
- We need to discriminate ~0.1 m/s differences in velocity (the between-shot variation)
- Required SNR: 0.1 / noise < 0.01, meaning noise must be < 0.01 m/s
- Current noise: ~1.2 m/s, needed: ~0.01 m/s = **120x improvement needed**

### 4.2 What Physics IK Gives Us
- Best case 3.7x noise reduction: 1.2 m/s -> 0.32 m/s
- Still 32x worse than needed
- Even a hypothetical 10x reduction: 1.2 m/s -> 0.12 m/s (barely useful)

### 4.3 Basketball-Specific Studies
- Lab studies achieve 0.01 m/s velocity accuracy using high-speed cameras (60fps+) with analytic trajectory models and least-squares fitting
- But they use **ball trajectory** fitting (parabolic + drag), NOT hand velocity differentiation
- The ball trajectory approach is fundamentally different: it fits a physics model to the ball's 2D image coordinates across multiple frames
- Source: [Basketball release variability (PMC8256521)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/)

---

## 5. Existing Tools and Libraries

### Physics-Based MoCap Processing
1. **Pose2Sim** - Markerless kinematics to OpenSim (Python, open-source)
   - Pipeline: 2D pose -> 3D triangulation -> OpenSim IK
   - Achieves 3-4 degree joint angle accuracy
   - [GitHub](https://github.com/perfanalytics/pose2sim)

2. **MuJoCo MJX** - GPU-accelerated differentiable physics
   - Supports gradient-based optimization for IK
   - Can enforce dynamics constraints
   - [MuJoCo Docs](https://mujoco.readthedocs.io/)

3. **mink** - Differential IK for MuJoCo (Python)
   - [GitHub](https://github.com/kevinzakka/mink)

4. **OSDCap** - Physics-based motion capture from video
   - Neural Kalman filter + physics simulation
   - [GitHub](https://github.com/cuongle1206/OSDCap)

5. **OpenSimRT** - Real-time OpenSim IK + ID
   - [GitHub](https://github.com/mitkof6/OpenSimRT)

### Filtering/Smoothing (Non-Physics)
6. **Butterworth zero-phase filter** - Best general-purpose for velocity estimation
7. **Savitzky-Golay filter** - Good for heavily quantized data
8. **Kalman filter/smoother** - Best for sequential estimation with physics priors

---

## 6. Applicability to Our 69-Keypoint 60fps Data

### What Would Work
- **Skeletal constraint enforcement**: Ensuring bone lengths are constant and joint angles are within physiological limits. This would remove ~30-50% of jitter from keypoint detection inconsistency.
- **Temporal smoothing with physics priors**: Using a Kalman filter that models human dynamics (inertia, joint torques) to smooth the trajectory. Expected improvement: 2-3x in velocity space.
- **Pose2Sim pipeline**: Could process our data through OpenSim IK to get physically consistent joint angles. But the output is joint angles, not velocities.

### What Would NOT Work
- **Direct velocity estimation improvement**: Even with 3.7x noise reduction, velocity from position differentiation remains too noisy for useful features.
- **Ball velocity from hand keypoints**: The ball is not tracked in our data, and hand keypoint noise is ~20-40mm, making velocity estimation at 60fps fundamentally limited.
- **Transfer from marker-based results**: Published IK accuracy numbers (sub-mm, sub-degree) assume marker-based input. Our markerless input is 200-400x noisier to start with.

### Implementation Effort vs. Benefit
- **High effort**: Requires skeleton model construction, IK solver setup, parameter tuning
- **Medium noise reduction**: 2-3x improvement in velocity space
- **Low impact on predictions**: Our best model (Sub 1350, LB 0.006776) already uses features extracted at specific frames (not velocities), which sidesteps the noise problem entirely
- Previous experiments showed velocity/acceleration features from full trajectories scored LB 0.007528 (11% WORSE than Sub 1350)

---

## 7. Comparison with Our Current Approach

| Approach | How It Handles Noise | LB Score |
|----------|---------------------|----------|
| Per-example regression (Sub 1350) | Uses static pose at optimal frames, no differentiation | 0.006776 |
| Velocity features (Sub 1492) | Differentiates noisy positions, many features | 0.007528 |
| Physics IK + velocity (hypothetical) | 2-3x noise reduction on velocity | ~0.0072 (estimated) |
| Physics IK + static features (hypothetical) | Marginal improvement on already-good features | ~0.00675 (estimated) |

The estimated improvement from physics IK on static features is minimal because:
1. Static pose features at a single frame are already relatively clean (no differentiation)
2. Per-example regression already handles per-shot variation well
3. The remaining error is likely irreducible noise in the target mapping, not feature noise

---

## 8. Conclusion and Recommendation

### NOT RECOMMENDED for this competition

**Reasons:**
1. **Insufficient noise reduction**: 2-3x improvement on a 100x+ problem
2. **Wrong bottleneck**: Our best model avoids velocity features entirely - it uses static poses at optimal frames
3. **High implementation cost**: Building a proper IK pipeline (skeleton model, solver, tuning) is days of work
4. **Marginal expected gain**: Even the best-case scenario improves LB by <0.0002 (from ~0.00676 to ~0.00674)
5. **Proven failure**: Velocity/acceleration features already tested and failed (Sub 1492, LB 0.007528)

### What WOULD Be Worth Trying Instead
1. **Better frame selection**: Fine-tuning the optimal extraction frames per-player (not just per-target)
2. **Ensemble of per-example models**: Different kernel bandwidths, different feature subsets
3. **Stacking**: Using per-example predictions as features for a meta-learner
4. **Feature engineering**: Novel static features (ratios, angles between joints) at optimal frames

### The Core Insight
Physics-based IK is designed to solve a different problem: making motion capture data physically plausible for animation and biomechanical analysis. Our problem is prediction accuracy on 3 specific targets from noisy keypoint data. The noise reduction from IK is real but insufficient, and our best approach (per-example regression on static features) already works around the noise problem by not computing velocities at all.
