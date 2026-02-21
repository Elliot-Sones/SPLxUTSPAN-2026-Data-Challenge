# Deep Research: Novel Approaches for Free Throw Prediction
**Date:** 2026-02-20
**Goal:** Find breakthrough approaches to close the gap from LB 0.006234 to 0.0059 (gap: 0.000334)
**Constraint:** 345 train, 113 test, 5 players, 69 keypoints x 3 coords x 240 frames

---

## Executive Summary: Ranked by Realistic Promise

| Rank | Approach | Promise | Implementation Time | Why |
|------|----------|---------|--------------------|----|
| 1 | DTW Kernel in Locally Weighted Regression | HIGH | 2-3 hours | Direct drop-in to core pipeline, proven on time series |
| 2 | Functional PCA Regression (proper) | MEDIUM-HIGH | 3-4 hours | Prior attempt was botched (Fourier basis), scikit-fda makes it easy |
| 3 | Symbolic Regression (PySR) | MEDIUM | 4-6 hours | Could discover non-obvious feature combinations |
| 4 | Koopman Autoencoder Features | MEDIUM-LOW | 6-8 hours | Clever but needs enough data to train the autoencoder |
| 5 | Conformal Prediction Calibration | LOW | 2-3 hours | Fixes calibration, not prediction quality |
| 6 | Riemannian/Lie Group Features | LOW | 8-12 hours | Elegant math, impractical for N=345 regression |
| 7 | Optimal Transport Distance | LOW | 4-6 hours | Expensive, marginal advantage over DTW |
| 8 | Neural Processes | LOW | 6-8 hours | Neural nets lose to trees/Ridge at N=345 |

---

## 1. DTW Kernel in Locally Weighted Regression

### What it is
Replace the Euclidean distance in the Gaussian kernel of our core locally weighted Ridge regression with Dynamic Time Warping distance. Currently, our pipeline computes distances between shots using feature vectors at a single frame. DTW would compare entire trajectory segments, allowing elastic alignment of shooting motions that happen at slightly different speeds.

### Why it could help THIS problem
- Our core pipeline uses Gaussian kernel: K(x_i, x_j) = exp(-d^2 / 2h^2) where d is Euclidean distance between feature vectors at a single frame
- Different players have slightly different timing in their shooting motion
- DTW captures "same motion, different tempo" which Euclidean distance at a fixed frame misses
- The per-player locally weighted regression framework already uses a distance-based kernel - DTW is a DROP-IN replacement

### Prior work in this project
- DTW was already tried (Sub 2851) but as a STANDALONE model (LOO 0.009826, diversity: angle r=0.966)
- It was NOT tried as a kernel replacement in the core locally weighted pipeline
- The standalone DTW model used only shooting arm trajectories (frames 100-200)
- Key insight: DTW should be used to compute the DISTANCE MATRIX for the existing Ridge pipeline, not as a separate model

### Validated on small datasets?
- DTW is inherently non-parametric, works well with any N
- Kernelized DTW (KDTW) has been proven to be a positive definite kernel, so it is valid in kernel regression
- The main risk is computational cost: DTW is O(n*m) per pair, so the distance matrix for 345 shots x ~50 frames = ~6M DTW computations, which is feasible

### Implementation plan
- Library: `dtaidistance` (fast C implementation) or `tslearn`
- Compute DTW distance matrix on key joint trajectories (right wrist, elbow, shoulder - 9 channels)
- Use frames 120-180 (release window) subsampled to 30 frames
- Replace Euclidean distance in the existing per_example_pipeline.py with DTW distance
- Keep everything else identical (Ridge, PLS, per-player, bandwidth optimization)
- Test with Sakoe-Chiba band constraint (window=5) to prevent pathological warping

### Expected diversity vs existing models
- HIGH diversity from velocity/position CNNs (different computation entirely)
- MODERATE diversity from base Ridge (same kernel framework, different distance metric)
- If it works, it would improve the core pipeline itself, not just add a blend source

### Honest assessment
**This is the most promising approach.** It is the minimal change to the best-performing pipeline that addresses a real limitation (fixed-frame feature extraction misses temporal variation). The risk is that shooting motions may already be well-aligned across shots (they are free throws, highly stereotyped), in which case DTW reduces to Euclidean distance and gives no benefit.

---

## 2. Functional PCA Regression (Proper Implementation)

### What it is
Treat each shot's trajectory as a continuous FUNCTION over time (not discrete frames), fit a smooth basis (B-splines), decompose the function space via FPCA, and use the FPCA scores as features for regression.

### Why it could help THIS problem
- Our current approach picks a SINGLE frame (frame 153 for angle, 150 for depth, 170 for LR) and uses features from that frame
- This discards 239/240 of the available temporal information
- FPCA captures the dominant modes of variation ACROSS THE ENTIRE TRAJECTORY
- Mode 1 might capture overall shooting height, Mode 2 might capture release timing, Mode 3 might capture arm speed
- These functional modes could encode information that no single-frame extraction can capture

### Prior work in this project
- scripts/functional_data.py tried FPCA but used Fourier basis with PCA on coefficients (NOT proper FPCA)
- Results: MSE 15.03 vs baseline 10.12 (48% worse)
- The problem: it used standard PCA on Fourier coefficients, NOT functional PCA which respects the function-space geometry
- Also used only 9 channels (right arm) and 20 Fourier coefficients - too crude
- The scikit-fda library provides PROPER FPCA with B-spline smoothing, roughness penalties, and function-space inner products

### Validated on small datasets?
- FPCA is designed for functional data and works well with small N (it is essentially a covariance decomposition)
- The Tecator dataset example in scikit-fda uses N=215 successfully
- With 345 shots, FPCA should work well if we use 5-10 components (not 30+)
- Per-player FPCA may be too few samples (66-74 per player) - consider global FPCA with player indicator

### Implementation plan
- Library: `scikit-fda` (pip install scikit-fda)
- For each joint channel (e.g., right_wrist_x), create an FDataGrid object from the 240-frame trajectory
- Smooth with B-spline basis (order 4, ~30 knots) with roughness penalty (GCV to select lambda)
- Compute FPCA, extract top 5-10 scores per channel
- Use a subset of important channels: right arm (9), left arm (3), hips (3), spine (3) = ~18 channels
- 18 channels x 8 FPCA scores = 144 features
- Feed into per-player Ridge (or PLS-compress first, then Ridge)
- Also extract FPCA DERIVATIVES: velocity and acceleration modes (scikit-fda supports differentiation)

### Expected diversity vs existing models
- HIGH diversity - captures global trajectory shape, not single-frame snapshots
- Similar philosophy to CNNs but linear/interpretable and no training required
- Expect depth correlation r < 0.7 with existing models (depth is hardest, most benefits from temporal info)

### Honest assessment
**Medium-high promise.** The prior implementation was fundamentally wrong (Fourier + PCA is not FPCA). A proper implementation with scikit-fda, B-spline smoothing, and roughness penalties could work much better. The key risk is that free throw trajectories may have very low functional variance (they are all similar), in which case FPCA scores will be dominated by noise. Mitigate by using per-player centering and limiting to 5-8 components.

---

## 3. Symbolic Regression (PySR) for Feature Discovery

### What it is
Use evolutionary search to discover mathematical formulas that predict targets from raw keypoint coordinates. Instead of hand-engineering features like "right_wrist_z - right_shoulder_z", let PySR find non-obvious combinations like "sin(wrist_z * elbow_angle) / hip_rotation^0.5".

### Why it could help THIS problem
- All our current features are manually designed: hoop-relative coordinates, joint angles, velocities
- There may be non-linear combinations we have not thought of
- PySR is particularly good at discovering physics-inspired relationships
- Free throw trajectories follow known physics (projectile motion) - there may be mathematical relationships between body configuration and release parameters that are more complex than linear features capture

### Validated on small datasets?
- PySR's evolutionary search evaluates many candidate formulas, so it NEEDS a validation set
- With N=345, splitting into train/validation/test is risky (230/115 split?)
- Overfitting is a MAJOR concern: PySR can find complex expressions that fit noise
- Mitigation: heavy parsimony penalty, limit expression complexity to depth 3-4, use LOO or cross-validation as the fitness function
- The Pareto frontier approach (accuracy vs complexity) helps, but still risky at N=345

### Implementation plan
- Library: `pysr` (pip install pysr; requires Julia backend)
- Input: ~30 hand-selected features at the release frame (joint angles, positions, velocities)
- Target: each of the 3 targets separately
- Config: maxsize=20, parsimony=0.01, populations=30, niterations=100
- Binary operators: +, -, *, /, ^
- Unary operators: sin, cos, abs, sqrt, exp, log
- Use 5-fold CV as evaluation metric, not single train/test split
- Extract the Pareto frontier, pick expressions at the "elbow" of accuracy/complexity
- Use discovered formulas as ADDITIONAL FEATURES in the existing pipeline (not as standalone predictions)

### Expected diversity vs existing models
- VERY HIGH diversity if it discovers genuinely new relationships
- But could also rediscover things like "joint angle" which we already have
- Best case: finds 2-3 novel features that capture release mechanics better than our hand-crafted ones

### Honest assessment
**Medium promise with high variance.** PySR is either a home run (discovers a key formula we missed) or a complete waste of time (overfits or rediscovers existing features). The main downside is implementation time: Julia compilation, tuning PySR parameters, and evaluating the Pareto frontier takes significant effort. Worth trying but manage expectations.

---

## 4. Koopman Autoencoder Features

### What it is
Train an autoencoder where the latent space evolves according to LINEAR dynamics. The encoder maps a frame's keypoints to a latent code, a linear matrix K advances the code by one timestep, and the decoder reconstructs the next frame. The Koopman operator K linearizes the nonlinear shooting dynamics.

### Why it could help THIS problem
- Shooting motion is a nonlinear dynamical system: the body moves through a complex trajectory
- Koopman theory says there exists a (possibly infinite-dimensional) linear representation of any nonlinear system
- A Koopman autoencoder learns a FINITE approximation of this representation
- The latent code at the release frame would encode the "linear momentum" of the shooting motion
- This is fundamentally different from raw keypoint positions - it captures the DYNAMICS, not the CONFIGURATION

### Validated on small datasets?
- THIS IS THE BIGGEST RISK. Koopman autoencoders need training data.
- With 345 shots x 240 frames = 82,800 frame transitions, there is enough data to train the autoencoder
- But each shot is independent - the dynamics reset between shots
- Can train on per-player data: 70 shots x 240 frames = 16,800 transitions per player
- Recent work (tcKAE, 2025) specifically addresses limited and noisy training data with temporal consistency regularization

### Implementation plan
- Library: PyTorch (custom implementation, ~200 lines)
- Architecture:
  - Encoder: 207 -> 64 -> 32 (latent dim)
  - Koopman K: 32x32 linear matrix
  - Decoder: 32 -> 64 -> 207
  - Loss: reconstruction + prediction (next frame) + linearity constraint
- Train on ALL 458 shots (train+test) since this is self-supervised (no labels needed)
- After training, extract latent codes at release frame as features
- Feed 32-dim latent codes into per-player Ridge or as PLS input

### Expected diversity vs existing models
- HIGH diversity - encodes dynamics, not static configuration
- Different from CNNs (which also capture dynamics but through convolution, not linear operator)
- Latent codes should have low correlation with hoop-relative features

### Honest assessment
**Medium-low promise.** Theoretically elegant but practically difficult. The autoencoder needs careful tuning (latent dim, K regularization, training schedule) and may not converge with 345 shots. The "use all frames as training data" trick helps but assumes the dynamics are stationary across the shooting motion, which they are not (preparation vs release vs follow-through are different phases). Would try this only after DTW kernel and proper FPCA have been tested.

---

## 5. Conformal Prediction for Calibration

### What it is
Use conformal prediction to calibrate the Ridge regression outputs. Instead of raw Ridge predictions, use a calibration set to produce prediction intervals and use the interval midpoints (or conformalized predictions) as the final output.

### Why it could help THIS problem
- Our error budget analysis shows calibration slopes ranging from 0.16 to 0.82
- Perfect calibration would have slope = 1.0
- Conformal prediction provides distribution-free calibration guarantees
- The key insight: if predictions are systematically biased (e.g., shrunk toward the mean), conformal calibration can detect and correct this

### Validated on small datasets?
- Conformal prediction requires splitting data into training and calibration sets
- With N=345, this is painful: ~230 train + ~115 calibration
- Recent methods (EPICSCORE, 2024) augment conformal scores with epistemic uncertainty, which helps with small calibration sets
- However, conformal prediction is primarily about COVERAGE (prediction intervals), not POINT predictions

### Implementation plan
- Library: `mapie` (Model-Agnostic Prediction Intervals Estimator)
- Split training data: 80% proper training, 20% calibration
- Fit the core pipeline on proper training set
- Use calibration set to compute conformal residuals
- Conformalize test predictions by adjusting for systematic bias
- Alternative: cross-conformal (uses all data via CV splits for calibration)

### Expected diversity vs existing models
- ZERO diversity - it adjusts the existing predictions, does not create new ones
- It could improve the existing best submission by fixing calibration bias

### Honest assessment
**Low promise for MSE improvement.** Conformal prediction is designed for coverage guarantees, not point prediction accuracy. The calibration slopes we observe (0.16-0.82) reflect genuine prediction uncertainty, not a fixable systematic bias. Cross-conformal might give a tiny improvement by averaging over more calibration splits, but the MSE impact is likely negligible (< 0.000050). Skip unless other approaches have been exhausted.

---

## 6. Riemannian/Lie Group Skeleton Features

### What it is
Represent each body pose as a point on the Lie group SE(3)^K (where K is the number of joints), compute inter-joint rotations as elements of SO(3), and use the Riemannian geometry (geodesic distances, Frechet means, tangent space projections) as features.

### Why it could help THIS problem
- Euclidean features (x,y,z coordinates) do not respect the geometric structure of the skeleton
- Joint rotations live on SO(3), a curved manifold - Euclidean distances are wrong
- The tangent space at the Frechet mean provides a linear approximation of the manifold
- Features computed in tangent space might better capture joint angle variations

### Validated on small datasets?
- Lie group methods are primarily validated on ACTION RECOGNITION (classification), not regression
- The original paper (Vemulapalli et al., 2014) used large datasets (thousands of samples)
- For N=345, computing the Frechet mean and tangent space projection is fine
- But the downstream regression task is the same (Ridge/PLS on transformed features)
- The question is whether SO(3) features are materially different from our existing joint angle features

### Implementation plan
- Library: `geomstats` (Riemannian geometry in Python)
- For each pair of connected joints, compute relative rotation as SO(3) element
- Compute Frechet mean per player on SO(3)
- Project each shot to tangent space at the Frechet mean
- Use tangent vectors as features (dimensionality: K_pairs x 3 for so(3))
- Feed into per-player Ridge

### Expected diversity vs existing models
- LOW diversity - our existing joint angle features already capture most of what SO(3) rotations encode
- The Riemannian treatment mainly matters for LARGE rotations; free throw shooting involves small angle variations where Euclidean approximation is accurate

### Honest assessment
**Low promise.** This is theoretically beautiful but practically redundant for our problem. Free throw shooting involves small, controlled joint rotations where the Euclidean approximation of SO(3) is accurate to first order. The tangent space features would be nearly identical to our existing joint angle features. The implementation complexity is high (geomstats, rotation matrices, Frechet means) for likely negligible benefit. Skip.

---

## 7. Optimal Transport Distance for Kernel

### What it is
Replace Euclidean distance in the Gaussian kernel with Wasserstein distance between shots' temporal distributions. Treat each shot's trajectory as a probability distribution over the position space, and compute the optimal transport cost between distributions.

### Why it could help THIS problem
- Wasserstein distance accounts for the "shape" of the entire trajectory distribution
- It is robust to temporal shifts (like DTW) but also handles differences in trajectory spread
- The Partial Ordered Wasserstein (POW) distance preserves temporal ordering (unlike standard Wasserstein)

### Validated on small datasets?
- Wasserstein distance computation is O(n^3) for n timesteps
- For 30-frame subsequences, the 345x345 pairwise distance matrix requires 345^2/2 x O(30^3) = ~1.6B operations
- Feasible but slow
- Works at any sample size (non-parametric)

### Implementation plan
- Library: `POT` (Python Optimal Transport)
- Represent each shot as empirical distribution over (time, position) pairs
- Use Sinkhorn-regularized OT for faster computation
- Compute 345x345 distance matrix
- Replace Euclidean distance in core pipeline

### Expected diversity vs existing models
- MODERATE diversity - similar philosophy to DTW but different metric
- Likely highly correlated with DTW distance (both capture temporal alignment)

### Honest assessment
**Low promise.** Optimal transport is more general than DTW but also more expensive and harder to tune (entropic regularization parameter). For temporally ordered sequences of the same length (which is our case - all shots have 240 frames), DTW is more natural and better understood. OT adds complexity without a clear advantage over DTW for this specific problem. Try DTW first; only consider OT if DTW shows promise and you want a different distance metric for blending.

---

## 8. Neural Processes

### What it is
A meta-learning framework that learns a distribution over regression functions. Given a "context set" of (x, y) pairs, a Neural Process predicts y* for a new x* by encoding the context into a latent representation.

### Why it could help THIS problem
- Designed for few-shot regression with uncertainty quantification
- Could learn player-specific patterns from each player's ~70 shots
- Attentive Neural Processes (ANPs) can attend to the most relevant training examples

### Validated on small datasets?
- Neural Processes are designed for few-shot scenarios (5-50 context points)
- BUT they require meta-training on many TASKS to learn the prior
- With 5 players, we have at most 5 "tasks" for meta-learning
- This is FAR too few tasks for meta-learning to work
- NPs shine when you have 1000+ tasks and need to generalize to new tasks with few examples

### Implementation plan
- Library: `neural-process-family` or custom PyTorch
- Would need to artificially create tasks (bootstrap subsets? cross-player transfer?)
- Architecture: encoder (MLP), aggregator, decoder
- Meta-train on subsets of training data, meta-test on held-out shots

### Honest assessment
**Low promise.** Neural Processes solve the wrong problem for us. They are designed for meta-learning across many tasks with few examples per task. We have 5 tasks (players) with ~70 examples each. The meta-learning component cannot learn a useful prior from 5 tasks. Standard Ridge regression with per-player fitting is already doing what NPs would do, but without the overhead of neural network training. Skip unless you can create hundreds of meaningful task variations.

---

## Detailed Implementation Priority

### Tier 1 - Try immediately (highest expected value)

**1. DTW Kernel in Core Pipeline (2-3 hours)**
- Modify `per_example_pipeline.py` to accept a precomputed distance matrix
- Compute DTW distance on right arm trajectories (frames 120-180, 9 channels)
- Run LOO with DTW kernel vs Euclidean kernel
- If improvement > 1%: submit blend with current best

**2. Proper FPCA Features with scikit-fda (3-4 hours)**
- Install scikit-fda
- Compute B-spline FPCA on 18 key channels
- Extract 5-8 scores per channel = ~120 features
- PLS compress to 15 features, feed into existing pipeline
- Compare standalone quality AND diversity with current best

### Tier 2 - Try if Tier 1 shows promise (medium expected value)

**3. PySR Feature Discovery (4-6 hours)**
- Run PySR on release-frame features targeting each of 3 targets
- Use strict parsimony and cross-validation
- Extract top 3-5 formulas from Pareto frontier
- Add as features to existing pipeline

### Tier 3 - Probably skip (low expected value for competition)

4. Koopman autoencoder (complex, uncertain payoff)
5. Conformal calibration (fixes wrong thing)
6. Riemannian features (redundant with joint angles)
7. Optimal transport (inferior to DTW for this use case)
8. Neural Processes (wrong problem structure)

---

## Key Insight: Where the Gap Actually Is

The gap from 0.006234 to 0.0059 is 0.000334 (5.4% relative). The error budget shows:
- P4 angle: 33.6% of angle error (worst)
- P5 angle: 29.6% of angle error
- P5 depth: 30.2% of depth error

The breakthrough will come from either:
1. **Better temporal information extraction** (DTW kernel, proper FPCA) - helps with ALL players
2. **Player-specific feature discovery** (PySR on P4/P5 data) - targeted improvement
3. **A fundamentally new signal** that existing features miss

The DTW kernel approach is promising because it replaces the weakest link in the current pipeline (single-frame feature extraction in the distance computation) with a principled temporal distance measure, without changing anything else about the proven Ridge+PLS framework.

---

## Sources

- [FDA in Sports Biomechanics (30 year review)](https://www.tandfonline.com/doi/full/10.1080/14763141.2024.2398508)
- [scikit-fda: FPCA Regression](https://fda.readthedocs.io/en/stable/auto_examples/plot_fpca_regression.html)
- [scikit-fda GitHub](https://github.com/GAA-UAM/scikit-fda)
- [Deep Learning on Lie Groups for Skeleton-Based Action Recognition](https://arxiv.org/abs/1612.05877)
- [Lie Group Skeleton Representation (Vemulapalli 2014)](https://ieeexplore.ieee.org/document/6909476/)
- [Partial Ordered Wasserstein Distance for Sequential Data](https://www.sciencedirect.com/science/article/abs/pii/S0925231224006799)
- [Wasserstein-Fourier Distance for Stationary Time Series](https://arxiv.org/abs/1912.05509)
- [Conformal Prediction: A Data Perspective](https://dl.acm.org/doi/10.1145/3736575)
- [EPICSCORE: Epistemic Uncertainty in Conformal Scores](https://openreview.net/pdf/76a322f280fd331646daba598b6fa2115ad083e1.pdf)
- [Neural Process Family Documentation](https://yanndubs.github.io/Neural-Process-Family/text/Intro.html)
- [Attentive Neural Processes](https://openreview.net/forum?id=SkE6PjC9KX)
- [Kernelized DTW (KDTW)](https://github.com/pfmarteau/KDTW)
- [DTW Kernel Evaluation](https://link.springer.com/article/10.1007/s10462-021-10050-y)
- [PySR: Symbolic Regression](https://github.com/MilesCranmer/PySR)
- [PySR Paper](https://arxiv.org/abs/2305.01582)
- [PySR Overfitting Discussion](https://github.com/MilesCranmer/PySR/discussions/995)
- [Koopman Operator Theory (SIAM Review)](https://epubs.siam.org/doi/10.1137/21M1401243)
- [Temporally Consistent Koopman Autoencoders](https://www.nature.com/articles/s41598-025-05222-7)
- [Physics-Informed Koopman Networks (ICLR 2024)](https://iclr.cc/virtual/2024/21345)
- [Deep Learning on Tabular Data Benchmarks](https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360)
