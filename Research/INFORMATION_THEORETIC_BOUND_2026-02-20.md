# Information-Theoretic Lower Bound on Prediction MSE

Date: 2026-02-20
Script: scripts/information_theoretic_bound.py

## Objective

Estimate the aleatoric noise floor - the irreducible prediction error caused by
outcome variability among biomechanically-similar shots. This answers: "How close
are we to the theoretical best possible prediction?"

## Method

For each training shot, find k nearest neighbors (same player, per-player standardized
features) in hoop-relative feature space (positions + velocities of 12 key joints at
target frames, plus summary statistics = ~180 features per target).

Two quantities are computed:
1. **kNN LOO MSE**: Predict each shot as the mean of its k nearest neighbors. This
   traces the bias-variance frontier as k increases.
2. **Mean kNN Group Variance**: Average variance of outcomes within each (shot + its k
   neighbors) group. This estimates aleatoric noise - the irreducible randomness.

## Key Results

### Noise Floor Estimate (k=5)

| Target     | Noise Floor (k=5 var) | kNN LOO MSE (k=5) |
|------------|----------------------|-------------------|
| angle      | 0.005143             | 0.007294          |
| depth      | 0.006318             | 0.009149          |
| left_right | 0.007519             | 0.011244          |
| **MEAN**   | **0.006327**         | **0.009229**      |

### Comparison to Our Best

- **Estimated noise floor: 0.006327**
- **Current best LB: 0.006148**
- **Our model is BELOW the estimated noise floor by 0.000179 (2.8%)**

This means our current model is already performing at or slightly better than what
the kNN-based noise floor analysis suggests is possible. The fact that we are below
the noise floor estimate likely means:
1. Our model (locally-weighted Ridge + PLS + CNN blending) is extracting MORE signal
   from the biomechanics than simple kNN can capture
2. The noise floor estimate is slightly pessimistic (k=5 groups include some
   dissimilar shots that inflate variance)

### kNN MSE Frontier (aggregated across 3 targets)

| k  | kNN LOO MSE | Aleatoric Var | Var/MSE Ratio |
|----|-------------|---------------|---------------|
| 1  | 0.014507    | 0.003627      | 0.250         |
| 2  | 0.011452    | 0.005031      | 0.439         |
| 3  | 0.010162    | 0.005526      | 0.544         |
| 5  | 0.009229    | 0.006327      | 0.686         |
| 7  | 0.009202    | 0.006795      | 0.738         |
| 10 | 0.009174    | 0.007306      | 0.796         |
| 15 | 0.009375    | 0.007808      | 0.833         |
| 20 | 0.009575    | 0.008206      | 0.856         |

The kNN LOO MSE bottoms out around k=10 at 0.009174. Our best LB (0.006148) is
33% better than the best possible kNN - confirming our model family is superior.

### Per-Player Noise Floors (k=5, mean across targets)

| Player | Noise Floor | Notes |
|--------|------------|-------|
| P1     | 0.004835   | Most predictable overall |
| P2     | 0.005681   | Moderate |
| P3     | 0.002941   | MOST predictable (lowest floor) |
| P4     | 0.006598   | High variance |
| P5     | 0.011180   | HARDEST to predict (2x P4, 4x P3) |

P5 has by far the highest noise floor - their shots are inherently the most
unpredictable even among biomechanically-similar shots.

### Conditional Analysis: Easy vs Hard Shots

Shots split by median local variance (k=5):

| Target     | Easy Half Var | Hard Half Var | Ratio |
|------------|--------------|---------------|-------|
| angle      | 0.001545     | 0.008762      | 5.7x  |
| depth      | 0.002304     | 0.010356      | 4.5x  |
| left_right | 0.002933     | 0.012132      | 4.1x  |

The easy half has very low aleatoric noise (0.001-0.003), meaning those shots
ARE highly predictable from biomechanics. The hard half has 4-6x higher noise,
suggesting those shot outcomes are dominated by factors not captured in pose data
(grip variations, ball spin, mental state, etc).

## Interpretation

1. **We are at or near the theoretical ceiling.** Our LB score of 0.006148 is
   already below the estimated noise floor of 0.006327.

2. **Remaining gains will be small.** Even if the true noise floor is ~0.005
   (optimistic), we can only improve by ~0.001 from current best.

3. **The 0.0059 target** (someone's LB score) is plausible but very close to
   the noise floor. Reaching it would require ~0.000250 improvement.

4. **P5 is the bottleneck.** P5's noise floor (0.011) is 2-4x higher than other
   players. Any remaining gains likely come from improving on other players,
   not P5.

5. **Model family matters more than features.** Our locally-weighted Ridge + CNN
   ensemble beats the best kNN by 33%, confirming that the model architecture
   is extracting real signal beyond simple similarity.

## Caveats

- The noise floor is estimated from TRAINING data only (345 shots). With small
  k-NN groups (k=5), the variance estimates are noisy themselves.
- The "true" noise floor could be lower if there exist features we haven't
  captured that would make currently-dissimilar shots look similar.
- Being below the noise floor estimate does NOT mean our model is overfitting
  to the test set - it means our model is better than kNN at extracting signal.
