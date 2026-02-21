# Physics Modeling Research and Integration Plan

Date: 2026-02-15
Scope: External physics modeling research + concrete integration path for this repo

## Objective

Extract higher-fidelity physical signal from noisy markerless keypoints, without relying on directly observed ball trajectory (which we do not have).

## Ground Truth From Existing Repo Findings

1. Physics signal exists in release-phase kinematics and player-specific timing:
   - `Research/FINAL_PHYSICS_RESULTS.md` reports strong per-player depth and moderate angle signal from compact physics features.
   - `Research/EXTENDED_PHYSICS_FEATURES_RESULTS.md` shows right-hand features as the strongest extension path.

2. Post-release fingertip ball tracking is not reliable:
   - `Research/POST_RELEASE_BALLISTIC_ANALYSIS.md` shows fitted gravity from fingertip trajectories is far from true gravity, with frequent non-physical estimates.
   - `Research/INDEX.md` already classifies fingertip ball tracking as a dead end.

Implication:
- We should treat physics as a structured prior on latent states, not as direct ball reconstruction from post-release fingertips.

## External Research - What Is Most Relevant

## 1) Physics-informed constraints are effective when observations are noisy or incomplete
- PINNs inject governing equations into training via residual losses and can learn with sparse supervision.
- Source: Raissi et al., "Physics-informed neural networks" (JCP, 2019), arXiv:1711.10561
  - https://arxiv.org/abs/1711.10561

Project implication:
- Add soft physics constraints to latent release-state estimation instead of hard-coding ballistic reconstruction.

## 2) Learn dynamics in energy-conserving forms instead of unconstrained black-box regression
- HNN learns Hamiltonian dynamics and uses gradients to recover dynamics.
- LNN extends this to systems where Lagrangian structure is more natural.
- Deep Lagrangian Networks (DeLaN) learn physically plausible robot dynamics and support differentiable control/identification.
- Sources:
  - HNN: https://arxiv.org/abs/1906.01563
  - LNN: https://arxiv.org/abs/2003.04630
  - DeLaN: https://arxiv.org/abs/1907.04490

Project implication:
- Replace ad hoc release-window dynamics features with energy or momentum-consistent latent features derived from shoulder-elbow-wrist-finger chain.

## 3) Discover compact governing equations for local motion regimes
- SINDy discovers sparse nonlinear governing equations from data.
- Source: Brunton et al., "Discovering governing equations..." (PNAS, 2016), arXiv:1509.03580
  - https://arxiv.org/abs/1509.03580

Project implication:
- Fit per-player sparse local dynamics in release windows and use equation coefficients as stable, low-dimensional features.

## 4) Differentiable biomechanics can improve markerless motion modeling
- A recent differentiable biomechanics framework directly targets markerless motion capture limitations and adds biomechanical consistency.
- Source: "Differentiable Biomechanics Unlocks Opportunities for Markerless Motion Capture", arXiv:2402.17192
  - https://arxiv.org/abs/2402.17192

Project implication:
- Use differentiable biomechanical constraints on upper-limb kinematics to reduce non-physical jitter in features that drive angle and depth.

## 5) Mature optimization and simulation tooling exists now
- OpenSim Moco provides direct-collocation-based trajectory optimization for musculoskeletal models.
- MuJoCo MJX provides batched JAX simulation with broad MuJoCo API support.
- Brax supports differentiable rigid-body simulation in JAX for large batched experiments.
- Sources:
  - OpenSim Moco paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7793308/
  - MuJoCo MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html
  - Brax repo: https://github.com/google/brax
  - Brax README (raw): https://raw.githubusercontent.com/google/brax/main/README.md

Project implication:
- If we need simulation-backed latent estimation, use batched differentiable simulation, not single-shot custom loops.

## Recommended Integration Strategy For This Repo

Priority is ranked by expected signal quality vs implementation risk for this dataset.

## Priority 1 - Physics-constrained latent release state features

What to build:
- A latent state estimator around release window:
  - latent variables: release position, release velocity vector, release timing, optional drag/spin nuisance terms.
- Train with:
  - supervised target loss (angle/depth/left_right),
  - plus soft physics consistency losses on short-window kinematics and projectile plausibility.

Why this fits current code:
- Existing release detection hooks already exist in:
  - `scripts/per_example_pipeline.py:164`
  - `scripts/extended_physics_features.py:158`
- Existing feature extraction insertion point:
  - `scripts/per_example_pipeline.py:204`

Output to model:
- A compact vector of latent physics descriptors per shot, then feed into existing PLS + Ridge/per-example stack.

What to avoid:
- Do not use post-release fingertip parabolic fitting as primary release estimator.

## Priority 2 - Energy-flow features from differentiable upper-limb chain

What to build:
- A small differentiable chain for shooting side: shoulder -> elbow -> wrist -> fingertip centroid.
- Extract energy and power-transfer descriptors:
  - kinetic/potential energy profiles,
  - power impulse near release,
  - timing offsets of peak power transfer.

Why this fits current code:
- You already use right-hand feature engineering and PLS compression in:
  - `scripts/extended_physics_features.py`
- This becomes an additional compact feature block, not a replacement.

## Priority 3 - SINDy coefficients as player-conditioned physics fingerprints

What to build:
- For each player and target, fit sparse local dynamics on release-window state vectors.
- Use discovered coefficient vectors and residual statistics as additional features.

Why this fits current code:
- Current pipeline already supports player-conditioned behavior and per-target feature handling.

## Execution Plan (Measure-First, Small-to-Large)

Phase A - Pilot, sub-minute:
- Run on a small subset and a reduced fold setup to verify end-to-end integration.
- Only scale parameter changes between pilot and full run.

Phase B - Full CV:
- Evaluate full Group/LOPO setting with exact reproducibility logging.

Phase C - Submission gating:
- Standalone new physics submission.
- Low-weight blends with current strong anchors.
- Angle-only blend variants if signal is concentrated there.

Metrics to measure:
- Per-target CV MSE and total CV MSE.
- Correlation/diversity versus current best anchor submissions.
- LB transfer behavior of low-weight blends.

Note:
- Improvement magnitude needs benchmarking in this dataset and cannot be assumed from literature.

## Concrete Code Touch Points

1. Add new module for latent physics features:
- Suggested path: `scripts/latent_release_physics_features.py`

2. Integrate feature block into existing extractor:
- `scripts/per_example_pipeline.py:204`
- Preserve existing compact features, append latent physics block.

3. Keep existing per-target frame configs for compatibility:
- `scripts/per_example_pipeline.py:53`
- `scripts/extended_physics_features.py:46`

4. Reuse existing PLS compression path:
- `scripts/per_example_pipeline.py:705`
- `scripts/per_example_pipeline.py:713`

## Decision Rule After First Full Run

Advance only if all are true:
1. CV improves on at least one target without materially harming the other two.
2. New predictions show useful diversity versus current anchor submissions.
3. First LB probes with conservative blend weights are non-regressive.

If not, stop and retain only the best ablation component.

## References

1. Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks. J Comput Phys (2019).
   - https://arxiv.org/abs/1711.10561
2. Greydanus S, Dzamba M, Yosinski J. Hamiltonian Neural Networks (2019).
   - https://arxiv.org/abs/1906.01563
3. Cranmer M, et al. Lagrangian Neural Networks (2020).
   - https://arxiv.org/abs/2003.04630
4. Lutter M, Ritter C, Peters J. Deep Lagrangian Networks (2019).
   - https://arxiv.org/abs/1907.04490
5. Brunton SL, Proctor JL, Kutz JN. Sparse identification of nonlinear dynamics (SINDy).
   - https://arxiv.org/abs/1509.03580
6. Differentiable Biomechanics Unlocks Opportunities for Markerless Motion Capture (2024).
   - https://arxiv.org/abs/2402.17192
7. Dembia CL, et al. OpenSim Moco: Musculoskeletal optimal control.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC7793308/
8. MuJoCo MJX documentation.
   - https://mujoco.readthedocs.io/en/stable/mjx.html
9. Brax repository.
   - https://github.com/google/brax
10. Brax README (raw source text).
    - https://raw.githubusercontent.com/google/brax/main/README.md

