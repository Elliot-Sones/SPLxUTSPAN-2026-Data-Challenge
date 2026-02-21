# Research Index

Master index of all experiments, analyses, and findings. Updated 2026-02-14.

Current best: Sub 2169, LB 0.006552 (30% 3-frame ensemble + 70% Sub 2063).
Target: below 0.005. Deadline: 2026-02-21.

---

## Table of Contents

- [Proven Winners (In Best Submission)](#proven-winners-in-best-submission)
- [Untapped Gains (LOO Tested, Not Yet in LB Best)](#untapped-gains-loo-tested-not-yet-in-lb-best)
- [Dead Ends (Conclusively Failed)](#dead-ends-conclusively-failed)
- [Weak / Marginal (Not Worth Pursuing)](#weak--marginal-not-worth-pursuing)
- [Analysis Documents (No Direct Experiment)](#analysis-documents-no-direct-experiment)
- [Infrastructure / Implementation Docs](#infrastructure--implementation-docs)
- [Full File List by Category](#full-file-list-by-category)

---

## Proven Winners (In Best Submission)

These techniques are integrated into Sub 2169 (LB 0.006552):

| Technique | LOO Improvement | LB Score | File | Script |
|-----------|----------------|----------|------|--------|
| Per-example locally weighted Ridge | Baseline | 0.006776 (Sub 1350) | AGENT_TEAM_MISSION_FINAL_REPORT.md | per_example_pipeline.py |
| Joint angle features (10 angles) | -7.48% angle, -6.05% depth, -3.55% LR | 0.006619 (Sub 2020) | FIVE_APPROACHES_RESULTS.md | five_approaches.py |
| Angle fix (safe_average) | angle-specific | 0.006619 (Sub 1828) | (in MEMORY.md) | angle_fix.py |
| LASSO feature selection (10% blend) | small | 0.006698 (Sub 1640) | (in MEMORY.md) | feature_selection_submission.py |
| Multi-frame ensemble (3-frame) | -4.68% mean | 0.006552 (Sub 2169) | MULTIFRAME_ENSEMBLE_RESULTS.md | multiframe_ensemble.py |
| Cauchy kernel (10% blend) | -6.67% mean LOO | 0.006589 (Sub 2194) | KERNEL_EXPERIMENTS_RESULTS.md | kernel_experiments.py |

---

## Untapped Gains (LOO Tested, Not Yet in LB Best)

These showed LOO improvement but are NOT yet in the best submission:

| Technique | LOO Improvement | LB Tested? | File | Script |
|-----------|----------------|-----------|------|--------|
| Target denoising (Ridge LOO smooth) | -30.6% | NO | TARGET_DENOISING_RESULTS.md | target_denoising.py |
| Trajectory distance blend (30/70) | -11.1% on angle | NO | TRAJECTORY_PER_EXAMPLE_RESULTS.md | (per_example variants) |
| Mixup augmentation (alpha=1.0, 2x) | -9% mean | NO | MIXUP_AUGMENTATION_RESULTS.md | mixup_augmentation.py |
| Multi-frame averaging (7-frame MSE-weighted) | -5.3% mean | Partial (3f/5f tested) | MULTIFRAME_AVERAGING_RESULTS.md | multiframe_averaging.py |
| Semisupervised (depth pseudo-labels) | -7.43% depth only | NO | SEMISUPERVISED_RESULTS.md | semisupervised_pipeline.py |
| Probabilistic gate ensemble | Candidates generated | NO | PROBABILISTIC_GATE_ENSEMBLE_RESULTS_2026-02-14.md | probabilistic_gate_ensemble.py |
| Energy Wave transfer (core + PLS-only) | -0.580141590572% mean | Subs 2602-2607 generated (LB pending) | ENERGY_WAVE_TRANSFER_RESULTS_2026-02-15.md | energy_wave_transfer.py |
| Cross-fitting | Weak (2-10% theoretical) | Subs 2202-2222 (not scored) | CROSS_FITTING_RESULTS.md | cross_fitting.py |

WARNING: Large LOO gains (target denoising -30.6%) may not translate to LB - the CV-LB gap is real.

---

## Dead Ends (Conclusively Failed)

These have been thoroughly tested and do NOT work:

| Technique | Why It Failed | File |
|-----------|--------------|------|
| Physics simulation (MuJoCo/ballistic) | Velocity gap 3.6x, 96% shots invalid, gravity wrong by 2x | PHYSICS_RELEASE_ACCURACY_TEST.md, POST_RELEASE_BALLISTIC_ANALYSIS.md, DEFINITIVE_PHYSICS_RESULTS.md |
| Fingertip ball tracking | Fingertips follow follow-through not ball; fitted gravity 13.9 vs 32.17 ft/s^2 | POST_RELEASE_BALLISTIC_ANALYSIS.md |
| Ball position from keypoints | Hand keypoints collapse to tight cluster (2-3in spread vs 9.4in ball) | BALL_POSITION_ESTIMATION.md |
| External data transfer (NBA/baseball) | Modality mismatch: our data is pose-before-release, external is ball-after-release | NBA_DATA_ANALYSIS.md, OPENBIOMECHANICS_RESULTS.md, EXTERNAL_DATA_RESULTS.md |
| SPL Open Data transfer | Only 125 shots, 1 player, 27 keypoints vs 69; target distributions incompatible | SPL_EXTERNAL_DATA_EXPLORATION.md |
| External motion pretraining (1D CNN) | Pretrained worse than baseline; synthetic+CMU mocap signal too weak | EXTERNAL_MOTION_PRETRAINING_RESULTS.md |
| Random subspace ensemble | Features too correlated (r=0.97-0.99); 3.6% WORSE than baseline | RANDOM_SUBSPACE_ENSEMBLE_RESULTS.md |
| Per-player mean subtraction | Ridge with intercept absorbs mean shift - mathematically identical (0% change) | PER_PLAYER_MEAN_SUBTRACTION_RESULTS.md |
| TTA (test-time augmentation) | Noise adds variance; per-example Ridge already smooth; LB 0.006717 (worse) | FIVE_APPROACHES_RESULTS.md |
| Mirror augmentation | LB 0.011905 - catastrophically worse | NOVEL_APPROACHES_RESULTS.md |
| Data augmentation (rotation+noise) | +9.2% worse MSE | TEST_RESULTS.md |
| Video SSL self-training | Student underperformed teacher across all folds | VIDEO_SSL_SELFTRAIN_RESULTS.md |
| Temporal CNN (1D CNN on 240 frames) | CV 0.010354 vs baseline 0.003743 (2.77x worse) | TEMPORAL_CNN_MPS_RESULTS.md |
| Inverse projectile (physics -> predict) | Direct prediction MSE 0.011041 (worse than 0.007224 baseline) | INVERSE_PROJECTILE_RESULTS.md |
| Kalman velocity estimation | Best RMSE 11.698 ft/s, far from needed accuracy | KALMAN_VELOCITY_ESTIMATION.md |

---

## Weak / Marginal (Not Worth Pursuing)

These showed some signal but not enough to justify further work:

| Technique | Result | File |
|-----------|--------|------|
| Biomech features (43 features) | Good LOO but LB 0.007794 (WORSE than baseline) - classic overfit | BIOMECH_FEATURES_RESULTS.md |
| Inverse dynamics (58 features) | Feature correlations present but no LOO validation completed | INVERSE_DYNAMICS_RESULTS.md |
| TabNet | CV +76% vs Ridge but LB ~0.007682 (large CV-LB gap) | TABNET_EXPERIMENT_RESULTS.md |
| GP variants (Matern, RBF) | Best CV 0.002836 but subs untested on LB; diverse (r=0.907) | GP_FAST_EXPERIMENT_RESULTS.md |
| Self-supervised pretraining | Pilot only; LOO 37.04 mean on 30 samples; full data not evaluated | SELF_SUPERVISED_PRETRAINING_RESULTS.md |
| Video SSL transfer | +5.58% CV improvement but weak external signal, likely overfit | VIDEO_SSL_TRANSFER_RESULTS.md |
| Video pose hands SSL | Pilot CV 0.036914; many subs generated but no confirmed LB gain | VIDEO_POSE_HANDS_SSL_RESULTS.md |
| Multitask learning | CV -32% but large CV-LB gap; 19 subs generated, no strong LB result | MULTITASK_RESULTS.md |
| Uncertainty ensemble | All variants r=1.000 with baseline (no diversity) | UNCERTAINTY_ENSEMBLE_RESULTS.md |
| Angle diverse ensemble | Physics angle r=0.9995 with ML angle (no diversity) | ANGLE_DIVERSE_ENSEMBLE_RESULTS.md |
| Temporal dynamics (full 240-frame) | 450-feature trajectory, subs 1478-1486 generated, no LB improvement | TEMPORAL_DYNAMICS_RESULTS.md |

---

## Analysis Documents (No Direct Experiment)

These contain analysis, strategy, or domain knowledge - not direct experiments:

| File | Category | Key Finding |
|------|----------|-------------|
| RESEARCH.md | Reference | Dataset structure: 344 train, 112 test, 207 features, 3 targets |
| COMPLETE_RESEARCH_SUMMARY.md | Summary | Tracks first 51 submissions; ensemble approach identified as winner |
| train_vs_testset_data.md | Distribution | NO train/test shift (adversarial AUC=0.5042); identical player splits |
| KEYPOINT_NOISE_ANALYSIS.md | Data quality | Noise floor 1.4mm (ankle) to 60mm (hand joints); 332/345 shots clean |
| MOCAP_NOISE_ANALYSIS.md | Data quality | Arm joints 16-26mm noise; finger joints 20-194mm (extremely unreliable) |
| OBSERVATION_SPACE_DECODED.md | Data format | SkillMimic uses heading-relative coordinate frame |
| PLATEAU_BREAKTHROUGH_RESEARCH.md | Strategy | CV-LB gap is overfitting (33-81%); proposed 8 novel approaches |
| VALIDATION_STRATEGY_REVIEW_2026-02-10.md | Strategy | Review of validation approaches |
| ALMOST_PERFECT_SCORE_STRATEGY_2026-02-10.md | Strategy | Strategy for approaching perfect score |
| PATH_TO_007_FINDINGS.md | Analysis | Per-player feature overlap extremely low (Jaccard 0.045-0.083) |
| PRECISION_RESEARCH.md | Analysis | Float32 precision loss (7 sig digits vs 18); recommend float64 |
| PHYSICS_FRAME_ANALYSIS.md | Analysis | Physics frame timing analysis |
| STRATEGY.md | Strategy | Transfer learning approach using SkillMimic data |
| SUBMISSION_STRATEGY.md | Strategy | depth_max as strongest LB predictor; Sub 51 with depth_max=0.7829 |
| SELECTIVE_AMPLIFICATION_ANALYSIS.md | Analysis | Sub 219 technique: pctl=91, alpha=1.1, contrast=Sub151 |
| SUB_219_ANALYSIS.md | Analysis | Sub 219 amplifies 11 high-disagreement samples for 1.63% gain |
| TOP_CANDIDATES_SUMMARY.md | Analysis | Sub 206-210 estimated 0.78% improvement over Sub 133 |
| FINAL_CANDIDATES_SUMMARY.md | Analysis | Sub 219 (LB 0.007682) recommended testing |
| NEW_SUBMISSIONS_SUMMARY.md | Log | 175+ submissions; Sub 183 achieves LB 0.007698 via selective amp |
| AMPLIFICATION_BREAKTHROUGH.md | Analysis | Selective amplification technique documentation |
| PROFILE_CONSTRAINED_BLEND.md | Analysis | Sub 164 validates profile-constrained optimization framework |
| NBA_COMPREHENSIVE_TEST_RESULTS.md | External data | Launch angle correlation -0.604 with angle target; per-player norm fails |
| EXTERNAL_DATA_COMPREHENSIVE.md | External data | Located 3 sources: OpenBiomechanics (411), SkillMimic (3), CMU Mocap |
| PHYSICS_FEATURES_ANALYSIS.md | Physics | Physics feature analysis |
| PHYSICS_FEATURES_SUMMARY.md | Physics | Physics feature summary |
| PHYSICS_SIGNAL_FINAL.md | Physics | Final physics signal analysis |
| PHYSICS_VELOCITY_FINDINGS.md | Physics | Velocity estimation findings |
| physics_approaches_deep_research.md | Physics | Survey of differentiable physics engines (MuJoCo MJX, Nimble, Brax) |
| PHYSICS_MODELING_RESEARCH_2026-02-15.md | Physics | External physics-modeling survey mapped to current pipeline and release-state integration plan |
| LATENT_RELEASE_PRIORITY1_RESULTS_2026-02-15.md | Physics | Priority 1 latent release-state benchmark; full-scale CV regressed vs baseline |
| SUB2503_PROBABILISTIC_ROW_GATE_RESULTS_2026-02-15.md | Physics | Row-level confidence/risk analysis for Sub 2503 with gated hard-model fallback |
| SUB2503_ROW_SURGERY_CANDIDATES_2026-02-15.md | Physics | Risk-targeted row surgery submissions (2579-2584) built from Sub 2503 diagnostics |
| physics_ball_release_research.md | Physics | Ball release detection research |
| physics_feature_extraction_results.md | Physics | Physics feature extraction results |
| MUJOCO_PHYSICS_FINDINGS.md | Physics | MuJoCo simulation findings |
| MUJOCO_PHYSICS_RESEARCH.md | Physics | MuJoCo research |
| MUJOCO_IK_FEASIBILITY.md | Physics | MuJoCo inverse kinematics feasibility |

---

## Infrastructure / Implementation Docs

| File | What It Documents |
|------|------------------|
| GPU_TRAINING.md | Vast.ai GPU setup guide with rsync and training commands |
| IMPLEMENTATION_SUMMARY.md | Implementation summary |
| ANGLE_MODEL_IMPLEMENTATION.md | Angle-specific pipeline: features.py, train/optimize scripts |
| SOLUTION_COMPLETE.md | SkillMimic 3D visualization with 53-joint skeleton |
| BALL_TRACKING_RESULTS.md | Ball tracking with hybrid weighted hand centroid (442-line module) |
| BALL_TRACKING_SUMMARY.md | 3-module ball tracking implementation (1383 lines total) |
| BALL_TRAJECTORY_FIX.md | Found best ball features X=53, Y=275, Z=2 |
| BALL_POSITION_ISSUE.md | Ball position coordinate frame mismatch (2.65 unit offset) |
| BALL_POSITION_FINAL_STATUS.md | 53-joint skeleton accurate; ball trajectory approximate |
| SKILLMIMIC_FORMAT.md | SkillMimic .pt format: 337 features per frame decoded |
| SKILLMIMIC_JOINTS.md | SkillMimic 53-body humanoid skeleton mapping |
| SKILLMIMIC_FIX_SUMMARY.md | Fixed visualization from 20 to 53 joints |
| SKILLMIMIC_TO_SPL_MAPPING.md | Complete feature map between SkillMimic and SPL formats |
| SKILLMIMIC_ANALYSIS_RESULTS.md | SkillMimic data analysis |
| SKILLMIMIC_APPLICATION_RESULTS.md | SkillMimic shows 70 deg release angle, 2.683 ball speed (3 shots) |
| physics_engine_implementation.md | Physics engine implementation details |
| TEST_RESULTS_LOG.md | Historical test results log (50+ submissions) |
| TEST_PLAN.md | Systematic experimental plan |
| EXPERIMENT_RESULTS.md | 12 advanced approaches tested (MTL, GP, LSTM, MAML) |
| FINAL_STATUS.md | Project status summary |
| FINAL_PHYSICS_RESULTS.md | Physics features CV 0.005241 (37% better than Sub 25) |
| VELOCITY_MULTIFRAME_RESULTS.md | Multi-frame physics velocity results |
| PHYSICS_SIMULATION_RESULTS.md | Physics simulation experiment results |
| PHYSICS_PER_EXAMPLE_RESULTS.md | Physics features in per-example pipeline |
| THREE_APPROACHES_EXPERIMENT.md | Hoop-relative coords (0.00752 avg CV), PLS raw timeseries |
| AGENT_TEAM_MISSION_FINAL_REPORT.md | 5-agent parallel exploration: 67 subs, Sub 1350 confirmed near-optimal |
| VIDEO_SSL_SELECTIVE_AMP_SUBMISSIONS.md | Video SSL + selective amplification submissions |
| TARGET_SPECIFIC_SSL_EMBED_RESULTS.md | Target-specific SSL embedding results |

---

## Full File List by Category

### Feature Engineering (7 files)
- BIOMECH_FEATURES_RESULTS.md - 43 biomech features, good LOO but LB worse (overfit)
- INVERSE_DYNAMICS_RESULTS.md - 58 inverse dynamics features, correlations but no LOO
- INVERSE_PROJECTILE_RESULTS.md - Physics -> predict, MSE 0.011 (failed)
- NOVEL_APPROACHES_RESULTS.md - 4 approaches: mirror aug worst (LB 0.011905)
- BALL_POSITION_ESTIMATION.md - Hand keypoints too clustered for ball position
- KALMAN_VELOCITY_ESTIMATION.md - Best RMSE 11.698 ft/s (insufficient)
- COMPREHENSIVE_FEATURE_ENGINEERING_SUMMARY.md - Feature engineering summary

### Physics / Simulation (19 files)
- DEFINITIVE_PHYSICS_RESULTS.md
- PHYSICS_PER_EXAMPLE_RESULTS.md
- PHYSICS_FEATURES_ANALYSIS.md
- PHYSICS_FEATURES_SUMMARY.md
- PHYSICS_SIGNAL_FINAL.md
- PHYSICS_SIMULATION_RESULTS.md
- PHYSICS_VELOCITY_FINDINGS.md
- PHYSICS_FRAME_ANALYSIS.md
- PHYSICS_RELEASE_ACCURACY_TEST.md
- POST_RELEASE_BALLISTIC_ANALYSIS.md
- physics_approaches_deep_research.md
- PHYSICS_MODELING_RESEARCH_2026-02-15.md
- LATENT_RELEASE_PRIORITY1_RESULTS_2026-02-15.md
- SUB2503_PROBABILISTIC_ROW_GATE_RESULTS_2026-02-15.md
- SUB2503_ROW_SURGERY_CANDIDATES_2026-02-15.md
- physics_engine_implementation.md
- physics_ball_release_research.md
- physics_feature_extraction_results.md
- FINAL_PHYSICS_RESULTS.md

### Deep Learning / SSL (8 files)
- SELF_SUPERVISED_PRETRAINING_RESULTS.md
- VIDEO_SSL_TRANSFER_RESULTS.md
- VIDEO_SSL_SELFTRAIN_RESULTS.md
- VIDEO_SSL_SELECTIVE_AMP_SUBMISSIONS.md
- VIDEO_POSE_HANDS_SSL_RESULTS.md
- TARGET_SPECIFIC_SSL_EMBED_RESULTS.md
- TEMPORAL_CNN_MPS_RESULTS.md
- TABNET_EXPERIMENT_RESULTS.md

### Ensemble / Blending (10 files)
- FIVE_APPROACHES_RESULTS.md
- MULTIFRAME_ENSEMBLE_RESULTS.md
- MULTIFRAME_AVERAGING_RESULTS.md
- KERNEL_EXPERIMENTS_RESULTS.md
- ANGLE_DIVERSE_ENSEMBLE_RESULTS.md
- UNCERTAINTY_ENSEMBLE_RESULTS.md
- RANDOM_SUBSPACE_ENSEMBLE_RESULTS.md
- CROSS_FITTING_RESULTS.md
- PER_PLAYER_MEAN_SUBTRACTION_RESULTS.md
- PROBABILISTIC_GATE_ENSEMBLE_RESULTS_2026-02-14.md

### Data / Augmentation (8 files)
- EXTERNAL_DATA_RESULTS.md
- EXTERNAL_DATA_FINAL_SUMMARY.md
- EXTERNAL_MOTION_PRETRAINING_RESULTS.md
- SPL_EXTERNAL_DATA_EXPLORATION.md
- MIXUP_AUGMENTATION_RESULTS.md
- SEMISUPERVISED_RESULTS.md
- TARGET_DENOISING_RESULTS.md
- MULTITASK_RESULTS.md

### External Data Sources (5 files)
- NBA_DATA_ANALYSIS.md
- NBA_COMPREHENSIVE_TEST_RESULTS.md
- EXTERNAL_DATA_COMPREHENSIVE.md
- OPENBIOMECHANICS_RESULTS.md
- SPL_EXTERNAL_DATA_EXPLORATION.md

### MuJoCo / SkillMimic (9 files)
- MUJOCO_PHYSICS_FINDINGS.md
- MUJOCO_PHYSICS_RESEARCH.md
- MUJOCO_IK_FEASIBILITY.md
- SKILLMIMIC_FORMAT.md
- SKILLMIMIC_ANALYSIS_RESULTS.md
- SKILLMIMIC_JOINTS.md
- SKILLMIMIC_FIX_SUMMARY.md
- SKILLMIMIC_TO_SPL_MAPPING.md
- SKILLMIMIC_APPLICATION_RESULTS.md

### Strategy / Analysis (12 files)
- PLATEAU_BREAKTHROUGH_RESEARCH.md
- VALIDATION_STRATEGY_REVIEW_2026-02-10.md
- ALMOST_PERFECT_SCORE_STRATEGY_2026-02-10.md
- STRATEGY.md
- SUBMISSION_STRATEGY.md
- SELECTIVE_AMPLIFICATION_ANALYSIS.md
- SUB_219_ANALYSIS.md
- TOP_CANDIDATES_SUMMARY.md
- FINAL_CANDIDATES_SUMMARY.md
- PATH_TO_007_FINDINGS.md
- PRECISION_RESEARCH.md
- PROFILE_CONSTRAINED_BLEND.md

### Noise / Data Quality (2 files)
- KEYPOINT_NOISE_ANALYSIS.md
- MOCAP_NOISE_ANALYSIS.md

### Temporal / Trajectory (3 files)
- TEMPORAL_DYNAMICS_RESULTS.md
- TEMPORAL_DYNAMICS_TESTING_PRIORITY.md
- TRAJECTORY_PER_EXAMPLE_RESULTS.md

### Infrastructure / Logs (10 files)
- GPU_TRAINING.md
- RESEARCH.md
- COMPLETE_RESEARCH_SUMMARY.md
- train_vs_testset_data.md
- TEST_PLAN.md
- TEST_RESULTS.md
- TEST_RESULTS_LOG.md
- EXPERIMENT_RESULTS.md
- NEW_SUBMISSIONS_SUMMARY.md
- AGENT_TEAM_MISSION_FINAL_REPORT.md

### Ball Tracking (5 files)
- BALL_TRACKING_RESULTS.md
- BALL_TRACKING_SUMMARY.md
- BALL_TRAJECTORY_FIX.md
- BALL_POSITION_ISSUE.md
- BALL_POSITION_FINAL_STATUS.md

### Implementation (4 files)
- IMPLEMENTATION_SUMMARY.md
- ANGLE_MODEL_IMPLEMENTATION.md
- SOLUTION_COMPLETE.md
- FINAL_STATUS.md
