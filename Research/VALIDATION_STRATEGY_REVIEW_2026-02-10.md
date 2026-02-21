# Validation Strategy Review - 2026-02-10

## Objective
Document the current validation approach in this repo and define a stricter protocol that better predicts both public and private leaderboard behavior.

---

## What Is Currently Used

### 1. Within-player 5-fold CV (core historical strategy)
- `src/ensemble_submission.py:241` uses `KFold(n_splits=5, shuffle=True, random_state=42)` **inside each participant**.
- `scripts/sub9_cv_score.py:182` does the same for Sub9 replication.
- This has been treated as representative because train/test participant composition is similar (`Research/train_vs_testset_data.md:17`).

### 2. Grouped CV / LOPO variants (also used)
- `src/experiment_runner.py:261` uses `GroupKFold` by group.
- `scripts/cv_lb_correlation.py:285` uses `LeaveOneGroupOut` (LOPO), and also compares within-player/LOPO/bootstrap.
- `scripts/oof_stacking_pipeline.py:9` and `scripts/oof_stacking_pipeline.py:953` use LOPO OOF generation and LOPO-based model ranking.

### 3. Submission-profile and surrogate LB heuristics (used heavily in late-stage selection)
- Correlation-based LB surrogate from submission statistics:
  - `scripts/predict_lb_score.py:17` uses only 8 known LB points.
- Manual profile constraints and post-hoc calibration:
  - `scripts/final_candidates.py:28` targets fixed depth mean.
  - `scripts/optimize_around_219.py:69` recalibrates depth mean to `0.5055`.
  - `scripts/optimal_blend_search.py:76` optimizes by correlation/diversity against Sub1640.
  - `scripts/angle_fix.py:559` swaps angle into Sub1640 and ranks by LOO plus anchor correlation.

### 4. Existing quantitative evidence in repo
- `output/cv_lb_correlation.csv:2-4` shows CV scores do not rank LB cleanly.
- `output/cv_lb_correlations.csv:2-5` shows weak/inconclusive CV-metric correlations with LB on tiny sample size.
- `Research/TEST_RESULTS.md:1087` vs `Research/TEST_RESULTS.md:1088` shows large CV-LB mismatch in at least one run.

---

## Main Problems

1. Model selection objective is not stable:
- Different scripts optimize different proxies (within-player CV, LOPO, profile distance, anchor correlation, heuristic predicted LB).

2. Public-LB overfitting risk:
- Many scripts optimize around known top submissions and profile stats (for example Sub133/Sub1350/Sub1640 anchoring), which can chase public-LB artifacts.

3. Surrogate LB model is statistically weak:
- `scripts/predict_lb_score.py` fits on very few labeled points, then extrapolates.

4. Leakage risk from non-nested tuning:
- In multiple workflows, feature/model/hyperparameter choices are made on the same validation structure used for reporting.

5. No single "go/no-go" robust metric:
- There is no enforced acceptance rule requiring both mean and worst-case validation improvement before submission.

---

## Recommended Replacement Protocol

Use one standardized validation stack for all model selection decisions.

### A. Primary proxy: Repeated participant-stratified holdout (RPSH)
- Repeat `R=50` times with fixed seeds.
- In each repeat:
  - For each participant, hold out exactly `round(0.20 * n_pid)` shots as validation.
  - Train on the remaining shots.
  - Score on held-out shots with official scaled MSE.
- Report:
  - `mean_mse`, `std_mse`, `p90_mse`, `worst_mse`.

Why:
- Mimics test composition (same participants, similar proportions) without reusing a single fold partition.
- Gives variance and tail risk, not just one average.

### B. Secondary proxy: LOPO stress test
- Keep LOPO, but use it only as robustness stress test, not as main ranking metric.
- Report per-player errors to detect collapse on specific players.

### C. Strict nested tuning
- Outer split = RPSH repeat split.
- Inner split = participant-aware CV only on outer-train data.
- Hyperparameters, feature selection, and blending weights must be chosen only in inner loop.
- Outer validation remains untouched until final scoring.

### D. Public/private simulation inside local validation
- For each RPSH validation set:
  - Split it into `public_like` (35%) and `private_like` (65%) stratified by participant.
  - Compute both scores.
- Accept model only if it improves:
  - overall validation mean,
  - and private_like mean,
  - and does not worsen p90 by more than tolerance.

### E. Single submission gate (hard rule)
- Define one scalar gate score:
  - `gate = mean_mse + 0.5 * std_mse + 0.5 * (p90_mse - mean_mse)`
- Submit only if new model beats baseline gate score by a minimum margin (for example 1-2% relative) across at least 70% of repeats.

### F. Stop using profile-only optimization as primary criterion
- Keep profile metrics for diagnostics only.
- Do not rank candidates by angle std/depth mean proximity alone.

---

## Immediate Practical Migration Steps

1. Freeze a baseline:
- Use current best trusted submission pipeline as fixed baseline.

2. Build one shared validation harness:
- Input: train data, participant ids, model training callable.
- Output: RPSH + LOPO metrics and gate score.

3. Enforce experiment logging schema:
- Every run records:
  - split seeds,
  - fold membership indices,
  - hyperparameters,
  - per-repeat metrics,
  - final gate decision.

4. Only then run leaderboard submissions:
- Submit only candidates that pass gate, not those selected by profile heuristics alone.

---

## Bottom Line

Current strategy mixes valid CV with weak LB surrogates and anchor/profile heuristics, which explains repeated CV-LB mismatch.  
A repeated participant-stratified nested protocol with explicit private-like stress metrics is the most direct way to make "good local score" correspond more reliably to both public and private leaderboard performance.
