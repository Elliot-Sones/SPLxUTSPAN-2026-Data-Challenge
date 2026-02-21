# P4 Angle Fix Results - 2026-02-19

## Context
- P4 has highest angle variance (std=2.684 raw, 0.0895 scaled) among typical players
- Sub 3411 P4 angle calibration: 0.632 (captures only 63% of true variation)
- P4 has 67 training samples, 22 test samples
- P4 angle was identified as the single biggest error source (33.6% of all angle error)

## Phase 1: Hyperparameter Sweep (5-fold honest CV)
- Tested 160 configs: bw x [0.15-0.80], alpha x [1-50], PLS nc x [3-10]
- Baseline (bw=0.30, alpha=10, PLS nc=10): P4 CV = 0.005339
- **Best config: bw=0.15, alpha=50.0, PLS nc=10**
- Best CV: 0.004763 (+10.78% vs baseline)
- Key insight: P4 benefits from NARROW bandwidth (0.15) and HIGH regularization (alpha=50)
  - This makes physical sense: P4 has high variance, so using very local neighbors
    with strong regularization prevents overfitting to noisy neighbors

## Phase 2: Frame Sweep (with best hyperparams)
- Frame 140: 0.005025 (+5.87%)
- Frame 145: 0.004829 (+9.55%)
- Frame 148: 0.004753 (+10.97%)
- Frame 150: 0.004763 (+10.78%)
- Frame 153: 0.004518 (+15.37%)
- Frame 155: 0.004307 (+19.33%)
- **Frame 160: 0.004306 (+19.34%) <-- BEST**
- Note: P4 optimal frame is 160, much later than the current PLAYER_FRAMES[angle][4]=150
  - Later frames = more of the release motion captured = better angle discrimination
  - The default frame 153 for angle was optimized globally, not per-player

## Phase 3: Multi-Frame Ensemble
- Top-2 [160, 155]: CV = 0.004270 (+20.02%) <-- best ensemble
- Top-3 [160, 155, 153]: CV = 0.004341 (+18.69%)
- Top-4: CV = 0.004419 (+17.22%)
- Top-5: CV = 0.004480 (+16.08%)
- Adding more frames dilutes the signal from the best frames

## Phase 4: Alternative Models (at best frame, best PLS)
- All non-Ridge models HURT significantly:
  - knn_5: 0.005957 (-11.58%)
  - knn_10: 0.005935 (-11.17%)
  - rf_small: 0.006443 (-20.68%)
  - rf_medium: 0.006076 (-13.81%)
  - gbr: 0.007251 (-35.82%)
  - lgb_cons: 0.008459 (-58.45%)
  - lgb_mod: 0.007545 (-41.33%)
  - ridge_1: 0.006267 (-17.39%)
- Ridge with high alpha works best:
  - ridge_50: 0.004623 (+13.41%)
  - **ridge_100: 0.004471 (+16.26%)**
- This confirms: locally weighted Ridge is the RIGHT model family,
  but it needs HIGHER regularization than the default alpha=10

## Phase 5: Cross-Player Transfer
- ALL configurations HURT P4 angle (-34% to -47%)
- bw=0.3, upweight=5.0 was least bad at 0.007165 (-34.21%)
- Conclusion: P4's shooting mechanics are UNIQUE enough that other players' data
  degrades the signal. P4-only models are clearly better.

## Final Ranking (5-fold CV with honest PLS)
1. best_hyper (bw=0.15, a=50, nc=10, f=160): CV = 0.004306 (+19.34%)
2. top2_ens (frames 160+155 ensemble): CV = 0.004308 (+19.30%)
3. multiframe3 (frames 160+155+153): CV = 0.004341 (+18.69%)
4. top3_ens: CV = 0.004344 (+18.63%)
5. alt_ridge_100: CV = 0.004471 (+16.26%)
6. all_ens: CV = 0.004592 (+13.98%)
7. baseline: CV = 0.005339 (reference)
8. cross_player: CV = 0.007165 (-34.21%)

## Calibration Analysis
- Sub3411 P4 angle: pred_std=0.05785, train_std=0.08948, ratio=0.632
- best_hyper P4: pred_std=0.05820, calibration=0.650
- multiframe3 P4: pred_std=0.05886, calibration=0.658
- Note: calibration improved from 0.632 to 0.650-0.658 (small but correct direction)

## P5 Angle Check
- P5 baseline CV: 0.019738
- P5 with P4-optimal params: 0.019240 (+2.53%)
- P5 best frame: 148 (CV = 0.017130, +13.21%)
- P5 also benefits from frame optimization (148 vs 150 default)
- P5 improvement is useful for P4+P5 splice submissions

## Submissions Generated
12 submissions total (Sub 3474-3485):

### P4-Only Splices (into Sub 3411)
- **Sub 3474**: P4 splice [best_hyper] - bw=0.15, alpha=50, PLS nc=10, frame=160. P4 CV=0.004306 (+19.34%)
- **Sub 3475**: P4+P5 splice [best_hyper + P5 frame=148]
- Sub 3476: P4 splice [multiframe top-3 frames 160,155,153]. P4 CV=0.004341 (+18.69%)
- Sub 3477: P4+P5 splice [multiframe + P5opt]
- Sub 3478: P4 splice [Ridge alpha=100 no kernel]. P4 CV=0.004471 (+16.26%)
- Sub 3479: P4+P5 splice [Ridge100 + P5opt]
- Sub 3480: P4 splice [top-2 ensemble]. P4 CV=0.004308 (+19.30%)
- Sub 3481: P4+P5 splice [top2_ens + P5opt]

### P4 Angle Blends (with Sub 3411)
- Sub 3482: 30% best_hyper + 70% Sub3411 P4 angle
- Sub 3483: 50% best_hyper + 50% Sub3411 P4 angle
- Sub 3484: 70% best_hyper + 30% Sub3411 P4 angle
- **Sub 3485**: 100% best_hyper + 0% Sub3411 P4 angle (same as Sub 3474)

## Recommended Priority for LB Testing
1. **Sub 3474** (best_hyper P4 splice) - single best CV, full replacement
2. **Sub 3475** (P4+P5 splice) - adds P5 improvement too
3. **Sub 3483** (50% blend) - safer if new model overfits
4. Sub 3476 (multiframe) - slightly different predictions

## Key Takeaways
1. P4 angle benefits from NARROW bandwidth (0.15 vs 0.30) and HIGH alpha (50 vs 10)
2. P4 optimal angle frame is 160 (not 150) - later in the release motion
3. Cross-player transfer HURTS for P4 angle (-34%)
4. Ridge is the right model family; tree models and k-NN are worse
5. 19% improvement in P4 angle CV translates to potential overall improvement of:
   - P4 contributes 22/113 = 19.5% of test rows
   - If P4 angle CV drops by 19%, angle MSE drops by ~3.8%
   - Overall MSE impact: ~1.3% improvement (~0.00008 absolute)

## Runtime
- Total: ~200s (feature extraction: 21s, sweep: 42s, rest: predictions and IO)


## Per-Player Angle Optimization (all 5 players)
Default overall angle CV: 0.007178
Optimized overall angle CV: 0.006227 (+13.25%)

### Per-Player Results
- P1: 0.001787 -> 0.001426 (+20.22%) bw=0.8, alpha=100.0, nc=8, frames=[145, 140, 170]
- P2: 0.004984 -> 0.004385 (+12.01%) bw=0.8, alpha=20.0, nc=3, frames=[145, 140]
- P3: 0.004002 -> 0.003073 (+23.21%) bw=0.2, alpha=100.0, nc=10, frames=[160, 140]
- P4: 0.005019 -> 0.004200 (+16.32%) bw=0.1, alpha=50.0, nc=10, frames=[165, 155]
- P5: 0.019108 -> 0.017144 (+10.28%) bw=0.1, alpha=50.0, nc=10, frames=[148, 153]

### Submissions
- Sub 3496: ALL-PLAYER angle splice into Sub3411. Overall angle CV: 0.006227 (+13.25% vs default). Per-player configs: {np.int64(1): {'bw': 0.8, 'alpha': 100.0, 'pls_nc': 8, 'frames': [145, 140, 170]}, np.int64(2): {'bw': 0.8, 'alpha': 20.0, 'pls_nc': 3, 'frames': [145, 140]}, np.int64(3): {'bw': 0.2, 'alpha': 100.0, 'pls_nc': 10, 'frames': [160, 140]}, np.int64(4): {'bw': 0.1, 'alpha': 50.0, 'pls_nc': 10, 'frames': [165, 155]}, np.int64(5): {'bw': 0.1, 'alpha': 50.0, 'pls_nc': 10, 'frames': [148, 153]}}
- Sub 3497: 50% all-player angle opt + 50% Sub3411 angle.
- Sub 3498: 30% all-player angle opt + 70% Sub3411 angle.
- Sub 3499: P4+P5 angle splice into Sub3411 (safest, biggest error contributors).
- Sub 3500: P2+P4+P5 angle splice into Sub3411.

Runtime: 418.2s
