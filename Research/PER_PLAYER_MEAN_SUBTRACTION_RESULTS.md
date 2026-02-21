# Per-Player Mean Subtraction Results

**Date**: 2026-02-12
**Script**: scripts/per_player_mean_subtraction.py
**Status**: DEFINITIVE NULL RESULT - No effect whatsoever

## Concept

Instead of predicting raw target values, subtract per-player target means from training targets, train models on residuals, then add means back. The hypothesis was that this reduces the model's burden of learning player-level baselines and could reduce overfitting.

## Phase 1: Per-Player Target Statistics

Per-player means in scaled space:

| Player | n  | angle mean | angle std | depth mean | depth std | LR mean | LR std |
|--------|-----|-----------|-----------|-----------|-----------|---------|--------|
| Global | 345 | 0.516136  | 0.162124  | 0.515685  | 0.128404  | 0.475553| 0.118606|
| 1      | 70  | 0.479495  | 0.043130  | 0.555486  | 0.107524  | 0.473786| 0.127986|
| 2      | 66  | 0.420551  | 0.068450  | 0.545905  | 0.100708  | 0.496804| 0.116125|
| 3      | 68  | 0.600510  | 0.054279  | 0.512962  | 0.054896  | 0.460970| 0.088511|
| 4      | 67  | 0.741706  | 0.089477  | 0.502040  | 0.114710  | 0.480089| 0.122003|
| 5      | 74  | 0.354284  | 0.135723  | 0.465936  | 0.192976  | 0.467563| 0.129242|

Key observations:
- Angle has the largest per-player mean differences (range 0.387, from 0.354 to 0.742)
- Left_right has the smallest per-player mean differences (range 0.036)
- Player 5 has the highest variance for both depth (0.193) and angle (0.136)

## Phase 2-5: All Variants Tested

| Variant | angle LOO | depth LOO | LR LOO | mean LOO | delta vs baseline |
|---------|-----------|-----------|--------|----------|-------------------|
| baseline (no mean sub) | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |
| naive_mean_sub (shrink=1.0) | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |
| shrunk_mean_sub (best w) | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |
| lopo_honest | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |
| standardized (best w) | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |
| lopo_standardized | 0.002645 | 0.004601 | 0.004331 | 0.003859 | 0.00% |

ALL variants produce IDENTICAL LOO MSE to 6 decimal places.

## Phase 6: Diversity Analysis

Correlation with Sub 2063 test predictions:
- Angle: r=0.9806 (all variants identical)
- Depth: r=0.9734 (all variants identical)
- Left_right: r=0.9818 (all variants identical)

All variants produce identical test predictions. No diversity benefit.

## Phase 7: Submissions Generated

Submissions 2172-2190 generated but are effectively identical across variants.
Since the mean subtraction produces no change, these are all equivalent to blending the same base prediction at different weights with Sub 2063.

## Why It Does Not Work

**The per-player Ridge regression with intercept already learns the per-player mean implicitly.**

The pipeline already processes each player separately (the `for pid in unique_pids` loop). Within each player's data, Ridge regression fits:

    y = X * beta + intercept

When we subtract the mean to get:

    (y - mean) = X * beta' + intercept'

The Ridge solution is mathematically equivalent because:
- The intercept absorbs the mean shift
- The coefficients are unchanged (Ridge penalizes coefficients, not intercept)
- The prediction is: X * beta' + intercept' + mean = X * beta + intercept

This is a well-known property of linear models with intercept terms. Mean subtraction is a no-op for Ridge regression. It would ONLY matter for:
1. Models without intercepts (rare)
2. Non-linear models where the mean shift changes the curvature
3. Cases where the mean is computed from a different set than the training set (cross-player transfer)

The same reasoning explains why shrinkage, LOPO-honest, and standardization also have no effect - they are all linear transformations of the target, which Ridge absorbs into its intercept and coefficients.

## Conclusion

Per-player mean subtraction is a NO-OP for per-player locally weighted Ridge regression. The approach cannot improve the current pipeline because Ridge regression already implicitly performs mean subtraction via its intercept term.

This approach would only be relevant if:
- Using tree-based models (which don't have a continuous intercept)
- Using cross-player training (where means from other players are used)
- Using models without bias terms

Since the pipeline already trains per-player models with Ridge regression (which has intercept), this direction is definitively closed.
