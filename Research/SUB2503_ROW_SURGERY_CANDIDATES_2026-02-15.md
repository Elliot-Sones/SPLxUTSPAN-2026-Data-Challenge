# Sub 2503 Row Surgery Candidates

Date: 2026-02-15
Script: `scripts/sub2503_row_surgery_candidates.py`

## Objective

Increase LB robustness by modifying only uncertain/risky rows from `submission_2503.csv`.

## Inputs

- Base: `submission/submission_2503.csv`
- Conservative fallback: `submission/submission_2475.csv`
- Angle alternative: `submission/submission_2506.csv`
- Risk table:
  - `output/sub2503_row_confidence_sub2503_prob_row_gate_20260215.csv`

## What We Ran - Exact Command

```bash
uv run python scripts/sub2503_row_surgery_candidates.py --base-sub 2503 --fallback-sub 2475 --angle-sub 2506 --risk-csv output/sub2503_row_confidence_sub2503_prob_row_gate_20260215.csv --run-tag sub2503_row_surgery_20260215
```

## Candidate Policies and Submission Files

1. `C1_uncertain_lr_fallback`
- Rule: For uncertain rows (confidence <= 0.20), replace only `scaled_left_right` with Sub 2475.
- Output: `submission/submission_2579.csv`
- Rows changed: `23`
- Mean abs deltas:
  - angle: `0.000000000000`
  - depth: `0.000000000000`
  - left_right: `0.000880010606`

2. `C2_uncertain_depth_lr_fallback`
- Rule: For uncertain rows, replace `scaled_depth` and `scaled_left_right` with Sub 2475.
- Output: `submission/submission_2580.csv`
- Rows changed: `23`
- Mean abs deltas:
  - angle: `0.000000000000`
  - depth: `0.001005438752`
  - left_right: `0.000880010606`

3. `C3_uncertain_angle2506_lr2475`
- Rule: For uncertain rows, replace angle with Sub 2506 and LR with Sub 2475.
- Output: `submission/submission_2581.csv`
- Rows changed: `23`
- Mean abs deltas:
  - angle: `0.000000000000`
  - depth: `0.000000000000`
  - left_right: `0.000880010606`
- Note: angle delta is zero vs Sub 2503 in this set, so this is effectively similar to C1 for current files.

4. `C4_highrisk_all_fallback`
- Rule: For high-risk rows (confidence <= 0.10), replace all three targets with Sub 2475.
- Output: `submission/submission_2582.csv`
- Rows changed: `12`
- Mean abs deltas:
  - angle: `0.000758500038`
  - depth: `0.000574626411`
  - left_right: `0.000626316618`

5. `C5_p5_uncertain_depth_lr_fallback_plus_harddisagree_lr`
- Rule:
  - For uncertain player-5 rows: replace depth+LR with Sub 2475.
  - For uncertain rows flagged `hard_model_disagrees_with_base`: replace LR with Sub 2475.
- Output: `submission/submission_2583.csv`
- Rows changed: `14`
- Mean abs deltas:
  - angle: `0.000000000000`
  - depth: `0.000411518822`
  - left_right: `0.000449760062`

6. `C6_ood_all_fallback_uncertain_angle2506`
- Rule:
  - For `ood_motion_pattern` rows: replace all targets with Sub 2475.
  - For uncertain rows: replace angle with Sub 2506.
- Output: `submission/submission_2584.csv`
- Rows changed: `12`
- Mean abs deltas:
  - angle: `0.000135110091`
  - depth: `0.000446575446`
  - left_right: `0.000240974497`

## Recommended LB Testing Order

1. `submission/submission_2583.csv` - targeted and conservative, focuses risk concentration in player 5 and LR instability.
2. `submission/submission_2579.csv` - clean LR-only rollback on uncertain rows.
3. `submission/submission_2584.csv` - OOD-specific fallback with minimal global perturbation.
4. `submission/submission_2582.csv` - stronger rollback for highest risk only.
5. `submission/submission_2580.csv` - broader depth+LR rollback.
6. `submission/submission_2581.csv` - expected near-duplicate behavior with current files.

## Reproducibility Artifact

- Run metadata:
  - `output/sub2503_row_surgery_run_sub2503_row_surgery_20260215.json`

