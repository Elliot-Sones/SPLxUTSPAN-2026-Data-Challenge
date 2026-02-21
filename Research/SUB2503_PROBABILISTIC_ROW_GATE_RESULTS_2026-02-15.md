# Sub 2503 Probabilistic Row Gate - Results

Date: 2026-02-15
Script: `scripts/sub2503_probabilistic_row_gate.py`

## Objective

Use `submission_2503` as baseline and build row-level confidence:
- Keep highly confident rows close to base prediction.
- "Try harder" on uncertain rows using a stronger local-weighted model.
- Return IDs that are most confident vs most likely wrong.

## What We Ran - Exact Command

```bash
uv run python scripts/sub2503_probabilistic_row_gate.py --base-sub 2503 --expert-subs 2475,2493,2497,2502,2504,2505,2506,2507 --ood-k 5 --hard-bw 0.50 --uncertain-quantile 0.20 --max-hard-weight 0.35 --top-n 25 --seed 20260215 --run-tag sub2503_prob_row_gate_20260215
```

## Data and Inputs

- Base submission: `submission/submission_2503.csv`
- Expert submissions:
  - `submission/submission_2475.csv`
  - `submission/submission_2493.csv`
  - `submission/submission_2497.csv`
  - `submission/submission_2502.csv`
  - `submission/submission_2504.csv`
  - `submission/submission_2505.csv`
  - `submission/submission_2506.csv`
  - `submission/submission_2507.csv`
- Train/test data for OOD and hard-model path:
  - `data/train.csv`
  - `data/test.csv`

## Confidence Model Components

Per-row risk used:
- Ensemble disagreement around Sub 2503 (std/iqr/deviation-from-median across expert submissions).
- OOD percentile from nearest-neighbor distance in per-example handcrafted feature space.
- Prediction extremeness vs train scaled target distribution.
- Disagreement between Sub 2503 and hard local-weighted model (`hard_bw=0.50`).

Confidence:
- `confidence = 1 - percentile_rank(risk_raw)`
- Uncertain rows: bottom 20% confidence (`23` rows out of `113`).

## Exact Summary Results

From `output/sub2503_probabilistic_gate_run_sub2503_prob_row_gate_20260215.json`:

- `n_rows`: `113`
- `confidence_threshold`: `0.200000000000`
- `n_uncertain_rows`: `23`
- `mean_confidence`: `0.500000000000`
- Hard model OOF MSE:
  - angle: `0.0025105530502189904`
  - depth: `0.004510395048347482`
  - left_right: `0.004208871868109801`
- Mean gated delta per target (vs Sub 2503 predictions):
  - angle: `0.0008969066135401778`
  - depth: `0.000921405835292281`
  - left_right: `0.0017891300856627023`

## Most Confident Rows (Top 10)

Source: `output/sub2503_most_confident_sub2503_prob_row_gate_20260215.csv`

| id | pid | confidence | disagreement_std_mean | ood_mean | hard_delta_mean |
|---|---:|---:|---:|---:|---:|
| `ee10a9cc-e5a4-4e0d-b8a8-92a44ba5647f` | 3 | 1.0000000000000000 | 0.0019111525366493 | 0.0784313725490196 | 0.0015345978350942 |
| `f1b1e504-f4df-49c9-9214-b3e8418ddfa4` | 1 | 0.9910714285714286 | 0.0016825829551850 | 0.0142857142857142 | 0.0129078540104136 |
| `15b2802d-15c6-486c-96d7-a0689cbc2f7c` | 4 | 0.9821428571428572 | 0.0005382918675024 | 0.4079601990049751 | 0.0054589125014685 |
| `4bdf6d0b-32fc-40ac-91b2-c4f596676a14` | 1 | 0.9732142857142856 | 0.0022362717102153 | 0.0571428571428571 | 0.0129452229051371 |
| `ac4245b5-53d0-4c93-bf2a-f4cbf9340fac` | 2 | 0.9642857142857144 | 0.0010351726780116 | 0.4797979797979798 | 0.0060055887599566 |
| `6fb475ff-1732-42bc-8385-9f80956199fe` | 1 | 0.9553571428571428 | 0.0016455719626906 | 0.3476190476190475 | 0.0031068481653507 |
| `429040df-7594-4803-90b1-9db6d53c8f3e` | 2 | 0.9464285714285714 | 0.0012308231583377 | 0.4595959595959595 | 0.0038944052411858 |
| `7fa9f4f8-0044-41c5-9af6-a74d57cc2abd` | 2 | 0.9375000000000000 | 0.0025388895692989 | 0.2878787878787878 | 0.0041589108812092 |
| `1336d1b5-00a6-42a5-a0ef-0fcd103c3157` | 3 | 0.9285714285714286 | 0.0016761338810948 | 0.4754901960784313 | 0.0066828584219458 |
| `39f95c12-deab-4d77-8a9c-feecda4d5a66` | 1 | 0.9196428571428572 | 0.0030136580604418 | 0.4238095238095238 | 0.0033711749101899 |

## Most Likely Wrong / Most Off Rows (Top 10 Risk)

Source: `output/sub2503_most_risky_sub2503_prob_row_gate_20260215.csv`

| id | pid | confidence | risk_reasons | disagreement_std_mean | ood_mean | hard_delta_mean |
|---|---:|---:|---|---:|---:|---:|
| `79363fbd-902e-4df3-95af-d8ab53e608df` | 3 | 0.0000000000000000 | high_model_disagreement;hard_model_disagrees_with_base | 0.0332113352565822 | 0.8578431372549020 | 0.0470050675462261 |
| `68b1c0f8-27a1-497c-ad48-cf3e42894cc7` | 5 | 0.0089285714285713 | high_model_disagreement;ood_motion_pattern;extreme_prediction_profile;hard_model_disagrees_with_base | 0.0303100008602607 | 0.9954954954954954 | 0.0771531376168506 |
| `f6095bd1-d599-4a3f-aa6c-455ae7296c49` | 3 | 0.0178571428571429 | high_model_disagreement;ood_motion_pattern;hard_model_disagrees_with_base | 0.0289043049559135 | 0.9852941176470588 | 0.0620250767263681 |
| `edb762ad-c1a9-4fd9-8cbf-b2131bfc0b42` | 2 | 0.0267857142857143 | high_model_disagreement | 0.0192062299948887 | 0.8888888888888888 | 0.0209091461798706 |
| `b686eea3-5bf9-42b7-b3d8-44c7cb96499a` | 5 | 0.0357142857142857 | high_model_disagreement;ood_motion_pattern;hard_model_disagrees_with_base | 0.0156758080693393 | 0.9594594594594592 | 0.0663806556939806 |
| `ab52b638-2bb2-4083-8d8c-49165ce97252` | 5 | 0.0446428571428570 | high_model_disagreement | 0.0141453445258779 | 0.9054054054054054 | 0.0160033892881261 |
| `35b0e24b-2a01-4c18-9b2f-fde61116c884` | 2 | 0.0535714285714286 | high_model_disagreement;ood_motion_pattern | 0.0151353339559933 | 0.9242424242424242 | 0.0153992993148526 |
| `e6e7870e-22ac-467d-bff4-f9ae912f3f0f` | 5 | 0.0625000000000000 | moderate_combined_risk | 0.0096770037053859 | 0.8468468468468467 | 0.0266214105608131 |
| `56159d6d-f377-4139-b511-0e47361d5d7d` | 4 | 0.0714285714285714 | high_model_disagreement | 0.0165755134399296 | 0.2686567164179104 | 0.0243193330670689 |
| `1d99c47c-e316-4d7b-843a-3740439a37b4` | 2 | 0.0803571428571429 | high_model_disagreement | 0.0151548519529680 | 0.4040404040404040 | 0.0209638052997245 |

## Distribution of Top-25 Risky Rows by Player

Exact counts:

- player 2: `5`
- player 3: `3`
- player 4: `6`
- player 5: `11`

## Outputs

- All rows with confidence/risk diagnostics:
  - `output/sub2503_row_confidence_sub2503_prob_row_gate_20260215.csv`
- Top confident rows:
  - `output/sub2503_most_confident_sub2503_prob_row_gate_20260215.csv`
- Top risky rows:
  - `output/sub2503_most_risky_sub2503_prob_row_gate_20260215.csv`
- Gated candidate predictions:
  - `output/sub2503_gated_candidate_sub2503_prob_row_gate_20260215.csv`
- Run metadata:
  - `output/sub2503_probabilistic_gate_run_sub2503_prob_row_gate_20260215.json`

