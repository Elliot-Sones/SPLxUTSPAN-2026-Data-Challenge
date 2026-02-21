# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_fb_combo_pilot_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 3 --feature-bank hybrid_advanced_plus_frame_triplet --models all --run-tag rs_nested_fb_combo_pilot_20260214`
- seed: `20260214`
- scale: `1`
- feature_bank: `hybrid_advanced_plus_frame_triplet`
- models_arg: `all`
- best_lb_reference: `0.006596000000000`
- n_outer_repeats: `1`
- outer_holdout_frac: `0.2`
- inner_folds: `2`
- n_bootstrap: `40`
- n_dirichlet_target: `80`
- uncertainty_lambda: `0.4`
- n_models: `5`
- n_train: `345`
- n_test: `113`

## Model Summary

- `elasticnet` `angle`: mean_mse=`0.010842755587499`, std_mse=`nan`
- `elasticnet` `depth`: mean_mse=`0.008393179711721`, std_mse=`nan`
- `elasticnet` `left_right`: mean_mse=`0.010772862290626`, std_mse=`nan`
- `extra_trees` `angle`: mean_mse=`0.008636794731280`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.008744813069200`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.011410042879152`, std_mse=`nan`
- `knn` `angle`: mean_mse=`0.010083623473393`, std_mse=`nan`
- `knn` `depth`: mean_mse=`0.011790426836420`, std_mse=`nan`
- `knn` `left_right`: mean_mse=`0.015121286538013`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.008612169768371`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.009612922744901`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.010330916992182`, std_mse=`nan`
- `ridge` `angle`: mean_mse=`0.016151749506967`, std_mse=`nan`
- `ridge` `depth`: mean_mse=`0.008996726213987`, std_mse=`nan`
- `ridge` `left_right`: mean_mse=`0.012886603764099`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2384.csv`
  - submission_num: `2384`
  - combo_id: `50`
  - score: `0.016646819839789`
  - mean_boot_mse: `0.016327086219144`
  - std_boot_mse: `0.000799334051613`
  - q10/q50/q90: `0.015171094732133` / `0.016502486836272` / `0.017227460463283`
  - w_angle: `{"ridge": 0.1429607690453901, "elasticnet": 0.16969549652419075, "knn": 0.05228148845423655, "random_forest": 0.4960782550628308, "extra_trees": 0.13898399091335187}`
  - w_depth: `{"ridge": 0.21299476684552807, "elasticnet": 0.14181804375812468, "knn": 0.020635054853913946, "random_forest": 0.08787122317456811, "extra_trees": 0.5366809113678652}`
  - w_lr: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.6, "extra_trees": 0.0}`
- file: `submission/submission_2385.csv`
  - submission_num: `2385`
  - combo_id: `20`
  - score: `0.016739488453442`
  - mean_boot_mse: `0.016378027144455`
  - std_boot_mse: `0.000903653272469`
  - q10/q50/q90: `0.015236722091911` / `0.016308141505722` / `0.017539373447103`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.2, "knn": 0.0, "random_forest": 0.8, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.29999999999999993, "random_forest": 0.0, "extra_trees": 0.0}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2386.csv`
  - submission_num: `2386`
  - combo_id: `24`
  - score: `0.017143336346490`
  - mean_boot_mse: `0.016760969511163`
  - std_boot_mse: `0.000955917088320`
  - q10/q50/q90: `0.015609745216655` / `0.016794353008922` / `0.017744122401317`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.2, "knn": 0.0, "random_forest": 0.8, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_fb_combo_pilot_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_fb_combo_pilot_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_fb_combo_pilot_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_fb_combo_pilot_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_fb_combo_pilot_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_fb_combo_pilot_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_fb_combo_pilot_20260214.md`
- elapsed_seconds: `32.233847`
