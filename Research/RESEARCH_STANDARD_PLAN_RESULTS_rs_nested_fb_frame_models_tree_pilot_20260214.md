# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_fb_frame_models_tree_pilot_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 1 --feature-bank frame_triplet_compact --models extra_trees,random_forest --run-tag rs_nested_fb_frame_models_tree_pilot_20260214`
- seed: `20260214`
- scale: `1`
- feature_bank: `frame_triplet_compact`
- models_arg: `extra_trees,random_forest`
- best_lb_reference: `0.006596000000000`
- n_outer_repeats: `1`
- outer_holdout_frac: `0.2`
- inner_folds: `2`
- n_bootstrap: `40`
- n_dirichlet_target: `80`
- uncertainty_lambda: `0.4`
- n_models: `2`
- n_train: `345`
- n_test: `113`

## Model Summary

- `extra_trees` `angle`: mean_mse=`0.008270254835723`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.008622959463788`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.010936962384144`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.008126802814665`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.009425859410595`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.011294843198997`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2387.csv`
  - submission_num: `2387`
  - combo_id: `52`
  - score: `0.016658454285696`
  - mean_boot_mse: `0.016313996912051`
  - std_boot_mse: `0.000861143434113`
  - q10/q50/q90: `0.015137423298077` / `0.016390629267721` / `0.017374086185777`
  - w_angle: `{"random_forest": 0.11729012092532398, "extra_trees": 0.8827098790746759}`
  - w_depth: `{"random_forest": 0.4220767963769785, "extra_trees": 0.5779232036230216}`
  - w_lr: `{"random_forest": 0.525639590254244, "extra_trees": 0.4743604097457561}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_fb_frame_models_tree_pilot_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_fb_frame_models_tree_pilot_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_fb_frame_models_tree_pilot_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_fb_frame_models_tree_pilot_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_fb_frame_models_tree_pilot_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_fb_frame_models_tree_pilot_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_fb_frame_models_tree_pilot_20260214.md`
- elapsed_seconds: `33.677331`
