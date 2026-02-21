# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_fb_frame_models_treeknn_pilot_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 1 --feature-bank frame_triplet_compact --models extra_trees,random_forest,knn --run-tag rs_nested_fb_frame_models_treeknn_pilot_20260214`
- seed: `20260214`
- scale: `1`
- feature_bank: `frame_triplet_compact`
- models_arg: `extra_trees,random_forest,knn`
- best_lb_reference: `0.006596000000000`
- n_outer_repeats: `1`
- outer_holdout_frac: `0.2`
- inner_folds: `2`
- n_bootstrap: `40`
- n_dirichlet_target: `80`
- uncertainty_lambda: `0.4`
- n_models: `3`
- n_train: `345`
- n_test: `113`

## Model Summary

- `extra_trees` `angle`: mean_mse=`0.009124261749883`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.008010133516976`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.010864613654194`, std_mse=`nan`
- `knn` `angle`: mean_mse=`0.009274401567210`, std_mse=`nan`
- `knn` `depth`: mean_mse=`0.010660798385732`, std_mse=`nan`
- `knn` `left_right`: mean_mse=`0.015950805886843`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.007688764294213`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.009531982354804`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.011498930822603`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2388.csv`
  - submission_num: `2388`
  - combo_id: `52`
  - score: `0.016657566362990`
  - mean_boot_mse: `0.016310829259774`
  - std_boot_mse: `0.000866842758040`
  - q10/q50/q90: `0.015106234644078` / `0.016366342658222` / `0.017369747330241`
  - w_angle: `{"knn": 0.0, "random_forest": 0.4, "extra_trees": 0.6}`
  - w_depth: `{"knn": 0.07492071090974958, "random_forest": 0.26093818761524135, "extra_trees": 0.6641411014750089}`
  - w_lr: `{"knn": 0.09551619919481016, "random_forest": 0.2853815724487754, "extra_trees": 0.6191022283564146}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_fb_frame_models_treeknn_pilot_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_fb_frame_models_treeknn_pilot_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_fb_frame_models_treeknn_pilot_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_fb_frame_models_treeknn_pilot_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_fb_frame_models_treeknn_pilot_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_fb_frame_models_treeknn_pilot_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_fb_frame_models_treeknn_pilot_20260214.md`
- elapsed_seconds: `46.583879`
