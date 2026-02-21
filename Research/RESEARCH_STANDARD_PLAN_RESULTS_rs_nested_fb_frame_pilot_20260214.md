# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_fb_frame_pilot_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 3 --feature-bank frame_triplet_compact --models all --run-tag rs_nested_fb_frame_pilot_20260214`
- seed: `20260214`
- scale: `1`
- feature_bank: `frame_triplet_compact`
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

- `elasticnet` `angle`: mean_mse=`0.010962770418453`, std_mse=`nan`
- `elasticnet` `depth`: mean_mse=`0.009046833261253`, std_mse=`nan`
- `elasticnet` `left_right`: mean_mse=`0.011725513925959`, std_mse=`nan`
- `extra_trees` `angle`: mean_mse=`0.009064312725848`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.008296814455946`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.010640636674838`, std_mse=`nan`
- `knn` `angle`: mean_mse=`0.009274401567210`, std_mse=`nan`
- `knn` `depth`: mean_mse=`0.010660798385732`, std_mse=`nan`
- `knn` `left_right`: mean_mse=`0.015950805886843`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.008957804506650`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.009742917019488`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.010410119466004`, std_mse=`nan`
- `ridge` `angle`: mean_mse=`0.013657984905594`, std_mse=`nan`
- `ridge` `depth`: mean_mse=`0.008352028176546`, std_mse=`nan`
- `ridge` `left_right`: mean_mse=`0.010900568778343`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2378.csv`
  - submission_num: `2378`
  - combo_id: `41`
  - score: `0.016598571496427`
  - mean_boot_mse: `0.016311439979412`
  - std_boot_mse: `0.000717828792539`
  - q10/q50/q90: `0.015422024903621` / `0.016223385398507` / `0.017174927039143`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.2, "knn": 0.0, "random_forest": 0.8, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.21299476684552807, "elasticnet": 0.14181804375812468, "knn": 0.020635054853913946, "random_forest": 0.08787122317456811, "extra_trees": 0.5366809113678652}`
  - w_lr: `{"ridge": 0.6000000000000001, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.3999999999999999}`
- file: `submission/submission_2379.csv`
  - submission_num: `2379`
  - combo_id: `20`
  - score: `0.016765654444271`
  - mean_boot_mse: `0.016402513190281`
  - std_boot_mse: `0.000907853134977`
  - q10/q50/q90: `0.015245423959389` / `0.016350943404475` / `0.017573333019342`
  - w_angle: `{"ridge": 0.11837204283697933, "elasticnet": 0.2216370220177595, "knn": 0.331781182576826, "random_forest": 0.3190774422433341, "extra_trees": 0.009132310325101145}`
  - w_depth: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.29999999999999993, "random_forest": 0.0, "extra_trees": 0.0}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2380.csv`
  - submission_num: `2380`
  - combo_id: `32`
  - score: `0.017248541163876`
  - mean_boot_mse: `0.016839547234357`
  - std_boot_mse: `0.001022484823795`
  - q10/q50/q90: `0.015578143750831` / `0.016785870325705` / `0.017917547731318`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.2, "knn": 0.0, "random_forest": 0.8, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_fb_frame_pilot_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_fb_frame_pilot_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_fb_frame_pilot_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_fb_frame_pilot_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_fb_frame_pilot_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_fb_frame_pilot_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_fb_frame_pilot_20260214.md`
- elapsed_seconds: `44.264674`
