# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_pilot2_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 5 --run-tag rs_nested_pilot2_20260214`
- seed: `20260214`
- scale: `1`
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

- `elasticnet` `angle`: mean_mse=`0.010497439981887`, std_mse=`nan`
- `elasticnet` `depth`: mean_mse=`0.009218537947632`, std_mse=`nan`
- `elasticnet` `left_right`: mean_mse=`0.016112156243119`, std_mse=`nan`
- `extra_trees` `angle`: mean_mse=`0.008526528820296`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.009349108283588`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.014354977886159`, std_mse=`nan`
- `knn` `angle`: mean_mse=`0.009642335960526`, std_mse=`nan`
- `knn` `depth`: mean_mse=`0.012294182268721`, std_mse=`nan`
- `knn` `left_right`: mean_mse=`0.015498212328201`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.009133706642172`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.011115386160605`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.013673740160828`, std_mse=`nan`
- `ridge` `angle`: mean_mse=`0.012209620698224`, std_mse=`nan`
- `ridge` `depth`: mean_mse=`0.009145085669483`, std_mse=`nan`
- `ridge` `left_right`: mean_mse=`0.022362642416417`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2278.csv`
  - submission_num: `2278`
  - combo_id: `52`
  - score: `0.016921653545781`
  - mean_boot_mse: `0.016565194296274`
  - std_boot_mse: `0.000891148123766`
  - q10/q50/q90: `0.015470044574558` / `0.016642885094894` / `0.017641870587041`
  - w_angle: `{"ridge": 0.11837204283697933, "elasticnet": 0.2216370220177595, "knn": 0.331781182576826, "random_forest": 0.3190774422433341, "extra_trees": 0.009132310325101145}`
  - w_depth: `{"ridge": 0.21299476684552807, "elasticnet": 0.14181804375812468, "knn": 0.020635054853913946, "random_forest": 0.08787122317456811, "extra_trees": 0.5366809113678652}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2279.csv`
  - submission_num: `2279`
  - combo_id: `29`
  - score: `0.017241251616321`
  - mean_boot_mse: `0.016879671410596`
  - std_boot_mse: `0.000903950514311`
  - q10/q50/q90: `0.015678896336456` / `0.016969138218114` / `0.017910845258599`
  - w_angle: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
  - w_depth: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.29999999999999993, "random_forest": 0.0, "extra_trees": 0.0}`
  - w_lr: `{"ridge": 0.0767377731605863, "elasticnet": 0.27078884387415175, "knn": 0.11568756689926343, "random_forest": 0.20119496887795715, "extra_trees": 0.3355908471880413}`
- file: `submission/submission_2280.csv`
  - submission_num: `2280`
  - combo_id: `31`
  - score: `0.017254324204772`
  - mean_boot_mse: `0.016897209467313`
  - std_boot_mse: `0.000892786843647`
  - q10/q50/q90: `0.015813931942446` / `0.016850775291840` / `0.017728154444564`
  - w_angle: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
  - w_depth: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.29999999999999993, "random_forest": 0.0, "extra_trees": 0.0}`
  - w_lr: `{"ridge": 0.008345629264262256, "elasticnet": 0.04714761120997761, "knn": 0.04391793114753858, "random_forest": 0.5470072000680378, "extra_trees": 0.3535816283101838}`
- file: `submission/submission_2281.csv`
  - submission_num: `2281`
  - combo_id: `19`
  - score: `0.017353271021406`
  - mean_boot_mse: `0.017027853228855`
  - std_boot_mse: `0.000813544481377`
  - q10/q50/q90: `0.015977001047322` / `0.017055594001430` / `0.017988854362025`
  - w_angle: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.008345629264262256, "elasticnet": 0.04714761120997761, "knn": 0.04391793114753858, "random_forest": 0.5470072000680378, "extra_trees": 0.3535816283101838}`
- file: `submission/submission_2282.csv`
  - submission_num: `2282`
  - combo_id: `33`
  - score: `0.017506939654412`
  - mean_boot_mse: `0.017172764780737`
  - std_boot_mse: `0.000835437184186`
  - q10/q50/q90: `0.015989823710023` / `0.017320469407804` / `0.018010125338057`
  - w_angle: `{"ridge": 0.2313770722882838, "elasticnet": 0.3573552476123591, "knn": 0.11227432173481672, "random_forest": 0.011869030001973075, "extra_trees": 0.28712432836256724}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.0767377731605863, "elasticnet": 0.27078884387415175, "knn": 0.11568756689926343, "random_forest": 0.20119496887795715, "extra_trees": 0.3355908471880413}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_pilot2_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_pilot2_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_pilot2_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_pilot2_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_pilot2_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_pilot2_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_pilot2_20260214.md`
- elapsed_seconds: `29.131115`
