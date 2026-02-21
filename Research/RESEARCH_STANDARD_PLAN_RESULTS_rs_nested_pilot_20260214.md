# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_pilot_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 1 --seed 20260214 --best-lb 0.006596 --top-k-submissions 5 --run-tag rs_nested_pilot_20260214`
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

- `elasticnet` `angle`: mean_mse=`0.010383577321817`, std_mse=`nan`
- `elasticnet` `depth`: mean_mse=`0.015507195436931`, std_mse=`nan`
- `elasticnet` `left_right`: mean_mse=`0.013024578205897`, std_mse=`nan`
- `extra_trees` `angle`: mean_mse=`0.008083255267703`, std_mse=`nan`
- `extra_trees` `depth`: mean_mse=`0.012862366103254`, std_mse=`nan`
- `extra_trees` `left_right`: mean_mse=`0.014375981605579`, std_mse=`nan`
- `knn` `angle`: mean_mse=`0.010274027387948`, std_mse=`nan`
- `knn` `depth`: mean_mse=`0.014858186222840`, std_mse=`nan`
- `knn` `left_right`: mean_mse=`0.013791124469532`, std_mse=`nan`
- `random_forest` `angle`: mean_mse=`0.009172539391406`, std_mse=`nan`
- `random_forest` `depth`: mean_mse=`0.015792474340414`, std_mse=`nan`
- `random_forest` `left_right`: mean_mse=`0.014645918977809`, std_mse=`nan`
- `ridge` `angle`: mean_mse=`0.011184983078308`, std_mse=`nan`
- `ridge` `depth`: mean_mse=`0.016288203263281`, std_mse=`nan`
- `ridge` `left_right`: mean_mse=`0.014343643276889`, std_mse=`nan`

## Selected Submissions

- file: `submission/submission_2273.csv`
  - submission_num: `2273`
  - combo_id: `52`
  - score: `0.017157662050255`
  - mean_boot_mse: `0.016782928755624`
  - std_boot_mse: `0.000936833236578`
  - q10/q50/q90: `0.015605387021769` / `0.016720046538144` / `0.018015445306455`
  - w_angle: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.29999999999999993, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.09693491365287892, "elasticnet": 0.127302600327352, "knn": 0.25611853241175936, "random_forest": 0.11548126635434125, "extra_trees": 0.4041626872536684}`
  - w_lr: `{"ridge": 0.35720138839888527, "elasticnet": 0.05709324225320837, "knn": 0.3845140195436345, "random_forest": 0.1353045458791773, "extra_trees": 0.0658868039250945}`
- file: `submission/submission_2274.csv`
  - submission_num: `2274`
  - combo_id: `63`
  - score: `0.017558643215551`
  - mean_boot_mse: `0.017146768892724`
  - std_boot_mse: `0.001029685807068`
  - q10/q50/q90: `0.015858377271365` / `0.017120404311579` / `0.018343907264939`
  - w_angle: `{"ridge": 0.7000000000000001, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.29999999999999993, "extra_trees": 0.0}`
  - w_depth: `{"ridge": 0.21299476684552807, "elasticnet": 0.14181804375812468, "knn": 0.020635054853913946, "random_forest": 0.08787122317456811, "extra_trees": 0.5366809113678652}`
  - w_lr: `{"ridge": 0.0767377731605863, "elasticnet": 0.27078884387415175, "knn": 0.11568756689926343, "random_forest": 0.20119496887795715, "extra_trees": 0.3355908471880413}`
- file: `submission/submission_2275.csv`
  - submission_num: `2275`
  - combo_id: `17`
  - score: `0.017587466643590`
  - mean_boot_mse: `0.017195754794181`
  - std_boot_mse: `0.000979279623523`
  - q10/q50/q90: `0.015860259176517` / `0.017204767159108` / `0.018364411850691`
  - w_angle: `{"ridge": 0.11328133589813505, "elasticnet": 0.02184826634446413, "knn": 0.07075248370853925, "random_forest": 0.5129201356389086, "extra_trees": 0.28119777840995297}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2276.csv`
  - submission_num: `2276`
  - combo_id: `25`
  - score: `0.017628965465817`
  - mean_boot_mse: `0.017230491701000`
  - std_boot_mse: `0.000996184412042`
  - q10/q50/q90: `0.016076183751073` / `0.017260810797271` / `0.018381417539248`
  - w_angle: `{"ridge": 0.11328133589813505, "elasticnet": 0.02184826634446413, "knn": 0.07075248370853925, "random_forest": 0.5129201356389086, "extra_trees": 0.28119777840995297}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.1, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.9}`
  - w_lr: `{"ridge": 0.0, "elasticnet": 0.4, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2277.csv`
  - submission_num: `2277`
  - combo_id: `18`
  - score: `0.017712145770316`
  - mean_boot_mse: `0.017299708471410`
  - std_boot_mse: `0.001031093247267`
  - q10/q50/q90: `0.015753591910172` / `0.017525330851563` / `0.018367949003934`
  - w_angle: `{"ridge": 0.11328133589813505, "elasticnet": 0.02184826634446413, "knn": 0.07075248370853925, "random_forest": 0.5129201356389086, "extra_trees": 0.28119777840995297}`
  - w_depth: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.5, "random_forest": 0.0, "extra_trees": 0.5}`
  - w_lr: `{"ridge": 0.39264036403809294, "elasticnet": 0.1621115461719553, "knn": 0.09139397467397595, "random_forest": 0.27762152047422417, "extra_trees": 0.07623259464175156}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_pilot_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_pilot_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_pilot_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_pilot_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_pilot_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_pilot_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_pilot_20260214.md`
- elapsed_seconds: `10.752453`
