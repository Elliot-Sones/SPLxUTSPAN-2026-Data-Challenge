# Research-Standard Nested OOF Ensemble Run

- run_tag: `rs_nested_fb_frame_full2_heavy_20260214`
- command: `scripts/research_standard_nested_oof_ensemble.py --scale 2 --seed 20260214 --best-lb 0.006596 --top-k-submissions 5 --feature-bank frame_triplet_compact --models all --run-tag rs_nested_fb_frame_full2_heavy_20260214`
- seed: `20260214`
- scale: `2`
- feature_bank: `frame_triplet_compact`
- models_arg: `all`
- best_lb_reference: `0.006596000000000`
- n_outer_repeats: `6`
- outer_holdout_frac: `0.2`
- inner_folds: `3`
- n_bootstrap: `120`
- n_dirichlet_target: `240`
- uncertainty_lambda: `0.4`
- n_models: `5`
- n_train: `345`
- n_test: `113`

## Model Summary

- `elasticnet` `angle`: mean_mse=`0.010658968032786`, std_mse=`0.002374692294525`
- `elasticnet` `depth`: mean_mse=`0.008732490389831`, std_mse=`0.001562187957478`
- `elasticnet` `left_right`: mean_mse=`0.010444864554744`, std_mse=`0.001421994115281`
- `extra_trees` `angle`: mean_mse=`0.007975437859906`, std_mse=`0.001942443315764`
- `extra_trees` `depth`: mean_mse=`0.008516625830152`, std_mse=`0.001329252105819`
- `extra_trees` `left_right`: mean_mse=`0.009963526025564`, std_mse=`0.001525054888837`
- `knn` `angle`: mean_mse=`0.009647137104521`, std_mse=`0.002804672266718`
- `knn` `depth`: mean_mse=`0.010178608953517`, std_mse=`0.001395063487140`
- `knn` `left_right`: mean_mse=`0.013414901481823`, std_mse=`0.002568775466585`
- `random_forest` `angle`: mean_mse=`0.008392563481005`, std_mse=`0.002009767614085`
- `random_forest` `depth`: mean_mse=`0.009144256422896`, std_mse=`0.000982591375811`
- `random_forest` `left_right`: mean_mse=`0.011167387789929`, std_mse=`0.001698385950594`
- `ridge` `angle`: mean_mse=`0.013523659527827`, std_mse=`0.002003920205495`
- `ridge` `depth`: mean_mse=`0.009959464142934`, std_mse=`0.003106336008248`
- `ridge` `left_right`: mean_mse=`0.010361898270872`, std_mse=`0.001673561167130`

## Selected Submissions

- file: `submission/submission_2405.csv`
  - submission_num: `2405`
  - combo_id: `15`
  - score: `0.009856190415267`
  - mean_boot_mse: `0.009611584435287`
  - std_boot_mse: `0.000611514949950`
  - q10/q50/q90: `0.008855379239979` / `0.009535463559693` / `0.010371517434056`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.5, "extra_trees": 0.5}`
  - w_depth: `{"ridge": 0.3193803691035493, "elasticnet": 0.1237733602233562, "knn": 0.10722145866165701, "random_forest": 0.11418068403946308, "extra_trees": 0.3354441279719745}`
  - w_lr: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`
- file: `submission/submission_2406.csv`
  - submission_num: `2406`
  - combo_id: `18`
  - score: `0.009976859700478`
  - mean_boot_mse: `0.009719342366307`
  - std_boot_mse: `0.000643793335428`
  - q10/q50/q90: `0.008934359110166` / `0.009653321530630` / `0.010494042626593`
  - w_angle: `{"ridge": 0.0, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.2, "extra_trees": 0.8}`
  - w_depth: `{"ridge": 0.29526988030305207, "elasticnet": 0.3918790468480418, "knn": 0.13381923239959148, "random_forest": 0.07581630002557199, "extra_trees": 0.10321554042374262}`
  - w_lr: `{"ridge": 0.6000000000000001, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.3999999999999999}`
- file: `submission/submission_2407.csv`
  - submission_num: `2407`
  - combo_id: `42`
  - score: `0.010112433201605`
  - mean_boot_mse: `0.009875047088782`
  - std_boot_mse: `0.000593465282057`
  - q10/q50/q90: `0.009103825705861` / `0.009877928372265` / `0.010540037439674`
  - w_angle: `{"ridge": 0.10584370223261319, "elasticnet": 0.034681132554365786, "knn": 0.0660517585295941, "random_forest": 0.45017919836239095, "extra_trees": 0.3432442083210358}`
  - w_depth: `{"ridge": 0.09472118320996491, "elasticnet": 0.5663439992144036, "knn": 0.07044156542985472, "random_forest": 0.08358647180387763, "extra_trees": 0.18490678034189906}`
  - w_lr: `{"ridge": 0.6000000000000001, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.3999999999999999}`
- file: `submission/submission_2408.csv`
  - submission_num: `2408`
  - combo_id: `61`
  - score: `0.010115672710090`
  - mean_boot_mse: `0.009852499158819`
  - std_boot_mse: `0.000657933878177`
  - q10/q50/q90: `0.009025456648275` / `0.009822716211737` / `0.010696088874197`
  - w_angle: `{"ridge": 0.08802457630021533, "elasticnet": 0.029489134420096933, "knn": 0.07926140409797408, "random_forest": 0.06065877803321453, "extra_trees": 0.742566107148499}`
  - w_depth: `{"ridge": 0.3193803691035493, "elasticnet": 0.1237733602233562, "knn": 0.10722145866165701, "random_forest": 0.11418068403946308, "extra_trees": 0.3354441279719745}`
  - w_lr: `{"ridge": 0.30000000000000004, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.7}`
- file: `submission/submission_2409.csv`
  - submission_num: `2409`
  - combo_id: `39`
  - score: `0.010134744363772`
  - mean_boot_mse: `0.009882432073542`
  - std_boot_mse: `0.000630780725576`
  - q10/q50/q90: `0.009135440000737` / `0.009898759652610` / `0.010635058595624`
  - w_angle: `{"ridge": 0.10584370223261319, "elasticnet": 0.034681132554365786, "knn": 0.0660517585295941, "random_forest": 0.45017919836239095, "extra_trees": 0.3432442083210358}`
  - w_depth: `{"ridge": 0.2879744021667605, "elasticnet": 0.44373090788700703, "knn": 0.12937237362325107, "random_forest": 0.11418464400398187, "extra_trees": 0.024737672318999586}`
  - w_lr: `{"ridge": 0.4, "elasticnet": 0.0, "knn": 0.0, "random_forest": 0.0, "extra_trees": 0.6}`

## Artifacts

- outer_metrics_csv: `output/research_standard_outer_metrics_rs_nested_fb_frame_full2_heavy_20260214.csv`
- tuning_csv: `output/research_standard_tuning_rs_nested_fb_frame_full2_heavy_20260214.csv`
- target_weight_csv: `output/research_standard_target_weight_search_rs_nested_fb_frame_full2_heavy_20260214.csv`
- ensemble_candidates_csv: `output/research_standard_ensemble_candidates_rs_nested_fb_frame_full2_heavy_20260214.csv`
- selected_csv: `output/research_standard_selected_rs_nested_fb_frame_full2_heavy_20260214.csv`
- run_json: `output/research_standard_run_rs_nested_fb_frame_full2_heavy_20260214.json`
- details_md: `output/research_standard_submission_details_rs_nested_fb_frame_full2_heavy_20260214.md`
- elapsed_seconds: `3235.178678`
