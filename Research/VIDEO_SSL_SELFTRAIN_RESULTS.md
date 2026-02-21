# Video SSL Teacher-Student Self-Training Results

Date: 2026-02-05

## Objective
Test a fundamentally different model that increases effective supervised data:
1. MAE-pretrain encoder on external video plus challenge unlabeled.
2. Train teacher ensemble on challenge labels.
3. Pseudo-label test with uncertainty estimates.
4. Train student on train labels plus weighted pseudo-labels.

Script:
- `scripts/video_ssl_selftrain.py`

## Pilot 1 - Exact command
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selftrain.py --max-external 120 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --teacher-epochs 1 --student-epochs 1 --teacher-seeds 2 --cv-folds 3 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-supervised 1e-3 --seed 42 --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pseudo-weight-min 0.10 --pseudo-weight-max 0.60 --base-submission 771 --blend-weights 0.2 0.4 0.7 1.0 --output-dir output/video_ssl_selftrain_pilot_e1_t1_s1
```

Exact data/model details:
- External samples used: `120`
- External positive rate: `0.500000000000`
- Pretrain corpus: `578`
- Pretrain last loss: `0.992303279032`
- Teacher CV total MSE: `0.042583927140`

Teacher-student CV (3 folds):
- Fold 1: teacher `0.040868405253`, student `0.089219093323`, delta `-0.048350688070`
- Fold 2: teacher `0.014965799637`, student `0.030005943030`, delta `-0.015040143393`
- Fold 3: teacher `0.043800909072`, student `0.042036436498`, delta `0.001764472574`
- Teacher avg: `0.033211704654`
- Student avg: `0.053753824284`
- Student minus teacher: `0.020542119630`

Submissions created:
- Direct student: `submission/submission_960.csv`
- Blend variants: `submission/submission_961.csv` to `submission/submission_964.csv`
- Metrics: `output/video_ssl_selftrain_pilot_e1_t1_s1/run_metrics.json`

Result:
- Student self-training under this config underperformed teacher.

## Pilot 2 - Lower pseudo-label weights
Exact command:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/video_ssl_selftrain.py --max-external 120 --num-frames 24 --frame-size 8 --pretrain-epochs 1 --teacher-epochs 1 --student-epochs 1 --teacher-seeds 2 --cv-folds 3 --batch-size 32 --hidden-dim 64 --dropout 0.1 --lr-pretrain 1e-3 --lr-supervised 1e-3 --seed 42 --pretrain-mask-ratio 0.3 --include-challenge-train-unlabeled --include-challenge-test-unlabeled --pseudo-weight-min 0.00 --pseudo-weight-max 0.20 --base-submission 771 --blend-weights 0.2 0.4 0.7 1.0 --output-dir output/video_ssl_selftrain_pilot_e1_t1_s1_loww
```

Exact results:
- External samples used: `120`
- External positive rate: `0.500000000000`
- Pretrain corpus: `578`
- Pretrain last loss: `0.992303279032`
- Teacher CV total MSE: `0.042583927140`

Teacher-student CV (3 folds):
- Fold 1: teacher `0.040868405253`, student `0.086635679007`, delta `-0.045767273754`
- Fold 2: teacher `0.014965799637`, student `0.027447737753`, delta `-0.012481938116`
- Fold 3: teacher `0.043800909072`, student `0.038793921471`, delta `0.005006987602`
- Teacher avg: `0.033211704654`
- Student avg: `0.050959112744`
- Student minus teacher: `0.017747408090`

Pseudo weight stats:
- min `0.000000000000`
- mean `0.146341204643`
- max `0.200000002980`

Submissions created:
- Direct student: `submission/submission_965.csv`
- Blend variants: `submission/submission_966.csv` to `submission/submission_969.csv`
- Metrics: `output/video_ssl_selftrain_pilot_e1_t1_s1_loww/run_metrics.json`

Result:
- Lower pseudo-label weight reduced damage but student still underperformed teacher.

## Conclusion
- This first teacher-student formulation is not yet ready for full-scale leaderboard use.
- It increases training data quantity, but current pseudo-labeling dynamics degrade within-player CV.
