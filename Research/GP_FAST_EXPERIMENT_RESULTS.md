# Fast Gaussian Process Regression Experiments

Date: 2026-02-08

## Setup
- Train: 345 shots, Test: 113 shots
- Features: 213 per target (198 HC + 15 PLS)
- Validation: 5-fold CV (faster than LOO)
- Baseline: Sub 784 (LB 0.007224)
- Current best: Sub 1350 (LB 0.006776)

## Results

| Sub | Kernel | LS | Alpha | PLS | Blend | Mean CV MSE | Angle r | Depth r | LR r | Mean r |
|-----|--------|----| ------|-----|-------|-------------|---------|---------|------|--------|
| 1497 | rbf | 1.0 | 0.1 | True | 0.5 | 0.002897 | 0.977 | 0.946 | 0.879 | 0.934 |
| 1498 | matern_1.5 | 1.0 | 0.1 | True | 0.5 | 0.002907 | 0.978 | 0.959 | 0.900 | 0.946 |
| 1499 | matern_2.5 | 1.0 | 0.1 | True | 0.5 | 0.002836 | 0.977 | 0.957 | 0.889 | 0.941 |
| 1500 | rational_quadratic | 1.0 | 1.0 | True | 0.5 | 0.002888 | 0.977 | 0.925 | 0.876 | 0.926 |
| 1501 | matern_2.5 | 0.5 | 0.1 | True | 0.5 | 0.002836 | 0.977 | 0.957 | 0.889 | 0.941 |
| 1502 | matern_2.5 | 2.0 | 0.1 | True | 0.5 | 0.002836 | 0.977 | 0.957 | 0.889 | 0.941 |
| 1503 | rbf | 1.0 | 0.1 | False | 0.5 | 0.008439 | 0.992 | 0.958 | 0.866 | 0.939 |
| 1504 | matern_2.5 | 1.0 | 0.1 | False | 0.5 | 0.008401 | 0.992 | 0.961 | 0.876 | 0.943 |
| 1505 | matern_2.5 | 1.0 | 0.1 | True | 0.3 | 0.002836 | 0.992 | 0.983 | 0.923 | 0.966 |
| 1506 | matern_2.5 | 1.0 | 0.1 | True | 0.7 | 0.002836 | 0.957 | 0.917 | 0.848 | 0.907 |

## Key Findings

**Best CV:** Sub 1499 (matern_2.5) - MSE 0.002836

**Most diverse:** Sub 1506 (matern_2.5) - r=0.907

**High diversity (r<0.90):** 0 submissions

## Recommendations

Test high diversity submissions on LB. GP models provide different inductive biases vs tree ensembles.
