# gplearn Symbolic Regression & GP Bayesian Model Results (2026-02-20)

## gplearn Symbolic Regression

### What
Used genetic programming (gplearn) to evolve mathematical feature formulas.
Two modes:
1. SymbolicTransformer: evolves features for downstream Ridge model
2. SymbolicRegressor: directly evolves prediction formula

### Features
93 compact features per shot: hip-relative positions for 11 key joints (6 per joint),
velocity for 6 shooting arm joints (3 per joint), acceleration for wrist/elbow,
plus geometric features (elbow angle, wrist angle, shoulder elevation, etc.)

### Results

| Target | Base LOO | +Symbolic LOO | Change |
|--------|----------|--------------|--------|
| Angle | 0.007315 | 0.007226 | -1.22% |
| Depth | 0.007405 | 0.007480 | +1.02% |
| LR | 0.009470 | 0.009176 | -3.11% |
| Mean | 0.008063 | 0.007961 | -1.28% |

### Diversity vs Sub3507
- Angle: r=0.9615
- Depth: r=0.8965
- **LR: r=0.8321** (decent)

### Key Discovered Formulas
- Angle: involves nose position, shoulder position, shoulder-hoop distance relationships
- Depth: SR degenerated to constant (0.513) - couldn't find pattern
- LR: SR degenerated (sin(sin(0.513)))

### Assessment
- Modest LOO improvement (-1.28% mean)
- LR diversity (r=0.83) is useful
- The SR for depth/LR couldn't find good formulas, indicating the signal is more complex
- Submissions: 3546 (standalone), 3547-3549 (3/5/8% blends with Sub3507)

---

## GP Bayesian Model

### What
Gaussian Process Regressor with kernel selection.
Per-player GP models, tested 4 kernels: RBF, Matern32, Matern52, RationalQuadratic.
Each with ConstantKernel + WhiteKernel.

### Results

| Target | LOO MSE | Best Kernel |
|--------|---------|-------------|
| Angle | 0.006229 | Matern32 |
| Depth | 0.007363 | RQ |
| LR | 0.007698 | Matern52 |
| **Mean** | **0.007097** | (varies) |

### Diversity vs Sub3507
- Angle: r=0.9643
- **Depth: r=0.8587** (good)
- **LR: r=0.7975** (very good!)

### Key Insights
1. Different kernels win for different targets - Matern family best overall
2. RBF consistently worst (too smooth for this problem)
3. LR diversity (r=0.80) is the BEST we've seen from any model
4. Standalone quality (0.007097) is better than gplearn (0.007961)
5. GP handles small per-player samples well (Bayesian regularization)

### Assessment
- **PROMISING for blending**, especially LR (r=0.80 diversity)
- Quality is decent: better than CNNs standalone but worse than full Ridge pipeline
- Should be tested on LB at 3-5% blend with Sub3507
- Submissions: 3550 (standalone), 3551-3553 (3/5/8% blends with Sub3507)

---

## Comparison with Previous Models

| Model | Mean LOO | Diversity (depth r) | Diversity (LR r) |
|-------|----------|-------------------|-----------------|
| Full Ridge pipeline | ~0.006830 | - | - |
| Position CNN | 0.00649 | 0.879 | 0.794 |
| Velocity CNN | 0.00766 | 0.879 | - |
| gplearn symbolic | 0.007961 | 0.897 | 0.832 |
| **GP Bayesian** | **0.007097** | **0.859** | **0.798** |

GP Bayesian has the best quality-diversity tradeoff for depth and LR.
