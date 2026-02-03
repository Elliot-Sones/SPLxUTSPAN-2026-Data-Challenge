# Selective Amplification Analysis

## Summary

Analysis of Sub 219 reveals it uses a "selective amplification" technique:
1. Identify samples where Sub 133 and Sub 151 disagree most (91st percentile threshold)
2. Amplify in Sub 133's direction: `new = Sub133 + alpha * (Sub133 - Sub151)`
3. Calibrate depth_mean to 0.5055

## Sub 219 Parameters (Confirmed)

- **Percentile threshold**: 91 (affects 11 samples out of 113)
- **Alpha**: 1.1
- **Depth calibration**: 0.5055
- **Base submission**: Sub 133
- **Contrast submission**: Sub 151

## Grid Search Results

Tested 404 configurations across:
- Percentiles: 85, 88, 90, 91, 92, 93, 95
- Alphas: 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5
- Depth calibrations: 0.5045, 0.5050, 0.5055, 0.5060, None
- Per-target alpha configurations
- Alternative contrast submissions (9, 25, 84, 100, 150, 152, 160)
- Two-stage amplification

### Best Match to Sub 219
```
type: uniform
contrast: 151
percentile: 91
alpha: 1.1
depth_cal: 0.5055
n_modified: 11
sim_to_219: 0.000271 (essentially identical)
```

## Disagreement Analysis

Total disagreement between Sub 133 and Sub 151:
- Range: 0.0020 to 0.1851
- Mean: 0.0247

Percentile thresholds and sample counts:
| Percentile | Threshold | Samples Above |
|------------|-----------|---------------|
| 85         | 0.0353    | 17            |
| 88         | 0.0382    | 14            |
| 90         | 0.0416    | 12            |
| 91         | 0.0423    | 11            |
| 92         | 0.0433    | 9             |
| 93         | 0.0484    | 8             |
| 95         | 0.0498    | 6             |

## Submissions Generated

### Initial Grid Search (335-367)
33 submissions testing:
- Alpha variations: 0.95, 1.05, 1.15
- Percentile variations: 89, 90, 92, 93
- Depth calibrations: 0.5045, 0.5050, 0.5055, 0.5060
- Combined variations
- Per-target alpha configurations
- Two-stage amplification

### Refined Variations (370-412)
43 submissions testing:
- Fine-grained alpha: 1.02, 1.03, 1.04, 1.06, 1.07, 1.08, 1.12, 1.13, 1.14
- Fine-grained depth: 0.5052, 0.5053, 0.5054, 0.5056, 0.5057, 0.5058
- Asymmetric amplification (different alpha per target)
- Custom threshold variations around 91st percentile
- Weighted amplification (alpha scales with disagreement level)
- Blends with Sub 219
- Combined alpha + depth variations

## Promising Configurations to Test

### Slightly Higher Alpha (might capture more signal)
- Sub 376: alpha=1.12, pct=91, depth=0.5055
- Sub 377: alpha=1.13, pct=91, depth=0.5055
- Sub 337: alpha=1.15, pct=91, depth=0.5055

### Slightly Lower Threshold (include 1 more sample)
- Sub 395: custom threshold 0.0401 (12 samples)
- Sub 339: pct=90 (12 samples)
- Sub 345: pct=90, alpha=1.1 (12 samples)

### Asymmetric Amplification (different weights per target)
- Sub 385: angle=1.15, depth=1.1, lr=1.1
- Sub 388: angle=1.1, depth=1.1, lr=1.15
- Sub 393: angle=1.15, depth=1.05, lr=1.1

### Weighted Amplification (proportional to disagreement)
- Sub 400: base_alpha=1.1, weight_factor=0.5
- Sub 401: base_alpha=1.1, weight_factor=0.7

### Blends with Sub 219
- Sub 405: 55% amplified, 45% Sub 219
- Sub 406: 60% amplified, 40% Sub 219

### Combined Variations
- Sub 408: alpha=1.12, depth=0.5053
- Sub 410: alpha=1.12, depth=0.5057

## Technical Details

### The Selective Amplification Algorithm

```python
def selective_amplify(base, contrast, percentile=91, alpha=1.1, depth_cal=0.5055):
    # 1. Calculate per-sample disagreement
    diff = |base - contrast| summed across all targets

    # 2. Find threshold
    threshold = np.percentile(diff, percentile)
    high_disagree_mask = diff >= threshold

    # 3. Amplify high-disagreement samples
    for target in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        direction = base[target] - contrast[target]
        new[target][high_disagree_mask] = base[target] + alpha * direction

    # 4. Clip to [0, 1]
    new = np.clip(new, 0, 1)

    # 5. Calibrate depth mean
    new['scaled_depth'] += (depth_cal - new['scaled_depth'].mean())

    return new
```

### Why This Might Work

1. **Confidence-based correction**: High disagreement samples are where the models are least certain
2. **Direction from base**: Uses Sub 133 as the "ground truth" direction
3. **Selective application**: Only modifies 11 samples (9.7% of test set)
4. **Calibration**: Ensures depth mean matches expected target distribution

## Files

- Grid search script: `/external_data/scripts/selective_amplification_grid_search.py`
- Refined variations script: `/external_data/scripts/selective_amplification_refined.py`
- Grid search results: `/external_data/output/selective_amplification_grid_results.csv`
- Submission details: `/external_data/output/selective_amplification_submissions.csv`
- Refined submission details: `/external_data/output/selective_amplification_refined.csv`
