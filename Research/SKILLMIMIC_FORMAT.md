# SkillMimic Dataset Format Documentation

## Overview

SkillMimic is a motion capture dataset for basketball shooting motions. Data is stored in PyTorch `.pt` pickle files containing temporal sequences of humanoid motion features.

**Paper**: "SkillMimic: Learning Reusable Basketball Skills from Broadcasts" (CVPR 2025)
**Project Page**: https://ingrid789.github.io/SkillMimic/
**GitHub**: https://github.com/wyhuai/SkillMimic
**arXiv**: https://arxiv.org/abs/2408.15270

## File Structure

### Format
- **File Type**: PyTorch pickle files (`.pt`)
- **Content**: `torch.Tensor` (single tensor per file)
- **Typical Shape**: `(num_frames, 337)`
  - **num_frames**: Temporal sequence length (e.g., 104 frames for a shot)
  - **337**: Total feature count per time step

### Example File
```
005_014pickle_shot_001.pt
  Shape: (104, 337)
  Data type: torch.float32
  Memory: ~137 KB
  Min: -1.68, Max: 2.88, Mean: 0.46, Std: 0.68
```

## Feature Composition (337 features)

Based on the SkillMimic paper, the 337 features represent:

1. **Humanoid Proprioception** (~156 features)
   - Joint positions (3D coordinates)
   - Joint rotations (quaternions or euler angles)
   - Joint velocities (linear)
   - Joint angular velocities
   - Estimated ~52 joints based on humanoid model

2. **Fingertip Forces** (30 features)
   - 10 fingers × 3 dimensions (x, y, z)
   - Contact force measurements

3. **Ball Observation** (18 features)
   - Ball position: 3 features (x, y, z)
   - Ball rotation: 9 features (3×3 rotation matrix flattened)
   - Ball linear velocity: 3 features
   - Ball angular velocity: 3 features

4. **Additional Features** (~133 features)
   - Derived kinematic features
   - Normalized and transformed representations
   - Possible dimensionality reduction from theoretical 673 features

### Reference Frame
- **Root-local coordinates**: Features are likely expressed relative to the humanoid root joint
- **Normalization**: Data is zero-centered and normalized (typical range: -2 to +3)

## Loading and Analysis

### Loading a Single File

```python
import torch

# Load the file
data = torch.load("005_014pickle_shot_001.pt", map_location='cpu')

# Inspect structure
print(f"Shape: {data.shape}")  # (104, 337)
print(f"Type: {data.dtype}")   # torch.float32
print(f"Device: {data.device}") # cpu

# Access specific frame
frame_0 = data[0]  # First frame (337 features)

# Access specific feature across time
feature_0_timeseries = data[:, 0]  # Feature 0 for all 104 frames
```

### Using the Inspector Tool

For comprehensive analysis of a single file:

```bash
cd visualisation
uv run python skillmimic_inspector.py <path_to_file.pt> --output-dir analysis_output
```

**Outputs**:
- Console summary with statistics
- CSV export (frames × features table)
- NumPy export (.npy format)
- JSON metadata (structure and statistics)
- HTML visualizations:
  - Time series plots (first 20 features)
  - Distribution histograms
  - Correlation heatmap
- Feature type inference (positions, velocities, accelerations)

### Batch Processing Multiple Files

```bash
cd visualisation
uv run python skillmimic_batch_processor.py <directory> --pattern "*.pt" --output batch_results
```

**Outputs**:
- `summary.csv`: Per-file statistics table
- `per_feature_cross_file_stats.csv`: Feature statistics across all files
- `report.html`: Comprehensive HTML report
- `comparative_plots/`: Visualizations comparing files
  - Feature comparison plots
  - Duration comparison
  - Feature variance analysis

## Typical Statistics

From sample file `005_014pickle_shot_001.pt`:

| Metric | Value |
|--------|-------|
| Num frames | 104 |
| Num features | 337 |
| Global min | -1.68 |
| Global max | 2.88 |
| Global mean | 0.46 |
| Global std | 0.68 |
| Frame-to-frame L2 change (mean) | 0.36 |
| Frame-to-frame L2 change (max) | 1.07 |

### Feature Type Distribution
- Likely positions: ~89 features (smooth, continuous)
- Likely velocities: ~0 features (in this sample)
- Likely accelerations: ~3 features
- Other features: ~245 features (forces, rotations, derived)

## Common Operations

### Extract Temporal Subsequence
```python
# Get frames 10-30
subsequence = data[10:30]  # Shape: (20, 337)
```

### Calculate Velocities
```python
# Calculate frame-to-frame differences
velocities = torch.diff(data, dim=0)  # Shape: (103, 337)
```

### Filter Specific Features
```python
# Extract first 50 features
subset = data[:, :50]  # Shape: (104, 50)

# Extract specific feature indices
feature_indices = [0, 5, 10, 15]
selected_features = data[:, feature_indices]  # Shape: (104, 4)
```

### Export to NumPy
```python
import numpy as np

# Convert to NumPy
numpy_data = data.numpy()  # Shape: (104, 337), dtype: float32

# Save as NumPy file
np.save("output.npy", numpy_data)
```

### Export to CSV
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame(
    data.numpy(),
    columns=[f"feature_{i:03d}" for i in range(data.shape[1])]
)
df.insert(0, 'frame', range(len(df)))

# Save to CSV
df.to_csv("output.csv", index=False)
```

## Comparison with SPL Dataset

| Aspect | SPL Dataset | SkillMimic |
|--------|-------------|------------|
| **Format** | CSV (tabular) | PyTorch tensor (.pt) |
| **Keypoints** | 70 joints (detailed hands) | ~52 humanoid joints |
| **Raw features** | 210 (3D positions only) | ~156 positions |
| **Total features** | 210 | 337 (6.6× expansion) |
| **Reference frame** | Global coordinates | Root-local frame |
| **Derived features** | None | Velocities, forces, rotations |
| **Normalization** | Raw feet units | Zero-centered normalized |
| **Frame rate** | 60 FPS | Unknown (likely 30-60 FPS) |
| **Temporal format** | Variable-length JSON arrays | Fixed tensor shape |

### Key Differences
1. **Feature richness**: SkillMimic includes velocities, forces, and rotations; SPL only has positions
2. **Normalization**: SkillMimic is normalized; SPL uses raw world coordinates
3. **Format**: SkillMimic uses PyTorch tensors; SPL uses CSV with JSON-encoded arrays
4. **Reference frame**: SkillMimic uses root-local; SPL uses global coordinates

## Interpretation Guidelines

### Identifying Feature Types

**Position-like features** (smooth, continuous):
- Low frame-to-frame second derivative
- Larger value range (>0.5)
- Smooth temporal evolution

**Velocity-like features** (first derivatives):
- Higher frame-to-frame change rate
- Zero-centered distributions
- Temporal spikes during motion changes

**Force-like features** (contact-dependent):
- Sparse activation (many zeros)
- Sudden spikes at contact events
- Concentrated in specific time windows

### Temporal Analysis

The temporal sequence represents the full shooting motion:
1. **Preparation phase**: Initial frames (ball hold, stance setup)
2. **Execution phase**: Middle frames (shooting motion)
3. **Release phase**: Ball release and follow-through
4. **Recovery phase**: Final frames (return to neutral pose)

Frame-to-frame L2 norm changes can identify motion phase transitions:
- Low change: Static poses
- High change: Dynamic motion (jump, release)

## Tools and Scripts

### Available Tools

1. **`skillmimic_inspector.py`**: Single file analysis
   - Comprehensive statistics
   - Multiple export formats
   - Visualizations
   - Feature type inference

2. **`skillmimic_batch_processor.py`**: Multi-file analysis
   - Batch processing
   - Consistency validation
   - Cross-file statistics
   - Comparative visualizations

### Dependencies

Required packages (install via `uv`):
- `torch`: Loading .pt files
- `numpy`: Numerical operations
- `pandas`: Tabular data export
- `plotly`: Interactive visualizations

## References

1. **SkillMimic Paper**: https://arxiv.org/abs/2408.15270
2. **Project Page**: https://ingrid789.github.io/SkillMimic/
3. **GitHub Repository**: https://github.com/wyhuai/SkillMimic
4. **CVPR 2025**: Conference proceedings (when published)

## Notes

- Feature indices and exact composition may vary between dataset versions
- The 337-feature count suggests dimensionality reduction from theoretical 673 features
- Root-local coordinates enable better generalization across different body sizes
- Normalized data facilitates neural network training and comparison across motions

## Example Workflow

```python
import torch
import pandas as pd
import numpy as np

# 1. Load file
data = torch.load("005_014pickle_shot_001.pt", map_location='cpu')

# 2. Inspect
print(f"Motion duration: {data.shape[0]} frames")
print(f"Features per frame: {data.shape[1]}")

# 3. Analyze temporal evolution
frame_changes = torch.norm(torch.diff(data, dim=0), dim=1)
peak_motion_frame = frame_changes.argmax().item()
print(f"Peak motion at frame: {peak_motion_frame}")

# 4. Export subset
interesting_features = data[:, :50]  # First 50 features
np.save("subset.npy", interesting_features.numpy())

# 5. Statistical analysis
per_feature_stats = {
    'mean': data.mean(dim=0).numpy(),
    'std': data.std(dim=0).numpy(),
    'min': data.min(dim=0).values.numpy(),
    'max': data.max(dim=0).values.numpy(),
}
stats_df = pd.DataFrame(per_feature_stats)
stats_df.to_csv("feature_statistics.csv")
```
