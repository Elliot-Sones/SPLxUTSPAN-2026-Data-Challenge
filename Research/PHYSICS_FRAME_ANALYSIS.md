# Physics Frame Analysis - Optimal Feature Extraction Timing

## Date: 2026-02-05

## Core Discovery

The physics-detected release frame (~frame 120) is NOT the best frame for feature extraction. The follow-through phase (frames 140-180) is MORE informative. Different targets are best predicted from different follow-through phases.

## Frame Analysis Results

CV MSE at different fixed extraction frames (hoop-relative features, Ridge+LGB+XGB ensemble):

### Angle
| Frame | MSE |
|-------|-----|
| 100 | 0.008742 |
| 110 | 0.008648 |
| 120 | 0.008351 |
| 130 | 0.008238 |
| 140 | 0.007095 |
| 150 | 0.006998 |
| **153** | **0.006546** |
| 160 | 0.006970 |
| 170 | 0.007467 |
| 180 | 0.007684 |

Optimal: frame 153. This is the mid follow-through.

### Depth
| Frame | MSE |
|-------|-----|
| 100 | 0.012368 |
| 110 | 0.012032 |
| 120 | 0.012153 |
| 130 | 0.010406 |
| 140 | 0.008328 |
| **150** | **0.007827** |
| 153 | 0.008037 |
| 160 | 0.008771 |
| 170 | 0.010120 |
| 180 | 0.009748 |

Optimal: frame 150. Adding release_frame as feature improves to 0.007548.

### Left-Right
| Frame | MSE |
|-------|-----|
| 100 | 0.015451 |
| 110 | 0.015097 |
| 120 | 0.013941 |
| 130 | 0.012709 |
| 140 | 0.013058 |
| 150 | 0.010705 |
| 153 | 0.010866 |
| 160 | 0.009289 |
| **170** | **0.009068** |
| 180 | 0.009728 |

Optimal: frame 170. This is 16.5% better than frame 153!

### Mean MSE across all 3 targets
| Config | Mean MSE |
|--------|----------|
| Frame 120 | 0.011482 |
| Frame 140 | 0.009493 |
| Frame 153 | 0.008483 |
| **Frame 160** | **0.008343** |
| Per-shot physics | 0.012794 |
| F153 + release_frame | 0.008375 |

## Release Frame as a Feature

Per-player correlation of physics-detected release frame with depth (scaled):
| Player | r |
|--------|---|
| 1 | -0.20 |
| 2 | +0.75 |
| 3 | +0.59 |
| 4 | +0.71 |
| 5 | +0.45 |

Adding release_frame as a feature to frame-153 extraction: depth MSE 0.008037 -> 0.007548 (6% improvement).

## Physics-Informed Feature Extraction (523 features, multi-frame)

| Config | Angle | Depth | LR | Mean |
|--------|-------|-------|------|------|
| Physics-informed HR (no PLS) | 0.007323 | 0.008516 | 0.011497 | 0.009112 |
| Physics-informed HR + PLS | 0.007306 | 0.008588 | 0.009154 | 0.008349 |

Multi-frame extraction (523 features) HURTS compared to single-frame. Too many features dilute signal on 345 samples.

## Physics-Optimal Frame Pipeline (per-target frame + release_frame + PLS)

| Config | Angle | Depth | LR | Mean |
|--------|-------|-------|------|------|
| Per-target optimal (no PLS) | 0.007164 | 0.007695 | 0.009104 | - |
| Per-target optimal + PLS | 0.007292 | 0.007891 | 0.008724 | 0.007969 |
| Combined multi-frame + PLS | 0.007356 | 0.008018 | 0.008753 | - |
| Frame 153 + RF + PLS (baseline) | 0.007292 | 0.008400 | 0.009021 | - |

Best config: per-target optimal frame + PLS gives mean MSE 0.007969 (vs 0.0081 baseline HR+PLS).

## Key Insights

1. **Follow-through > Release**: The body's follow-through movement (frames 140-180) is more informative about shot outcome than the actual release instant (~frame 120). This is consistent with sports science - the follow-through reflects how the body guided the ball.

2. **Different targets peak at different times**: Angle peaks at frame 153 (arm extension), depth at frame 150 (distance control), left_right at frame 170 (lateral follow-through correction).

3. **Release frame timing encodes depth**: When a player releases earlier (lower frame), the ball has more/less time to travel, affecting depth. This is a genuine physics signal: release timing affects ball flight time which determines depth at the hoop plane.

4. **Per-shot frame detection is too noisy**: The release frame has std=18 frames. Using it to select the extraction frame amplifies noise. Using it as a SCALAR FEATURE works much better.

5. **Multi-frame features overfit**: Extracting at 6+ frames produces 500+ features on 345 samples. The extra features don't add information - they just add noise dimensions for the model to overfit.

## Submissions Generated

### From physics_informed_features.py (multi-frame, poor)
| Sub | Config | Notes |
|-----|--------|-------|
| 1000-1005 | Physics-informed HR + PLS, blended with Sub 784 | w=0.05 to 0.50 |

### From physics_optimal_frame.py (per-target optimal, good)
| Sub | Config | Notes |
|-----|--------|-------|
| 1011-1016 | Per-target optimal + PLS, blended with Sub 784 | w=0.05 to 0.50 |
| 1017-1021 | Per-target optimal + PLS, target-specific blend with Sub 784 | Various weights |

### From physics_enhanced_blend.py (full pipeline, recommended)
| Sub | Config | Notes |
|-----|--------|-------|
| 1022-1026 | Physics-enhanced models, blended with Sub 771 | Same structure as Sub 784 creation |
| 1027-1031 | Physics-enhanced models, blended with Sub 784 | Various blend weights |
| 1032 | Standalone physics-enhanced pipeline | No blending |

**Recommended for LB testing:**
- **Sub 1026**: Same blend weights as Sub 784 (aw=0.00, dw=0.30, lw=0.50 with Sub 771) but physics-enhanced models. Direct comparison with Sub 784.
- **Sub 1029**: Blended with Sub 784 (aw=0.00, dw=0.20, lw=0.30). Adds physics diversity to Sub 784.

## Reproduction

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge

# Frame analysis (analysis only, no submissions)
uv run python /path/to/scratchpad/frame_analysis.py

# Physics-informed multi-frame (Sub 1000-1005)
uv run python scripts/physics_informed_features.py

# Per-target optimal frame (Sub 1011-1021)
uv run python scripts/physics_optimal_frame.py

# Full physics-enhanced blend (Sub 1022-1032)
uv run python scripts/physics_enhanced_blend.py
```
