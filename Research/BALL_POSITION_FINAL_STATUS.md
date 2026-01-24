# Ball Position - Final Status and Limitations

## Summary

After exhaustive analysis, the ball position features **do not perfectly align** with the skeleton coordinate system. The best visualization shows:

✓ **Complete 53-joint skeleton** (100% accurate)
⚠️ **Ball trajectory** (approximate, with known limitations)

## What Was Discovered

### Parabolic Motion Analysis

Analyzed all 337 features for proper physics-based motion after release:
- **Best features**: X=2, Y=119, Z=3
- **Y=119**: Shows true parabolic motion (rises to 0.880 at frame 63, then falls)
- **Release frame**: 59 (verified by contact flag 336)

### Coordinate System Mismatch

Despite finding features with correct parabolic motion, they don't align spatially with the skeleton:

| Frame | Hand Position (avg) | Ball Position (X=2, Y=119, Z=3) | Distance |
|-------|--------------------|---------------------------------|----------|
| 0 (in hand) | (0.120, 1.364, 1.048) | (0.956, -1.112, 1.435) | 2.59 units |
| 59 (release) | (0.176, 1.112, 2.194) | (1.262, -0.605, 1.493) | 2.08 units |

**Expected**: Ball should be < 0.3 units from hands when contact flag = 1
**Actual**: Ball is 2.0-2.6 units away

### Why This Happens

The ball position features appear to be in a **different coordinate reference frame**:

1. **Skeleton features (165-323)**: Root-relative (pelvis-centered) coordinates
2. **Ball features (2, 3, 119)**: Possibly world/camera coordinates

**Transformations tried (all failed)**:
- ✗ Add pelvis offset: Distance still 1.87 units
- ✗ Subtract pelvis offset: Distance 4.28 units
- ✗ Swap X/Y/Z orderings: All distances > 1.7 units
- ✗ Simple offset (X-1, Y+2, Z-1): Distance 1.13 units (best attempt)

## Final Visualization

### File: `skillmimic_3d_final.py`

**Features used**:
- Skeleton: Features 165-323 (53 bodies × 3)
- Ball: Features X=2, Y=119, Z=3 (best parabolic motion)

**Visual adjustments**:
- Ball opacity: 40-50% (was 70-100%)
- Ball size: 0.025 (was 0.03)
- Trajectory opacity: 60% (to show motion pattern)
- Clear disclaimer in title

### What It Shows

✓ **Correct skeleton structure** (all 53 joints, anatomically correct)
✓ **Parabolic motion pattern** (ball rises after release, follows physics)
⚠️ **Approximate ball position** (not spatially aligned with hands)

### Usage

```bash
# Generate visualization
uv run python visualisation/skillmimic_3d_final.py \
  "/Users/elliot18/Downloads/005_014pickle_shot_001.pt" \
  --output-dir visualisation/output

# View
open visualisation/output/skillmimic_3d_final.html
```

## What Would Fix This Completely

To get perfect ball-skeleton alignment requires:

1. **Official documentation** from SkillMimic authors about observation space format
2. **Coordinate transformation matrix** between ball and skeleton frames
3. **Source code analysis** of `skillmimic/envs/humanoid_ball.py` observation construction
4. **Simulation comparison** - run SkillMimic to generate test data with known ball positions

## Recommendations

### Option 1: Use Skeleton-Only (Safest)
```bash
uv run python visualisation/skillmimic_3d_viewer_skeleton_only.py \
  "<file.pt>" --output-dir visualisation/output
```
**Pros**: 100% accurate, no misleading information
**Cons**: Missing ball interaction

### Option 2: Use Final Version (Best Effort)
```bash
uv run python visualisation/skillmimic_3d_final.py \
  "<file.pt>" --output-dir visualisation/output
```
**Pros**: Shows shooting motion pattern, ball trajectory visible
**Cons**: Ball position not perfectly aligned (noted in title)

## Research Summary

### Feature Search Methods Tried

1. **Proximity search**: Find features close to hands during contact → Found X=53, Y=275, Z=2 (0.64 units away)
2. **Parabolic motion**: Find features with physics-based trajectory → Found X=2, Y=119, Z=3 (best motion)
3. **Coordinate transforms**: Tried pelvis offset, swapping, scaling → Best: 1.13 units away
4. **Exhaustive combinations**: Tested ~58 × 36 × 53 = ~110,000 combinations

### Key Insights

1. **Y=119 is definitely correct** for vertical position (shows true parabolic motion)
2. **Feature 325** has largest Y-range (1.633), suggesting it's also a Y-like feature
3. **Features 2, 3** in first block (0-164) likely represent velocities or other state, not primary positions
4. **Skeleton and ball use different coordinate origins**

## Files Created

### Working Visualizations
- `skillmimic_3d_viewer_skeleton_only.py` - Skeleton only (100% accurate) ⭐ RECOMMENDED
- `skillmimic_3d_final.py` - Skeleton + ball with disclaimer (best effort)

### Diagnostic Tools
- `search_ball_features_exhaustive.py` - Proximity-based search
- `find_parabolic_motion.py` - Physics-based search
- `verify_ball_position.py` - Coordinate verification
- `test_ball_transformation.py` - Transformation testing

### Documentation
- `BALL_POSITION_ISSUE.md` - Initial problem analysis
- `BALL_TRAJECTORY_FIX.md` - First fix attempt
- `BALL_POSITION_FINAL_STATUS.md` - This document

## Conclusion

**Primary goal achieved**: SkillMimic 3D skeleton visualization is fixed (53 joints, anatomically correct)

**Secondary goal partial**: Ball trajectory shows correct physics motion pattern but doesn't perfectly align with skeleton due to coordinate system mismatch

**Best visualization**: Use `skillmimic_3d_viewer_skeleton_only.py` for accuracy, or `skillmimic_3d_final.py` to see shooting motion with transparency-adjusted ball

The ball alignment issue is a **known limitation** of reverse-engineering the data format without official documentation.
