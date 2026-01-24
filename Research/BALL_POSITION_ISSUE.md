# SkillMimic Ball Position Issue

## Problem

The ball trajectory in the SkillMimic 3D visualization appears incorrect because the ball position features (119, 325, 326) are in a different coordinate system than the skeleton features (165-323).

## Investigation

### Skeleton Coordinates (Verified Correct)
- Features 165-323: 53 bodies × 3 coordinates
- Coordinate system: Root-relative (pelvis-centered)
- Frame 0 L_Wrist position: (0.313, 1.358, 0.991)
- Range: X=[-0.20, 0.56], Y=[0.86, 1.80], Z=[0.02, 2.39]

### Ball Coordinates (Issues Found)
- Features 119 (Y?), 325 (X?), 326 (Z?)
- Based on previous analysis in `visualize_ball_trajectory.py`
- Frame 0 ball raw: (1.272, -1.112, 1.032)
- Range: X=[-1.52, 1.32], Y=[-1.50, 0.88], Z=[0.95, 2.88]

### Problem Details

**At Frame 0** (ball should be in hands, contact flag = 1):
- L_Wrist: (0.313, 1.358, 0.991)
- Ball (features 325, 119, 326): (1.272, -1.112, 1.032)
- **Distance: 2.65 units** (ball is nowhere near hands!)

**Transformation attempts failed:**
1. Adding pelvis offset: Distance still 1.769
2. Swapping X/Y/Z orderings: All distances > 1.7
3. Other simple transforms: No match found

## Root Cause

The ball position features are likely:

1. **Different coordinate frame**: Ball may be in global/world coordinates or camera coordinates, while skeleton is in root-relative (pelvis-centered) coordinates

2. **Complex transformation required**: May need rotation matrix or other transformation not just translation

3. **Wrong feature identification**: Features 119, 325, 326 may not actually be ball XYZ position, or they're encoded differently

## Evidence

From `visualize_ball_trajectory.py` analysis:
- Feature 336 correctly identifies contact (1=in hand, 0=released)
- Features 119, 325, 326 show ball-like motion patterns (parabolic trajectory after release)
- But absolute positions don't match skeleton coordinate system

## Attempted Solutions

1. ✗ Add pelvis position (root offset)
2. ✗ Try all 6 permutations of X/Y/Z ordering
3. ✗ Search all feature triplets for proximity to hands
4. ✗ Check if features are consecutive (they're not: 119, 325, 326)

## Workaround Options

### Option 1: Remove Ball from Visualization
Show only the skeleton (which is correct) and omit ball trajectory entirely.

**Pros**: Clean, accurate skeleton visualization
**Cons**: Misses the interaction aspect

### Option 2: Show Ball with Disclaimer
Display ball using features 119, 325, 326 but add warning that coordinates may not be aligned.

**Pros**: Shows motion pattern even if position is offset
**Cons**: Confusing/misleading visualization

### Option 3: Further Investigation Required
Contact SkillMimic authors or deep-dive into codebase to find correct transformation.

**Required**: Access to SkillMimic source code or documentation
**Timeline**: Unknown

## Recommendation

**For now: Option 1 - Remove ball, show skeleton only**

Reasons:
1. Skeleton is verified correct (100% accurate)
2. Wrong ball position is misleading
3. Can add ball back once transformation is discovered
4. Primary goal was fixing skeleton (achieved)

## Future Work

To properly fix ball position:

1. **Check SkillMimic source code**: Look in `skillmimic/envs/humanoid_ball.py` for observation construction
2. **Find transformation**: Identify coordinate frame conversion between ball and skeleton
3. **Verify against simulation**: Run SkillMimic simulator to check coordinate systems
4. **Alternative features**: Search for other ball position features in remaining 178 features

## Files Affected

- `visualisation/skillmimic_3d_viewer_fixed.py` - Currently shows incorrect ball
- Need to create: `visualisation/skillmimic_3d_viewer_skeleton_only.py` - Skeleton without ball

## Status

- Skeleton: ✓ Fixed (53 joints, anatomically correct)
- Ball trajectory: ✗ Not working (coordinate system mismatch)
- Overall: Partial success (main issue solved, secondary issue remains)
