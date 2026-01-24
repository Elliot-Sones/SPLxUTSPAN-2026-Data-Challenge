# SkillMimic Observation Space - Complete Decode

## Source Code Analysis

From `skillmimic/env/tasks/humanoid_object_task.py` and `humanoid_task.py`:

### Observation Structure

```
[Humanoid Obs (~808 features) | Object Obs (15 features) | Task Obs (optional)]
```

### Humanoid Observation (~808 features)
1. Root height: 1 feature
2. Local body positions: (52 bodies - 1 root) × 3 = 153 features
3. Local body rotations: 52 × 6 = 312 features (tan-norm)
4. Local body velocities: 52 × 3 = 156 features
5. Local body angular velocities: 52 × 3 = 156 features
6. Contact forces: 10 bodies × 3 = 30 features

**Total: 808 features** (but our data has 337, so it's a reduced version)

### Object (Ball) Observation (15 features)

Located AFTER humanoid features, the 15-dimensional ball observation contains:
- **Indices 0-2**: Local ball position (X, Y, Z) relative to humanoid
- **Indices 3-8**: Ball rotation (tangent-norm, 6 values)
- **Indices 9-11**: Local ball linear velocity
- **Indices 12-14**: Local ball angular velocity

## Critical Discovery: Heading-Relative Frame

**All coordinates use a heading-relative frame**, not the root-relative frame!

### Transformation Applied (from source):

```python
# 1. Ball position relative to root
local_tar_pos = tar_pos - root_pos

# 2. Rotate into heading-aligned frame
heading_rot = calc_heading_quat_inv(root_rot)
local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
```

**This means**:
- Ball coordinates are rotated so the humanoid is always "facing forward"
- Skeleton coordinates (165-323) are likely also in heading-relative frame
- Both use the same frame, but the rotation is baked in!

## Why Ball Didn't Align

We were trying to align heading-rotated ball coordinates with heading-rotated skeleton coordinates, but **we need to account for the rotation**.

The rotation makes the humanoid always face the same direction in the observation space, even as it rotates in world space.

## Solution: Extract Heading Rotation

To get ball position in skeleton coordinates, we need to:
1. Extract the heading rotation from the skeleton data
2. Apply the same rotation to understand alignment
3. OR use the ball features directly if they're already aligned

## For SPL Data Mapping

This is critical for your data:
1. **If SPL uses root-relative**: Ball is at pelvis + ball_obs
2. **If SPL uses heading-relative**: Ball and skeleton are already aligned, just need to find the right features
3. **Need to check**: Does SPL rotate observations by heading direction?

## Next Steps

1. Find which features in our 337-dimensional vector are the ball observations
2. Identify if our data is the "reduced" version (337 vs 808 features)
3. Determine what reduction was applied (which features were kept)
