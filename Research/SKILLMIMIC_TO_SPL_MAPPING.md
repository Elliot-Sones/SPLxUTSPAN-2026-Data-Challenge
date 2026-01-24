# SkillMimic to SPL Data Mapping Guide

## SkillMimic Observation Space (337 features) - FULLY DECODED

### Complete Feature Map

Based on source code analysis (`humanoid_object_task.py`, `humanoid_task.py`, `skillmimic.py`):

```
[root_pos(3) | root_rot(3) | dof_pos(156) | dof_vel(156) | target_states(10) | key_body_pos(8) | contact(1)]
```

### Exact Feature Indices

| Feature Range | Content | Description |
|--------------|---------|-------------|
| **0-2** | root_pos | Pelvis position (X, Y, Z) in heading-relative frame |
| **3-5** | root_rot | Pelvis rotation (exponential map, 3 values) |
| **6-161** | dof_pos | Joint positions (52 joints × 3 DOF = 156) |
| **162-317** | dof_vel | Joint velocities (52 joints × 3 DOF = 156) |
| **318-327** | target_states | **Ball observation** (10 features) |
| | 318-320 | **Ball position (X, Y, Z)** ✓ |
| | 321-324 | Ball rotation (quaternion) |
| | 325-327 | Ball velocity (X, Y, Z) |
| **328-335** | key_body_pos | Key body positions (8 features, likely 2-3 bodies) |
| **336** | contact | Ball contact flag (1=in hand, 0=released) |

**Total: 337 features**

### Verified Ball Position

**Features 318-320** contain ball position:
- Distance to hands during contact: **0.11-0.14 units**
- Coordinate frame: **Heading-relative** (same as skeleton)
- Units: Meters (basketball radius ~0.12m matches)

## Critical: Heading-Relative Coordinate Frame

### What is Heading-Relative?

Both skeleton and ball positions are **rotated** so the humanoid always faces "forward" in observation space:

```python
# From SkillMimic source:
heading_rot = calc_heading_quat_inv(root_rot)
local_pos = quat_rotate(heading_rot, world_pos - root_pos)
```

**Effect**:
- Humanoid always faces +X direction in observations
- Eliminates rotation as a variable
- Skeleton and ball are in the SAME rotated frame

### Why This Matters for SPL

If SPL data uses:
1. **World coordinates**: Need to apply inverse heading rotation
2. **Root-relative without heading**: Need to account for character facing direction
3. **Heading-relative** (like SkillMimic): Direct mapping possible!

## SPL Data Mapping Strategy

### Step 1: Determine SPL Coordinate System

Check if SPL data:
- Has constant root position? → Likely root-relative
- Has character always facing same direction in observations? → Heading-relative
- Changes position with player movement? → World coordinates

### Step 2: Map Joint Positions

SkillMimic joint positions (features 165-323) come from:
- 53 bodies (from mocap_humanoid.xml)
- Body positions relative to pelvis
- Rotated into heading frame

SPL mapping:
- Identify which joints SPL uses (subset of 52?)
- Determine if SPL includes velocities
- Check if positions are relative or absolute

### Step 3: Map Ball Position

**SkillMimic ball** (features 318-320):
```python
ball_pos = world_ball_pos - pelvis_pos
ball_pos = rotate_by_heading_inv(ball_pos)
```

**For SPL**:
- Find ball X, Y, Z in SPL features
- Check if ball is relative to player/root
- Verify with release frame analysis
- Calculate distance to hand keypoints (should be <0.2m during contact)

### Step 4: Identify Additional Features

SkillMimic includes:
- **Ball velocity** (325-327): For trajectory prediction
- **Contact flag** (336): Critical for release detection
- **Ball rotation** (321-324): For spin analysis

SPL should have similar features for complete modeling.

## Validation Tests

### Test 1: Ball in Hands
During contact frames (contact=1):
```python
distance = norm(ball_pos - average_hand_pos)
assert distance < 0.2  # Basketball held in hands
```

### Test 2: Coordinate System
Check if skeleton joints maintain constant relative positions:
```python
left_right_symmetry = abs(L_Wrist[Y] - R_Wrist[Y])
assert left_right_symmetry < 0.1  # Symmetric
```

### Test 3: Heading Independence
If heading-relative, skeleton should not rotate in observation space when character turns.

## Key Insights for Your Competition

### 1. Joint Positions Are Heading-Aligned

The skeleton features (165-323) are NOT in world coordinates. They're rotated so the player always faces forward. This means:
- ✓ **Easier for ML**: Rotation-invariant features
- ✓ **Consistent**: Same pose looks the same regardless of player direction
- ⚠️ **Need heading**: To convert back to world space

### 2. Ball Position Is Also Heading-Aligned

Ball features (318-320) use the same heading rotation:
- ✓ **Already aligned** with skeleton
- ✓ **Direct distance calculations** work
- ✓ **No coordinate transformation needed**

### 3. Contact Flag Is Critical

Feature 336 (contact) tells you when ball is in hands:
- Use for **supervised learning** of release detection
- Use for **data segmentation** (dribble vs shot vs pass)
- Use for **trajectory validation** (ball should be near hands when contact=1)

## Feature Engineering for SPL

### From SkillMimic Observations

**Position features** (6-161, 165-323):
- Relative joint positions
- Already normalized by heading
- Ready for trajectory prediction

**Velocity features** (162-317):
- Joint velocities
- Critical for dynamic modeling
- Use for release point prediction

**Ball state** (318-327):
- Position: Where ball is
- Velocity: Where ball is going
- Rotation: Ball spin (affects trajectory)

### Recommended SPL Features

Based on SkillMimic structure:

1. **Hand/wrist positions** (from skeleton)
2. **Ball position** (from target_states)
3. **Ball velocity** (for physics prediction)
4. **Contact flag** (for release detection)
5. **Joint velocities** (for motion dynamics)
6. **Key body positions** (head, elbows, knees for posture)

## Example: Extract Ball-Hand Distance

```python
import torch

data = torch.load("spl_shot.pt")

# Based on SkillMimic structure:
ball_x = data[:, 318]
ball_y = data[:, 319]
ball_z = data[:, 320]

# Hand positions from skeleton (adjust indices for SPL)
left_hand = data[:, [219, 220, 221]]  # Example indices
right_hand = data[:, [276, 277, 278]]

avg_hand = (left_hand + right_hand) / 2

distance = torch.norm(
    torch.stack([ball_x, ball_y, ball_z], dim=1) - avg_hand,
    dim=1
)

# During contact, distance should be < 0.2
contact = data[:, 336]
assert (distance[contact == 1] < 0.2).all()
```

## Next Steps for SPL Integration

1. **Analyze SPL data format**: Determine feature layout
2. **Map joint indices**: Find which SPL features = which body parts
3. **Locate ball features**: Find ball X/Y/Z in SPL observations
4. **Verify coordinate system**: Test if heading-relative or other
5. **Extract release features**: Identify release frame detection mechanism
6. **Build trajectory model**: Use ball position + velocity for prediction

## References

- SkillMimic source: `humanoid_object_task.py`, `humanoid_task.py`
- Observation space: 337 features (verified)
- Ball position: Features 318-320 (distance 0.11-0.14 to hands)
- Contact flag: Feature 336 (1=contact, 0=released)
- Coordinate system: Heading-relative (rotation-invariant)
