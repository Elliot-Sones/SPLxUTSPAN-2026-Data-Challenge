# SkillMimic Humanoid Joint Structure

Complete documentation of the SkillMimic humanoid skeleton based on `mocap_humanoid.xml`.

## Summary

- **Total Bodies**: 53
- **Position Features**: 165-323 (53 bodies × 3 coordinates = 159 features)
- **Total Features**: 337 (159 position + 178 other state variables)
- **Coordinate System**: X (horizontal), Y (vertical/up), Z (forward/depth)

## Body Hierarchy

```
Pelvis (root)
├── L_Hip → L_Knee → L_Ankle → L_Toe
├── R_Hip → R_Knee → R_Ankle → R_Toe
└── Torso → Spine → Spine2 → Chest
    ├── Neck → Head
    ├── L_Thorax → L_Shoulder → L_Elbow → L_Wrist
    │   ├── L_Index1 → L_Index2 → L_Index3
    │   ├── L_Middle1 → L_Middle2 → L_Middle3
    │   ├── L_Pinky1 → L_Pinky2 → L_Pinky3
    │   ├── L_Ring1 → L_Ring2 → L_Ring3
    │   └── L_Thumb1 → L_Thumb2 → L_Thumb3
    └── R_Thorax → R_Shoulder → R_Elbow → R_Wrist
        ├── R_Index1 → R_Index2 → R_Index3
        ├── R_Middle1 → R_Middle2 → R_Middle3
        ├── R_Pinky1 → R_Pinky2 → R_Pinky3
        ├── R_Ring1 → R_Ring2 → R_Ring3
        └── R_Thumb1 → R_Thumb2 → R_Thumb3
```

## Feature Index Mapping

### Core Body (Pelvis → Head): Features 165-209

| Body Part | Feature Indices | Notes |
|-----------|----------------|-------|
| Pelvis | 165, 166, 167 | Root joint, minimal movement |
| L_Hip | 168, 169, 170 | Left hip joint |
| L_Knee | 171, 172, 173 | Left knee |
| L_Ankle | 174, 175, 176 | Left ankle |
| L_Toe | 177, 178, 179 | Left foot |
| R_Hip | 180, 181, 182 | Right hip joint |
| R_Knee | 183, 184, 185 | Right knee |
| R_Ankle | 186, 187, 188 | Right ankle |
| R_Toe | 189, 190, 191 | Right foot |
| Torso | 192, 193, 194 | Lower torso |
| Spine | 195, 196, 197 | Mid spine |
| Spine2 | 198, 199, 200 | Upper spine |
| Chest | 201, 202, 203 | Chest/shoulders |
| Neck | 204, 205, 206 | Neck |
| Head | 207, 208, 209 | Head |

### Left Arm: Features 210-221

| Body Part | Feature Indices |
|-----------|----------------|
| L_Thorax | 210, 211, 212 |
| L_Shoulder | 213, 214, 215 |
| L_Elbow | 216, 217, 218 |
| L_Wrist | 219, 220, 221 |

### Left Hand Fingers: Features 222-266

| Finger | Segment 1 | Segment 2 | Segment 3 |
|--------|-----------|-----------|-----------|
| Index | 222-224 | 225-227 | 228-230 |
| Middle | 231-233 | 234-236 | 237-239 |
| Pinky | 240-242 | 243-245 | 246-248 |
| Ring | 249-251 | 252-254 | 255-257 |
| Thumb | 258-260 | 261-263 | 264-266 |

### Right Arm: Features 267-278

| Body Part | Feature Indices |
|-----------|----------------|
| R_Thorax | 267, 268, 269 |
| R_Shoulder | 270, 271, 272 |
| R_Elbow | 273, 274, 275 |
| R_Wrist | 276, 277, 278 |

### Right Hand Fingers: Features 279-323

| Finger | Segment 1 | Segment 2 | Segment 3 |
|--------|-----------|-----------|-----------|
| Index | 279-281 | 282-284 | 285-287 |
| Middle | 288-290 | 291-293 | 294-296 |
| Pinky | 297-299 | 300-302 | 303-305 |
| Ring | 306-308 | 309-311 | 312-314 |
| Thumb | 315-317 | 318-320 | 321-323 |

## Other Important Features

### Ball Position
- **X coordinate**: Feature 325
- **Y coordinate**: Feature 119
- **Z coordinate**: Feature 326
- **Contact flag**: Feature 336 (1 = in hand, 0 = released)

### Remaining Features
Features 0-164 and 324, 327-335, 337+ likely contain:
- Joint velocities
- Joint rotations (quaternions or euler angles)
- Root velocity/angular velocity
- Previous positions/velocities
- Other state information

## Measured Dimensions (Frame 0)

### Limb Lengths
- Upper arm (shoulder → elbow): 0.2525 units
- Forearm (elbow → wrist): 0.2439 units
- Thigh (hip → knee): 0.4619 units
- Shin (knee → ankle): 0.3653 units

### Position Variability (across 104 frames)
- Pelvis: x=0.0085, y=0.0739, z=0.1604 (root is relatively stable)
- Wrists/hands: Much higher variability (active shooting motion)

### Symmetry (Left vs Right at Frame 0)
- Wrists Y difference: 0.0131 (excellent symmetry)
- Wrists Z difference: 0.1128 (shooting pose - not symmetric)
- Knees Y difference: 0.0060 (excellent)
- Ankles Y difference: 0.0358 (good)

## Validation Results

The skeleton structure was validated and confirmed:
- ✓ Anatomically correct hierarchy
- ✓ Symmetric left/right limbs
- ✓ Reasonable limb lengths (human-like proportions)
- ✓ Proper movement patterns (shooting motion visible)
- ✓ Ball trajectory originates from hand position

## Visualization Color Coding

- **Blue (Torso/Spine)**: Pelvis, Torso, Spine, Spine2, Chest
- **Gold (Head)**: Neck, Head
- **Green (Legs)**: Hip, Knee, Ankle, Toe
- **Red (Arms)**: Thorax, Shoulder, Elbow, Wrist
- **Orange (Hands)**: All finger segments

## Source Files

- **MJCF Definition**: `mocap_humanoid.xml` from SkillMimic repository
- **Joint Mapper**: `visualisation/skillmimic_joint_mapper.py`
- **Joint Map JSON**: `visualisation/skillmimic_joint_map.json`
- **Visualization**: `visualisation/skillmimic_3d_viewer_fixed.py`

## References

- SkillMimic Paper: https://arxiv.org/abs/2408.15270
- SkillMimic GitHub: https://github.com/wyhuai/SkillMimic
- MJCF File: https://github.com/wyhuai/SkillMimic/blob/main/skillmimic/data/assets/mjcf/mocap_humanoid.xml
- Isaac Gym Documentation: https://developer.nvidia.com/isaac-gym

## Previous Implementation (Broken)

The original `skillmimic_3d_viewer_with_skeleton.py` only extracted 20 joints:
- Features 51-59, 108-116, 220-264
- This missed: complete torso, all legs, head, most of arms
- Connections were nonsensical (treated finger joints as entire body)

**Fix**: Now extracts all 53 bodies (features 165-323) with proper anatomical connections.
